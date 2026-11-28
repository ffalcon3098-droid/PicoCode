#!/usr/bin/env python3
"""
PicoCode Automated Test Suite.

Validates:
  1. Registry schema, Apache-2.0 licenses, and download profiles
  2. Hardware-aware MoE & dense command generation across 4GB/6GB/8GB/24GB VRAM (Vulkan & CUDA)
  3. Live-VRAM WDDM thrashing protection (context cap when free VRAM < 3.6 GiB)
  4. Hardware-aware model recommender (16 GB vs 32 GB RAM)
  5. Web Studio HTTP endpoints (/, /api/status, /api/models) without requiring GPU inference

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
import pathlib
import sys
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import agent_cli  # noqa: E402
import download_models  # noqa: E402
import probe_hardware  # noqa: E402
import serve  # noqa: E402
import tui  # noqa: E402
import web_ui  # noqa: E402


class TestPicoCodeSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reg = serve.load_registry()

    def test_01_registry_integrity(self) -> None:
        self.assertIn("engine_releases", self.reg)
        self.assertIn("win_vulkan_x64", self.reg["engine_releases"])
        self.assertIn("win_cuda_12_4_x64", self.reg["engine_releases"])
        for fid, item in self.reg["files"].items():
            self.assertEqual(item["license"], "apache-2.0", f"Non-Apache file: {fid}")
            self.assertGreater(int(item["expected_bytes"]), 100_000_000)
            self.assertTrue(item["url"].startswith("https://"))
        for pid, prof in self.reg["download_profiles"].items():
            for fid in prof["required"]:
                self.assertIn(fid, self.reg["files"], f"Profile {pid} references missing file {fid}")

    def test_02_inkling2_mini_6gb_vulkan_moe_offload(self) -> None:
        hw = {
            "gpu_name": "AMD Radeon RX 5600 XT",
            "backend": "vulkan",
            "vram_total_gib": 6.0,
            "live_free_ded_gib": 5.2,
            "cpu_threads": 12,
            "ram_total_gib": 32.0,
        }
        cmd, plan = serve.build_server_command(
            "inkling2-mini", hw, self.reg, server_exe="llama-server.exe", allow_missing_files=True
        )
        self.assertEqual(plan["moe_cpu_override"], "--cpu-moe")
        self.assertIn("--cpu-moe", cmd)
        self.assertIn("--no-op-offload", cmd)
        self.assertNotIn("--no-host", cmd)  # Never combine --no-host with DSpark
        self.assertIn("-ctk", cmd)
        self.assertIn("q8_0", cmd)
        self.assertIn("--jinja", cmd)
        self.assertIn("--spec-type", cmd)
        self.assertIn("draft-dspark", cmd)
        self.assertIn("--spec-draft-n-max", cmd)
        self.assertIn("-md", cmd)
        self.assertIn("--mmproj", cmd)
        self.assertIn("--no-reasoning-preserve", cmd)
        self.assertEqual(plan["threads"], 6)
        self.assertEqual(plan["batch_size"], 1024)
        self.assertEqual(plan["ubatch_size"], 512)
        self.assertEqual(plan["ctx_size"], 16384)
        self.assertIsNone(plan["warning"])

    def test_03_live_vram_guard_caps_context(self) -> None:
        hw = {
            "gpu_name": "AMD Radeon RX 5600 XT",
            "backend": "vulkan",
            "vram_total_gib": 6.0,
            "live_free_ded_gib": 2.9,  # e.g. Chrome + game eating VRAM
            "cpu_threads": 12,
            "ram_total_gib": 32.0,
        }
        _, plan = serve.build_server_command(
            "inkling2-mini", hw, self.reg, server_exe="llama-server.exe", allow_missing_files=True
        )
        self.assertEqual(plan["ctx_size"], 8192)
        self.assertIsNotNone(plan["warning"])

    def test_04_4gb_vram_disables_draft_and_vision_and_caps_dense_ngl(self) -> None:
        hw = {
            "gpu_name": "GTX 1650",
            "backend": "cuda",
            "vram_total_gib": 4.0,
            "cpu_threads": 8,
            "ram_total_gib": 32.0,
        }
        cmd_moe, plan_moe = serve.build_server_command(
            "inkling2-mini", hw, self.reg, server_exe="llama-server.exe", allow_missing_files=True
        )
        self.assertEqual(plan_moe["ctx_size"], 8192)
        self.assertIsNone(plan_moe["draft_model"])
        self.assertIsNone(plan_moe["mmproj_model"])
        self.assertNotIn("-md", cmd_moe)

        _, plan_dense = serve.build_server_command(
            "qwen25-coder-7b", hw, self.reg, server_exe="llama-server.exe", allow_missing_files=True
        )
        self.assertEqual(plan_dense["n_gpu_layers"], 18)

    def test_05_24gb_vram_runs_moe_100pct_gpu(self) -> None:
        hw = {
            "gpu_name": "RTX 4090",
            "backend": "cuda",
            "vram_total_gib": 24.0,
            "cpu_threads": 16,
            "ram_total_gib": 64.0,
        }
        cmd, plan = serve.build_server_command(
            "inkling2-mini", hw, self.reg, server_exe="llama-server.exe", allow_missing_files=True
        )
        self.assertIsNone(plan["moe_cpu_override"])
        self.assertNotIn("--cpu-moe", cmd)
        self.assertNotIn("-ot", cmd)
        self.assertEqual(plan["ctx_size"], 32768)
        self.assertEqual(plan["batch_size"], 1024)

    def test_06_dense_7b_6gb_full_gpu(self) -> None:
        hw = {
            "gpu_name": "RX 5600 XT",
            "backend": "vulkan",
            "vram_total_gib": 6.0,
            "cpu_threads": 12,
            "ram_total_gib": 16.0,
        }
        _, plan = serve.build_server_command(
            "qwen25-coder-7b", hw, self.reg, server_exe="llama-server.exe", allow_missing_files=True
        )
        self.assertEqual(plan["n_gpu_layers"], 99)
        self.assertIsNone(plan["moe_cpu_override"])

    def test_06b_qwen3_coder_30b_adaptive_hybrid_moe_and_ngram(self) -> None:
        hw_free = {
            "gpu_name": "AMD Radeon RX 5600 XT",
            "backend": "vulkan",
            "vram_total_gib": 6.0,
            "live_free_ded_gib": 4.7,
            "cpu_threads": 12,
            "ram_total_gib": 32.0,
        }
        cmd, plan = serve.build_server_command(
            "qwen3-coder-30b", hw_free, self.reg, server_exe="llama-server.exe", allow_missing_files=True
        )
        self.assertGreaterEqual(plan["gpu_moe_layers"], 4)
        self.assertIn("--n-cpu-moe", cmd)
        self.assertIn("--spec-type", cmd)
        self.assertIn("ngram-mod", cmd)
        self.assertIn("--poll", cmd)
        self.assertIn("--cache-reuse", cmd)
        self.assertIn("--load-mode", cmd)
        self.assertEqual(cmd[cmd.index("--load-mode") + 1], "none")
        self.assertEqual(plan["batch_size"], 1024)
        self.assertEqual(plan["ubatch_size"], 256)

        # Under heavy desktop VRAM pressure, hybrid MoE gracefully falls back to --cpu-moe without spilling
        hw_tight = dict(hw_free, live_free_ded_gib=2.0)
        cmd_tight, plan_tight = serve.build_server_command(
            "qwen3-coder-30b", hw_tight, self.reg, server_exe="llama-server.exe", allow_missing_files=True
        )
        self.assertEqual(plan_tight["gpu_moe_layers"], 0)
        self.assertIn("--cpu-moe", cmd_tight)
        self.assertIn("ngram-mod", cmd_tight)

    def test_07_recommender_respects_ram_and_disk(self) -> None:
        p32, _ = download_models.recommend_profile({"ram_total_gib": 32.0, "disk_free_gib": 40.0, "vram_total_gib": 6.0})
        self.assertEqual(p32, "inkling2-mini")
        p16, _ = download_models.recommend_profile({"ram_total_gib": 16.0, "disk_free_gib": 40.0, "vram_total_gib": 6.0})
        self.assertEqual(p16, "starter-7b")

    def test_08_web_studio_http_endpoints(self) -> None:
        mgr = web_ui.ServerManager(api_host="127.0.0.1", api_port=18080)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), web_ui.make_handler(mgr))
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
                self.assertEqual(r.status, 200)
                html = r.read().decode("utf-8")
                self.assertIn("PicoCode Studio", html)
                self.assertIn("http://127.0.0.1:8080/v1", html)

            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=5) as r:
                st = json.loads(r.read().decode("utf-8"))
                self.assertEqual(st["state"], "stopped")
                self.assertIn("hardware", st)

            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/models", timeout=5) as r:
                mods = json.loads(r.read().decode("utf-8"))
                ids = {m["id"] for m in mods}
                self.assertIn("inkling2-mini", ids)
                self.assertIn("qwen3-coder-30b", ids)
                self.assertIn("qwen25-coder-7b", ids)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_09_terminal_agent_cli_tools_undo_and_stream(self) -> None:
        import tempfile
        from http.server import BaseHTTPRequestHandler
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            (root / "calc.py").write_text("def mul(a, b):\n    return a + b\n", encoding="utf-8")
            ws = agent_cli.AgentWorkspace(root)

            # 1) Path traversal blocked
            with self.assertRaises(PermissionError):
                ws.safe_path("../../Windows/win.ini")

            # 2) AST repo_map + instant syntax error feedback + /undo rollback
            rmap, _ = agent_cli.execute_cli_tool(ws, "repo_map", {})
            self.assertIn("calc.py", rmap)
            self.assertIn("def mul()", rmap)

            bad_txt, bad_chg = agent_cli.execute_cli_tool(
                ws, "edit_file",
                {"path": "calc.py", "target": "return a + b", "replacement": "return (a *"},
            )
            self.assertIn("[SYNTAX ERROR]", bad_txt)
            self.assertIsNotNone(bad_chg)
            self.assertEqual(ws.undo_turn([bad_chg["undo_id"]]), 1)
            self.assertIn("return a + b", (root / "calc.py").read_text(encoding="utf-8"))

            # 3) Destructive command safety guard & @file context extraction + TUI helpers
            self.assertIsNotNone(agent_cli.is_dangerous_command("rm -rf /"))
            blocked, _ = agent_cli.execute_cli_tool(ws, "run_command", {"command": "git reset --hard HEAD"}, auto_approve=True)
            self.assertIn("Blocked destructive command", blocked)

            at_ctx = agent_cli.extract_at_files(ws, "incele @calc.py ve @../../Windows/win.ini")
            self.assertIn("def mul", at_ctx)
            self.assertNotIn("win.ini", at_ctx)
            self.assertEqual(tui.format_tool_arg_preview("read_file", {"path": "calc.py", "start_line": 1, "end_line": 20}), "calc.py · L1-20")
            ui = tui.TerminalUI()
            self.assertIsNotNone(ui)
            self.assertEqual(agent_cli.resolve_model_alias("inkling"), "inkling2-mini")
            self.assertEqual(agent_cli.resolve_model_alias("qwen"), "qwen3-coder-30b")
            self.assertEqual(agent_cli.resolve_model_ctx_limit("inkling2-mini"), 16384)
            self.assertEqual(agent_cli.resolve_model_ctx_limit("qwen3-coder-30b"), 32768)
            mods = agent_cli.list_available_models()
            self.assertTrue(any(m["id"] == "inkling2-mini" for m in mods))
            self.assertTrue(any(m["id"] == "qwen3-coder-30b" for m in mods))

            # 4) Mock OpenAI SSE streaming turn with tool call + undo_turn
            call_n = {"n": 0}

            class MockAgentServer(BaseHTTPRequestHandler):
                def log_message(self, fmt, *args):
                    pass

                def do_POST(self):
                    n = int(self.headers.get("Content-Length", "0"))
                    _ = json.loads(self.rfile.read(n).decode("utf-8"))
                    call_n["n"] += 1
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()

                    def send(delta_obj):
                        self.wfile.write(f"data: {json.dumps({'choices': [{'delta': delta_obj}]})}\n\n".encode("utf-8"))

                    if call_n["n"] == 1:
                        send({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "edit_file", "arguments": json.dumps({"path": "calc.py", "target": "return a + b", "replacement": "return a * b"})}}]})
                    else:
                        send({"content": "mul fonksiyonu duzeltildi."})
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()

            srv = ThreadingHTTPServer(("127.0.0.1", 0), MockAgentServer)
            srv.daemon_threads = True
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            try:
                conv = [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "fix mul"},
                ]
                res = agent_cli.run_agent_turn(
                    f"http://127.0.0.1:{srv.server_address[1]}/v1",
                    "qwen3-coder-30b", ws, conv, auto_approve=True, quiet=True,
                )
                self.assertEqual(res["tools_called"], ["edit_file"])
                self.assertIn("duzeltildi", res["answer"])
                self.assertIn("return a * b", (root / "calc.py").read_text(encoding="utf-8"))
                self.assertEqual(ws.undo_turn(), 1)
                self.assertIn("return a + b", (root / "calc.py").read_text(encoding="utf-8"))
            finally:
                srv.shutdown()
                srv.server_close()

    def test_10_v050_trust_store_project_discovery_and_pinned_files(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            p1 = root / "ProjAlpha"
            p2 = root / "ProjBeta"
            p1.mkdir(); p2.mkdir()
            (p1 / ".git").mkdir()
            (p1 / "main.py").write_text("SECRET_VAL = 123\n", encoding="utf-8")

            # 1) TrustStore persistence (trust / is_trusted / untrust)
            ts = agent_cli.TrustStore(root / "trust.json")
            self.assertFalse(ts.is_trusted(p1))
            ts.trust(p1)
            self.assertTrue(ts.is_trusted(p1))
            self.assertFalse(ts.is_trusted(p2))
            self.assertEqual(len(ts.list_trusted()), 1)
            self.assertTrue(ts.untrust(p1))
            self.assertFalse(ts.is_trusted(p1))

            # 2) Project discovery + resolution
            ts.trust(p1)
            projs = agent_cli.discover_projects(p1, ts)
            by_name = {p["name"]: p for p in projs}
            self.assertIn("ProjAlpha", by_name)
            self.assertIn("ProjBeta", by_name)
            self.assertTrue(by_name["ProjAlpha"]["trusted"])
            self.assertTrue(by_name["ProjAlpha"]["is_git"])
            self.assertFalse(by_name["ProjBeta"]["trusted"])
            self.assertEqual(agent_cli._resolve_project_target("ProjBeta", p1, projs), p2.resolve())

            # 3) Pinned files injected into system prompt
            ws = agent_cli.AgentWorkspace(p1)
            sys_p = agent_cli._build_system_prompt(ws, plan_only=False, pinned_files=["main.py"])
            self.assertIn("Pinned file `main.py`", sys_p)
            self.assertIn("SECRET_VAL = 123", sys_p)

    def test_11_v060_audit_hardening_and_model_optimizations(self) -> None:
        import urllib.error

        # 1) Vulkan DEVICE_LOCAL heap parser handles hex parentheticals and ignores System RAM heaps
        sample_vk = """
        memoryHeaps[0]:
            size   = 5351931904 (0x13f000000) (4.98 GiB)
            flags  = count = 1
                MEMORY_HEAP_DEVICE_LOCAL_BIT
        memoryHeaps[1]:
            size   = 17123573760 (0x3fca50000) (15.95 GiB)
            flags  = None
        memoryHeaps[2]:
            size   = 268435456 (0x10000000) (256.00 MiB)
            flags  = count = 1
                MEMORY_HEAP_DEVICE_LOCAL_BIT
        """
        self.assertEqual(probe_hardware.parse_vulkan_device_local_gib(sample_vk), 6.0)

        # 2) Turkish 'h' (Hayir) in confirm_command MUST deny, NEVER auto-approve!
        ui = tui.TerminalUI()
        self.assertEqual(ui.confirm_command("echo hi", input_fn=lambda _: "h"), "n")
        self.assertEqual(ui.confirm_command("echo hi", input_fn=lambda _: "hayir"), "n")
        self.assertEqual(ui.confirm_command("echo hi", input_fn=lambda _: "t"), "a")
        self.assertEqual(ui.confirm_command("echo hi", input_fn=lambda _: "e"), "y")

        # 3) Epoch KV-cache-friendly compaction does NOT mutate messages below high-water mark
        conv = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        for i in range(9):
            conv.append({"role": "tool", "tool_call_id": f"t{i}", "content": "a" * 600})
        kept = agent_cli.compact_messages(conv, ctx_limit=32768)
        self.assertEqual(kept[2]["content"], conv[2]["content"])  # Prefix untouched for 100% KV hit!
        compacted = agent_cli.compact_messages(conv, ctx_limit=1000)  # Above 72% threshold
        self.assertIn("[Compacted:", compacted[2]["content"])

        # 4) Hardened dangerous command detector catches separated flags, aliases, git push -f, -enc
        for bad in ("rm -r -f .", "ri -Rec -Fo .", "git push origin main -f", "powershell -enc AAA", "py -c 'import shutil; shutil.rmtree(\".\")'"):
            self.assertIsNotNone(agent_cli.is_dangerous_command(bad), f"Failed to block: {bad}")

        # 5) serve.py passes -fit off, registry sampling defaults, -ctkd q8_0, and f16 KV scale=2.0
        hw = {"gpu_name": "RX 5600 XT", "backend": "vulkan", "vram_total_gib": 6.0, "live_free_ded_gib": 4.9, "cpu_threads": 12, "ram_total_gib": 32.0}
        cmd_i, _ = serve.build_server_command("inkling2-mini", hw, self.reg, server_exe="llama-server.exe", allow_missing_files=True)
        self.assertIn("-fit", cmd_i)
        self.assertIn("-ctkd", cmd_i)
        self.assertIn("--reasoning-budget", cmd_i)
        self.assertIn("--top-p", cmd_i)
        self.assertNotIn("--cache-reuse", cmd_i)  # Skipped on looped SWA/ShortConv

        _, plan_f16 = serve.build_server_command("qwen3-coder-30b", hw, self.reg, server_exe="llama-server.exe", kv_type="f16", allow_missing_files=True)
        _, plan_q8 = serve.build_server_command("qwen3-coder-30b", hw, self.reg, server_exe="llama-server.exe", kv_type="q8_0", allow_missing_files=True)
        self.assertLessEqual(plan_f16["gpu_moe_layers"], plan_q8["gpu_moe_layers"])

        # 6) Web UI CSRF + Host shield blocks unauthenticated POST
        mgr = web_ui.ServerManager(api_host="127.0.0.1", api_port=18080)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), web_ui.make_handler(mgr, csrf_token="tok123"))
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(urllib.request.Request(
                    f"http://127.0.0.1:{port}/api/stop",
                    data=b"{}", headers={"Content-Type": "application/json"}, method="POST",
                ), timeout=5)
            self.assertEqual(ctx.exception.code, 403)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_12_multi_gpu_legacy_cuda_fallback_and_8gb_guard(self) -> None:
        # 1) Multi-GPU rig (3x RTX 3060 12 GB = 36 GiB total) -> aggregates VRAM & runs 30B MoE 100% GPU
        smi_3x3060 = "\n".join([
            "NVIDIA GeForce RTX 3060, 12288, 11800, 8.6, 566.36",
            "NVIDIA GeForce RTX 3060, 12288, 12000, 8.6, 566.36",
            "NVIDIA GeForce RTX 3060, 12288, 12000, 8.6, 566.36",
        ])
        hw_3x = probe_hardware.parse_nvidia_smi_rows(smi_3x3060, os_name="Windows")
        assert hw_3x is not None
        self.assertEqual(hw_3x["gpu_count"], 3)
        self.assertEqual(hw_3x["gpu_name"], "3x NVIDIA GeForce RTX 3060")
        self.assertEqual(hw_3x["vram_total_gib"], 36.0)
        self.assertEqual(hw_3x["backend"], "cuda")
        hw_3x.update({"cpu_threads": 16, "ram_total_gib": 32.0})
        cmd_3x, plan_3x = serve.build_server_command(
            "qwen3-coder-30b", hw_3x, self.reg, server_exe="llama-server.exe", allow_missing_files=True
        )
        self.assertIsNone(plan_3x["moe_cpu_override"])
        self.assertNotIn("--cpu-moe", cmd_3x)
        self.assertNotIn("--n-cpu-moe", cmd_3x)
        self.assertEqual(plan_3x["ctx_size"], 65536)

        # 2) Maxwell Quadro K1200 (sm_50) -> automatic Vulkan fallback (avoids CUDA 12.4 kernel crash)
        hw_k1200 = probe_hardware.parse_nvidia_smi_rows(
            "Quadro K1200, 4096, 3600, 5.0, 552.22", os_name="Windows"
        )
        assert hw_k1200 is not None
        self.assertEqual(hw_k1200["backend"], "vulkan")
        self.assertIn("compute capability 5.0", hw_k1200.get("backend_note", ""))

        # 3) Windows driver < 551.61 -> automatic Vulkan fallback
        hw_old_drv = probe_hardware.parse_nvidia_smi_rows(
            "NVIDIA GeForce RTX 3060, 12288, 11500, 8.6, 536.99", os_name="Windows"
        )
        assert hw_old_drv is not None
        self.assertEqual(hw_old_drv["backend"], "vulkan")
        self.assertIn("551.61", hw_old_drv.get("backend_note", ""))

        # 4) Tesla P40 24 GB (Pascal sm_61) -> stays CUDA (TCC compatible), disables FA automatically, 100% GPU
        hw_p40 = probe_hardware.parse_nvidia_smi_rows(
            "Tesla P40, 24576, 24000, 6.1, 552.22", os_name="Windows"
        )
        assert hw_p40 is not None
        self.assertEqual(hw_p40["backend"], "cuda")
        self.assertEqual(hw_p40["compute_cap"], 6.1)
        hw_p40.update({"cpu_threads": 16, "ram_total_gib": 32.0})
        cmd_p40, plan_p40 = serve.build_server_command(
            "qwen3-coder-30b", hw_p40, self.reg, server_exe="llama-server.exe", allow_missing_files=True
        )
        self.assertEqual(plan_p40["flash_attn"], "off")
        self.assertNotIn("-ctv", cmd_p40)
        self.assertIsNone(plan_p40["moe_cpu_override"])

        # 5) 8 GB GPU defaults to 32K (not 64K) and caps to 16K when live free VRAM < 4.5 GiB
        hw_8gb_clean = {
            "gpu_name": "RTX 4060", "backend": "cuda", "vram_total_gib": 8.0,
            "live_free_ded_gib": 6.8, "cpu_threads": 12, "ram_total_gib": 32.0,
        }
        _, plan_8_clean = serve.build_server_command(
            "qwen3-coder-30b", hw_8gb_clean, self.reg, server_exe="llama-server.exe", allow_missing_files=True
        )
        self.assertEqual(plan_8_clean["ctx_size"], 32768)

        hw_8gb_busy = dict(hw_8gb_clean, live_free_ded_gib=3.9)
        _, plan_8_busy = serve.build_server_command(
            "qwen3-coder-30b", hw_8gb_busy, self.reg, server_exe="llama-server.exe", allow_missing_files=True
        )
        self.assertEqual(plan_8_busy["ctx_size"], 16384)
        self.assertIsNotNone(plan_8_busy["warning"])

        # 6) Low available RAM (< 18.5 GiB) skips `--load-mode none` to prevent malloc OOM crash
        hw_low_avail = dict(hw_8gb_clean, ram_total_gib=32.0, ram_avail_gib=11.0)
        cmd_low_avail, _ = serve.build_server_command(
            "qwen3-coder-30b", hw_low_avail, self.reg, server_exe="llama-server.exe", allow_missing_files=True
        )
        self.assertNotIn("--load-mode", cmd_low_avail)


if __name__ == "__main__":
    unittest.main(verbosity=2)
