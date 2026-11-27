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

import download_models  # noqa: E402
import probe_hardware  # noqa: E402
import serve  # noqa: E402
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
        self.assertEqual(plan["threads"], 6)
        self.assertEqual(plan["ubatch_size"], 256)
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
        self.assertEqual(plan["batch_size"], 2048)
        self.assertEqual(plan["ubatch_size"], 512)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
