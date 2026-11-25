#!/usr/bin/env python3
"""
PicoCode Hardware-Aware Local Coding LLM Server & Benchmark Runner.

Runs frontier MoE & dense coding models (Inkling 2 Mini, Qwen3-Coder-30B-A3B,
Qwen2.5-Coder-7B) smoothly on 4-6 GB VRAM + 32 GB RAM consumer PCs via official
llama.cpp (Vulkan or CUDA) with smart GPU+RAM MoE expert offloading, KV-cache
quantization, optional DSpark speculative decoding, and multimodal vision.

Exposes an OpenAI-compatible API at http://127.0.0.1:8080/v1 ready for
VS Code (Continue, Cline, Roo Code), OpenCode, and PicoCode Web Studio.

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "models_registry.json"
ENGINE_DIR = REPO_ROOT / "engine"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import probe_hardware  # noqa: E402


def configure_utf8() -> None:
    for name in ("stdout", "stderr"):
        st = getattr(sys, name, None)
        if st is not None and hasattr(st, "reconfigure"):
            try:
                st.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def load_registry(path: pathlib.Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(rel_or_abs: str) -> pathlib.Path:
    p = pathlib.Path(rel_or_abs)
    return p if p.is_absolute() else (REPO_ROOT / p)


def find_llama_server(preferred_backend: str = "vulkan") -> pathlib.Path | None:
    """Locate prebuilt llama-server.exe in engine/ or system PATH."""
    order = ["llama-cuda", "llama-vulkan"] if preferred_backend == "cuda" else ["llama-vulkan", "llama-cuda"]
    for sub in order:
        cand = ENGINE_DIR / sub / ("llama-server.exe" if os.name == "nt" else "llama-server")
        if cand.exists():
            return cand
    found = shutil.which("llama-server.exe") or shutil.which("llama-server")
    return pathlib.Path(found) if found else None


def resolve_model_file(model_cfg: dict[str, Any], reg: dict[str, Any], allow_missing: bool = False) -> pathlib.Path | None:
    files_map = reg["files"]
    primary = resolve_path(files_map[model_cfg["model_file"]]["path"])
    if primary.exists():
        return primary
    compact_key = model_cfg.get("compact_file")
    if compact_key and compact_key in files_map:
        compact = resolve_path(files_map[compact_key]["path"])
        if compact.exists():
            return compact
    return primary if allow_missing else None


def pick_installed_model(reg: dict[str, Any], requested: str | None = None) -> str:
    models = reg["models"]
    if requested and requested in models:
        return requested
    for mid in ("inkling2-mini", "qwen3-coder-30b", "qwen25-coder-7b"):
        if mid in models and resolve_model_file(models[mid], reg, allow_missing=False) is not None:
            return mid
    return "inkling2-mini"


def compute_optimal_threads(cpu_threads: int) -> int:
    """Pick physical-core count to maximize DDR4/DDR5 MoE bandwidth without SMT contention."""
    if cpu_threads <= 4:
        return max(2, cpu_threads - 1)
    if cpu_threads <= 8:
        return max(4, cpu_threads // 2)
    # Benchmarked on Ryzen 5 5500 (6C/12T): physical cores (6) beats 8/10 SMT threads by ~9%
    return max(6, cpu_threads // 2)


def build_server_command(
    model_id: str,
    hw: dict[str, Any],
    reg: dict[str, Any],
    *,
    server_exe: pathlib.Path | str | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    ctx_size: int | None = None,
    threads: int | None = None,
    flash_attn: str = "auto",
    use_draft: bool = True,
    use_vision: bool = True,
    n_gpu_layers: int | None = None,
    allow_missing_files: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    """
    Build hardware-tuned llama-server command line & human-readable tuning plan.
    Pure & testable across 4GB/6GB/8GB/16GB+ VRAM and Vulkan/CUDA profiles.
    """
    if model_id not in reg["models"]:
        raise ValueError(f"Unknown model id: {model_id}")

    mcfg = reg["models"][model_id]
    files_map = reg["files"]
    model_path = resolve_model_file(mcfg, reg, allow_missing=allow_missing_files)
    if model_path is None:
        expected = files_map[mcfg["model_file"]]["path"]
        raise FileNotFoundError(f"Model file not found: {expected}")

    exe_str = str(server_exe or find_llama_server(hw.get("backend", "vulkan")) or "llama-server")
    vram_gib = float(hw.get("vram_total_gib", 6.0))
    live_free = hw.get("live_free_ded_gib")
    cpu_threads = int(hw.get("cpu_threads", 8))
    backend = str(hw.get("backend", "vulkan")).lower()
    arch = mcfg.get("arch", "dense_small")
    defaults = mcfg.get("defaults", {})

    # 1. Determine context window based on VRAM class & live free VRAM headroom
    if ctx_size is not None:
        chosen_ctx = int(ctx_size)
    elif vram_gib <= 4.5:
        chosen_ctx = int(defaults.get("ctx_size_4gb", 8192))
    elif vram_gib <= 6.8:
        chosen_ctx = int(defaults.get("ctx_size_6gb", 16384))
    else:
        chosen_ctx = int(defaults.get("ctx_size_8gb", 32768))

    # Guard against WDDM Shared VRAM thrashing when browser/desktop eats >1.5 GiB
    live_warning = None
    if live_free is not None and vram_gib <= 6.8 and float(live_free) < 3.6:
        capped = min(chosen_ctx, 8192)
        if capped < chosen_ctx:
            live_warning = (
                f"Canli bos VRAM dusuk ({live_free:.2f} GiB) — WDDM tasmasini onlemek icin "
                f"baglam {chosen_ctx} -> {capped} dusuruldu."
            )
            chosen_ctx = capped

    # 2. GPU offloading & MoE routed-expert CPU routing
    ngl = n_gpu_layers
    moe_cpu_offload = False
    moe_override = None
    if arch in ("moe", "moe_looped"):
        # Full MoE weights (~15-16 GiB) fit on GPU only with 20+ GiB VRAM
        if vram_gib < 20.0:
            if mcfg.get("moe_cpu_mode", "cpu_moe") == "cpu_moe":
                moe_cpu_offload = True
            else:
                moe_override = mcfg.get("moe_cpu_regex", r"\.ffn_.*_exps\.=CPU")
            ngl = 99 if ngl is None else ngl
        else:
            ngl = 99 if ngl is None else ngl
    else:
        # Dense 7B model (~4.36 GiB)
        if ngl is None:
            ngl = 99 if vram_gib >= 5.5 else 22

    # 3. Flash attention & batching (RDNA1/Vulkan TDR safety)
    fa_mode = flash_attn.lower()
    if fa_mode == "auto":
        fa_mode = "on"

    # Keep ubatch moderate on Vulkan to prevent Windows 2s TDR timeout on long code prefill
    batch_size = 512 if (backend == "vulkan" or vram_gib <= 6.8) else 1024
    ubatch_size = 256 if (backend == "vulkan" or vram_gib <= 6.8) else 512
    chosen_threads = int(threads) if threads else compute_optimal_threads(cpu_threads)

    cmd = [
        exe_str,
        "-m", str(model_path),
        "--alias", model_id,
        "--host", host,
        "--port", str(port),
        "-np", "1",
        "-c", str(chosen_ctx),
        "-ngl", str(ngl),
        "-t", str(chosen_threads),
        "-tb", str(min(cpu_threads, max(chosen_threads, cpu_threads - 1))),
        "-b", str(batch_size),
        "-ub", str(ubatch_size),
        "-ctk", "q8_0",
        "-fa", fa_mode,
        "--jinja",
        "--metrics",
    ]

    if fa_mode != "off":
        cmd += ["-ctv", "q8_0"]

    if moe_cpu_offload:
        # --no-op-offload prevents Vulkan/CUDA from allocating a ~7.6 GiB compute scratchpad
        # for CPU MoE tensors (drops GPU VRAM from 9.28 GiB -> 2.55 GiB and speeds up pp 23x!).
        # --no-host avoids pinning 14.5 GiB in Vulkan_Host shared memory while keeping fast mmap loads.
        cmd += ["--cpu-moe", "--no-op-offload", "--no-host"]
    elif moe_override:
        cmd += ["-ot", moe_override, "--no-op-offload", "--no-host"]

    # 4. Optional DSpark / Draft model (only if present & enough VRAM)
    draft_active = None
    draft_key = mcfg.get("draft_file")
    if use_draft and draft_key and draft_key in files_map and vram_gib >= 5.0:
        dpath = resolve_path(files_map[draft_key]["path"])
        if dpath.exists() or allow_missing_files:
            spec_type = mcfg.get("draft_spec_type")
            if spec_type:
                cmd += ["--spec-type", str(spec_type)]
            cmd += ["-md", str(dpath), "-ngld", "99"]
            if mcfg.get("draft_n_max"):
                cmd += ["--spec-draft-n-max", str(mcfg["draft_n_max"])]
            draft_active = str(dpath)

    # 5. Optional Vision projector (mmproj)
    vision_active = None
    mm_key = mcfg.get("mmproj_file")
    if use_vision and mm_key and mm_key in files_map and vram_gib >= 5.0:
        mpath = resolve_path(files_map[mm_key]["path"])
        if mpath.exists() or allow_missing_files:
            cmd += ["--mmproj", str(mpath)]
            vision_active = str(mpath)

    plan = {
        "model_id": model_id,
        "label": mcfg["label"],
        "arch": arch,
        "model_path": str(model_path),
        "backend": backend,
        "gpu_name": hw.get("gpu_name", "Unknown"),
        "vram_total_gib": vram_gib,
        "live_free_ded_gib": live_free,
        "ctx_size": chosen_ctx,
        "n_gpu_layers": ngl,
        "moe_cpu_override": "--cpu-moe" if moe_cpu_offload else moe_override,
        "threads": chosen_threads,
        "batch_size": batch_size,
        "ubatch_size": ubatch_size,
        "flash_attn": fa_mode,
        "kv_cache": "q8_0",
        "draft_model": draft_active,
        "mmproj_model": vision_active,
        "api_base": f"http://{host}:{port}/v1",
        "warning": live_warning,
    }
    return cmd, plan


def wait_for_health(host: str, port: int, proc: subprocess.Popen[str], timeout_s: float = 180.0) -> bool:
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8", errors="replace"))
                    if data.get("status") == "ok":
                        return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def run_benchmark(cmd: list[str], plan: dict[str, Any], host: str, port: int, max_tokens: int = 192) -> int:
    """Launch llama-server, run a real coding prompt, measure tok/s + peak VRAM, and shut down."""
    print(f"\n[PicoCode Bench] Model: {plan['label']}")
    print(f"[PicoCode Bench] GPU: {plan['gpu_name']} ({plan['backend'].upper()}, {plan['vram_total_gib']:.1f} GiB) | ctx={plan['ctx_size']} | fa={plan['flash_attn']} | draft={'ON' if plan['draft_model'] else 'OFF'} | vision={'ON' if plan['mmproj_model'] else 'OFF'}")
    print(f"[PicoCode Bench] Komut: {' '.join(cmd)}\n")

    peak_ded = 0.0
    peak_shr = 0.0
    stop_mon = threading.Event()

    def _vram_monitor() -> None:
        nonlocal peak_ded, peak_shr
        while not stop_mon.is_set():
            d, s = probe_hardware.probe_live_vram_mib()
            if d is not None and d > peak_ded:
                peak_ded = d
            if s is not None and s > peak_shr:
                peak_shr = s
            stop_mon.wait(0.8)

    mon_thread = threading.Thread(target=_vram_monitor, daemon=True)
    mon_thread.start()

    t_start = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    log_lines: list[str] = []

    def _drain() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            log_lines.append(line.rstrip())
            if any(k in line for k in ("error", "failed", "loaded", "Vulkan", "offloaded", "draft", "mmproj", "listening")):
                print(f"  [llama] {line.rstrip()}")

    drain_thread = threading.Thread(target=_drain, daemon=True)
    drain_thread.start()

    try:
        if not wait_for_health(host, port, proc, timeout_s=240.0):
            print("\nHATA: llama-server hazir duruma gelemedi! Son log satirlari:", file=sys.stderr)
            for ln in log_lines[-25:]:
                print("  " + ln, file=sys.stderr)
            return 1

        load_s = time.monotonic() - t_start
        print(f"\n[PicoCode Bench] Sunucu hazir ({load_s:.1f}s). Kodlama promptu gonderiliyor...")

        payload = {
            "model": plan["model_id"],
            "messages": [
                {"role": "system", "content": "You are a concise, expert Python systems engineer. Output clean, bug-free code."},
                {"role": "user", "content": "Write a thread-safe Python LRU cache class with O(1) get and put using OrderedDict, plus a 3-line unit test. Keep it concise."},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        req = urllib.request.Request(
            f"http://{host}:{port}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t_req = time.monotonic()
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
        req_s = time.monotonic() - t_req

        timings = body.get("timings") or {}
        usage = body.get("usage") or {}
        content = ""
        choices = body.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            if reasoning and not content:
                content = f"[reasoning]\n{reasoning}"

        pp_tps = timings.get("prompt_per_second")
        tg_tps = timings.get("predicted_per_second")
        draft_acc = timings.get("draft_n_accepted")
        draft_tot = timings.get("draft_n")

        print("\n================ BENCHMARK SONUCU ================")
        print(f"  Yukleme suresi    : {load_s:.2f} s")
        print(f"  Istek suresi      : {req_s:.2f} s")
        print(f"  Prompt eval (pp)  : {pp_tps:.2f} tok/s" if isinstance(pp_tps, (int, float)) else f"  Prompt tokens     : {usage.get('prompt_tokens')}")
        print(f"  Uretim hizi (tg)  : {tg_tps:.2f} tok/s" if isinstance(tg_tps, (int, float)) else f"  Completion tokens : {usage.get('completion_tokens')}")
        if draft_tot:
            rate = (draft_acc or 0) / max(1, draft_tot) * 100.0
            print(f"  DSpark Kabul Orani: {draft_acc}/{draft_tot} ({rate:.1f}%)")
        print(f"  Peak Dedicated    : {peak_ded:.0f} MiB ({peak_ded/1024:.2f} GiB)")
        print(f"  Peak Shared       : {peak_shr:.0f} MiB ({peak_shr/1024:.2f} GiB)")
        print("--------------------------------------------------")
        print(content[:700] + ("..." if len(content) > 700 else ""))
        print("==================================================\n")
        return 0
    finally:
        stop_mon.set()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
        mon_thread.join(timeout=2)
        drain_thread.join(timeout=2)


def main() -> int:
    configure_utf8()
    reg = load_registry()
    models = reg["models"]

    parser = argparse.ArgumentParser(description="PicoCode Local Coding LLM Server")
    parser.add_argument("--model", choices=list(models.keys()), default=None, help="Model ID to serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ctx-size", type=int, default=None, help="Override context window (tokens)")
    parser.add_argument("--threads", type=int, default=None, help="Override CPU threads for MoE experts")
    parser.add_argument("--fa", choices=["auto", "on", "off"], default="auto", help="Flash attention mode")
    parser.add_argument("--ngl", type=int, default=None, help="Override -ngl GPU layers")
    parser.add_argument("--no-draft", action="store_true", help="Disable DSpark speculative decoding")
    parser.add_argument("--no-vision", action="store_true", help="Disable mmproj vision projector")
    parser.add_argument("--bench", action="store_true", help="Run a live coding benchmark and exit")
    parser.add_argument("--max-tokens", type=int, default=192, help="Max tokens for --bench")
    parser.add_argument("--dry-run", action="store_true", help="Print hardware plan & command without starting")
    args = parser.parse_args()

    hw = probe_hardware.probe_hardware(REPO_ROOT / "models")
    model_id = pick_installed_model(reg, args.model)

    server_exe = find_llama_server(hw.get("backend", "vulkan"))
    if server_exe is None and not args.dry_run:
        print("HATA: llama-server.exe bulunamadi! Once `py scripts\\download_models.py --install-engine auto` calistirin.", file=sys.stderr)
        return 1

    try:
        cmd, plan = build_server_command(
            model_id,
            hw,
            reg,
            server_exe=server_exe,
            host=args.host,
            port=args.port,
            ctx_size=args.ctx_size,
            threads=args.threads,
            flash_attn=args.fa,
            use_draft=not args.no_draft,
            use_vision=not args.no_vision,
            n_gpu_layers=args.ngl,
            allow_missing_files=args.dry_run,
        )
    except FileNotFoundError as exc:
        print(f"HATA: {exc}\nModeli indirmek icin: py scripts\\download_models.py --profile {model_id}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(json.dumps({"hardware": hw, "plan": plan, "command": cmd}, indent=2, ensure_ascii=False))
        return 0

    if args.bench:
        return run_benchmark(cmd, plan, args.host, args.port, max_tokens=args.max_tokens)

    print("============================================================")
    print(f"  PicoCode OpenAI API Server — {plan['label']}")
    print("============================================================")
    print(f"  GPU / Backend : {plan['gpu_name']} ({plan['backend'].upper()}, {plan['vram_total_gib']:.1f} GiB VRAM)")
    print(f"  MoE Offload   : {plan['moe_cpu_override'] or 'Tam GPU (-ngl ' + str(plan['n_gpu_layers']) + ')'}")
    print(f"  Baglam / KV   : {plan['ctx_size']} token | KV={plan['kv_cache']} | FA={plan['flash_attn']} | Threads={plan['threads']}")
    print(f"  DSpark / Viz  : Draft={'AKTIF' if plan['draft_model'] else 'KAPALI'} | Vision={'AKTIF' if plan['mmproj_model'] else 'KAPALI'}")
    print(f"  OpenAI API    : {plan['api_base']}")
    if plan["warning"]:
        print(f"  [UYARI]       : {plan['warning']}")
    print("  Durdurmak icin Ctrl+C")
    print("============================================================\n")

    proc = subprocess.Popen(cmd)

    def _sig_handler(_signum: int, _frame: Any) -> None:
        if proc.poll() is None:
            proc.terminate()

    signal.signal(signal.SIGINT, _sig_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sig_handler)

    try:
        return proc.wait()
    except KeyboardInterrupt:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=8)
        return 0


if __name__ == "__main__":
    sys.exit(main())
