#!/usr/bin/env python3
"""
PicoCode Engine & Model Downloader (driven by configs/models_registry.json).

Features:
  - Downloads official prebuilt llama.cpp (Win64 Vulkan or CUDA) — ZERO C++ compilation required!
  - Bulletproof resumable GGUF downloads (curl with conditional -C - + pure-Python urllib Range fallback)
  - Hardware-aware interactive model wizard (detects GPU VRAM, RAM, and free disk space)

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
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


def format_gb(nbytes: int) -> str:
    return f"{nbytes / (1024**3):.2f} GiB"


def load_registry(path: pathlib.Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(rel_or_abs: str) -> pathlib.Path:
    p = pathlib.Path(rel_or_abs)
    return p if p.is_absolute() else (REPO_ROOT / p)


def check_file(item: dict[str, Any]) -> tuple[bool, str, int]:
    p = resolve_path(item["path"])
    exp = int(item["expected_bytes"])
    if not p.exists():
        return False, "MISSING", exp
    sz = p.stat().st_size
    if sz == exp:
        return True, f"OK ({format_gb(sz)})", 0
    return False, f"PARTIAL/MISMATCH ({format_gb(sz)} / {format_gb(exp)})", max(0, exp - sz)


def _download_with_python_urllib(url: str, dest: pathlib.Path, expected_bytes: int = 0) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 5):
        existing = dest.stat().st_size if dest.exists() else 0
        if expected_bytes > 0 and existing > expected_bytes:
            dest.unlink(missing_ok=True)
            existing = 0
        headers = {"User-Agent": "PicoCode/0.1"}
        mode = "wb"
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            mode = "ab"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                status = getattr(resp, "status", 200)
                if existing > 0 and status == 200:
                    existing = 0
                    mode = "wb"
                cl = int(resp.headers.get("Content-Length", "0") or "0")
                total = (existing + cl) if (cl > 0 and status == 206) else (cl or expected_bytes)
                downloaded = existing
                t0 = t_last = time.monotonic()
                with dest.open(mode) as fh:
                    while True:
                        buf = resp.read(4 * 1024 * 1024)
                        if not buf:
                            break
                        fh.write(buf)
                        downloaded += len(buf)
                        now = time.monotonic()
                        if now - t_last >= 0.4:
                            spd = (downloaded - existing) / max(0.01, now - t0) / (1024 * 1024)
                            pct = (downloaded / total * 100.0) if total > 0 else 0.0
                            print(f"\r         [Python] {pct:5.1f}%  {format_gb(downloaded)} / {format_gb(total)} ({spd:.1f} MB/s)  ", end="", flush=True)
                            t_last = now
                print()
                if expected_bytes <= 0 or dest.stat().st_size == expected_bytes:
                    return True
        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                dest.unlink(missing_ok=True)
                continue
            print(f"\n  [WARN] HTTP {exc.code}: {exc.reason}", file=sys.stderr)
        except Exception as exc:
            print(f"\n  [WARN] Download error (attempt {attempt}/4): {exc}", file=sys.stderr)
        time.sleep(min(10, attempt * 2))
    return False


def download_url_resumable(url: str, dest: pathlib.Path, expected_bytes: int = 0, label: str = "") -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if expected_bytes > 0 and dest.exists() and dest.stat().st_size > expected_bytes:
        dest.unlink(missing_ok=True)
    if label:
        print(f"  [GET ] {label}")
        print(f"         -> {dest}")

    curl = shutil.which("curl.exe") or shutil.which("curl")
    curl_ok = False
    if curl:
        has_partial = dest.exists() and dest.stat().st_size > 0
        cmd = [
            curl, "-L", "--fail",
            "--connect-timeout", "30",
            "--retry", "6", "--retry-delay", "4",
            "--speed-limit", "10000", "--speed-time", "45",
            "-#",
        ]
        if has_partial:
            cmd += ["-C", "-"]
        cmd += [url, "-o", str(dest)]
        p = subprocess.run(cmd)
        if p.returncode == 0:
            curl_ok = True
        else:
            # Never delete a multi-GB partial download on transient network drop; hand off to urllib Range resume
            print(f"  [WARN] curl exited with code {p.returncode} (partial file kept)", file=sys.stderr)

    if not curl_ok:
        print("  [INFO] Switching to built-in Python downloader (resuming partial file)...")
        _download_with_python_urllib(url, dest, expected_bytes)

    return dest.exists() and (expected_bytes <= 0 or dest.stat().st_size == expected_bytes)


def download_file(item: dict[str, Any]) -> bool:
    p = resolve_path(item["path"])
    ok, msg, _ = check_file(item)
    if ok:
        print(f"  [SKIP] {item['path']} — {msg}")
        return True
    download_url_resumable(item["url"], p, int(item.get("expected_bytes", 0)), label=item["desc"])
    ok, msg, _ = check_file(item)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {item['path']} — {msg}")
    return ok


def _safe_extract_zip(zpath: pathlib.Path, target_dir: pathlib.Path) -> None:
    """Extract zip safely while blocking Zip-Slip path traversal."""
    root = target_dir.resolve()
    with zipfile.ZipFile(zpath, "r") as zf:
        for info in zf.infolist():
            dest = (root / info.filename).resolve()
            if not dest.is_relative_to(root):
                raise PermissionError(f"Zip-Slip blocked: {info.filename}")
        zf.extractall(root)


def install_llama_engine(reg: dict[str, Any], backend: str = "vulkan") -> pathlib.Path | None:
    """Download & extract official prebuilt llama.cpp (llama-server.exe) for Win64 Vulkan or CUDA."""
    sub = "llama-cuda" if backend == "cuda" else "llama-vulkan"
    target_dir = ENGINE_DIR / sub
    server_exe = target_dir / "llama-server.exe"
    if server_exe.exists():
        print(f"  [SKIP] Engine ready: {server_exe}")
        return server_exe

    rels = reg["engine_releases"]
    entry = rels["win_cuda_12_4_x64"] if backend == "cuda" else rels["win_vulkan_x64"]
    target_dir.mkdir(parents=True, exist_ok=True)

    zips_to_get = [(entry["url"], entry["desc"], target_dir / "engine_tmp.zip")]
    if backend == "cuda" and entry.get("cudart_url"):
        zips_to_get.append((entry["cudart_url"], "CUDA 12.4 Runtime DLLs", target_dir / "cudart_tmp.zip"))

    for url, desc, zpath in zips_to_get:
        if not download_url_resumable(url, zpath, 0, label=desc):
            return None
        _safe_extract_zip(zpath, target_dir)
        zpath.unlink(missing_ok=True)

    # Flatten if extracted inside a single subfolder
    if not server_exe.exists():
        for found in target_dir.rglob("llama-server.exe"):
            for item in found.parent.iterdir():
                dst = target_dir / item.name
                if dst != item:
                    if dst.exists():
                        if dst.is_dir():
                            shutil.rmtree(dst)
                        else:
                            dst.unlink()
                    shutil.move(str(item), str(dst))
            break

    if server_exe.exists():
        print(f"  [OK  ] Installed llama-server.exe: {server_exe}")
        return server_exe
    print("ERROR: Could not extract llama-server.exe from archive.", file=sys.stderr)
    return None


def recommend_profile(hw: dict[str, Any]) -> tuple[str, str]:
    ram = float(hw.get("ram_total_gib", 16.0))
    disk = float(hw.get("disk_free_gib", 20.0))
    if ram >= 24.0 and disk >= 19.0:
        return "inkling2-mini", f"Looped MoE Inkling 2 Mini for {ram:.1f} GiB RAM + {hw['vram_total_gib']:.1f} GiB VRAM"
    if ram >= 20.0 and disk >= 17.0:
        return "inkling2-compact", f"Compact 4-bit Inkling 2 Mini (RAM {ram:.1f} GiB, Disk {disk:.1f} GiB)"
    return "starter-7b", f"100% GPU Qwen2.5-Coder-7B for compact RAM/Disk (RAM {ram:.1f} GiB)"


def main() -> int:
    configure_utf8()
    reg = load_registry()
    profiles = reg["download_profiles"]
    files_map = reg["files"]

    parser = argparse.ArgumentParser(description="PicoCode Engine & Model Downloader")
    parser.add_argument("--install-engine", choices=["auto", "vulkan", "cuda"], default=None, help="Install official prebuilt llama.cpp binary")
    parser.add_argument("--profile", choices=list(profiles.keys()), default=None, help="Model profile to download")
    parser.add_argument("--check", action="store_true", help="Verify downloaded files")
    parser.add_argument("--list-profiles", action="store_true", help="List profiles & sizes")
    parser.add_argument("--force-low-disk", action="store_true")
    args = parser.parse_args()

    hw = probe_hardware.probe_hardware(REPO_ROOT / "models")

    if args.list_profiles:
        print("=== PicoCode Model Profiles ===")
        for k, v in profiles.items():
            tot = sum(int(files_map[f]["expected_bytes"]) for f in v["required"])
            print(f"  {k:18s} ({format_gb(tot):>9s})  {v['desc']}")
        return 0

    if args.install_engine:
        b = hw["backend"] if args.install_engine == "auto" else args.install_engine
        exe = install_llama_engine(reg, backend=b)
        if exe is None:
            return 1
        if args.profile is None and not args.check:
            return 0

    prof = args.profile
    if prof is None:
        rec, reason = recommend_profile(hw)
        print(f"\n[PicoCode] GPU: {hw['gpu_name']} ({hw['backend'].upper()}, {hw['vram_total_gib']:.1f} GiB) | RAM: {hw['ram_total_gib']:.1f} GiB | Free Disk: {hw['disk_free_gib']:.1f} GiB")
        print(f"           Recommended: {rec} — {reason}")
        keys = list(profiles.keys())
        for idx, k in enumerate(keys, 1):
            tot = sum(int(files_map[f]["expected_bytes"]) for f in profiles[k]["required"])
            rem = sum(check_file(files_map[f])[2] for f in profiles[k]["required"])
            mark = " ★ Recommended" if k == rec else ""
            print(f"  [{idx}] {k:18s} ({format_gb(tot)}, remaining {format_gb(rem)}){mark} — {profiles[k]['desc']}")
        def_i = str(keys.index(rec) + 1)
        ch = input(f"Select package to download [{def_i}]: ").strip() or def_i
        prof = keys[int(ch) - 1] if ch.isdigit() and 1 <= int(ch) <= len(keys) else rec

    req_ids = profiles[prof]["required"]
    if args.check:
        all_ok = True
        for fid in req_ids:
            ok, msg, _ = check_file(files_map[fid])
            print(f"  [{'OK  ' if ok else 'MISS'}] {files_map[fid]['path']}: {msg}")
            if not ok:
                all_ok = False
        return 0 if all_ok else 1

    needed = sum(check_file(files_map[fid])[2] for fid in req_ids)
    free_b = shutil.disk_usage(REPO_ROOT).free
    if needed > 0 and (free_b - needed) < 1_500_000_000 and not args.force_low_disk:
        print(f"ERROR: Insufficient free disk space! Required: {format_gb(needed)}, Free: {format_gb(free_b)}", file=sys.stderr)
        return 1

    print(f"[PicoCode] Downloading '{prof}' (remaining: {format_gb(needed)}, free disk: {format_gb(free_b)})...")
    ok_all = True
    for fid in req_ids:
        if not download_file(files_map[fid]):
            ok_all = False
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
