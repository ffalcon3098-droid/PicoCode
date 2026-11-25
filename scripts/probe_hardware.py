#!/usr/bin/env python3
"""
PicoCode Hardware & Live VRAM Probe (CUDA + Vulkan, Windows + Linux).

Detects GPU backend, total/usable VRAM, live free VRAM, system RAM, CPU threads,
and target drive free disk space for zero-config hybrid GPU+RAM MoE serving.

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import ctypes
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
from typing import Any


def probe_ram_gib() -> tuple[float | None, float | None]:
    """Return (total_ram_gib, avail_ram_gib) quickly without WMI stalls."""
    if platform.system() == "Windows":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return round(st.ullTotalPhys / (1024**3), 2), round(st.ullAvailPhys / (1024**3), 2)
        except Exception:
            return None, None
    try:
        info: dict[str, int] = {}
        for line in pathlib.Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                info[parts[0].rstrip(":")] = int(parts[1])
        tot = info.get("MemTotal")
        avail = info.get("MemAvailable")
        return (round(tot / (1024**2), 2) if tot else None), (round(avail / (1024**2), 2) if avail else None)
    except Exception:
        return None, None


def probe_live_vram_mib() -> tuple[float | None, float | None]:
    """Return (dedicated_used_mib, shared_used_mib) on Windows/NVIDIA."""
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True, timeout=3, stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                return round(float(out.splitlines()[0].strip()), 1), 0.0
        except Exception:
            pass

    if platform.system() == "Windows":
        ps = (
            "$d=(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue)"
            ".CounterSamples|Measure-Object -Property CookedValue -Maximum|Select-Object -ExpandProperty Maximum;"
            "$s=(Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage' -ErrorAction SilentlyContinue)"
            ".CounterSamples|Measure-Object -Property CookedValue -Maximum|Select-Object -ExpandProperty Maximum;"
            "if($null -ne $d){Write-Host ([long][math]::Round([double]$d/1MB)) ([long][math]::Round([double]$s/1MB))}"
        )
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", ps],
                text=True, timeout=5, stderr=subprocess.DEVNULL, creationflags=flags,
            ).strip()
            parts = out.split()
            if len(parts) >= 2:
                return float(parts[0]), float(parts[1])
        except Exception:
            pass
    return None, None


def probe_hardware(target_dir: pathlib.Path | None = None) -> dict[str, Any]:
    ram_tot, ram_avail = probe_ram_gib()
    check_path = target_dir or pathlib.Path.cwd()
    while not check_path.exists() and check_path.parent != check_path:
        check_path = check_path.parent
    disk = shutil.disk_usage(check_path)
    cpu_threads = max(2, os.cpu_count() or 8)

    hw: dict[str, Any] = {
        "os": platform.system(),
        "cpu_threads": cpu_threads,
        "ram_total_gib": ram_tot or 16.0,
        "ram_avail_gib": ram_avail or 12.0,
        "disk_free_gib": round(disk.free / (1024**3), 2),
        "vendor": "Unknown",
        "gpu_name": "CPU Fallback",
        "backend": "vulkan",
        "vram_total_gib": 6.0,
        "vram_class": "6GB",
    }

    # 1. Check NVIDIA CUDA via nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                text=True, timeout=4, stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                name, tot_m, free_m = [x.strip() for x in out.splitlines()[0].split(",")[:3]]
                tot_g = round(float(tot_m) / 1024.0, 2)
                free_g = round(float(free_m) / 1024.0, 2)
                hw.update({
                    "vendor": "NVIDIA",
                    "gpu_name": name,
                    "backend": "cuda",
                    "vram_total_gib": tot_g,
                    "live_used_ded_gib": round(max(0.0, tot_g - free_g), 2),
                    "live_free_ded_gib": free_g,
                    "vram_class": "4GB" if tot_g <= 4.5 else ("6GB" if tot_g <= 6.5 else ("8GB" if tot_g <= 10.0 else "16GB+")),
                })
                return hw
        except Exception:
            pass

    # 2. Check Vulkan via vulkaninfo
    vkinfo = shutil.which("vulkaninfo") or shutil.which("vulkaninfoSDK")
    if vkinfo:
        try:
            txt = subprocess.check_output([vkinfo, "--summary"], text=True, timeout=6, stderr=subprocess.DEVNULL)
            m_name = re.search(r"deviceName\s*=\s*(.+)", txt)
            m_vid = re.search(r"vendorID\s*=\s*(0x[0-9a-fA-F]+)", txt)
            if m_name:
                hw["gpu_name"] = m_name.group(1).strip()
            if m_vid:
                vid = m_vid.group(1).lower()
                hw["vendor"] = "AMD" if vid == "0x1002" else ("Intel" if vid == "0x8086" else "Vulkan")
        except Exception:
            pass

    # Refine VRAM total on Windows via registry / WMI-free DxDiag-like fallback or vulkaninfo heaps
    if vkinfo:
        try:
            full = subprocess.check_output([vkinfo], text=True, timeout=8, stderr=subprocess.DEVNULL)
            heaps = [int(m) for m in re.findall(r"size\s*=\s*(\d+)\s*\([^)]*GiB\)", full)]
            if heaps:
                # Filter typical 2-24 GiB dedicated GPU heaps
                gpu_heaps = [h / (1024**3) for h in heaps if 1.5 <= h / (1024**3) <= 48.0]
                if gpu_heaps:
                    tot_g = round(max(gpu_heaps), 2)
                    hw["vram_total_gib"] = tot_g
        except Exception:
            pass

    tot_g = float(hw["vram_total_gib"])
    hw["vram_class"] = "4GB" if tot_g <= 4.5 else ("6GB" if tot_g <= 6.5 else ("8GB" if tot_g <= 10.0 else "16GB+"))

    ded_m, shr_m = probe_live_vram_mib()
    if ded_m is not None:
        used_g = round(ded_m / 1024.0, 2)
        hw["live_used_ded_gib"] = used_g
        hw["live_free_ded_gib"] = round(max(0.0, tot_g - used_g), 2)
    if shr_m is not None:
        hw["live_used_shr_gib"] = round(shr_m / 1024.0, 2)
    return hw


if __name__ == "__main__":
    print(json.dumps(probe_hardware(), indent=2, ensure_ascii=False))
