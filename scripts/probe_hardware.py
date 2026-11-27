#!/usr/bin/env python3
"""
PicoCode Hardware & Live VRAM Probe (CUDA + Vulkan, Windows + Linux).

Detects GPU backend, total/usable VRAM (strictly DEVICE_LOCAL heaps), live free VRAM
via zero-subprocess native Windows PDH (`pdh.dll` `PdhAddEnglishCounterW` with TTL cache),
system RAM, CPU threads, and target drive free disk space.

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
import threading
import time
from typing import Any

_VRAM_CACHE_LOCK = threading.Lock()
_VRAM_CACHE_TS = 0.0
_VRAM_CACHE_VAL: tuple[float | None, float | None] = (None, None)
_VRAM_CACHE_TTL_S = 2.5


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


class _PDH_FMT_COUNTERVALUE(ctypes.Structure):
    _fields_ = [
        ("CStatus", ctypes.c_uint32),
        ("doubleValue", ctypes.c_double),
    ]


class _PDH_FMT_COUNTERVALUE_ITEM_W(ctypes.Structure):
    _fields_ = [
        ("szName", ctypes.c_wchar_p),
        ("FmtValue", _PDH_FMT_COUNTERVALUE),
    ]


def _probe_windows_pdh_vram_mib() -> tuple[float | None, float | None]:
    """Read Windows GPU Adapter Dedicated & Shared Usage via native pdh.dll (~2ms, zero PowerShell)."""
    if platform.system() != "Windows":
        return None, None
    try:
        pdh = ctypes.windll.pdh
        PDH_FMT_DOUBLE = 0x00000200
        PDH_MORE_DATA = 0x800007D2

        query = ctypes.c_void_p()
        if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) != 0:
            return None, None
        try:
            c_ded = ctypes.c_void_p()
            c_shr = ctypes.c_void_p()
            r1 = pdh.PdhAddEnglishCounterW(query, r"\GPU Adapter Memory(*)\Dedicated Usage", 0, ctypes.byref(c_ded))
            r2 = pdh.PdhAddEnglishCounterW(query, r"\GPU Adapter Memory(*)\Shared Usage", 0, ctypes.byref(c_shr))
            if r1 != 0 or r2 != 0:
                return None, None
            if pdh.PdhCollectQueryData(query) != 0:
                return None, None

            def _read_max_mib(counter: ctypes.c_void_p) -> float | None:
                buf_sz = ctypes.c_uint32(0)
                item_cnt = ctypes.c_uint32(0)
                st = pdh.PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, ctypes.byref(buf_sz), ctypes.byref(item_cnt), None)
                if (st & 0xFFFFFFFF) != PDH_MORE_DATA or buf_sz.value == 0:
                    return None
                raw = ctypes.create_string_buffer(buf_sz.value)
                st2 = pdh.PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, ctypes.byref(buf_sz), ctypes.byref(item_cnt), raw)
                if st2 != 0 or item_cnt.value == 0:
                    return None
                arr_t = _PDH_FMT_COUNTERVALUE_ITEM_W * item_cnt.value
                items = arr_t.from_buffer(raw)
                vals = [
                    it.FmtValue.doubleValue / (1024.0 * 1024.0)
                    for it in items
                    if it.FmtValue.CStatus == 0 and it.FmtValue.doubleValue >= 0.0
                ]
                return round(max(vals), 1) if vals else None

            return _read_max_mib(c_ded), _read_max_mib(c_shr)
        finally:
            pdh.PdhCloseQuery(query)
    except Exception:
        return None, None


def probe_live_vram_mib(*, force: bool = False) -> tuple[float | None, float | None]:
    """Return (dedicated_used_mib, shared_used_mib) with TTL cache and zero PowerShell overhead."""
    global _VRAM_CACHE_TS, _VRAM_CACHE_VAL
    now = time.monotonic()
    with _VRAM_CACHE_LOCK:
        if not force and _VRAM_CACHE_TS > 0.0 and (now - _VRAM_CACHE_TS) < _VRAM_CACHE_TTL_S:
            return _VRAM_CACHE_VAL

    result: tuple[float | None, float | None] = (None, None)
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True, timeout=3, stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                result = (round(float(out.splitlines()[0].strip()), 1), 0.0)
        except Exception:
            pass

    if result == (None, None) and platform.system() == "Windows":
        result = _probe_windows_pdh_vram_mib()

    with _VRAM_CACHE_LOCK:
        _VRAM_CACHE_TS = time.monotonic()
        _VRAM_CACHE_VAL = result
    return result


def parse_vulkan_device_local_gib(full_text: str) -> float | None:
    """
    Parse `vulkaninfo` memoryHeaps blocks and return total DEVICE_LOCAL VRAM in GiB.
    Handles hex parentheticals like `size = 5351931904 (0x13f000000) (4.98 GiB)`
    and strictly ignores host System RAM heaps (`flags = None`).
    """
    if not full_text:
        return None
    blocks = re.split(r"memoryHeaps\[\d+\]\s*:", full_text)[1:]
    dev_local_bytes = 0
    for blk in blocks:
        # Restrict inspection to this heap block before the next section
        chunk = blk[:450]
        m_sz = re.search(r"\bsize\s*=\s*(\d+)\b", chunk)
        if not m_sz:
            continue
        if "MEMORY_HEAP_DEVICE_LOCAL_BIT" in chunk:
            sz = int(m_sz.group(1))
            if sz > 0:
                dev_local_bytes += sz
    if dev_local_bytes <= 0:
        return None
    gib = dev_local_bytes / (1024**3)
    # Windows WDDM often splits a 6 GiB card into 4.98 GiB + 256 MiB BAR or reports ~5.25 GiB usable
    # Round close standard physical VRAM tiers cleanly (e.g. 5.2-6.1 -> 6.0, 7.2-8.1 -> 8.0, etc.)
    for std in (4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0):
        if abs(gib - std) <= 0.85 and gib <= std + 0.25:
            return std
    return round(gib, 2)


def _probe_windows_registry_gpu() -> tuple[str | None, float | None]:
    """Fast fallback reading GPU name & HardwareInformation.qwMemorySize from Windows registry."""
    if platform.system() != "Windows":
        return None, None
    try:
        import winreg
        base_key = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        best_name: str | None = None
        best_gib: float = 0.0
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base_key) as root:
            for idx in range(16):
                try:
                    sub_name = winreg.EnumKey(root, idx)
                except OSError:
                    break
                if not sub_name.isdigit():
                    continue
                try:
                    with winreg.OpenKey(root, sub_name) as sk:
                        desc, _ = winreg.QueryValueEx(sk, "DriverDesc")
                        mem_val = None
                        for vname in ("HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"):
                            try:
                                v, _ = winreg.QueryValueEx(sk, vname)
                                if isinstance(v, bytes) and len(v) >= 4:
                                    mem_val = int.from_bytes(v, "little")
                                elif isinstance(v, int) and v > 0:
                                    mem_val = v
                                break
                            except OSError:
                                continue
                        if mem_val and mem_val > 0:
                            g = mem_val / (1024**3)
                            if g > best_gib:
                                best_gib = g
                                best_name = str(desc)
                except OSError:
                    continue
        if best_gib > 0.5:
            return best_name, round(best_gib, 2)
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

    # 2. Check Windows Registry GPU info (instant, works even without Vulkan SDK!)
    reg_name, reg_gib = _probe_windows_registry_gpu()
    if reg_name:
        hw["gpu_name"] = reg_name
        low = reg_name.lower()
        hw["vendor"] = "AMD" if ("amd" in low or "radeon" in low) else ("Intel" if "intel" in low else ("NVIDIA" if "nvidia" in low else "Vulkan"))
    if reg_gib is not None:
        hw["vram_total_gib"] = reg_gib

    # 3. Refine via vulkaninfo if installed
    vkinfo = shutil.which("vulkaninfo") or shutil.which("vulkaninfoSDK")
    if vkinfo:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            txt = subprocess.check_output([vkinfo, "--summary"], text=True, timeout=6, stderr=subprocess.DEVNULL, creationflags=flags)
            m_name = re.search(r"deviceName\s*=\s*(.+)", txt)
            m_vid = re.search(r"vendorID\s*=\s*(0x[0-9a-fA-F]+)", txt)
            if m_name:
                hw["gpu_name"] = m_name.group(1).strip()
            if m_vid:
                vid = m_vid.group(1).lower()
                hw["vendor"] = "AMD" if vid == "0x1002" else ("Intel" if vid == "0x8086" else "Vulkan")
        except Exception:
            pass
        if reg_gib is None:
            try:
                full = subprocess.check_output([vkinfo], text=True, timeout=8, stderr=subprocess.DEVNULL, creationflags=flags)
                vk_gib = parse_vulkan_device_local_gib(full)
                if vk_gib is not None:
                    hw["vram_total_gib"] = vk_gib
            except Exception:
                pass

    tot_g = float(hw["vram_total_gib"])
    hw["vram_class"] = "4GB" if tot_g <= 4.5 else ("6GB" if tot_g <= 6.5 else ("8GB" if tot_g <= 10.0 else "16GB+"))

    ded_m, shr_m = probe_live_vram_mib(force=True)
    if ded_m is not None:
        used_g = round(ded_m / 1024.0, 2)
        hw["live_used_ded_gib"] = used_g
        hw["live_free_ded_gib"] = round(max(0.0, tot_g - used_g), 2)
    if shr_m is not None:
        hw["live_used_shr_gib"] = round(shr_m / 1024.0, 2)
    return hw


if __name__ == "__main__":
    print(json.dumps(probe_hardware(), indent=2, ensure_ascii=False))
