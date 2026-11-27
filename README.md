# PicoCode v0.7.0 — Local 30B Coding LLMs & Claude-Code-Style Terminal Agent on Low-End 4–6 GB GPUs (AMD & NVIDIA)

[![Regression Tests](https://github.com/ffalcon3098-droid/PicoCode/actions/workflows/tests.yml/badge.svg)](https://github.com/ffalcon3098-droid/PicoCode/actions/workflows/tests.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> [Türkçe dokümantasyon için README.tr.md](README.tr.md) · **Platform:** Windows 10/11 (x64) · **GPUs:** AMD Radeon (Vulkan), NVIDIA GeForce GTX/RTX (CUDA 12.4 & Vulkan), Intel Arc (Vulkan)

**PicoCode** brings frontier 26B–30B Mixture-of-Experts (MoE) and fast 7B local coding models — **Qwen3-Coder-30B-A3B (`97.5 tok/s` pp via `--load-mode none` + `18–30 tok/s` tg via Adaptive Hybrid GPU MoE + lossless `ngram-mod`)**, **Inkling 2 Mini (26.4B / 3.4B Looped MoE + DSpark + Vision)**, and **Qwen2.5-Coder-7B** — to ordinary **4–6 GB VRAM** consumer PCs with **zero C++ compilation**, an **English-first Claude-Code-Style 24-bit TrueColor Terminal Coding Agent (`.\agent.cmd`)**, an **OpenAI-compatible API** for VS Code (Cline, Roo Code, Continue, OpenCode) & **PicoStudio IDE**, and a hardened dark **Web Studio**.

All bundled code and default models are **100% Apache 2.0**.

---

## AMD & NVIDIA GPU Compatibility + Honest Hardware Tiers

PicoCode automatically probes your GPU (`scripts/probe_hardware.py`), downloads the matching prebuilt **`llama.cpp` Vulkan or CUDA 12.4** binary, and tunes layer offloading, KV quantization, and context limits so powerful models run smoothly without OOM crashes:

| GPU Vendor / Models | Backend (Auto-Selected) | Status | Notes |
|---|---|---|---|
| **AMD Radeon RX** (RX 5500 XT / **RX 5600 XT 6 GB** / RX 6600 / RX 7600+) | **Vulkan** (`ggml-vulkan`) | **Physically Verified (RX 5600 XT 6 GB)** | Zero ROCm hassle on Windows; prebuilt official `llama.cpp b12807` Vulkan binary. |
| **NVIDIA GeForce GTX / RTX** (GTX 1650 / 1660 / RTX 2060 / 3050 / 3060 / 4060+) | **CUDA 12.4** (or Vulkan) | **Auto-Configured (`setup.cmd`)** | Auto-detects `nvidia-smi`, downloads official `llama.cpp b12807` CUDA 12.4 + `cudart` DLLs; native FP16 Tensor Core speed. |
| **Intel Arc** (A380 / A580 / A750 / B580) | **Vulkan** (`ggml-vulkan`) | **Supported via Vulkan** | Uses the same prebuilt Vulkan engine and automatic VRAM budget probing. |

| System Tier (VRAM + System RAM) | Recommended Model | Expected Experience |
|---|---|---|
| **4–6 GB VRAM + 16 GB RAM** *(Entry / Low-End)* | `qwen25-coder-7b` (`4.36 GiB`) | Fits **100% inside 6 GB VRAM** (or hybrid on 4 GB) — instant startup, ultra-fast coding & tool-calling. |
| **6 GB VRAM + 32 GB RAM** *(Sweet Spot — Verified)* | `qwen3-coder-30b` (`15.25 GiB`) & `inkling2-mini` (`16.26 GiB`) | Adaptive Hybrid GPU MoE + `--load-mode none`: **30B/3B coding at `18–30 tok/s`** + Vision + DSpark! |
| **8 GB+ VRAM + 32 GB RAM** *(RTX 3060/4060 / RX 6600+)* | All models (`32K–64K` ctx) | Even more MoE layers resident on GDDR6 for higher sustained throughput. |

---

## Key Highlights (`v0.7.0`)

- **Claude-Code-Style 24-bit Terminal Coding Agent (`.\agent.cmd` / `.\picocode.cmd agent`):**
  - **English-First UI (with multilingual model replies):** Clean English TUI, Web Studio, installer, and CLI output; the agent automatically responds in whatever language you prompt in.
  - **10 Workspace Tools Built for Real Codebases & Reviews:** `project_overview` (1-shot manifest/language/hotspot summary), multi-language `repo_map` (Python, JS/TS, **Rust `.rs` + `#[tauri::command]`**, **Go `.go`**), `read_file`, `search_code`, `list_files`, `git_status`, `git_diff`, `edit_file` (instant syntax feedback + CRLF preservation), `write_file`, and `run_command` (destructive-command guard).
  - **100% KV-Cache Preservation & Anti-Empty-Review Loop:** Keeps identical `tools` schema with `tool_choice="none"` on guaranteed final synthesis turns (`f_keep = 1.000`), epoch/hysteresis compaction (`>72%` ctx), per-step parallel tool output budget, and an evidence-first review verification rule.
  - **Folder Trust Gate (`~/.picocode_trusted_folders.json`), Project Picker (`/project` / `/cd`), Pinning (`/add` / `/drop`) & Live `/model` Hot-Swap:** Switch projects in **0 seconds** while keeping the LLM warm in VRAM, and hot-swap `/model inkling` ↔ `/model qwen` without losing conversation history.
- **Zero-Compile Windows Setup (`.\setup.cmd`):** Automatically installs official prebuilt `llama.cpp b12807` Win64 **Vulkan** or **CUDA 12.4** binaries, TUI dependencies, and verified GGUF models (resumable + Zip-Slip protected). **No Visual Studio or C++ Build Tools required.**
- **Empirically Tuned for 6 GB VRAM + 32 GB RAM (`RX 5600 XT` verified):**
  - **`Qwen3-Coder-30B-A3B`:** Adaptive Hybrid GPU MoE (`--n-cpu-moe 42`, `6` layers on GDDR6), **`--load-mode none`** (`+18.1%` pp: `82.5 → 97.45 tok/s`, `+3.4%` tg: `20.24 tok/s`), **`-b 1024 -ub 256 -t 6 -tb 12`** (`+10.8%` long-context tg without Vulkan scratch spill), zero-VRAM lossless **`--spec-type ngram-mod`**, and **`--cache-reuse 256`**.
  - **`Inkling 2 Mini`:** **`-b 1024 -ub 512 -t 6 -tb 12 --cpu-moe --no-op-offload --no-reasoning-preserve`** + **`DSpark`** (`q8_0` draft KV, `64–83%` accept) + **Vision `mmproj`**, with crisp intermediate thinking (`reasoning_budget=512`) and deep final synthesis (`768`).
- **5 ms Native Windows PDH VRAM Telemetry & Hardened Web Studio:** Instant `pdh.dll` Dedicated/Shared VRAM monitoring (replaces slow PowerShell polling) and full Host + Origin + Content-Type + CSRF (`X-PicoCode-Token`) protection on `http://127.0.0.1:7870`.

---

## Verified Real-World Benchmarks (AMD Radeon RX 5600 XT 6 GB Vulkan + Ryzen 5 5500 + 32 GB RAM)

### 1. `Qwen3-Coder-30B-A3B-Instruct` (`IQ4_XS`, `15.25 GiB`, Full `32,768` Context, Live `llama-server`)

| Configuration (`32K ctx`, `q8_0` KV) | Warm Load | Fresh Code (`tg`) | Real File Refactor (`pp` / `tg`) | Multi-Turn Follow-up (`KV Cache`) | Speculative (`0 MB VRAM`) | Peak Dedicated / Shared |
|---|---:|---:|---:|---:|---:|---|
| Previous Baseline (`b12720`, `--cpu-moe`, `b512/ub256`) | `3.1 s` | `16.20 tok/s` | `58.8` / `16.89 tok/s` (`36.7s`) | `1267/1305` · `17.05 tok/s` | OFF | `3.77 / 0.50 GiB` |
| **`v0.2.0` Tuned (`b12807`, `-ncmoe 44`, `b2048/ub512`, `poll 100`, `ngram-mod`)** | **`3.5 s`** | **`18.25 tok/s`** | **`71.6` (+22%) / `30.21 tok/s` (+79%!) (`26.1s`)** | **`1267/1305` (`97%`) · `18.25 tok/s`** | **`232/232` (100%)** | **`4.89 / 0.59 GiB`** |
| **`v0.2.0` `q4_0` KV (`-ncmoe 42`, 6 GPU MoE, `ngram-mod`)** | **`3.5 s`** | **`18.42 tok/s`** | **`71.8` (+22%) / `29.50 tok/s` (+75%!) (`26.3s`)** | **`1267/1305` (`97%`) · `18.92 tok/s`** | **`232/232` (100%)** | **`4.88 / 0.59 GiB`** |

### 2. `Inkling-2-Mini` (`UD-Q4_K_M`, `15.47 GiB`, `16K` Context + Vision `mmproj` + DSpark)

Measured live on Windows 11 with desktop + browser active (`~1.65 GiB` baseline Dedicated VRAM), `Inkling-2-Mini-UD-Q4_K_M.gguf` (`15.47 GiB`), `16,384` context (`q8_0` KV cache), Vision `mmproj` enabled, `256` generated tokens:

| Configuration | Warm Load | Prompt Eval (`pp`) | Generation (`tg`) | Total Time | Peak Dedicated (incl. Desktop) | Peak Shared |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (`--cpu-moe`, default op-offload, DSpark) | 24.6 s | 0.70 tok/s | 8.13 tok/s | 136.8 s | 4.38 GiB | 9.04 GiB |
| **PicoCode Tuned (`--no-op-offload --no-host -t 6`, DSpark OFF)** | **4.8 s** | **15.5 tok/s** | **7.68 tok/s** | **37.0 s** | **4.21 GiB** | **1.16 GiB** |
| **PicoCode Tuned + DSpark (`--spec-draft-n-max 6`, Default)** | **6.8 s** | **14.4 tok/s** | **8.32 tok/s** | **34.7 s** | **4.26 GiB** | **2.12 GiB** |

- **DSpark Acceptance Rate:** `72%–83%` on Python coding tasks (`207/287` accepted).
- **Projected `llama.cpp` GPU VRAM (`16K ctx`):** **`2.55 GiB`** (`1.24 GiB` model + `0.38 GiB` KV cache + `0.88 GiB` compute) — leaves plenty of headroom even on 6 GB cards!

---

## Supported Models (`configs/models_registry.json`)

| Model ID | Architecture | Size | Hardware Recommendation | Notes |
|---|---|---:|---|---|
| `inkling2-mini` | 26.4B / 3.4B Looped MoE | 16.26 GiB (incl. DSpark + Vision) | **6 GB VRAM + 32 GB RAM** | Default · Coding + Reasoning + Screenshot Vision |
| `qwen3-coder-30b` | 30B / 3B MoE | 15.25 GiB (`IQ4_XS`) | **6 GB VRAM + 32 GB RAM** | Pure high-context coding (up to 32K ctx) |
| `qwen25-coder-7b` | 7B Dense | 4.36 GiB (`Q4_K_M`) | **4–6 GB VRAM + 16 GB RAM** | Fits 100% in 6 GB VRAM (ultra-fast) |

---

## Quick Start (Windows)

```powershell
# 1. One-click zero-compile setup (installs llama.cpp b12807 + TUI deps + models)
.\setup.cmd

# 2. Launch Claude-Code-style Terminal Agent (zero-flag project picker + Trust Gate)
.\agent.cmd

# 3. Or launch Dark Web Studio (http://127.0.0.1:7870)
.\web-studio.cmd

# 4. Or launch headless OpenAI API server for VS Code (http://127.0.0.1:8080/v1)
.\picocode.cmd --model inkling2-mini

# 5. Run a live coding speed & VRAM benchmark on your own GPU
.\picocode.cmd --model inkling2-mini --bench
```

### Connect VS Code (Cline / Roo Code / Continue)

- **Provider:** `OpenAI Compatible`
- **Base URL:** `http://127.0.0.1:8080/v1`
- **API Key:** `picocode`
- **Model ID:** `inkling2-mini` (or `qwen3-coder-30b`, `qwen25-coder-7b`)

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
