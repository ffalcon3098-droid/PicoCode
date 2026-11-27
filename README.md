# PicoCode v0.6.0 — 30 tok/s Local 30B Coding LLMs & Claude-Code-Style Terminal Agent on 4–6 GB VRAM PCs

> [Türkçe dokümantasyon için README.tr.md](README.tr.md)

**PicoCode** runs frontier sparse Mixture-of-Experts (MoE) and dense local coding models — led by **Qwen3-Coder-30B-A3B (up to `30.2 tok/s` via Adaptive Hybrid GPU MoE + lossless `ngram-mod`)**, **Inkling 2 Mini (26.4B / 3.4B Looped MoE + DSpark + Vision)**, and **Qwen2.5-Coder-7B** — smoothly on ordinary **4–6 GB VRAM + 32 GB RAM** consumer PCs (AMD Vulkan & NVIDIA CUDA), with **zero C++ compilation**, a **Claude-Code-Style 24-bit TrueColor Terminal Coding Agent (`.\agent.cmd`)**, an **OpenAI-compatible API** for VS Code (Cline, Roo Code, Continue, OpenCode) & **PicoStudio IDE**, and a dark **Web Studio**.

All bundled code and default models are **100% Apache 2.0**.

---

## Key Highlights

- **Claude-Code-Style 24-bit Terminal Coding Agent (`v0.5.0` · `.\agent.cmd` / `.\picocode.cmd agent`):**
  - **Folder Trust Gate (`~/.picocode_trusted_folders.json`) & Interactive Project Picker (`/project` / `/cd`):** Launch `.\agent.cmd` with zero flags to pick or create a project folder, approve trust (`Trust` / `Read-Only Plan` / `Cancel`), and switch projects mid-session in **0 seconds** while keeping the LLM warm in VRAM.
  - **File Browser, Context Pinning & Live Model Hot-Swap:** `/files`, `/add <file>` (`Tab`), `/drop`, `@file` autocomplete, and live `/model inkling` ↔ `/model qwen` hot-swapping without losing chat history.
  - Rich 24-bit TrueColor TUI (`scripts/tui.py` + `scripts/agent_cli.py`): rounded 2-column welcome header with gradient `PICOCODE` logo, animated braille spinner (`⠋`), live syntax-highlighted Markdown streaming, `● Tool(args)` / `⎿` call tree, red/green Diff Cards, and bottom status toolbar (`mode · model · project · git · pinned · ctx%`).
- **Zero-Compile Windows Setup (`.\setup.cmd`):** Automatically installs official prebuilt `llama.cpp b12807` Win64 **Vulkan** or **CUDA 12.4** binaries and verified GGUF models. **No Visual Studio or C++ Build Tools required.**
- **Adaptive Hybrid GPU+RAM MoE + Zero-VRAM Lossless `ngram-mod` (`v0.2.0`):**
  - Reads live free VRAM before startup and places `4–6` MoE expert layers directly onto 6 GB GDDR6 (`--n-cpu-moe N`, `336 GB/s` vs `35 GB/s` DDR4) with a `650 MiB` anti-spill guard, keeping full **`32K` `q8_0` KV cache** inside VRAM.
  - Enables **`--spec-type ngram-mod`** (0 MB VRAM, 100% lossless speculative decoding), **`--poll 100`**, **`--cache-reuse 256`**, and **`-b 2048 -ub 512 -tb 12`** — hitting **`18.3–18.9 tok/s`** on fresh code and **`30.2 tok/s` (`232/232` acceptance!)** on real code edits/refactors!
- **Smart Inkling 2 Mini Engine:** Keeps attention, loop layers, `q8_0` KV cache (`16K`), `DSpark`, and `mmproj` Vision inside **~2.55 GiB VRAM** (`23×` faster prompt eval).
- **Live VRAM Protection:** Automatically caps context & GPU MoE layers if browsers/desktop apps consume heavy VRAM.

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

# 3. Or launch headless OpenAI API server for VS Code (http://127.0.0.1:8080/v1)
.\picocode.cmd --model inkling2-mini

# 4. Run a live coding speed & VRAM benchmark on your own GPU
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
