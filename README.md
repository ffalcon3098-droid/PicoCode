# PicoCode v0.1.0 — Frontier Local Coding LLMs on 4–6 GB VRAM PCs

> [Türkçe dokümantasyon için README.tr.md](README.tr.md)

**PicoCode** runs frontier sparse Mixture-of-Experts (MoE) and dense local coding models — led by **Inkling 2 Mini (26.4B total / 3.4B active Looped MoE + DSpark + Vision)**, **Qwen3-Coder-30B-A3B**, and **Qwen2.5-Coder-7B** — smoothly on ordinary **4–6 GB VRAM + 32 GB RAM** consumer PCs (AMD Vulkan & NVIDIA CUDA), with **zero C++ compilation**, **zero Visual Studio**, an **OpenAI-compatible API** for VS Code (Cline, Roo Code, Continue, OpenCode), and a dark **Web Studio**.

All bundled code and default models are **100% Apache 2.0**.

---

## Key Highlights

- **Zero-Compile Windows Setup (`.\setup.cmd`):** Automatically installs Python (if needed), downloads official prebuilt `llama.cpp b12720` Win64 **Vulkan** or **CUDA 12.4** binaries, and downloads verified GGUF models with resumable downloads. **No Visual Studio or C++ Build Tools required.**
- **Smart Hybrid GPU+RAM MoE Engine:**
  - Keeps attention, shared/loop layers, `q8_0` KV cache (`16K` context), `DSpark` block-drafting model, and `mmproj` Vision projector inside **~2.55 GiB GPU VRAM** while streaming sparse routed MoE experts from system RAM (`--cpu-moe --no-op-offload --no-host`).
  - Eliminates Vulkan's `7.63 GiB` compute scratchpad overflow on 6 GB cards — cutting projected VRAM from **9.28 GiB → 2.55 GiB** and accelerating prompt eval by **23×**!
- **Live VRAM Protection:** Automatically checks live free Dedicated VRAM before startup and caps context cleanly if browser/desktop apps are consuming heavy VRAM.
- **Dark Web Studio (`http://127.0.0.1:7870`) + OpenAI API (`http://127.0.0.1:8080/v1`):**
  - Streaming code chat with collapsible reasoning (`<think>`), 1-click code-block copy, screenshot/UI image upload (Inkling 2 Mini Vision), live Dedicated/Shared VRAM + RAM bars, and 1-click VS Code config snippets.
  - Uses ports `7870` / `8080` so it runs cleanly side-by-side with **PicoVid** (`7860`).

---

## Verified Real-World Benchmarks (AMD Radeon RX 5600 XT 6 GB Vulkan + Ryzen 5 5500 + 32 GB RAM)

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
# 1. One-click zero-compile setup (installs llama.cpp b12720 + models)
.\setup.cmd

# 2. Launch Dark Web Studio (http://127.0.0.1:7870)
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
