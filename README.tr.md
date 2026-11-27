# PicoCode v0.7.0 — Düşük Donanımlı 4–6 GB GPU'larda (AMD & NVIDIA) Yerel 30B Kodlama Yapay Zekası & Terminal Ajanı

[![Regression Tests](https://github.com/ffalcon3098-droid/PicoCode/actions/workflows/tests.yml/badge.svg)](https://github.com/ffalcon3098-droid/PicoCode/actions/workflows/tests.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> [For English documentation, see README.md](README.md) · **Platform:** Windows 10/11 (x64) · **GPU Desteği:** AMD Radeon (Vulkan), NVIDIA GeForce GTX/RTX (CUDA 12.4 & Vulkan), Intel Arc (Vulkan)

**PicoCode**, **Qwen3-Coder-30B-A3B (`--load-mode none` ile `97.5 tok/s` prompt işleme + Adaptif Hibrit GPU MoE ve kayıpsız `ngram-mod` ile `18–30 tok/s` üretim)**, **Inkling 2 Mini (26.4B / 3.4B Looped MoE + DSpark + Vision)** ve **Qwen2.5-Coder-7B** gibi güncel açık kaynaklı kodlama modellerini sıradan **4–6 GB VRAM** bilgisayarlarda **sıfır C++ derlemesi**, **İngilizce ana arayüzlü 24-bit TrueColor Terminal Kodlama Ajanı (`.\agent.cmd`)**, **VS Code uyumlu OpenAI API** ve güvenlik sertleştirmeli **Web Studio** ile çalıştırır.

Tüm proje kodu ve varsayılan modeller **%100 Apache 2.0** lisanslıdır.

---

## AMD & NVIDIA GPU Uyumluluğu + Dürüst Donanım Katmanları

PicoCode ekran kartınızı otomatik algılar (`scripts/probe_hardware.py`), uygun resmi hazır **`llama.cpp` Vulkan veya CUDA 12.4** ikilisini indirir ve katman dağıtımını / KV önbelleğini OOM olmadan en yüksek hıza ayarlar:

| GPU Üreticisi / Modeller | Otomatik Backend | Durum | Notlar |
|---|---|---|---|
| **AMD Radeon RX** (RX 5500 XT / **RX 5600 XT 6 GB** / RX 6600 / RX 7600+) | **Vulkan** (`ggml-vulkan`) | **Fiziksel Doğrulandı (RX 5600 XT 6 GB)** | Windows'ta ROCm derdi yok; resmi `llama.cpp b12807` Vulkan motoruyla tak-çalıştır. |
| **NVIDIA GeForce GTX / RTX** (GTX 1650 / 1660 / RTX 2060 / 3050 / 3060 / 4060+) | **CUDA 12.4** (veya Vulkan) | **Otomatik Kurulum (`setup.cmd`)** | `nvidia-smi` algılar, resmi `llama.cpp b12807` CUDA 12.4 + `cudart` DLL'lerini otomatik indirir. |
| **Intel Arc** (A380 / A580 / A750 / B580) | **Vulkan** (`ggml-vulkan`) | **Vulkan ile Desteklenir** | Aynı hazır Vulkan ikilisi ve otomatik VRAM bütçesiyle çalışır. |

| Sistem Katmanı (VRAM + Sistem RAM) | Önerilen Model | Beklenen Deneyim |
|---|---|---|
| **4–6 GB VRAM + 16 GB RAM** *(Giriş / Düşük Donanım)* | `qwen25-coder-7b` (`4.36 GiB`) | **6 GB VRAM'e %100 sığar** (4 GB'da hibrit) — anlık açılış, çok hızlı kodlama & araç çağrısı. |
| **6 GB VRAM + 32 GB RAM** *(En İdeal — Ölçüldü)* | `qwen3-coder-30b` (`15.25 GiB`) & `inkling2-mini` (`16.26 GiB`) | Adaptif Hibrit GPU MoE + `--load-mode none`: **`18–30 tok/s` 30B/3B kodlama** + Vision + DSpark! |
| **8 GB+ VRAM + 32 GB RAM** *(RTX 3060/4060 / RX 6600+)* | Tüm modeller (`32K–64K` bağlam) | Daha fazla MoE katmanı GDDR6'da tutulur, daha yüksek hız elde edilir. |

---

## Öne Çıkan Özellikler (`v0.7.0`)

- **Claude Code Tarzı 24-bit Terminal Kodlama Ajanı (`.\agent.cmd` / `.\picocode.cmd agent`):**
  - **İngilizce Varsayılan Arayüz (Çok Dilli Model Yanıtı):** TUI, Web Studio, kurulum ve CLI çıktıları evrensel İngilizce standarttadır; model ise sorduğunuz dilde (Türkçe/İngilizce) yanıt verir.
  - **Gerçek Projeler & İnceleme (Review) İçin 10 Araç:** `project_overview` (tek adımda mimari/manifest/sıcak dosya özeti), çok dilli `repo_map` (Python, JS/TS, **Rust `.rs` + `#[tauri::command]`**, **Go `.go`**), `read_file`, `search_code`, `list_files`, `git_status`, `git_diff`, `edit_file` (anlık syntax kontrolü + CRLF koruma), `write_file` ve `run_command`.
  - **%100 KV-Cache Koruması & Garantili Final Sentez:** Final adımda `tools` şemasını sabit tutup `tool_choice="none"` göndererek `f_keep = 1.000` KV-cache isabeti korur; adım-başı paralel çıktı bütçesi ve kanıt-öncelikli inceleme kuralıyla yanlış alarmları engeller.
  - **Klasör Güven Onayı (`Trust Folder`), Proje Seçici (`/project` / `/cd`), Dosya Sabitleme (`/add` / `/drop`) & Canlı `/model` Geçişi:** Model VRAM'de sıcak kalırken **0 saniyede** proje değiştirir, sohbeti kaybetmeden `/model inkling` ↔ `/model qwen` geçişi yapar.
- **Tek Tıkla Sıfır Derlemeli Kurulum (`.\setup.cmd`):** Visual Studio gerektirmez! Resmi `llama.cpp b12807` Win64 **Vulkan/CUDA**, TUI paketleri ve doğrulanmış GGUF modellerini (Zip-Slip korumalı ve kaldığı yerden devam eden indiriciyle) kurar.
- **RX 5600 XT 6 GB + 32 GB RAM Ampirik Optimizasyonları:**
  - **`Qwen3-Coder-30B-A3B`:** `--n-cpu-moe 42` (`6` GPU MoE), **`--load-mode none`** (`+18.1%` pp: `82.5 → 97.45 tok/s`, `20.24 tok/s` tg), **`-b 1024 -ub 256 -t 6 -tb 12`** ve `0 MB VRAM` kayıpsız `ngram-mod`.
  - **`Inkling 2 Mini`:** **`-b 1024 -ub 512 -t 6 -tb 12 --cpu-moe --no-op-offload --no-reasoning-preserve`** + `DSpark` (`q8_0` draft KV, `%64–83` kabul) + Vision `mmproj`.
- **5 ms Native Windows PDH VRAM Ölçümü & Web Studio Güvenlik Kalkanı:** `pdh.dll` ile anlık Dedicated/Shared VRAM ölçümü ve `http://127.0.0.1:7870` üzerinde tam Host + Origin + Content-Type + CSRF (`X-PicoCode-Token`) koruması.

---

## Gerçek Donanım Test Sonuçları (AMD Radeon RX 5600 XT 6 GB Vulkan + Ryzen 5 5500 + 32 GB RAM)

### 1. `Qwen3-Coder-30B-A3B-Instruct` (`IQ4_XS`, `15.25 GiB`, Tam `32,768` Bağlam, Canlı `llama-server`)

| Profil (`32K` bağlam, `q8_0` KV) | Yükleme | Sıfırdan Kod (`tg`) | Gerçek Dosya Refactor (`pp` / `tg`) | Takip Turu (`KV Önbellek`) | Spekülatif (`0 MB VRAM`) | Pik Dedicated / Shared |
|---|---:|---:|---:|---:|---:|---|
| Eski Varsayılan (`b12720`, `--cpu-moe`, `b512/ub256`) | `3.1 sn` | `16.20 tok/sn` | `58.8` / `16.89 tok/sn` (`36.7s`) | `1267/1305` · `17.05 tok/sn` | Kapalı | `3.77 / 0.50 GiB` |
| **`v0.2.0` Optimize (`b12807`, `-ncmoe 44`, `b2048/ub512`, `poll 100`, `ngram-mod`)** | **`3.5 sn`** | **`18.25 tok/sn`** | **`71.6` (+22%) / `30.21 tok/sn` (+79%!) (`26.1s`)** | **`1267/1305` (%97) · `18.25 tok/sn`** | **`232/232` (%100)** | **`4.89 / 0.59 GiB`** |
| **`v0.2.0` `q4_0` KV (`-ncmoe 42`, 6 GPU MoE, `ngram-mod`)** | **`3.5 sn`** | **`18.42 tok/sn`** | **`71.8` (+22%) / `29.50 tok/sn` (+75%!) (`26.3s`)** | **`1267/1305` (%97) · `18.92 tok/sn`** | **`232/232` (%100)** | **`4.88 / 0.59 GiB`** |

### 2. `Inkling-2-Mini` (`UD-Q4_K_M`, `15.47 GiB`, `16K` Bağlam + Vision `mmproj` + DSpark)

Windows 11 üzerinde masaüstü + tarayıcı açıkken (`~1.65 GiB` taban Dedicated VRAM), `Inkling-2-Mini-UD-Q4_K_M.gguf` (`15.47 GiB`), `16,384` bağlam (`q8_0` KV), Vision `mmproj` açık, `256` token üretim testi:

| Profil | Sıcak Yükleme | Prompt İşleme (`pp`) | Üretim Hızı (`tg`) | Toplam Süre | Pik Dedicated (Masaüstü Dahil) | Pik Shared |
|---|---:|---:|---:|---:|---:|---:|
| İlk Hal (`--cpu-moe`, varsayılan op-offload, DSpark) | 24.6 sn | 0.70 tok/sn | 8.13 tok/sn | 136.8 sn | 4.38 GiB | 9.04 GiB |
| **PicoCode Optimize (`--no-op-offload --no-host -t 6`, DSpark Kapalı)** | **4.8 sn** | **15.5 tok/sn** | **7.68 tok/sn** | **37.0 sn** | **4.21 GiB** | **1.16 GiB** |
| **PicoCode Optimize + DSpark (`--spec-draft-n-max 6`, Varsayılan)** | **6.8 sn** | **14.4 tok/sn** | **8.32 tok/sn** | **34.7 sn** | **4.26 GiB** | **2.12 GiB** |

- **DSpark Kabul Oranı:** Python kodlama görevlerinde **%72 – %83** (`207/287` kabul).
- **`llama.cpp` GPU VRAM İhtiyacı (`16K` bağlam):** Sadece **`2.55 GiB`** (`1.24 GiB` model + `0.38 GiB` KV + `0.88 GiB` compute).

---

## Desteklenen Modeller (`configs/models_registry.json`)

| Model ID | Mimari | Boyut | Önerilen Donanım | Açıklama |
|---|---|---:|---|---|
| `inkling2-mini` | 26.4B / 3.4B Looped MoE | 16.26 GiB (DSpark + Vision dahil) | **6 GB VRAM + 32 GB RAM** | Varsayılan · Kodlama + Akıl Yürütme + Ekran Görüntüsü |
| `qwen3-coder-30b` | 30B / 3B MoE | 15.25 GiB (`IQ4_XS`) | **6 GB VRAM + 32 GB RAM** | Saf yüksek bağlamlı kodlama (32K'ya kadar) |
| `qwen25-coder-7b` | 7B Dense | 4.36 GiB (`Q4_K_M`) | **4–6 GB VRAM + 16 GB RAM** | 6 GB VRAM'e %100 sığar (çok hızlı) |

---

## Hızlı Başlangıç (Windows)

```powershell
# 1. Tek tıkla kurulum (llama.cpp b12807 + TUI paketleri + model indirme, Visual Studio gerektirmez!)
.\setup.cmd

# 2. Claude Code tarzı Terminal Ajanını aç (proje seçici + klasör güven onayı)
.\agent.cmd

# 3. Veya karanlık Web Studio arayüzünü aç (http://127.0.0.1:7870)
.\web-studio.cmd

# 4. Veya VS Code için sadece OpenAI API sunucusunu başlat (http://127.0.0.1:8080/v1)
.\picocode.cmd --model inkling2-mini

# 5. Kendi ekran kartınızda canlı hız & VRAM testi çalıştırın
.\picocode.cmd --model inkling2-mini --bench
```

### VS Code Bağlantısı (Cline / Roo Code / Continue)

- **Provider:** `OpenAI Compatible`
- **Base URL:** `http://127.0.0.1:8080/v1`
- **API Key:** `picocode`
- **Model ID:** `inkling2-mini` (veya `qwen3-coder-30b`, `qwen25-coder-7b`)

---

## Lisans

[Apache License, Version 2.0](LICENSE) altında lisanslanmıştır.
