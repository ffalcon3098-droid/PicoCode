# PicoCode v0.2.0 — 4–6 GB VRAM Bilgisayarlarda 30 tok/s Yerel 30B Kodlama Yapay Zekası

> [For English documentation, see README.md](README.md)

**PicoCode**, **Qwen3-Coder-30B-A3B (Adaptif Hibrit GPU MoE + kayıpsız `ngram-mod` ile `30.2 tok/s`'ye kadar)**, **Inkling 2 Mini (26.4B / 3.4B Looped MoE + DSpark + Vision)** ve **Qwen2.5-Coder-7B** gibi güncel açık kaynaklı kodlama modellerini sıradan **4–6 GB VRAM + 32 GB RAM** bilgisayarlarda (AMD Vulkan ve NVIDIA CUDA) **sıfır C++ derlemesi**, **VS Code uyumlu OpenAI API** ve karanlık **Web Studio** arayüzüyle çalıştırır.

Tüm proje kodu ve varsayılan modeller **%100 Apache 2.0** lisanslıdır.

---

## Öne Çıkan Özellikler

- **Tek Tıkla Sıfır Derlemeli Kurulum (`.\setup.cmd`):** Visual Studio veya C++ Build Tools gerektirmez! Resmi derlenmiş `llama.cpp b12807` Win64 **Vulkan** veya **CUDA 12.4** motorunu ve doğrulanmış GGUF modellerini otomatik kurar.
- **Adaptif Hibrit GPU+RAM MoE + 0 MB VRAM Kayıpsız `ngram-mod` (`v0.2.0`):**
  - Başlatmadan önce canlı boş Dedicated VRAM'i ölçer; `650 MiB` taşma güvenliği bırakarak **`4–6` MoE uzman katmanını doğrudan 6 GB GDDR6'ya (`--n-cpu-moe N`)** alır ve tam **`32K` `q8_0` KV önbelleğini** VRAM'de tutar.
  - **`--spec-type ngram-mod`** (0 MB VRAM, %100 kayıpsız spekülatif çözümleme), **`--poll 100`**, **`--cache-reuse 256`** ve **`-b 2048 -ub 512 -tb 12`** ile sıfırdan kodda **`18.3–18.9 tok/s`**, gerçek kod düzenleme/refactor işlerinde **`30.2 tok/s` (`232/232` %100 kabul!)** hızına ulaşır.
- **Canlı VRAM Koruması:** Tarayıcı veya masaüstü fazla VRAM tüketiyorsa GPU MoE katman sayısını ve bağlamı otomatik daraltarak WDDM takılmasını önler.

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
# 1. Tek tıkla kurulum (llama.cpp b12720 + model indirme, Visual Studio gerektirmez!)
.\setup.cmd

# 2. Karanlık Web Studio arayüzünü aç (http://127.0.0.1:7870)
.\web-studio.cmd

# 3. Veya VS Code için sadece OpenAI API sunucusunu başlat (http://127.0.0.1:8080/v1)
.\picocode.cmd --model inkling2-mini

# 4. Kendi ekran kartınızda canlı hız & VRAM testi çalıştırın
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
