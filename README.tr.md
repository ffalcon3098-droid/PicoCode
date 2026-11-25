# PicoCode v0.1.0 — 4–6 GB VRAM Bilgisayarlarda Güncel Yerel Kodlama Yapay Zekası

> [For English documentation, see README.md](README.md)

**PicoCode**, **Inkling 2 Mini (26.4B toplam / 3.4B aktif Looped MoE + DSpark + Vision)**, **Qwen3-Coder-30B-A3B** ve **Qwen2.5-Coder-7B** gibi güncel açık kaynaklı kodlama modellerini sıradan **4–6 GB VRAM + 32 GB RAM** bilgisayarlarda (AMD Vulkan ve NVIDIA CUDA) **sıfır C++ derlemesi**, **Visual Studio gerektirmeyen tek tık kurulum**, **VS Code uyumlu OpenAI API** ve karanlık **Web Studio** arayüzüyle çalıştırır.

Tüm proje kodu ve varsayılan modeller **%100 Apache 2.0** lisanslıdır.

---

## Öne Çıkan Özellikler

- **Tek Tıkla Sıfır Derlemeli Kurulum (`.\setup.cmd`):** Visual Studio veya C++ Build Tools gerektirmez! Resmi derlenmiş `llama.cpp b12720` Win64 **Vulkan** veya **CUDA 12.4** motorunu ve seçilen GGUF modellerini kaldığı yerden devam eden indiriciyle otomatik kurar.
- **Akıllı Hibrit GPU + RAM MoE Motoru:**
  - Attention, döngü (loop) çekirdeği, `q8_0` KV önbelleği (`16K` bağlam), `DSpark` hızlandırıcı ve `mmproj` Vision projektörünü yalnızca **~2.55 GiB GPU VRAM** içinde tutar; seyrek MoE uzmanlarını (`--cpu-moe --no-op-offload --no-host`) sistem RAM'inden akıtır.
  - Vulkan'ın 6 GB kartlarda `7.63 GiB` geçici `compute` tamponu taşmasını tamamen ortadan kaldırır: öngörülen GPU VRAM **9.28 GiB → 2.55 GiB** düşer, prompt işleme hızı **23×** artar!
- **Canlı VRAM Koruması:** Başlatmadan önce canlı boş Dedicated VRAM'i ölçer; tarayıcı/masaüstü fazla VRAM tüketiyorsa bağlamı otomatik daraltarak WDDM takılmasını önler.
- **Karanlık Web Studio (`http://127.0.0.1:7870`) + OpenAI API (`http://127.0.0.1:8080/v1`):**
  - Canlı akışlı kodlama sohbeti, katlanabilir düşünme (`<think>`) bölümü, kod blokları için tek tık kopyala, **ekran görüntüsü yükleme (Inkling 2 Mini Vision)**, canlı Dedicated/Shared VRAM + RAM göstergeleri ve tek tık **VS Code (Cline / Roo Code / Continue)** ayarları.
  - `7870` / `8080` portlarını kullandığı için **PicoVid** (`7860`) ile çakışmadan aynı anda çalışır.

---

## Gerçek Donanım Test Sonuçları (AMD Radeon RX 5600 XT 6 GB Vulkan + Ryzen 5 5500 + 32 GB RAM)

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
