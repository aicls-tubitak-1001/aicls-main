# El yazısı matematik tanıma — kod tabanı taraması, WAP kararı ve yol haritası

- **Tarih:** 2026-10-22   **Hazırlayan:** Ç. Şahin (Claude Code; 3 araştırma ajanı + proje taraması)   **Dal:** docs/hmer-literatur
- **Besledikleri:** ADR-0013 (el yazısı girdi biçimi, karar ~2026-10-30), `experiments/2026-10-22-wap-baseline/NOTE.md`, İP-3 OCR taban çizgisi
- **Veri setleri:** ayrı belge → [veri-setleri.md](veri-setleri.md)
- **Doğrulama notu:** GitHub/arXiv/HF sayfaları 2026-10-22'de çekildi; "DOĞRULANMADI" işaretli satırlar yalnızca arama özetine dayanır.

## 1. Soru ve kısa yanıt

**Soru.** Zhang ve ark. (Pattern Recognition 71, 2017) "Watch, Attend and Parse" (WAP) reposu, el yazısı matematiği API'siz ve yerelde tanımak için AI-CLS'nin başlangıç kod tabanı olabilir mi? Olamazsa akademik temelli (tercihen SCI) hangi kod tabanları uygun?

**Yanıt.** **Hayır, WAP kod tabanı olarak başlangıç noktası olamaz; makale yöntem/kıyas referansı olarak kalır.** Dört bağımsız ret nedeni var (§2). Projenin kendi `aicls.ocr` seq2seq altyapısı WAP'tan daha moderndir ve ADR-0006/0013'e zaten uyar; dışarıdan devralınacak şey *kod tabanı* değil, *ağırlık ve fikir*dir. Öneri (§4): (i) kendi altyapıda MathWriting taban çizgisi; (ii) Uni-MuMER (NeurIPS 2025, Apache-2.0, Qwen3-VL-2B/4B) öğretmen/yüksek doğruluk referansı; (iii) kapsama dikkatini (WAP/CoMER fikri) çözücüye ablasyon bayrağı olarak ekleme; (iv) CROHME 2023 ile literatürle karşılaştırılabilir ölçüm.

## 2. WAP: smoke test sonucu ve ret gerekçesi

Ayrıntılar `experiments/2026-10-22-wap-baseline/NOTE.md`. Özet:

| Boyut | Bulgu | Sonuç |
|---|---|---|
| **Lisans** | Repo'da LICENSE yok; issue #16 (2020) yorumsuz kapanmış | Kod varsayılan "tüm hakları saklı"; proje lisansı beklemedeyken (ADR-0002) kopyalanamaz — **tek başına yeterli ret** |
| **Yığın** | CUDA 8.0 + cuDNN 7 + Theano 0.10 + libgpuarray, Python 2.7 | Theano 2017'de terk edildi; libgpuarray sm_120 (RTX 5090) için derlenemez; 15 dosyadan 13'ü Py3'te derlenmiyor (2to3 sonrası 10/10 derleniyor ama `scipy.misc` kalıyor) |
| **CPU yolu** | BatchNorm `theano.gpuarray.dnn`'e sabitlenmiş | Sahte veriyle bile CPU ileri geçişi yok → smoke test "derlenir mi?" düzeyinde kalır |
| **Modalite** | Çevrimdışı görüntü (BMP → GRU) | ADR-0013 çevrimiçi izi birincil aday sayıyor; WAP'ın iz karşılığı TAP (IEEE T-MM 2018) aynı yığın, aynı lisanssızlık |
| **Doğruluk** | Yazarın bildirdiği tek model ~44–50 % ExpRate (CROHME 2014); 5'li topluluk kodu repoda yok | 2017 düzeyi; güncel küçük modeller 60–65 %, VLM ince ayarı ~80 % |
| **Veri** | Repo CROHME 2014 train/test + 2016 test BMP'lerini ve 41 MB ağırlığı doğrudan içeriyor | CROHME "araştırma amaçlı" lisanslı; resmi kaynaktan (TC11/Zenodo) ve DATASET_DOWNLOAD defter satırıyla alınmalı, repodan değil |
| **Projeye katkı** | Kapsama (coverage) dikkati | `aicls.ocr.models.decoders`'a ~50 satırlık isteğe bağlı ek; kod tabanı devralmayı gerektirmez |

Sözlük kapsamı (111 sembol): `\lim \limits \int \sum \frac \sqrt \infty \rightarrow` var; **`\partial \ln \to \prod \left \right` yok** — kalkülüs müfredatımız için eksik. Eğitim setinde `\lim` 287, `\int` 550, `\sum` 552, `\frac` 2 494 ifade (n = 8 835).

## 3. Alternatif kod tabanları — karşılaştırma ve yargı

Değerlendirme ölçütleri, önem sırasıyla: (1) lisans (proje lisansı beklemede → yalnız MIT/Apache/BSD güvenli); (2) torch 2.11 + cu128 + Windows'ta çalışma (eski pin'ler sm_120'de çalışmaz); (3) indirilebilir ağırlık; (4) akademik temel (SCI dergi / üst düzey konferans); (5) modalite (iz/görüntü) ve `aicls.ocr.Seq2Seq` arayüzüne uyum; (6) çıkarım maliyeti (ADR-0014 WebSocket bütçesi).

### 3.1 Çevrimdışı (görüntü → LaTeX)

| Model | Yayın | Lisans | Yığın pin'i | Ağırlık | ExpRate CROHME 14/16/19 | Yargı |
|---|---|---|---|---|---|---|
| **WAP** | Pattern Recognition 2017 (SCI) | **yok** | Theano/CUDA 8/Py2 | repoda | ~44–50 (tek model) | **Bırak** |
| whywhs PyTorch-WAP | yok (topluluk) | MIT | torch 1.0, Py3.6 | HMER ağırlığı yok | 38.6 (2016) | Bırak |
| BTTR | ICDAR 2021 | **MIT** | torch 1.7, PL, Py3.7 (arşiv) | yok | 54–57 | Kod referansı (MIT) — çift yönlü eğitim fikri |
| CoMER | ECCV 2022 | **yok** (#15 yanıtsız) | torch 1.8.1, PL 1.4.9 (arşiv) | 52 MB ckpt | 59.3 / 59.8 / 63.0 | Fikir (kapsama); kod kopyalanamaz |
| CAN | ECCV 2022 | **MIT** | belirsiz; yığın 8 → 32 GB VRAM | yok (#42 açık) | ~57 / 56 / 55 | MIT ama ağırlıksız; sayma kaybı fikri |
| ABM | AAAI 2022 | CC BY-NC-SA (README) | torch 1.6, Py3.6 | .pkl (yükleme hatası #12) | 56.9 / 52.9 / 54.0 | Bırak (NC) |
| SAN | CVPR 2022 | **yok** | torch 1.6 | yok | 56.2 / 53.6 / 53.5 (aug. 63/61/62) | Bırak |
| PosFormer | ECCV 2024 | "BSD-2, yalnız akademik" (dosya yok) | torch 1.8.1, PL 1.4.9 | 78 MB ckpt | 62.7 / 61.0 / 65.0 | Yalnız akademik kıyas |
| TAMER | AAAI 2025 | **yok** | torch 1.8.1, PL 1.4.9 | 95 MB ckpt | 61.2 / 60.3 / 62.0; HME100K 68.5 | Kıyas; kod kopyalanamaz |
| ICAL | ICDAR 2024 | **yok** | torch 1.8.1 | 57 MB ckpt | 60.6 / 58.8 / 60.5 | Kıyas |
| **Uni-MuMER** | **NeurIPS 2025 (spotlight)** | **Apache-2.0** | transformers ≥ 5 (Qwen3.5), LLaMA-Factory; vLLM **gerekmez** | HF `phxember/*` (Qwen3-VL-2B/4B, Qwen3.5-2B/4B, Qwen2.5-VL-3B) | **82.3 / 78.3 / 79.8** (3B); MathWriting test EM 50.7–54.3 | **Aday #1 (öğretmen / yüksek doğruluk)** |
| TexTeller 3.0 | arXiv 2508.09220 (ön baskı) | Apache-2.0 | `pip install texteller` | HF, 298 M | 88.0 / 85.9 / 85.8 (eğitim verisi örtüşmesi DOĞRULANMADI) | Fotoğraf girişi için hazır araç; Windows ONNX sorunları (#97) |
| UniMERNet | arXiv 2404.15254 | Apache-2.0 | Py3.10+ | HF tiny/small/base | CROHME yok; UniMER-Test HWE 6 332 örnek | Basılı+el yazısı kıyas |
| TrOCR ince ayarı (`fhswf/TrOCR_Math_handwritten`) | TrOCR: AAAI 2023 | MIT (taban) / AFL-3.0 (ince ayar) | HF transformers (projede zaten var) | HF | MathWriting alt kümesinde 77.8 % tam eşleşme (bölünme DOĞRULANMADI) | Hızlı taban çizgisi; `VisionEncoderDecoder` ile 1 günde kurulur |
| Pix2Text / Texify / Nougat / Donut | — | MIT / GPL-3 / MIT(ağırlık NC) / MIT | — | — | el yazısı iddiası yok | Bırak |
| **tuanfptu UniversityHMER (bitirme tezi, 2026-08)** | tez | **MIT** | TAMER-A3 adaptörü + Uni-MuMER Qwen3.5-2B LoRA | HF | **Gerçek sınıf görüntüsünde** 71.2 % (TAMER-A3, 0.31 s/görüntü) ve 74.8 % (Uni-MuMER LoRA, 2.55 s, RTX 3090) | Üniversite kalkülüsü — doğrudan ilgili tarif; olgunluğu düşük (2 yıldız) |

### 3.2 Çevrimiçi (iz / InkML → LaTeX)

| Model | Yayın | Lisans | Durum | Yargı |
|---|---|---|---|---|
| TAP | IEEE T-MM 2018 (SCI) | **yok** | Theano/CUDA 8, 2020'den beri ölü | Bırak (WAP ile aynı nedenler) |
| SCAN (iz-kısıtlı dikkat) | Pattern Recognition 119, 2021 (SCI) | kod yok | — | Makaleden yeniden uygulama adayı (iz + görüntü füzyonu) |
| Google MathWriting CTC-Transformer / PaLI | arXiv 2404.10690 (KDD '25 DOĞRULANMADI) | veri CC BY-NC-SA 4.0, kod Apache-2.0 | **model yayınlanmadı**; CER 5.49 % / EM 60 % (CTC-Tfm), PaLIGemma-3B EM 69 % | **Yeniden üretilecek taban çizgisi** — mimari basit (11 katman, 512, CTC) ve bizim `stroke_transformer`'a yakın |
| InkFM (PaLIGemma-3B) | arXiv 2503.23081 | yayın sözü var, çıkmadı | CER 3.88 % | İzle |
| Seshat | Pattern Recognition 2016 (SCI) | **GPL-3** | C++ CPU, ağırlıklı | Klasik (gramer) kıyas noktası; GPL → yalnız ayrı süreç |
| InkSight | arXiv 2402.05804 | Apache-2.0 | TF 2.20 | Yanlış görev (fotoğraf → iz "derendering") |

**Yargı.** Çevrimiçi tarafta sağlıklı, PyTorch, ağırlıklı, lisanslı bir akademik repo **yok**. Bu, projenin kendi `seq2seq_stroke` ailesini sürdürmesini zorunlu kılıyor; literatür kıyası için MathWriting makalesinin CTC-Transformer'ı aynı tokenizer ve CER tanımıyla yeniden üretilir.

### 3.3 Yerel VLM'ler (sıfır-atış el yazısı matematik)

Uni-MuMER Tablo 1 (CROHME 14/16/19 ExpRate): Qwen2.5-VL-3B 38.6/37.8/37.5; -7B 56.0/50.9/49.6; -72B 59.7/54.3/55.1; GPT-4o 50.6/46.0/49.8. **Sıfır-atış VLM, ince ayarlı küçük modellerin altında** — ince ayar şart. Lisans notu: Qwen3-VL ve Qwen3.5 Apache-2.0; Qwen2.5-VL-3B "Qwen Research" lisansı (DOĞRULANMADI) → **Qwen3-VL-2B/4B varyantı seçilmeli.** Windows'ta vLLM yok; transformers + SDPA (flash-attn sm_120 wheel'i DOĞRULANMADI). Türkçe çevre metni için Qwen3-VL-8B / Qwen3.5-4B sıfır-atış; Türkçe satır düzeyi el yazısı veri seti bulunamadı.

### 3.4 Lisans zinciri uyarısı

MathWriting **CC BY-NC-SA 4.0** (`catalog.yaml` zaten uyarıyor), CROHME 2023 CC BY-NC-SA 3.0, HME100K talep usulü. Uni-MuMER ağırlıkları bu NC setlerle eğitildi → türev ağırlıklar ticari aşamaya taşınamaz (hukuki inceleme). Projede üçüncü taraf kod/ağırlık politikası ADR'si yok; **bu belge o ADR'nin açılmasını öneriyor** (bkz. §4 Faz 0).

## 4. Yol haritası (8 hafta; ADR-0013 karar tarihi ~2026-10-30 ile hizalı)

| Faz | Hafta | İş | Kapı / çıktı |
|---|---|---|---|
| **0 — Karar** | 0 (bu hafta) | WAP → "bırak" (NOTE.md). ADR-0013'e `Girdi:` satırı. **Yeni ADR: üçüncü taraf kod/ağırlık/lisans politikası** (vendoring yok; yalnız MIT/Apache/BSD; NC ağırlıklar yalnız araştırma aşaması). Defter: `DATASET_DOWNLOAD` CROHME 2023 (Zenodo) satırı; `WEIGHTS_DOWNLOAD` Uni-MuMER Qwen3-VL satırı. | ADR PR'ları, defter satırları |
| **1 — Kendi taban çizgisi** | 1–2 | MathWriting (kapı açık) ile `online_stroke_tfm` ve `offline_resnet34_tfm` tam eğitim; ADR-0015 CER/TER/ExpRate; MathWriting tokenizer + Ek M CER tanımıyla **makale kıyası** (hedef CER ≤ 6 %, EM ≥ 55 %). Çözücüye `coverage: true` ablasyonu (WAP/CoMER fikri). | `reports/metrics/latest.json` ilk citable OCR CER'i |
| **2 — Öğretmen model** | 2–4 | Uni-MuMER Qwen3-VL-2B/4B: MathWriting test (render) + 200 örneklik iç kalkülüs setinde sıfır-atış; RTX 5090'da bf16 + SDPA gecikme ölçümü (ADR-0014 bütçesi). Karar: üretim (fotoğraf/PNG girişi) mi, yalnız sahte-etiket/damıtma öğretmeni mi? TrOCR-Math (`fhswf`) 1 günlük yedek taban çizgisi. | Deney notu `exp/unimumer-zeroshot`; gecikme tablosu |
| **3 — Kalkülüs verisi** | 4–6 | MathWriting'i müfredat sözlüğüne (`\lim_{x\to}`, `\frac{d}{dx}`, sınırlı `\int`, `\sum`) göre süzme + iç toplama sekmesiyle (etik kurul referansı, ADR-0012) 1–2 k iz; sentetik sahte-etiket (öğretmen). İnce ayar; Aida (sentetik limit) yalnız çevrimdışı artırma. | Veri manifesti (ADR-0008), ince ayar koşusu |
| **4 — Literatürle kıyas ve karar** | 6–8 | CROHME 2023 çevrimiçi test (Zenodo, açık) ile ExpRate — literatürle aynı dil; iç kalkülüs test setinde CER/ExpRate; ADR-0013 → "kabul edildi", `models/registry.yaml` birincil aile. | Gelişme raporu tablosu; ADR-0013 kapanışı |

**Riskler.** (1) NC lisanslar → ticari aşama öncesi hukuki inceleme; (2) VLM gecikmesi (3090'da 2.5 s) WebSocket bütçesini aşabilir → damıtma yolu hazır tutulmalı; (3) Windows'ta flash-attn yok → SDPA; (4) Türkçe el yazısı metin verisi yok → yalnız matematik, metin için sıfır-atış VLM; (5) MathWriting ~%1–2 etiket gürültüsü → ADR-0015 normalizasyonu ve `symeq` metriği.

## 5. Başlangıç kod tabanı sorusuna son söz

"Bu repoyu başlangıç kabul edebilir miyiz?" → **Hayır.** "Hangi repoyu başlangıç kabul edelim?" → **Hiçbirini kod tabanı olarak devralmayın; `aicls.ocr` zaten doğru soyutlamalara sahip.** Devralınacaklar: Uni-MuMER ağırlıkları (Apache-2.0) öğretmen olarak, BTTR (MIT) kodundan çift yönlü eğitim ve CoMER/WAP makalelerinden kapsama dikkati *fikir* olarak, MathWriting makalesinden tokenizer/CER sözleşmesi, CROHME 2023'ten karşılaştırılabilir test. SCI atıf omurgası: WAP (PR 2017), TAP (T-MM 2018), SCAN (PR 2021), Seshat (PR 2016), DAN (TPAMI 2023), HTR-VT (PR 2024); konferans: BTTR, CoMER, PosFormer, TAMER, Uni-MuMER.
