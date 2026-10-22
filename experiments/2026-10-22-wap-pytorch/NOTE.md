# WAP (Watch, Attend and Parse) — Theano deposunun PyTorch'ta ayağa kaldırılması

- **Tarih:** 2026-10-22   **Kişi:** csahin   **Dal:** deneme_Branch   **Git SHA:** (commit sonrası)
- **Soru:** Zhang ve ark. (2017) WAP mimarisi, bu depodaki el yazısı→LaTeX OCR hattına (`aicls.ocr`,
  ADR-0013 `seq2seq_image`) uyarlanabilir mi; CROHME 2014 üzerinde kağıttaki ExpRate'e ne kadar yaklaşılır?
- **Kurulum:** veri `C:\aicls-data\datasets\crohme_wap` (JianshuZhang/WAP `data/`, commit 47cce26;
  `manifest.json` sha256: train `8ac09c99…`, test2014 `7b7c28af…`); model `wap.py` (1,82 M parametre,
  orijinal `VGG/nmt.py` birebir); `config.yaml` = orijinal `train_nmt.py` hiperparametreleri;
  donanım RTX 5090 (Windows-yerli venv); ~1 adım/s, dönem ~17 dk.
- **Ölçüm:** token WER (`aicls.eval.metrics.edit_distance`), ExpRate (tam dizi eşleşmesi, WAP tanımı);
  n = 985 (CROHME 2014 test; 1 örnek `maxImagesize` 500 000 px sınırıyla elendi, orijinalde de öyle).

## Makale özeti (Pattern Recognition 71, 2017, s. 196–206)

| Bileşen | Kağıt / orijinal kod | Bu port |
|---|---|---|
| Watcher | VGG-tipi FCN: 4 blok × 4 konv (3×3), kanal 32/64/64/128, BN+ReLU, blok sonu 2×2 max-pool; çıktı 128 kanallı özellik haritası (H/16 × W/16) | `Encoder` aynı; `-1` dolgu + pool ile `ceil` boyut, maske `[::2, ::2]` |
| Parser | GRU (256) + Bahdanau dikkati (128); **kapsama (coverage)** vektörü: geçmiş dikkat toplamı 5×5 evrişimle (128 kanal) dikkat enerjisine eklenir → aşırı/eksik ayrıştırma azalır | `CoverageAttentionGRU` (iki aşamalı `gru_cond`) |
| Okuma | `W_h h + W_e y_{t-1} + W_c c_t` → maxout(2) → softmax (111 sembol + `<eol>`) | `WAP.readout` |
| Eğitim | Adam 2e-4, L2 1e-4, gradyan kırpma 100, dropout 0.2, yığın 8 (≤ 500 k px), erken durdurma (WER, sabır 15, lr/2) | aynı |
| Çözümleme | Işın araması k=10 | `WAP.beam` (uzunluk-normalize) |
| Sonuç | CROHME 2014 ExpRate **46,55 %** (tekil model ~40 %, ensemble ile 46,55); CROHME 2016 44,55 % | aşağıda |

Not: ScienceDirect sayfası botlara 403 verdiğinden makale metni doğrudan çekilemedi; mimari ve
hiperparametreler deponun `VGG/nmt.py` + `train_nmt.py` dosyalarından, sayılar yayımlanmış tablolardan.

## Sonuç
| Koşul | Metrik | Değer | n |
|---|---|---|---|
| (eğitim sürüyor — `C:\aicls-data\runs\wap-pytorch\train_log.jsonl`) | | | |

## Yorum
- Orijinal kod Theano 0.10 + CUDA 8 + Python 2 ister; bu yığın RTX 5090 / Python 3.11'de kurulamaz.
  Bu yüzden depo "ayağa kaldırma" = aynı mimari ve hiperparametrelerle PyTorch portu (`wap.py`).
- WAP deposu CROHME 2014 train/test görüntülerini ve 2016 test görüntülerini (etiketsiz) kendi
  içinde dağıtıyor; ayrı indirme gerekmedi. CROHME 2016 için etiket yok → yalnızca 2014 ölçülür.
- Darboğaz: çözücü adım döngüsü Python'da (T ≤ 200 adım × küçük çekirdekler) → GPU gecikme-sınırlı,
  ~1 adım/s. Daha büyük yığın veya CUDA graph ile 5–10× hızlanır ama kağıt kurulumundan sapar.
- `aicls.ocr` ile uyum: `aicls.ocr.models.encoders` zaten `vgg16`/`cnn_small` + dikkatli çözücü
  sunuyor; WAP'tan taşınacak özgün parça **kapsama dikkati** (`CoverageAttentionGRU.step`) ve
  görüntü alanına göre kovalı yığınlama (`run.py: make_batches`).

## Karar / öneri
- [ ] `src/`'a taşı: `aicls.ocr.models.decoders`'a `coverage_gru` çözücü seçeneği (PR: …)
- [ ] ADR-0013'e `Girdi:` satırı (WAP kapsama dikkati karşılaştırması)
- [ ] Bırak (neden)
- Okuma programına dönüş: Zhang ve ark. 2018 (DenseWAP, çok ölçekli dikkat) — depodaki `DenseNet/`.
