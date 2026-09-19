# ADR-0006: ML altyapısı — PyTorch + Hugging Face, pydantic yapılandırma, OCR için kendi Trainer, Whisper için Seq2SeqTrainer sarmalayıcısı

- **Durum:** kabul edildi
- **Tarih:** 2026-09-07
- **Karar vericiler:** Ç. Şahin, M. M. Turhan
- **İlgili İP:** İP-3

## Bağlam ve problem

İP-3 iki model (OCR encoder-decoder, MathWriting; ASR Whisper ince ayarı) ve "mimarisini, eğitim sürecini ve başarım değerlendirme sonuçlarını detaylandıran 2 teknik rapor" ister. Ortamda torch 2.11 (CPU) ve transformers 4.57 var; CUDA kurulumu ve eğitim kapılı. Altyapı bugün kodlanıp sentetik veriyle test edilmeli, iki yıl boyunca üç kişi tarafından anlaşılır kalmalı.

## Değerlendirilen seçenekler

1. **PyTorch Lightning / Hydra** — kolaylık; Windows'ta Hydra çalışma dizini sürprizleri, Lightning sürüm kırılmaları, bağımlılık ağırlığı.
2. **Her iki model için kendi Trainer** — tek döngü; Whisper'ın keskin köşeleri (dil/görev zorunlu kimlikleri, etiket kaydırma, -100 dolgu, generate ile değerlendirme) yeniden yazılır → literatürle karşılaştırılamayan WER riski.
3. **Her iki model için HF Trainer** — OCR seq2seq için HF veri/collator kalıbı zorlayıcı; CNN kodlayıcı ile uyumsuz kısımlar.
4. **OCR için kendi ~400 satırlık Trainer; Whisper için HF `Seq2SeqTrainer`'ı ortak manifest/kontrol noktası/metrik sözleşmesiyle saran `WhisperFineTuner`; pydantic yapılandırma** — her modelin doğal aracı, tek raporlama sözleşmesi.

## Karar

Seçenek 4. `aicls.ml.config.TrainConfig` (pydantic, `extra="forbid"`, `_base_` kalıtımı, noktalı CLI ezme, config hash). Model havuzu: torchvision kodlayıcılar `weights=None`, iz kodlayıcılar, LSTM/GRU/Transformer çözücüler; Whisper `from_pretrained(local_dir)`; Wav2Vec2 kayıtta. `Trainer` (bf16 autocast, grad birikimi, kırpma, warmup-cosine, periyodik CER/ExpRate değerlendirmesi, en-iyi/son atomik kontrol noktası, erken durdurma, kesinti kontrol noktası, RNG'li devam). `WhisperFineTuner`: `Seq2SeqTrainer` + `AiclsManifestCallback` (`run_manifest.json` eğitim başında) + `compute_metrics` → `aicls.eval.metrics.corpus_wer` (ham + tr_v1) + aynı kontrol noktası dizin düzeni. 8-bit optimizer yok (Windows); büyük model yalnızca LoRA. Profil koşusu (200 adım) her yapılandırmanın ilk kapılı eylemi; GPU bütçesi ölçümden türetilir. Tek GPU işi kuyruğu.

## Sonuçlar

- Olumlu: iki modelin raporları aynı manifest/metrik zincirinden çıkar; hakemlerin "iki eğitim döngüsü" çekincesi ortak sözleşmeyle karşılanır; CPU'da 2 adımlık duman testleri CI'da.
- Olumsuz: iki eğitim yolu bakımı; `transformers` sürüm yükseltmeleri Whisper sarmalayıcısını etkileyebilir → sürüm sabit, `test_whisper_ft_builds_from_tiny_random_config_without_download`.

## Öneri metniyle uyum

A.2.1: kodlayıcı-çözücü mimarisi; ResNet/VGG/EfficientNet; LSTM/GRU/Transformer; Gauss bulanıklaştırma, Laplace keskinleştirme, gri ton, [0,1] normalizasyon — hepsi yapılandırma anahtarı olarak ablasyonlu; MathWriting; CER. A.2.3: Wav2Vec/DeepSpeech/Whisper (DeepSpeech terk edilmiş; Whisper birincil, Wav2Vec2 kayıtta), log-Mel, SpecAugment, perde kaydırma, gürültü enjeksiyonu, TTS; WER. İP-3 risk planı: veri artırma, ensemble, niceleme (`ml.export` kancası).

## Doğrulama

`test_trainer_two_steps_synthetic_ocr_loss_decreases_cpu`, `test_trainer_two_steps_and_resume`, `test_checkpoint_atomic_write`, `test_run_manifest_captures_git_sha_and_dirty_flag`, `test_registry_builds_every_model_at_tiny_size`, `test_torchvision_encoders_built_with_weights_none`, `test_whisper_ft_builds_from_tiny_random_config_without_download`.

## İlgili

[ml/egitim-altyapisi.md](../ml/egitim-altyapisi.md), ADR-0007, ADR-0012, ADR-0013.
