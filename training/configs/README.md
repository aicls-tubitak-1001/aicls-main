# training/configs — eğitim yapılandırmaları

Şema: `aicls.ml.config.TrainConfig` (pydantic v2, `extra="forbid"`). Belge:
[docs/ml/egitim-altyapisi.md §1](../../docs/ml/egitim-altyapisi.md), kararlar
[ADR-0006](../../docs/adr/0006-ml-altyapisi-pytorch-hf-yapilandirma.md),
[ADR-0007](../../docs/adr/0007-deney-takibi-jsonl-mlflow-manifest.md),
[ADR-0012](../../docs/adr/0012-onay-kapilari.md).

Her dosya `_base_:` ile kalıtır (derin birleştirme: sözlükler birleşir, listeler ve skalerler
değiştirir) ve yalnızca kendi farkını yazar. CLI'dan noktalı ezme yapılır: `--set optim.lr=1e-4`.
`${AICLS_DATA_ROOT}` / `${AICLS_MODELS_ROOT}` / `${AICLS_RUNS_ROOT}` yükleme sırasında çözülür;
depoda makineye özgü mutlak yol yoktur.

## Hangi dosya ne için

| Dosya | Ne | Kapı |
|---|---|---|
| `_base.yaml` | ortak taban; koşulmaz, yalnızca kalıtılır | — |
| `ocr/offline_resnet34_tfm.yaml` | çevrimdışı OCR **referans hattı** (ResNet34 + Transformer, MathWriting) | `DATASET_DOWNLOAD`, `CUDA_INSTALL`, `TRAINING` |
| `ocr/offline_effnet_b0_tfm.yaml` | kodlayıcı ablasyonu: tek fark `model.encoder: efficientnet_b0` | aynı |
| `ocr/ablation_preproc.yaml` | ön işleme ablasyonu: `+ gaussian_blur, laplacian_sharpen` | aynı |
| `ocr/online_stroke_tfm.yaml` | çevrimiçi iz hattı (ADR-0013), batch 128 | aynı |
| `ocr/profile.yaml` | 200 adımlık verim/VRAM ölçümü | aynı |
| `ocr/smoke_synthetic.yaml` | CI duman koşusu: 64 sentetik iz, 2 adım, CPU | **kapısız** |
| `asr/whisper_small_ft.yaml` | whisper-small **tam ince ayar**, batch 32 × 30 s | `WEIGHTS_DOWNLOAD`, `CUDA_INSTALL`, `TRAINING` |
| `asr/whisper_large_v3_turbo_lora.yaml` | large-v3-turbo **yalnızca LoRA** (r=16), batch 16 + grad_accum 2 | aynı |
| `asr/profile.yaml` | 200 adımlık verim/VRAM ölçümü | aynı |
| `asr/smoke_synthetic.yaml` | CI duman koşusu: rastgele küçük WhisperConfig, sentetik log-Mel, 2 adım | **kapısız** |
| `sweeps/ocr_lr_sweep.yaml` | referans hat üzerinde öğrenme oranı taraması | `DATASET_DOWNLOAD`, `CUDA_INSTALL`, `TRAINING` |

## Kapısız olan yalnızca ikisidir

`ocr/smoke_synthetic.yaml` ve `asr/smoke_synthetic.yaml`. Kapısızlığın tek ölçütü
`data.dataset`in `synthetic_` ile başlamasıdır; `gates.synthetic` anahtarı **salt belgedir** ve
hiçbir kapıyı açamaz (ADR-0012). Gerçek veriye bakan her dosya ilk satırlarında ihtiyaç duyduğu
kapıları `# GATE: ...` başlığıyla bildirir; `tests/unit/ml/test_configs_validate.py` bunu denetler.

Bu iki duman yapılandırmasından çıkan hiçbir sayı **sonuç değildir**: rastgele başlatılmış küçük
bir ağın sentetik fikstür üzerindeki kaybı bağlantı denetimidir. CER/WER kanıtı ancak veri, ağırlık,
CUDA ve eğitim kapıları açıldıktan sonra üretilebilir.

## Epoch bütçesi ölçümden gelir, tahminden değil

Hiçbir yapılandırmada ve hiçbir belgede elle yazılmış epoch süresi yoktur. Bir aile için ilk kapılı
eylem daima profil koşusudur:

```text
aicls profile --config training/configs/ocr/profile.yaml --steps 200 --i-have-approval
```

`steps_per_s`, tepe VRAM ve dataloader bekleme yüzdesi ölçülür, `run_manifest.steps_per_s`'e ve
`reports/metrics/profile_*.json` içine yazılır; epoch/bütçe **yalnızca** bu ölçümden türetilir
(docs/ml/egitim-altyapisi.md §8). Dosyalardaki `epochs`, `max_steps` ve `batch_size` değerleri
niyet beyanıdır; profil sonrası güncellenmeleri beklenir.

## Tarama nasıl koşulur

Şemada `sweep:` bloğu yoktur. `sweeps/ocr_lr_sweep.yaml` ızgarasını başlığındaki makine-okunur
satırda taşır ve N ayrı koşu olarak çalıştırılır:

```text
# AICLS-SWEEP: optim.lr = [1.0e-4, 2.0e-4, 3.0e-4, 5.0e-4, 1.0e-3]
```

Her nokta ayrı bir koşu dizini, ayrı bir `run_manifest.json` ve ayrı bir `config_hash` üretir
(`optim.lr` hash'e girer, `run.name` ve `run.notes` girmez).

## Yerel ağırlık kuralı

`model.base_dir` **her zaman yerel bir dizindir** (`${AICLS_MODELS_ROOT}/pretrained/...`); hiçbir
yapılandırmada hub kimliği yoktur ve `model.weights` her zaman `null`'dır. `tools/check_gates.py`
bu ağacı statik olarak tarar.

## Kontrol noktası metriği

OCR yapılandırmaları `checkpoint.metric: val_cer`, ASR yapılandırmaları `val_wer` yazar
(docs/ml/egitim-altyapisi.md §4.1: "en iyi = en düşük CER"). Bu adlar eğiticinin doğrulama
sözlüğüne yazdığı anahtarlarla birebir aynı olmak zorundadır; eğitici (`aicls.ml.trainer`) henüz
yazılmadığı için bu bir **sözleşmedir**, doğrulanmış bir davranış değil.
