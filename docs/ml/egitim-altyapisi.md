# Eğitim Altyapısı

Paketler: `aicls.ml` (eğitim), `aicls.data` (veri kümeleri), `aicls.eval` (metrikler). Kararlar: [ADR-0006](../adr/0006-ml-altyapisi-pytorch-hf-yapilandirma.md), [ADR-0007](../adr/0007-deney-takibi-jsonl-mlflow-manifest.md), [ADR-0012](../adr/0012-onay-kapilari.md). Altyapının tamamı **kodlanır ve sentetik fikstürlerle CPU'da test edilir**; gerçek veri indirme, ağırlık indirme, CUDA kurulumu ve eğitim koşusu onay kapılarının arkasındadır.

Kapılar açıldığı gün izlenecek adım adım yol: [Eğitim el kitabı](egitim-el-kitabi.md). Bu sayfa **tasarımı** anlatır; el kitabı **işletimi** anlatır. Uygulamanın tasarımdan ayrıldığı her nokta [§11](#11-uygulama-notları-aiclsml-aiclscli-training-2026-09-08) içindedir.

!!! danger "Durum — 2026-09-08: ortada model yok, sayı da yok"

    Bu sayfada anlatılan altyapı **yazıldı ve CPU'da sentetik fikstürlerle test edildi**
    (`aicls.ml`, `aicls.cli`, `training/scripts/`, `training/configs/`; depo genelinde
    3.319 test geçiyor, 1'i sıraya bağımlı olarak başarısız — bkz. §11.1).
    Bunun ötesinde bugün hiçbir şey yoktur:

    * **Hiçbir model eğitilmedi.** Tek koşan döngü, rastgele başlatılmış 53.415
      parametrelik bir ağın üretilmiş Bezier izleri üzerindeki 2 adımıdır.
    * **Hiçbir veri kümesi indirilmedi.** MathWriting diskte yok; `DATASET_DOWNLOAD` kapalı.
    * **Hiçbir ağırlık indirilmedi.** `whisper-small` dahil hiçbir kontrol noktası yerelde
      yok; `WEIGHTS_DOWNLOAD` kapalı.
    * **Kurulu torch CPU derlemesidir** (`2.11.0+cpu`, `torch.cuda.is_available() == False`).
      Makinede RTX 5090 (sm_120, 32 GB, sürücü 610.88) *vardır* ama PyTorch onu göremez;
      `CUDA_INSTALL` kapalı.
    * Dolayısıyla projenin **OCR CER sayısı yoktur ve ASR WER sayısı yoktur.**
      `reports/metrics/` içinde model içeren ne varsa (`ocr_test_*.json`, `asr_test_*.json`,
      `latest.json`'ın `ocr`/`asr` blokları) bir **stub veya rastgele ağdan** çıkmıştır ve
      yalnızca **bağlantı denetimidir (wiring check)**. `latest.json` bu yüzden 2026-09-08'den
      beri `ocr` ve `asr` için **sayı değil `null`** yayımlar: girdide `measurement:
      "wiring_check"`, gerekçe ve stub'ın gerçekte aldığı değerler (`wiring_check.metrics`)
      durur. Daha önce aynı alan `ocr.cer = 0.0` diyordu; bu "mükemmel tanıma" değil, "stub
      kendisine verilen etiketi geri veriyor" demekti — koşucu artık her örneği modele vermeden
      önce körleştirdiği için o sayı üretilemez.

    Bu paragrafı okumadan `reports/metrics/latest.json` içindeki hiçbir sayı alıntılanamaz.
    Gerçek CER/WER kanıtı için dört kapının da (`DATASET_DOWNLOAD`, `WEIGHTS_DOWNLOAD`,
    `CUDA_INSTALL`, `TRAINING`) açılması gerekir; bkz. [§9](#9-onay-kapılı-betikler),
    [kurulum.md §3](../gelistirme/kurulum.md#3-onay-kapıları) ve
    [degerlendirme.md §4.1](degerlendirme.md#41-koşudan-rapora-bir-eğitim-koşusu-nasıl-alıntılanabilir-sayıya-dönüşür).

## 1. Yapılandırma sistemi

`aicls.ml.config.TrainConfig` — pydantic v2, `extra="forbid"` (yazım hatası koşudan önce yakalanır), YAML'dan `_base_` kalıtımıyla yüklenir, CLI'dan noktalı yol ile ezilir (`train.lr=1e-4`), çözümlenmiş yapılandırma ve SHA-256'sı koşu dizinine yazılır.

```yaml
# training/configs/ocr/offline_resnet34_tfm.yaml
_base_: ../_base.yaml
run:   {name: ocr_offline_resnet34_tfm, seed: 1234, deterministic: true, tags: [ocr, offline]}
data:
  dataset: mathwriting            # mathwriting | audio_commands | synthetic_ink | synthetic_audio
  root: ${AICLS_DATA_ROOT}/mathwriting/shards
  mode: offline                   # online | offline
  image_size: [96, 384]
  preprocess: [grayscale, normalize01, resize_pad]   # ablasyon: + gaussian_blur, laplacian_sharpen
  synthetic_ratio: 0.5            # MathWriting sentetik yarısının gerçek örneklere oranı (müfredat: 2 epoch yalnız sentetik ısınma, sonra 0.5)
  synthetic_warmup_epochs: 2
  augment: {affine_jitter: 0.05, stroke_dropout: 0.02}
  max_label_len: 160
  batch_size: 64
  num_workers: 6
model:
  family: seq2seq_image           # seq2seq_image | seq2seq_stroke | whisper | wav2vec2
  encoder: resnet34               # resnet18|resnet34|vgg16|efficientnet_b0 (torchvision, weights=None) | stroke_transformer | stroke_bilstm
  decoder: transformer            # transformer | lstm_attn | gru_attn
  d_model: 256
  layers: 6
  heads: 8
  dropout: 0.1
  vocab_path: ${AICLS_DATA_ROOT}/mathwriting/vocab.json
optim: {name: adamw, lr: 3.0e-4, weight_decay: 0.01, betas: [0.9, 0.98], scheduler: cosine, warmup_steps: 2000}
train: {epochs: 30, max_steps: null, grad_accum: 1, amp: bf16, grad_clip: 1.0, eval_every: 2000, ckpt_every: 2000, early_stop_patience: 25, allow_dirty: false}
eval:  {beam: 4, max_n_val: 2000, metrics: [cer, ter, exprate, symeq]}
tracking: {backend: jsonl, mlflow_uri: null, experiment: ocr}
```

Whisper yapılandırması aynı ağaçta `model.family: whisper`, `model.base_dir: ${AICLS_MODELS_ROOT}/pretrained/whisper-small` (yerel dizin; asla hub kimliği), `model.lora: {enabled: false, r: 16, alpha: 32, targets: [q_proj, v_proj]}`, `model.freeze_encoder: false`, `data.feature_extractor: whisper` (HF `WhisperFeatureExtractor`), `data.augment: {spec_augment: {time_masks: 2, freq_masks: 2}, pitch_semitones: [-2, 2], noise_snr_db: [10, 25], speed: [0.9, 1.1], tts_ratio: 0.0}`.

CI'da kullanılan küçük yapılandırmalar: `training/configs/ocr/smoke_synthetic.yaml` (d_model 32, 2 katman, 64 sentetik iz, 2 adım, CPU) ve `training/configs/asr/smoke_synthetic.yaml` (rastgele başlatılmış küçük `WhisperConfig`, sentetik log-Mel, 2 adım). Her koşu tipi için `profile.yaml` (bkz. §8).

## 2. Veri kümeleri (`aicls.data`)

| Sınıf | Kaynak | Örnek | Not |
|---|---|---|---|
| `MathWritingDataset(root, split, mode)` | `training/scripts/prepare_mathwriting.py` ile InkML → `npz` parçaları (`strokes float32 (T,3)`, `label_normalized`, `writer_id`, `is_synthetic`, resmi split); kaynak açılmış dizin **ya da doğrudan `mathwriting-2024.tgz` akışı** (`--archive`) | online: `(features (T,F), label_ids)`; offline: `render_ink` → `preprocess_image` → `(img (1,H,W), label_ids)` | 630 bin XML dosyasını her epoch'ta ayrıştırmak Windows'ta I/O duvarıdır; parça dönüşümü tek seferliktir, memory-map ile okunur; `synthetic_ratio` ile gerçek/sentetik karışımı örnekleyici düzeyinde |
| `AudioCommandDataset(manifest, split, feature_fn)` | `aicls contrib export-manifest` çıktısı (JSONL: path, text, speaker_pseudonym, university, gender, age_band, split) | `(wave16k, text, speaker)` → `feature_fn` (Whisper yolunda `WhisperProcessor`) | konuşmacı-ayrık split dosyası zorunlu; yoksa yükleme hatası |
| `SyntheticInkDataset(n, seed)` | `synth_ink` Bezier izleri + şablon LaTeX (`x^{k}`, `\frac{a}{b}`, `\int x dx`) | CI | 
| `SyntheticAudioDataset(n, seed)` | `synth_audio` sinüs süpürme + gürültü, sahte transkript (envanter cümleleri) | CI |
| `load_curriculum(path)` | `datasets/curriculum/problems.jsonl` | `Problem` | CAS değerlendirmesi |

Bölmeler (`aicls.data.splits`): MathWriting resmi train/valid/test/symbols/synthetic korunur. Ses: `make_splits(ids, group_key=speaker_pseudonym, stratify=(university, gender), ratios=(0.8,0.1,0.1), seed=1234)` → `datasets/manifests/audio_commands_splits.json` (SHA'sı her ASR raporuna yazılır); konuşmacı hiçbir zaman iki bölmede olamaz (`test_speaker_disjoint_split_has_no_overlap`), tabaka oranları %5 tolerans içinde (`test_speaker_disjoint_split_is_stratified_within_5pct`).

Manifestler (`aicls.data.manifest`): her büyük veri kökünde `manifest.json` (dosya, boyut, sha256, lisans, kaynak URL, alınma tarihi, sürüm); `aicls data verify mathwriting` doğrular; parmak izi her `EvalReport.dataset_fingerprint` alanına gider. Katalog `src/aicls/data/catalog.yaml`: MathWriting (**CC BY-NC-SA 4.0** — ticari kullanım yasak; `mathwriting-2024.tgz`, 2,88 GiB), MathWriting resmi örneklemi (aynı lisans, 1,55 MiB, `--excerpt`), `openai/whisper-small` (MIT), `openai/whisper-large-v3-turbo` (MIT), `facebook/wav2vec2-xls-r-300m` (Apache-2.0) — URL, sha256 (indirmeden sonra doldurulur), kapı adı.

Parçalama iki kaynaktan besleniyor (`aicls.data.sources`): açılmış bir dizin ağacı (`DirectoryInkmlSource`) ya da doğrudan `.tgz` akışı (`TarInkmlSource`). İkincisi ~630 bin küçük dosyayı diske hiç yazmaz — `drvfs` üzerindeki bir veri kökünde belirleyici fark budur — ve istenen tüm bölmeleri **tek geçişte** üretir, çünkü gzip akışında geri sarma yoktur. Örnek sırası kaynağa bağlıdır; `tar` sıralı dizin gezintisi yazdığından iki yol pratikte aynı parçaları ve aynı parmak izini üretir (2026-09-08'de 20 bin dosyalık sentetik fikstürde beş bölmede de doğrulandı; açma+parçalama 301 s'ye karşı akıtma 6,5 s), ama bu bir garanti değildir — `shards_manifest.json.source_root` hangisinin kullanıldığını yazar.

### 2.1 Uygulama notları (`aicls.data`, 2026-09-07)

Paket kodlandı ve sentetik fikstürlerle CPU'da test edildi (143 test, kapsam %94,2). Tablodaki adlar korundu; ayrıntılar ve sapmalar [veri-yonetimi.md §7](veri-yonetimi.md#7-uygulama-notları-aiclsdata-2026-09-07) içinde toplu olarak belgelendi. Eğitim tarafını doğrudan ilgilendirenler:

- `MathWritingDataset` torch'tan **türemez**; map-style bir nesnedir ve `DataLoader`'a `ds.to_torch()` ile verilir (tembel torch içe aktarma; `import aicls.data` torch'u yüklemez). Collate işlevleri numpy döner, `to_torch=True` ile tensöre çevirir: `collate_fn=functools.partial(collate_ink, to_torch=True)`.
- Öğe tipleri `InkSample(features, label_ids, label, sample_id, writer_id, is_synthetic)` ve `AudioSample(features, text, speaker, sample_id, sample_rate, duration_s)`; ikisi de belgelenen demet açılımını (`features, label_ids = sample`) destekler.
- `synthetic_ratio` **rastgeleliksiz**dir: `i` konumu sentetiktir ⇔ `floor((i+1)·r) > floor(i·r)`; epoch uzunluğu `round(n_real/(1−r))`, `synthetic_ratio: 1.0` (ısınma epoch'ları) yalnızca sentetik parçaları sunar. `r>0` iken `data.root` altında hazırlanmış bir `synthetic` bölmesi zorunludur.
- Parçalar sıkıştırmasız `npz`'dir ve `ShardReader` zip üyesini `np.memmap` ile eşler (`np.load(..., mmap_mode=...)` `.npz` için çalışmaz). `num_workers>0` ile her işçi kendi eşlemesini açar; parça manifestinin `fingerprint` alanı `run_manifest.json`'a ve `EvalReport.dataset_fingerprint`'e girer.
- `data.dataset: synthetic_ink|synthetic_audio` duman yapılandırmaları için `SyntheticInkDataset(n, seed=..., tokenizer=..., mode=..., preprocess=...)` ve `SyntheticAudioDataset(n, seed=..., feature_fn=..., seconds=...)` hazırdır; ikisi de indirme kapısı gerektirmez.
- `data.preprocess` listesi doğrudan `aicls.ocr.PreprocessConfig(steps=..., size=image_size)` olarak geçirilir; çevrimdışı yolda parça önceden çizilmiş raster taşıyorsa (`prepare_shards(..., render=True)`) yeniden çizim yapılmaz.
- `max_label_len` (`data.max_label_len`) veri kümesi düzeyinde uygulanır; etiketler `<s> ... </s>` ile sarılır (`add_special=True`).

## 3. Model havuzu (`aicls.ml.registry` + `aicls.ocr.models`, `aicls.asr`)

| Ad | Yapı | Not |
|---|---|---|
| `seq2seq_image` | torchvision `resnet18/34`, `vgg16`, `efficientnet_b0` (hepsi `weights=None`; `tools/check_gates.py` bunu grep'ler) + 2B konum kodlaması → Transformer / LSTM-attn / GRU-attn çözücü | öneri A.2.1'deki kodlayıcı-çözücü aileleri; ablasyon tablosu için config-anahtarlı |
| `seq2seq_stroke` | iz özellikleri (Δx, Δy, kalem, Bezier) → `stroke_transformer` veya `stroke_bilstm` kodlayıcı → aynı çözücüler | çevrimiçi yol (ADR-0013) |
| `whisper` | HF `WhisperForConditionalGeneration.from_pretrained(local_dir)`; `peft` LoRA isteğe bağlı | öneri A.2.3; `whisper-small` tam ince ayar ilk, `large-v3-turbo` + LoRA raporlanan model adayı |
| `wav2vec2` | HF `Wav2Vec2ForCTC` | İP-3 risk planı "ensemble" için kayıtta tutulur |

Her kayıt `build(cfg, vocab_size)` fabrikası ve parametre sayısı testi taşır (`test_registry_builds_every_model_at_tiny_size`). Tembel `torch` import; `aicls.api` bu modülleri yalnızca `TorchRecognizer/WhisperTranscriber.load()` üzerinden çağırır.

## 4. Eğitici

### 4.1 OCR: `aicls.ml.trainer.Trainer` (kendi döngümüz, ~400 satır)

`fit()`: `seed_everything` → `RunManifest.capture()` (ilk optimizer adımından **önce** `run_manifest.json`) → yükleyiciler (seed'li `worker_init_fn`, `generator`) → adım döngüsü: `autocast(bf16)` ileri, kayıp, grad birikimi, kırpma, AdamW + warmup-cosine → her `eval_every` adımda doğrulama alt kümesinde greedy/beam çözme ve `aicls.eval.metrics` (CER/TER/ExpRate/SymEq) → `CheckpointSaver` (best-by-CER + last, atomik yazım) → `EarlyStopping` → `KeyboardInterrupt`'ta `interrupt/` kontrol noktası. `resume(ckpt_dir)` model/optimizer/scheduler/adım/RNG (python, numpy, torch, cuda) durumlarını geri yükler; `test_trainer_two_steps_and_resume` adım sayacı ve RNG eşitliğini doğrular.

### 4.2 ASR: `aicls.ml.whisper_ft.WhisperFineTuner` (HF `Seq2SeqTrainer` sarmalayıcısı)

Whisper ince ayarının keskin köşeleri (zorunlu çözücü kimlikleri `language="tr", task="transcribe"`, etiket kaydırma, `-100` dolgu, değerlendirmede `generate`, gradyan kontrol noktası) yeniden yazılmaz; `Seq2SeqTrainer` kullanılır. Bizim sarmalayıcımız üç şeyi garanti eder: (1) `AiclsManifestCallback.on_train_begin` aynı `run_manifest.json`'ı yazar, (2) `compute_metrics` → `aicls.eval.metrics.corpus_wer(norm="tr_v1")` ve ham WER, (3) kontrol noktası dizini bizim düzenimizle tamamlanır (`config.yaml`, `run_manifest.json`, `metrics.json` HF dosyalarının yanına). Hakemlerin "iki eğitim döngüsü" çekincesi bu ortak manifest/kontrol noktası/metrik sözleşmesiyle karşılanır; ADR-0006'da kayıtlıdır. `test_whisper_ft_builds_from_tiny_random_config_without_download` ağ olmadan küçük rastgele `WhisperConfig` ile kurulumu doğrular.

## 5. Kontrol noktaları

Dizin: `AICLS_MODELS_ROOT/<component>-<version>/` → `model.safetensors`, `optimizer.pt` (yalnızca son), `scheduler.pt`, `config.yaml` (çözümlenmiş), `run_manifest.json`, `vocab.json`, `metrics.json`, `rng_state.pt`. Yazım `tmp` dizine + `os.replace` (atomik; `test_checkpoint_atomic_write` çökme simülasyonu). `sha256` model dosyası üzerinden hesaplanır ve `aicls models import` ile `model_versions` tablosuna yazılır; API yalnızca `is_active` satırı yükler. `models/registry.yaml` yalnızca tohum dosyasıdır (üç yerde sürüm tutulmaz).

## 6. Deney takibi

`Tracker` protokolü: `JsonlTracker` (varsayılan, bağımlılıksız; `run_dir/metrics.jsonl`, `params.json`) ve `MlflowTracker` (opsiyonel, `mlflow` saf Python olduğu için `requirements/dev.txt` içinde; `file:AICLS_DATA_ROOT/mlruns`). Raporlarda birincil alıntı anahtarı `run_manifest.json` (git SHA + config hash), MLflow `run_id` ikincildir. TensorBoard yoktur (üç örtüşen çıktı yerine iki).

## 7. Tekrarlanabilirlik

- `seed_everything(seed, deterministic=True)`: `random`, `numpy`, `torch`, `torch.cuda`; `torch.use_deterministic_algorithms(True, warn_only=True)`; `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- `RunManifest`: git SHA, kirli bayrağı, kirliyse diff SHA-256, config hash, veri manifest parmak izleri, `pip freeze` SHA-256 (tam liste `env.json`), Python/torch/CUDA/cuDNN sürümleri, GPU adı/sürücü, hostname, başlangıç zamanı, ölçülen adım/s. **Kirli çalışma ağacı** `train.allow_dirty=false` iken koşuyu reddeder; `--allow-dirty` ile koşu `citable=false` işaretlenir ve `poe report-check` bu koşudan gelen metrikleri rapora almaz.
- `scripts/reproduce_report.ps1`: `report-YYYY-MM` etiketinden temiz klon, `aicls models import`, rapora giren her `aicls eval` komutunu yeniden koşar, sonuçları `reports/metrics/` ile karşılaştırır.

## 8. GPU planı (tek RTX 5090, 32 GB, Windows)

- **Blackwell sm_120 → cu128 tekerleği zorunlu.** `scripts/install_ml.ps1` (kapı `CUDA_INSTALL`) tam torch sürümünü ve `--index-url .../cu128`'i sabitler, `torch.cuda.get_device_capability() == (12, 0)` ve bf16 matmul duman testi geçmeden başarı bildirmez; sürücü/araç takımı sürümü onay defterine yazılır. WSL2 yedeği `docs/gelistirme/kurulum.md` içinde.
- **Bütçenin baskın terimi doğrulamadır, adım döngüsü değil.** `aicls profile` artık adım döngüsünden sonra **gerçek bir doğrulama geçişi** de ölçer (`eval_samples`, varsayılan 8) ve `seconds_per_val_sample`'ı yazar; `projection` bloğu yapılandırmanın istediği takvimi (`total_steps`, `eval_count`, `n_val_per_eval`) bu iki ölçülmüş birim maliyetle fiyatlar. Doğrulama maliyeti ölçülmediyse **tahmin üretilmez**: `projected: false` ve `reason: eval_cost_not_measured`. Gerekçe: 6. dalganın gerçek koşusunda duvar saatinin %96,6'sı doğrulamadaydı; yalnız `steps_per_s`'ten türetilen bir bütçe iki mertebe yanlış **ve** alıntılanabilir olurdu.
- **Önce profil, sonra bütçe.** Hakemlerin işaret ettiği gibi elle yazılmış epoch süreleri güvenilmez. Kapı açıldıktan sonra her yapılandırma için ilk kapılı eylem `aicls profile --config ... --steps 200`'dür: `steps_per_s`, tepe VRAM, dataloader bekleme yüzdesi ölçülür, `run_manifest.steps_per_s`'e yazılır ve epoch/bütçe bu ölçümden türetilir. Bu belge sabit bir süre tahmini vermez; tahminler `reports/metrics/profile_*.json` içinde yaşar.
- **Bellek zarfı (başlangıç değerleri):** OCR offline ResNet34+Transformer 96×384, batch 64, bf16; online iz modeli batch 128; whisper-small tam ince ayar batch 32 × 30 s bf16; whisper-large-v3-turbo yalnızca LoRA (r=16) batch 16 + grad_accum 2. 8-bit optimizer (bitsandbytes) Windows'ta kullanılmaz; gerekirse gradyan kontrol noktası.
- **Tek iş.** `aicls.ml.queue.TrainingQueue` `training/queue.lock` kilidi (git-ignore; `training/queue.txt` yalnızca açıklamadır) ve `nvidia-smi` ile başka bir `aicls` sürecinin GPU'yu tuttuğunu görürse reddeder; eğitim sırasında API model çıkarımı `GET /health/ready` üzerinden `training` durumu bildirir.
- **CPU:** bugünkü makinede `num_workers=6/8`; satın alınan iş istasyonunda 12/16. Veri parçaları NVMe'de, `AICLS_DATA_ROOT` OneDrive dışında.

## 9. Onay kapılı betikler

| Betik | Kapı | Ne yapar | Kapı kapalıyken |
|---|---|---|---|
| `training/scripts/download_mathwriting.py` | `DATASET_DOWNLOAD` | katalog girdisini (URL, boyut, lisans, sha256) yazdırır, disk denetler, indirir, `manifest.json` üretir | çıkış kodu 3, üç koşulu listeler |
| `training/scripts/download_weights.py --model openai/whisper-small` | `WEIGHTS_DOWNLOAD` | `AICLS_MODELS_ROOT/pretrained/<name>/` içine `snapshot_download`, sha256 kaydı | 3 |
| `scripts/install_ml.ps1` | `CUDA_INSTALL` | `requirements/ml.txt` (cu128 indeksi) kurulumu + doğrulama | 3 |
| `training/scripts/train.py` / `aicls train` | `TRAINING` | `Trainer`/`WhisperFineTuner`; `--synthetic` veya `data.dataset: synthetic_*` ise kapı **aranmaz** | 3 |
| `aicls profile` | `TRAINING` | 200 adımlık ölçüm (gerçek veri) | 3 |

`aicls.core.gates.require_gate(name, cli_flag=args.i_have_approval)` üç koşulu birlikte ister: `--i-have-approval` bayrağı, `AICLS_GATE_<NAME>=approved` ortam değişkeni, `docs/gelistirme/onay-defteri.md` içinde `| <NAME> | approved | <tarih> | <kim> | <kapsam> |` satırı. CI hiçbirini ayarlamaz ve `tests/gated/test_gates.py` her betiğin 3 ile çıktığını doğrular. `tools/check_gates.py` ayrıca kaynak ağacında kapılı modüller dışında `urllib`, `requests.get`, `hf_hub_download`, `from_pretrained("openai/` ve `weights=` (None dışı) kullanımını arar. `poe_tasks.toml` kapılı komut içermez; bunlar yalnızca açık betik çağrısıyla çalışır.

## 10. Testlerin altyapıyı nasıl kullandığı

`tests/unit/ml/`: `test_train_config_override_dotted_path`, `test_train_config_rejects_unknown_key`, `test_train_config_base_inheritance`, `test_trainer_two_steps_synthetic_ocr_loss_decreases_cpu`, `test_trainer_two_steps_and_resume`, `test_checkpoint_roundtrip_contains_manifest_and_vocab`, `test_checkpoint_atomic_write`, `test_run_manifest_captures_git_sha_and_dirty_flag`, `test_run_manifest_refuses_dirty_tree_without_allow_dirty`, `test_seed_everything_reproducible_forward_pass`, `test_jsonl_tracker_schema`, `test_registry_builds_every_model_at_tiny_size`, `test_whisper_ft_builds_from_tiny_random_config_without_download`, `test_profile_report_fields_on_synthetic`, `test_training_queue_refuses_second_job`; `tests/unit/data/`: `test_prepare_mathwriting_shard_roundtrip_on_synthetic_inkml`, `test_data_manifest_verify_detects_tampered_sha`, `test_speaker_disjoint_split_has_no_overlap`, `test_speaker_disjoint_split_is_stratified_within_5pct`, `test_synthetic_ratio_sampler_mixes_as_configured`; `tests/gated/`: `test_gated_download_refuses_without_flag_and_env`, `test_gated_train_refuses_in_ci`, `test_gated_train_allows_synthetic_without_gate`, `test_check_gates_tool_flags_hub_id_usage`.

## 11. Uygulama notları (aicls.ml, aicls.cli, training, 2026-09-08)

Bu bölüm, 2026-09-08'de kodlanan altyapının tasarımdan (yukarıdaki §1–§10) **ayrıldığı** her noktayı ve **kanıtlanmamış** kalan her yolu kayda geçirir. Amaç, 2027-05-01 gelişme raporunu yazan kişinin ve projeyi devralacak araştırmacının belgeye bakıp koda güvenebilmesidir. Bölüm 2.1'in (`aicls.data`) devamıdır.

### 11.1 Ne kuruldu, ne ölçüldü

| Alan | Durum |
|---|---|
| `src/aicls/ml/` | `config.py`, `registry.py`, `manifest.py`, `tracking.py`, `checkpoint.py`, `queue.py`, `trainer.py`, `profile.py`, `whisper_ft.py` — dokuz modül, hepsi yazıldı |
| `src/aicls/cli/` | `main.py` + `common.py` + 10 komut modülü (`serve`, `db`, `data`, `train`, `evaluate`, `book`, `models`, `contrib`, `nlu`, `report`) |
| `training/configs/` | 12 YAML: `_base.yaml` + 10 görev yapılandırması (6 OCR, 4 ASR) + `sweeps/ocr_lr_sweep.yaml`; hepsi `load_config` ile doğrulanıyor |
| `training/scripts/` | 7 betik: 3 kapılı (`download_mathwriting`, `download_weights`, `train`), 4 kapısız |
| Testler | Depo toplamı **3.319 geçti, 1 başarısız, 2 atlandı** (2026-09-08, tam koşum). `tests/unit/ml` + `tests/unit/cli` + `tests/gated` yalnız koşturulduğunda **446 geçti, 0 başarısız** |
| Kapsam | `aicls.ml` + `aicls.cli` birlikte **%94,7** (hedef: ml ≥ %75, cli ≥ %70). En düşük modül `aicls.cli.models` %88,3; `aicls.ml.whisper_ft` %90,4 |

Başarısız tek test **sıraya bağımlıdır**: `tests/unit/ml/test_trainer.py::test_trainer_warns_that_the_synthetic_warmup_curriculum_is_not_wired` tam koşumda düşer, `pytest tests/unit/ml` tek başına koşturulduğunda geçer. Sebep, daha önceki bir testin yapılandırdığı structlog akışının bu testin günlük yakalamasına sızmasıdır (aynı kök nedenin CLI tarafındaki görünümü için 11.10'a bakın). Ürün kodunda bir hata değildir, ama CI'da kırmızı görünür; sahibi `aicls.ml`.

**Ölçülen tek "eğitim" sayısı** (2026-09-08, bu makine, CPU): `training/configs/ocr/smoke_synthetic.yaml`, 2 adım → `train_loss 5,206`, `val cer 1,130`, `steps_per_s 8,62`, `dataloader_wait_pct %2,4`. Aynı yapılandırmanın 20 adımlık profili: `15,5 adım/s`, `62,1 örnek/s`, `peak_vram_mb: null` (CPU). **Bunların hiçbiri sonuç değildir**; rastgele başlatılmış bir ağın üretilmiş veriye bakışıdır ve doğrulama kümesi eğitim kümesiyle aynı tohumdan üretilir. Her ikisi de `wiring_check: true`, `citable: false` taşır.

### 11.2 Yapılandırma şeması (§1'e göre sapmalar)

- **Şema, belgedeki örnekten geniştir.** Kaydediciler (`build_*`) olmadan çalışamayacağı için şu alanlar eklendi: `data.n_samples`, `data.sample_seconds`, `data.max_seconds`, `data.splits_path`, `data.split`, `model.max_len`, `model.ff_dim`, `model.num_mel_bins`, `model.pretrained_path`, `optim.momentum`, `train.log_every`; ayrıca `paths:`, `checkpoint:` ve `gates:` blokları tümüyle. §1'deki örnek hâlâ geçerlidir, eksiktir.
- **`model.weights` tipi `None`'dır.** Torchvision ağırlık numaralandırması yazılırsa bu bir *şema hatası* olur; ADR-0012 böylece yalnızca `tools/check_gates.py` grep'iyle değil, doğrulama katmanında da uygulanır.
- **`gates.synthetic` bağlayıcı değildir.** `TrainConfig.is_gate_free` **yalnızca** `data.dataset` adının `synthetic_` ile başlamasına bakar. Bir yapılandırma dosyası kendini kapısız ilan ederek `TRAINING` kapısını konuşarak aşamaz. `gates.synthetic` yalnızca belgeleyicidir; `data.dataset` ile çeliştiğinde `ml.config.gates_synthetic_ignored` günlüğü düşer. (Bu, faz 1'de `_base.yaml`'ın `gates: {synthetic: true}` taşıdığının fark edilmesi üzerine alınmış bir güvenlik kararıdır; `_base.yaml` o gün düzeltildi ve alan artık `null`.)
- **`_base_` normatif, `extends` kabul edilir.** İkisi birlikte varsa `_base_` kazanır; `_base_` bir liste de olabilir (soldan sağa birleştirilir).
- **Eski anahtar göçü.** `load_config` eski yerleşimleri taşır ve `ml.config.legacy_keys_migrated` günlüğü düşer (üst düzey `seed:`/`deterministic:` → `run.*`; `optim.grad_clip`/`optim.amp` → `train.*`; `train.early_stopping_patience` → `train.early_stop_patience`; `model.kind` → `model.family`; `synthetic_ocr` → `synthetic_ink`). Depodaki hiçbir dosya artık bu yolu tetiklemez; göç tabloları yalnızca kullanıcıların eski yapılandırmaları için durur.
- **`config_hash()` çözümlenmiş yapılandırma üzerinden alınır**, yani mutlak yollar hash'e girer. Aynı dosya farklı `AICLS_DATA_ROOT` altında farklı hash üretir. Raporlar bu hash'i alıntıladığı için bilinmesi gerekir.
- **`aicls.ml.config` modül düzeyinde `aicls.ocr`'i içe aktarır** (kodlayıcı ve ön işleme adı listeleri tek kaynaktan gelsin diye). ~0,33 s maliyeti vardır ve torch'suzdur.
- **Sapma listesi olarak `sweep:` bloğu yoktur.** `TrainConfig` `extra="forbid"` olduğu için tarama ızgarası şemaya sığmaz; `training/configs/sweeps/ocr_lr_sweep.yaml` ızgarayı makine tarafından okunabilir bir başlık yorumunda taşır: `# AICLS-SWEEP: optim.lr = [...]`. Dosyanın kendisi geçerli tek noktalı bir yapılandırmadır. **Tarama koşturucusu yoktur**; bugün beş nokta beş ayrı `--set optim.lr=` çağrısıyla koşar.

### 11.3 Model ve veri kaydı (§3'e göre sapmalar)

- Kayıtlı fabrikalar çözümlenmiş `ModelSpec` / `DatasetSpec` kayıtları alır (bir model kurucusunun `data.image_size`'a, bir veri kurucusunun `run.seed`'e ihtiyacı vardır). **Genel giriş noktaları** `build_model(cfg, vocab_size)` ve `build_dataset(cfg, split)` imzasını korur ve artık tam `TrainConfig` de kabul eder.
- `whisper` / `wav2vec2` ailesi `model.base_dir: null` ile **rastgele başlatılmış küçük bir model** kurar (CI yolu; §1'in adlandırdığı davranış). `base_dir` verilmişse dizin yerelde bulunmak zorundadır; hiçbir koşulda indirilmez.
- Yerel bir Whisper kontrol noktasının sözlüğü `vocab_size` argümanından farklıysa **gömme katmanı yeniden boyutlandırılmaz**; kontrol noktasının kendi sözlüğü kazanır. Bilinçli bir karardır (ince ayar kararı, kayıt kararı değil) ve `aicls.ml.whisper_ft` sahibinindir.

### 11.4 OCR eğiticisi (§4.1'e göre sapmalar)

- **`fit(max_steps=N)` mutlak bir bütçedir**, "N adım daha" değil. Aksi hâlde `resume()` + `fit()` bileşiminin tanımlı bir durma noktası olmazdı. Etkin bütçe `min(max_steps, train.max_steps, train.epochs × adım/epoch)`; hiçbiri verilmezse döngü tam bir epoch koşar ve `ml.trainer.default_budget` günlüğü düşer.
- **`resume(ckpt_dir)` yüklenen `Checkpoint` nesnesini döndürür** (moduller.md `-> None` yazıyordu). Dönüşü yok saymak hiçbir çağıranı bozmaz.
- **En iyi kontrol noktası politikası:** doğrulama iyileştiğinde **her zaman** bir kontrol noktası yazılır; `train.ckpt_every`, son adım ve erken durdurma buna eklenir. Aksi hâlde "CER'e göre en iyi", "`ckpt_every`'e denk gelen adımlar arasında en iyi" demek olurdu — kodun ilk sürümünde tam olarak bu hata vardı ve bir test yakaladı.
- **Kesinti kontrol noktası** `<run_dir>/checkpoints/interrupt` altındadır (§4.1 ebeveynsiz `interrupt/` yazıyor). `CheckpointSaver.prune` yalnızca `step-\d+` dizinlerini eşlediği için asla budanmaz. `KeyboardInterrupt`'ta izleyici koşusu `killed`, beklenmedik istisnada `failed` durumuyla kapanır.
- **Eğitici sessizce yok saymaz, reddeder:** `model.family` `whisper`/`wav2vec2` (→ `WhisperFineTuner`), aile ile `data.mode` uyuşmazlığı (bir görüntü modeline iz özniteliği vermek hiçbir şey öğretmez), `model.lora.enabled`. Hepsi `InputValidationError` + `details["reason"]`.
- **`data.synthetic_warmup_epochs` uygulanmıyor.** §1'in "2 epoch yalnız sentetik ısınma, sonra `synthetic_ratio: 0.5`" müfredatı, veri kümesinin epoch sınırında yeniden kurulmasını gerektirir; `Trainer` ise hazır kurulmuş bir veri kümesi alır. Şema alanı kabul eder, döngü `ml.trainer.synthetic_warmup_not_implemented` uyarısı düşer (bir test bunu sabitler). Doğru çözüm, eğiticinin epoch başına çağırabileceği bir veri kümesi fabrikasıdır — `aicls.ml` sahibinin tasarım kararı.
- **`evaluate()` toplu (batched) çözer ve döndürdüğü her metriğin yanına kapsamını yazar.** Anahtarlar: `loss`, `val_loss`, `eval_seconds`, `n_val`, `n_val_total`, `val_truncated` ve `cfg.eval.metrics`. `eval.max_n_val` ayarlıysa kayıp ortalaması da o alt kümeyle sınırlıdır: rapor edilen sayı bir **alt küme kaybıdır** — ve artık bunu söyleyen alanlar metrik satırının içindedir. Veri kümesi uzunluğunu bildiremiyorsa `n_val_total`/`val_truncated` **yazılmaz**; uydurulmuş bir bölünme büyüklüğü eksik olandan kötüdür.
- **Çözme yolu `fit_report.json`'da adlandırılır (`eval_decoder`).** Üç seçenek, bu sırayla: `aicls.ocr.models.seq2seq.greedy_decode_batch` / `beam_search_batch` (asıl yol), `aicls.ml.trainer.batched_greedy_ids` (bu modülün `encode`/`decode` üzerinden kendi toplu greedy'si; yalnızca `ocr` toplu giriş noktasını sunmuyorsa) ve örnek başına `beam_search` (yavaş yol — yanlış bir CER yavaş olandan kötüdür, bu yüzden korunur ve raporda adı geçer).
- **`FitReport` duvar saatini böler:** `train_seconds`, `eval_seconds`, `checkpoint_seconds`, `dataloader_wait_s`, `other_seconds` (+ `evaluations`, `seconds_per_eval`, `seconds_per_val_sample`, `eval_share_pct`). 6. dalgada `wall_time_s 613` ile `steps_per_s 8.66` yan yana yazılıyordu; ikisi de doğruydu ve birlikte yanıltıcıydı (1,4 s eğitim, 592 s doğrulama). Ölçüm: `runs/ocr_excerpt_wiring` (6. dalga) → `runs/ocr_excerpt_wave7` (7. dalga), aynı 20 adım ve aynı iki doğrulama: **613,4 s → 25,3 s**, doğrulama örneği başına **18,5 s → 0,139 s** (RTX 5090, MathWriting alıntısı, beam 4, batch 8). Kazancın bir bölümü `aicls.ocr`'ın artımlı önbelleğine, bir bölümü buradaki toplu döngüye aittir; aynı koşullarda ölçülen toplu/örnek-başına oranı 6,1×'tir.
- **GPU'da toplu çözme jeton-eşdeğer değildir, aritmetik-eşdeğerdir.** 8 satırlık bir matmul ile 1 satırlık matmul farklı çekirdek ve farklı toplama sırası kullanır; rastgele ağırlıklarla logitler neredeyse eşit olduğu için 5090'da 8 satırın 6'sı jeton jeton aynı çıktı, ikisi 84 ve 108. jetondan sonra ayrıştı. Aynı ayrışma `greedy_decode_batch` ile `greedy_decode` arasında da görülür (yani mantık değil, aritmetiktir) ve CPU'da hiç görülmez (`test_batched_greedy_ids_reproduces_per_sample_greedy_decode` CPU'yu zorlar). **Sonuç: iki koşunun CER'i ancak aynı cihazda ve aynı batch büyüklüğünde bit-bit karşılaştırılabilir.**
- **`eval.max_n_val` ASR yolunda hâlâ uygulanmıyor** (6. dalga önceliği 3 §4). `aicls.ml.trainer.limit_val_dataset(dataset, max_n)` artık bu işi tek çağrıda yapan, kesmeyi `ValCoverage` olarak kaydeden ortak yardımcıdır; `aicls.cli.train._run_whisper` içindeki `eval_ds` kurulumuna eklenmesi gerekir (sahibi: `aicls.cli`).
- **`score_predictions()`** üç durumlu sembolik eşdeğerliği "equal oranı" olarak toplar; `unparsable` burada eşit-değil sayılır. Bu, `aicls.eval.report`'un ayrı sütun tutan davranışından **farklıdır** ve raporda karıştırılmamalıdır.
- **Kuyruk kilidi `fit()` içinde alınmaz.** §8'in tek-iş kilidi süreç düzeyindedir ve `training/scripts/train.py` (gerçek veri koşularında varsayılan olarak) alır. `fit()` içine konsaydı her birim testi `nvidia-smi` yoklardı.
- **`fit()` koşu dizinine `env.json` ve `fit_report.json` de yazar** ve ikincisini izleyici artefaktı olarak kaydeder. §4.1 ikisini de saymıyor.
- **Cihaz seçimi enjekte edilebilir bir yoklama üzerinden yapılır** (`DeviceProbe` → `select_device`), böylece bf16/fp16/CUDA karar tablosu bu CPU makinesinde test edilebilir. `probe_device()` `torch.cuda`'ya dokunan tek yerdir ve CUDA yarısı `# pragma: no cover`'dır.

### 11.5 Whisper sarmalayıcısı (§4.2'ye göre sapmalar)

- **`training_args()` ikiye bölündü.** `Seq2SeqTrainingArguments.__post_init__` `accelerate>=0.26` şart koşar ve bu venv'de `accelerate` **kurulu değildir** (kapılı `ml` extra'sında). Bu yüzden bütün `TrainConfig` → HF eşlemesi `training_args_dict(*, with_eval=True) -> dict` içindedir ve `accelerate` olmadan tam olarak test edilebilir; `training_args()` bu sözlüğü nesneye veren iki satırlık sarmalayıcıdır. Şartnamedeki imza değişmedi.
- **`AiclsManifestCallback` bir `transformers.TrainerCallback` alt sınıfı değildir**, ördek-tipli düz bir sınıftır; böylece `import aicls.ml.whisper_ft` transformers'ı yüklemez (bir test bunu sabitler). HF'nin `CallbackHandler.call_event`'i olayları `getattr(callback, event)` ile çağırır ve `isinstance` denetimi yapmaz (kurulu transformers 4.57 kaynağında doğrulandı).
- **`forced_decoder_ids` ayarlanmaz.** transformers 4.57 bu alanı Whisper için kullanımdan kaldırıyor ve istemi `generation_config.language`/`.task` üzerinden çözüyor. `configure_generation()` bu ikisini kurar, `forced_decoder_ids`'i temizler ve `config.suppress_tokens`'ı boşaltır. §4.2'nin "zorunlu çözücü kimlikleri `language=tr, task=transcribe`" ifadesi modern API ile karşılanmıştır. **Bu, `transformers` sürümüne bağımlı tek noktadır**; her sürüm yükseltmesinde duman testi yeniden koşturulmalıdır.
- **`compute_metrics` üç anahtar döndürür:** `wer` (`tr_v1`, hedef metrik), `wer_raw` (`none`) ve `n_samples`. Sayı olmadan oran rapor edilmesin diye örnek sayısı da vardır.
- **`gradient_checkpointing` bir yapıcı argümanıdır**, yapılandırma anahtarı değil: `TrainConfig`'te `train.gradient_checkpointing` alanı yoktur. §4.2 ve §8 gradyan kontrol noktasını VRAM daraldığında geri düşüş olarak adlandırdığına göre **şemaya eklenmelidir** (istek `aicls.ml.config` sahibine iletildi).
- **`full_determinism` HF'ye geçilmez** (`enable_full_determinism` `warn_only` olmadan `use_deterministic_algorithms(True)` çağırır ve deterministik çekirdeği olmayan işlemleri kırar). Yerine §7 anlamıyla kendi `seed_everything(seed, deterministic=cfg.run.deterministic)` çağrımız kullanılır; HF'ye yalnızca `seed`/`data_seed` verilir.
- **`report_to=[]`.** Kayıt bizim JSONL/MLflow izleyicimizdir (ADR-0007); geri çağırma HF'nin `on_log`/`on_evaluate` yüklerini `Tracker.log_metrics`'e (train/valid ayrımıyla) aktarır, böylece HF döngüsü OCR eğiticisinin yazdığı `metrics.jsonl`'ın aynısını besler.
- **`fit()` `TRAINING` kapısını kendisi de sorar** (ilk iş olarak, tohumlamadan ve model yüklemeden önce). §9'un tablosu kapıyı CLI/betiğe koyar; burada da sorulması, CLI'yı atlayan bir çağıranın gerçek veriyle eğitememesi demektir. Yalnızca **reddedebilir**: hiçbir yerde `AICLS_GATE_*` ayarlanmaz, deftere satır eklenmez, `--i-have-approval` geçilmez.

### 11.6 Kontrol noktası düzeni (§5'e göre sapmalar)

§5, sekiz dosya listeliyor; gerçek dizin dokuz taşır. 2026-09-08'de üretilmiş gerçek bir kontrol noktası dizini:

```text
<run_dir>/checkpoints/step-00000002/
├── model.safetensors     # (paylaşımlı tensör varsa model.pt'ye düşer)
├── optimizer.pt
├── scheduler.pt
├── rng_state.pt
├── config.yaml           # çözümlenmiş
├── run_manifest.json
├── vocab.json
├── metrics.json
└── checkpoint.json       # EKLENDİ: model sha256, dosya listesi, adım, format
```

`checkpoint.json` iki iş yapar: `aicls models import`'un ihtiyaç duyduğu model SHA-256'sını taşır ve **tamamlanma işareti** olarak davranır — "yarım bir dizin yüklenemez" kuralı ancak böyle uygulanabilir. Ayrıca `CheckpointSaver` kontrol noktası kökünde `best.json` ve `last.json` **işaretçi dosyaları** yazar (Windows'ta sembolik bağ yok).

`save_checkpoint`'ün ilk parametresi `dir` değil `ckpt_dir`'dir (genel API'de yerleşik ad gölgelenmez).

!!! warning "Açık kusur: `checkpoint.metric` ile eğiticinin metrik adı uyuşmuyor"

    Sevk edilen OCR yapılandırmaları `checkpoint: {metric: val_cer}`, ASR yapılandırmaları
    `val_wer` diyor. `Trainer.evaluate()` ise doğrulama sözlüğünü `loss`, `val_loss`, `n_val`
    ve `cfg.eval.metrics` adlarıyla (yani **`cer`**, `wer`) döndürüyor. Sonuç: `CheckpointSaver`
    aradığı anahtarı hiç bulamıyor, **en iyi kontrol noktası hiç seçilmiyor**. 2026-09-08'de
    koşturulan gerçek duman koşusunun `fit_report.json`'u bunu açıkça gösteriyor:

    ```json
    {"best_metric": "val_cer", "best_value": null, "best_step": null, "best_checkpoint": null}
    ```

    Koşu yine de `last.json` üretir ve son kontrol noktası doğrudur; kaybolan şey
    "CER'e göre en iyi" seçimidir. İki taraftan biri düzeltilmelidir: ya `Trainer.evaluate()`
    doğrulama metriklerini `val_` önekiyle de yazmalı, ya da yapılandırmalar `cer`/`wer`
    demelidir. Hiçbir test bugün bu sözleşmeyi denetlemiyor; düzeltmeyle birlikte bir sözleşme
    testi de gelmelidir. Sahipleri: `aicls.ml.trainer` + `training/configs`.

### 11.7 Deney takibi (§6'ya göre sapmalar)

- **`metrics.jsonl` satır şeması** (bir testle sabitlenmiştir): `{"schema_version", "run_id", "step", "split", "wall_time", "ts", "metrics": {...}}`. Metrikler bir alt nesnede durur, böylece `step` veya `split` adlı bir metrik satır alanlarıyla çakışamaz. Sonlu olmayan değerler (ıraksamış kayıp) JSON `null` yazılır — dosya her zaman katı JSON olarak ayrışır, metriğin adı yine satırda görünür.
- **`Tracker.log_metrics` anahtar-kelimeli `split` alır**, `log_artifact` yazdığı kaydı döndürür.
- **`params.json` bir koşu kaydıdır**, düz yapılandırma değil: `{schema_version, run_id, run_name, started_at, ended_at, status, duration_s, params}`. Çözümlenmiş yapılandırma `params` altındadır; `read_params`/`read_metrics` yardımcıları vardır ki kimse elle ayrıştırmasın.
- **Varsayılan koşu dizini ULID son eki taşır** (`runs/<run.name>-<ulid>`): aynı yapılandırmanın iki koşusu tek bir `metrics.jsonl`'a eklenmemelidir (ADR-0007).

!!! warning "Açık kusur: bir koşu dizininde iki farklı `run_id` var"

    `RunManifest` kendi ULID'sini üretir, `JsonlTracker` de kendi ULID'sini üretir; ikisi
    birbirini görmez. 2026-09-08 duman koşusunda `run_manifest.json` `01M1Z95W53YDF29E69B8D6XGGP`,
    `metrics.jsonl`/`params.json`/`artifacts.jsonl` ise `01M1Z95V66DDYTEXDAMCEGX2YT` yazdı.
    Pratik sonuç: **ölçümleri manifeste `run_id` üzerinden bağlayamazsınız**; birleştirme
    anahtarı koşu DİZİNİDİR. `aicls models import` ve `report check` dizin üzerinden çalıştığı
    için bugün bir şey kırılmıyor, ama MLflow'a geçildiğinde veya koşular tek bir dizinde
    toplandığında bu yanlış birleştirmeye yol açar. Düzeltme: `Trainer` izleyiciyi manifest
    `run_id`'siyle kurmalı. Sahibi: `aicls.ml`.

### 11.8 Tekrarlanabilirlik (§7'ye göre sapmalar)

- `seed_everything(...)` kullanılan tohumu **`int` olarak döndürür** (moduller.md `-> None` yazıyordu).
- Kirli ağaç `RuntimeError` değil `DirtyWorkingTreeError` (bir `InputValidationError` alt sınıfı) fırlatır: kodlama standardı `AiclsError` dışı istisna yasaklar. `details["reason"] == "dirty_working_tree"`.
- `citable`, pydantic doğrulayıcısıyla **yeniden hesaplanır**: git yanıt verdi **ve** ağaç temiz. Git durumu bilinmiyorsa da alıntılanamaz — belgedeki "kirli ⇒ alıntılanamaz" kuralından daha sıkıdır.
- `RunManifest.capture(...)` dört ek anahtar-kelime alır (`repo_root`, `run_name`, `data_fingerprints`, `data_splits`) ve `allow_dirty` argümanını `cfg.train.allow_dirty` ile VEYA'lar; manifest **etkin** değeri kaydeder.
- **`data_fingerprints` hazırlanmış parça kökünü tanır** (2026-09-08 eki). `prepare_mathwriting.py` parmak izini **bölme başına** `<kök>/<bölme>/shards_manifest.json` içine yazar; okuyucu yalnızca kökteki `manifest.json`'a bakıyordu, bu yüzden RTX 5090'daki ilk gerçek MathWriting koşusunun manifesti `data_fingerprints: {}` ile kapandı — hangi kodun koştuğu yazılıydı, hangi veriyle koştuğu yazılı değildi. Artık koşum, açtığı her bölmenin parmak izini `mathwriting.train`, `mathwriting.valid`, ... olarak ve bunlardan türetilen tek bir `mathwriting` kimliğini kaydeder (`EvalReport.dataset_fingerprint` bunu alıntılar). Yerleşim bilgisi yalnızca `aicls.data.shards` içindedir (`discover_shard_splits`, `shard_fingerprints`, `shards_root_fingerprint`); manifest okuyucu dosya adı tahmin etmez. Varsayılan okuma **ucuzdur**: hazırlık sırasında hesaplanmış özet okunur, parçalar yeniden hash'lenmez (2,9 GiB'ı ilk adımdan önce okumamak için); baytların kendisi denetlenecekse `shard_fingerprints(..., verify=True)` ya da `aicls data verify` kullanılır.
- `pip freeze` kaydı **alt süreçle değil** `importlib.metadata` ile üretilir: aynı içerik, süreç yok, yapısal olarak ağa dokunamaz.
- `RunManifest.steps_per_s`, `aicls profile` koşana kadar `None` kalır. Profil `TRAINING` kapısının arkasındadır; dolayısıyla **bugün hiçbir epoch/bütçe tahmininin ölçüm dayanağı yoktur**.
- `nvidia-smi` yolu CPU derlemesinde de çalışır: 2026-09-08 manifesti `gpu_name: "NVIDIA GeForce RTX 5090"`, `driver: "610.88"`, `gpu_memory_total_mb: 32607.0` yazdı — ama aynı manifestte `cuda: null`, `cuda_available: false`, `torch: "2.11.0+cpu"`. Manifest donanımı görüyor, PyTorch görmüyor.

### 11.9 GPU planı ve kuyruk (§8'e göre sapmalar)

- **Her CUDA yolu yazıldı, hiçbiri koşmadı.** `# pragma: no cover` ile işaretli olanlar: `DeviceSpec.autocast_context`, `_backward`/`_optimizer_step` içindeki `GradScaler` dalları, `probe_device`'ın CUDA yarısı (`is_bf16_supported`, `get_device_capability`), `peak_vram_mb`/`reset_peak_vram`, `resume`'un `map_location="cuda"` dalı, `seed_everything`'in `torch.cuda.manual_seed_all` çağrısı ve RNG durumundaki CUDA bölümü. cu128 tekerleği geldiği gün **hepsi yeniden doğrulanmalıdır**.
- **fp16 + `GradScaler` tümüyle test dışıdır.** 5090'da amaçlanan kip bf16'dır; fp16 yalnızca şema izin verdiği için vardır.
- **`data.num_workers > 0` test edilmedi.** Testler `num_workers=0` koşar (Windows'ta `fork` yok). `seed_worker` ve `worker_init_fn` bağlantısı yazıldı, hiç çalışmadı; iş istasyonunun 6/8 ayarı bu yolu ilk kez sürecek.
- **`process_is_alive` Windows'ta `os.kill(pid, 0)` KULLANMAZ.** CPython'da `os.kill` Windows'ta `TerminateProcess` çağırır; bir canlılık denetimi sorduğu işi öldürürdü. Bunun yerine `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)` + `GetExitCodeProcess` ctypes prototipleriyle kullanılır.
- **KAPANDI (2026-09-08, 6. dalga): `training/queue.txt` artık kaybolmaz.** Kilit `training/queue.lock` adına taşındı ve `.gitignore`'a eklendi; `_live_holder()` hiçbir dosyayı silmez, `_write_lock()` sahibi yeniden denetledikten sonra üzerine yazar. Kilit koşunun *çevresinde* tutulduğu ve `RunManifest.capture()` git durumunu tam o sırada okuduğu için, izlenen herhangi bir kilit dosyası — üzerine yazılsa bile — her kuyruklu koşuyu `citable=false` yapardı (ADR-0007). `training/queue.txt` yalnızca belge olarak depoda kalır; hiçbir kod ona yazmaz.
- **KAPANDI (2026-09-08, 6. dalga): `TrainingQueueBusyError` artık `ErrorCode.TRAINING_BUSY` kullanır** (HTTP 503, başlık "GPU meşgul") ve metni `MSG_TRAINING_QUEUE_BUSY`'dir; `RATE_LIMITED`'ın "çok fazla istek" metni geride bırakıldı. Ayrıntı yine `details["reason"] ∈ {training_queue_busy, gpu_busy}` içindedir.

### 11.10 Kapılı betikler ve CLI (§9'a göre sapmalar)

- **Her kapılı betikte `--dry-run` vardır ve kapıyı hiç sormaz.** Planı (katalog girdisi, hedef dizin, boş/gerekli disk, kapının o anki durumu) yazar ve 0 ile çıkar. Onay talebi hazırlamak böylece kapısız bir iştir; `--json` aynı planı onay PR'ına eklenebilecek bir belge olarak verir. §9'un tablosu bunu saymıyor.
- **`--synthetic` kapıyı atlamaz, yapılandırmayı yeniden yazar.** `data.dataset` ailenin sentetik kardeşine çevrilir; kapısızlık kararını yine yalnızca `data.dataset`'e bakan `is_gate_free` verir. §9'un "`--synthetic` ise kapı aranmaz" ifadesi harfiyen okunsaydı `--synthetic --config <mathwriting>` gerçek veride kapıyı atlardı — bu bir güvenlik açığıdır ve kapatılmıştır (`tests/gated/test_gates.py::test_gated_train_synthetic_flag_rewrites_the_dataset_instead_of_skipping_the_gate`).
- **`--ledger` diye bir komut satırı bayrağı yoktur.** Gated betiklerin `main()` işlevi yalnızca testlerin geçtiği anahtar-kelimeli bir `ledger_path` alır. Gerekçe: defter üçlü kilidin **denetlenebilir** yarısıdır (git geçmişinde durur); onu işaret eden bir bayrak, operatörün kendi onayını üretmesine izin verirdi.
- **`train.py` kapıyı kendisi sorar** (torch içe aktarılmadan, veri açılmadan, koşu dizini oluşturulmadan önce); `Trainer` ikinci kez sorar. Onaylı bir koşuda `gate_invocations.log`'a bir satır fazladan düşer — "reddedilen koşu arkasında hiçbir şey bırakmaz" karşılığında kabul edilmiş bir bedeldir.
- **`download_mathwriting.py`**, katalog URL'sinin üzerine bir **konak beyaz listesi** (`storage.googleapis.com`), yalnızca-https denetimi, tar-slip koruması (yalnızca düz dosya ve dizin üyeleri açılır; sembolik/sert bağlantı reddedilir) ve `.part` → `os.replace` yayınlama ekler. Arşiv `.tgz`'dir; sunucunun bildirdiği `Content-Length` **ilk bayt yazılmadan önce** katalogla karşılaştırılır ve kesilen aktarım `Range` isteğiyle `.part` dosyasından devam eder. Disk denetimi varsayılan olarak yalnızca arşiv boyutudur (arşiv açılmaz); `--extract` verilirse `beklenen_bayt × (1 + --extract-factor)` (varsayılan 7,0) formülüne döner ve **kapıdan sonra, ilk bayttan önce** uygulanır. Bu çarpan ölçüm değil, katalog yorumundan alınmış bir tahmindir ve planda yazdığı için operatör görebilir/ezebilir. `--excerpt` aynı kapının arkasında 1,55 MiB'lık resmi örneklemi indirir.
- **`download_weights.py`**, hub deposu kimliğini katalogdan `<source>/<version>` olarak türetir (`openai/whisper-small`). Hub koordinatının yazıldığı tek yer `src/aicls/data/catalog.yaml` olarak kalır — kod değil, incelenebilir veri.
- **CLI hata sınırı tek yerdedir:** kök `typer.Typer(cls=AiclsGroup)` `Group.invoke`'u sarar ve tüm ağaç için `AiclsError` yakalar. `GateError` → 3, diğer `AiclsError` → 1, click kullanım hataları → 2.
- **CLI günlükleri stderr'e gider.** `configure_cli_logging()` structlog'u yazma anında `sys.stderr`'i çözen bir aracıya yönlendirir; olmasaydı `aicls train --json | jq` bozulurdu. CLI günlükçüleri bu yüzden modül düzeyinde değil, kullanım yerinde (`cli_logger(__name__)`) bağlanır.
- **`aicls eval` asla terfi ettirmez.** Değerlendirme koşumu `latest.json` ve `summary.md` yazdığı için komut ikisini sonradan geri koyar (`--promote` bundan vazgeçer). Terfi `aicls report promote`'un işidir (ADR-0015 §4).
- **`aicls eval --model`** bir etiket değil, davranış değişikliğidir: sürümün `model_versions` içinde kayıtlı **ve** terfi ettirilmiş olmasını arar, sonra ölçümü API'nin gerçekten servis ettiği `ModelRegistry` üzerinden koşar. Diskte kontrol noktası yoksa kayıt stub'a düşer ve rapor `stub: true` yazar — bayrak, bir stub koşusunu kâğıt üzerinde model koşusuna çeviremez.
- **`aicls models import` varsayılan olarak SHA-256 doğrular** ve uyuşmazlıkta dosyanın tamamını reddeder; `--no-verify` yalnızca kapı açılmadan önce kaydı tohumlamak içindir.
- **`aicls train` aileye göre yönlendirir:** `seq2seq_image`/`seq2seq_stroke` → `Trainer`; `whisper` → `WhisperFineTuner`; `wav2vec2` → tipli ret (`family_not_supported_by_cli`), çünkü ince ayar sarmalayıcısı yoktur. Reddetmek numara yapmaktan iyidir.
- **`aicls` konsol komutu kurulu DEĞİLDİR.** `pyproject.toml` `aicls = "aicls.cli.main:app"` ilan eder ve hedef gerçektir, ama paket venv'e kurulmamıştır; venv'in `Scripts/` dizininde `aicls.exe` yoktur. Bugünkü çağrı biçimi `PYTHONPATH=src python -m aicls.cli.main ...` veya `pip install -e .` (bkz. [el kitabı §1](egitim-el-kitabi.md#1-ön-koşullar-ve-çağrı-biçimi)).

### 11.11 Bugün kanıtlanmamış olanlar

Aşağıdaki yolların **hiçbiri hiç çalışmadı**. Bir kapı açıldığında ilk yapılacak iş, karşısındaki satırı doğrulamaktır.

| Kanıtlanmamış yol | Neden | Açılması gereken kapı |
|---|---|---|
| `download_archive()`, `download_snapshot()` gövdeleri | tek ağ çağrıları; kapı kapalı | `DATASET_DOWNLOAD`, `WEIGHTS_DOWNLOAD` |
| Her CUDA/bf16/fp16 dalı (11.9) | torch CPU derlemesi | `CUDA_INSTALL` |
| `WhisperFineTuner.build_trainer()` ve `fit()` gövdesi | `accelerate` kurulu değil, `Seq2SeqTrainingArguments` kurulamıyor | `CUDA_INSTALL` (accelerate'i getirir) |
| `apply_lora()`'nın başarı dalı | `peft` kurulu değil; LoRA hiç uygulanmadı | `CUDA_INSTALL` |
| `MlflowTracker`'ın istemci çağrıları | `mlflow` kurulu değil; yalnızca sahte modülle test edildi | — (saf Python, kapısız kurulabilir) |
| `data.num_workers > 0`, `seed_worker` | testler `num_workers=0` | — |
| safetensors → `torch.save` geri düşüşü (bağlı gömme) | küçük test modelinde paylaşımlı tensör yok | — |
| `TrainingQueue._write_lock` yarış dalı, `checkpoint._publish` geri alma | gerçek yarış / dosya sistemi hatası gerekir | — |
| `prepare_mathwriting.py --render` / `--limit`, gerçek ~230 bin ifadelik arşiv | veri yok | `DATASET_DOWNLOAD` |
| `prepare_audio_commands.py --verify-files` | diskte WAV yok | — (İP-3 toplama) |
| `train.py --resume` betik düzeyi bağlantısı | **KAPANDI (2026-09-09):** adım 18.000 kontrol noktasından gerçek devretme koştu — ağırlıklar, optimizer, planlayıcı ve RNG durumu geri yüklendi, koşu 215.520 adımlık bütçesine kaldığı yerden devam etti. `aicls train` de artık `--resume` alır (2026-09-09'a kadar yalnızca betikte vardı) | — |
| Eğitilmiş bir kontrol noktasından rapor üretme | **KAPANDI (2026-09-09):** `training/scripts/evaluate.py --component ocr --shards ... --split test --model-dir <kontrol noktası>` gerçek kontrol noktasıyla koştu. Daha önce hiç denenmemişti ve **çalışmıyordu** — bkz. §11.12 | — |
| ASR eğitim yolu uçtan uca | **KAPANDI (2026-09-08):** `aicls train --config training/configs/asr/smoke_synthetic.yaml --synthetic` kapsayıcı içinde RTX 5090'da uçtan uca koştu (2 adım, 2 üretim tabanlı değerlendirme, HF kontrol noktaları + üç köken dosyası). Gerçek Whisper ağırlıklarıyla yol hâlâ koşmadı | `WEIGHTS_DOWNLOAD` (gerçek ağırlıklar için) |

**KAPANDI (2026-09-08, 6. dalga).** `training/configs/asr/smoke_synthetic.yaml` artık CLI üzerinden de koşar: `WhisperFineTuner`, `weights_dir` ve `model.base_dir` birlikte `null` olduğunda **yalnızca `cfg.is_gate_free` ise** `random_init` moduna geçer, modeli `aicls.ml.registry.build_model` ile, işlemciyi `build_smoke_processor()` ile kurar. Adı verilmiş ama diskte olmayan bir kontrol noktası hâlâ hata verir ve `WEIGHTS_DOWNLOAD` kapısını adıyla söyler — gerçek veri üzerinde rastgele ağırlık uydurma sayı üretmek olurdu. Duman koşusunun tokenizer'ı indirme gerektirmeyen bir **karakter** modelidir; oradan çıkan her WER bir bağlantı denetimidir, sonuç değildir.

### 11.12 İlk gerçek OCR eğitimi: iki arıza, ikisi de ölçümde (2026-09-08/09)

MathWriting üzerinde ilk gerçek eğitim koşusu iki arıza ortaya çıkardı. İkisinin de ortak yanı şu:
**model iyiydi, modeli okuduğumuzu sandığımız yol bozuktu.** Bu bölüm ikisini de kayda geçirir,
çünkü ikisi de sessizdi ve ikisi de bir raporu zehirleyebilirdi.

#### Değerlendirme modeli değil, normalizasyon istatistiklerini ölçüyordu

`runs/ocr_r34_tfm_full2` adım 18.000'de `val_cer` 7,34 ve `ExpRate` 0,0 bildiriyordu; eğitim kaybı
ise 0,30-0,75 arasında tertemizdi. Aynı ağırlıklar, aynı batch: **batch istatistikleriyle 0,32,
biriken çalışan istatistiklerle 8,52** öğretmen zorlamalı kayıp. Tek değişken BatchNorm kipi.

Arıza yapısı gereği sessizdir: eğitim çalışan istatistikleri hiç okumaz, dolayısıyla eğitim kaybı
temiz kalırken doğrulama, en iyi kontrol noktası seçimi ve erken durdurma hep bozuk sayıya bakar.
O koşu 7,34 üzerinden erken durduruyordu.

| Adım 18.000 kontrol noktası | CER | ExpRate |
|---|---|---|
| olduğu gibi | 7,3356 | 0,000 |
| istatistikler 4096 örnekle yenilendi | 0,152 | 0,551 |
| devretme testinde 20 adım (öğrenme oranı tükenmiş) | 0,1439 | 0,528 |

Üçüncü satır en güçlü kanıttır: öğrenme oranı sıfıra yakınken 20 adım hiçbir şey öğretemez, yani
düzelen tek şey tamponlardır.

**Kök neden** (düzeltme sonrası taze tahminlerle ölçüldü; düzeltmeden önce ölçülemiyordu, çünkü
tamponlar kovaladıkları hedefin gerisindeki bir ortalamaydı): ağın iç etkinlik ölçeği hem ağır
kuyruklu hem hızlı hareketli. Adım 26.000'de ortanca kanal varyansı 19,8, %99'luk dilim 278, tepe
604; en üstteki %1 kanal toplam varyansın %10,4'ünü taşıyor. Toplu ölçek 2000 adımda +%3,80 büyür
ama bu sayı asıl olayı gizler: 8512 kanalın **2373'ü %10'dan fazla büyümüş, 1052'si %10'dan fazla
küçülmüştür**. Momentum 0,1 ≈ on batch'lik pencere demektir; bu pencereyle ağır kuyruklu ve hızlı
oynayan bir hedef kötü tahmin edilir ve 36 ardışık BatchNorm katmanında küçük hatalar birbirinin
üzerine biner.

**Düzeltme** `eval.recalibrate_norm_samples` (varsayılan kapalı; OCR referans yapılandırmasında
4096). Değerlendirmeden hemen önce çalışan istatistikler sıfırlanır ve **eğitim** bölmesinden
kümülatif olarak yeniden kurulur. Üç ayrıntı korrektlik kuralıdır, tercih değil:

- tahmin **yalnızca eğitim bölmesinden** alınır — puanlanacak bölmeden alınsaydı model kısmen test
  verisine uydurulmuş olurdu; gerçek kontrol noktasında dürüst seçim zaten bedavadır (0,152'ye
  karşı 0,157);
- yalnızca BatchNorm katmanları eğitim kipine alınır, **dropout kapalı kalır** — açık kalsaydı
  varyans tahmini değerlendirmede bulunmayacak gürültüyle şişerdi;
- geçiş **kümülatif** ortalar ve özgün momentum sonradan geri verilir, böylece devam eden eğitim
  kendi ortalama takvimini korur.

Yeni istatistikler **saklanır** ve kontrol noktasına yazılır; dışa aktarılan model kendini tarif
eden istatistikleri taşısın diye. Her ölçüm satırı `norm_recalibrated_n` taşır — sayının hangi tür
istatistikten geldiği tahmin edilmek zorunda kalınmasın diye. BatchNorm taşıyan bir model
kalibrasyonsuz puanlanırsa döngü `ml.trainer.norm_statistics_not_recalibrated` uyarısı düşer;
ilk koşuya mal olan şey tam olarak bu sessizlikti.

#### Eğitilmiş hiçbir kontrol noktası rapora çevrilemiyordu

`TorchRecognizer.load()` eğiticinin yazdığı her kontrol noktasında pydantic hatasıyla düşüyordu.
Eğitici tüm şemasını diske yazar, isteğe bağlı alanlar dahil, dolayısıyla `config.yaml` içinde
`model.max_len: null` durur — ve bu `null`, alanın 160'lık varsayılanını taşıyamayacağı bir değerle
eziyordu. Yazan ile okuyan aynı anahtarı farklı yorumluyordu: `aicls.ml.config` `self.max_len or
DEFAULT_MAX_LEN` der, okuyucu ise `None`'ı geçerli bir değer sanıyordu. Serileştirilmiş bir
yapılandırmadaki `null` "belirtilmemiş" demektir; `_load_config` artık böyle davranır.

2026-09-09'da raporlama yolu ilk kez gerçek bir kontrol noktasına doğrultularak bulundu — eğitimin
bitmesine sekiz saat vardı. Bulunmasaydı sonuç, eğitilmiş ama raporlanamayan bir modeldi.

### 11.13 Belge borçları

- `docs/mimari/moduller.md` `aicls.ml`/`aicls.cli` bölümleri 2026-09-08'de bu bölümle birlikte güncellendi.
- `training/README.md` ve `training/configs/README.md` gerçek ağacı anlatır; bu sayfayla çelişmez.
- Ses bölme dosyasının yeri iki belgede farklı yazıyor: bu sayfanın §2'si depo göreli `datasets/manifests/...` diyor, [veri-yonetimi.md §3](veri-yonetimi.md#3-manifestler-ve-sürümleme) depoya **özetin** girdiğini söylüyor. Kod `out.parent/audio_commands_splits.json` arar ve kayıtlar KVKK gereği depo dışındadır; sevk edilen yapılandırmalar bu yüzden `${AICLS_DATA_ROOT}/manifests/` kullanır. Tek cümlelik bir karar `docs/ml` sahibinden beklenmektedir.
- `docs/gelistirme/kurulum.md` satır 74'ün vaat ettiği `training/scripts/download_dataset.py --name` betiği **yoktur**; kurulum sayfası 2026-09-08'de düzeltildi, betik hâlâ yazılmadı.
