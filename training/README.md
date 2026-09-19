# training/ — yapılandırmalar ve kapılı betikler

Eğitim altyapısının belgesi: [docs/ml/egitim-altyapisi.md](../docs/ml/egitim-altyapisi.md); kararlar [ADR-0006](../docs/adr/0006-ml-altyapisi-pytorch-hf-yapilandirma.md), [ADR-0007](../docs/adr/0007-deney-takibi-jsonl-mlflow-manifest.md), [ADR-0012](../docs/adr/0012-onay-kapilari.md). Yapılandırma ağacının haritası ayrıca [configs/README.md](configs/README.md) içindedir.

## Düzen

```text
training/
├── configs/_base.yaml                 # ortak alanlar (seed, tracking, checkpoint, data root)
├── configs/ocr/*.yaml                 # offline_resnet34_tfm, offline_effnet_b0_tfm, online_stroke_tfm, ablation_preproc, profile, smoke_synthetic
├── configs/asr/*.yaml                 # whisper_small_ft, whisper_large_v3_turbo_lora, profile, smoke_synthetic
├── configs/sweeps/ocr_lr_sweep.yaml
├── scripts/download_mathwriting.py    # KAPILI: DATASET_DOWNLOAD
├── scripts/download_weights.py        # KAPILI: WEIGHTS_DOWNLOAD
├── scripts/prepare_mathwriting.py     # kapısız (yereldeki InkML → npz parçaları)
├── scripts/prepare_audio_commands.py  # kapısız (katkı dışa aktarımı → konuşmacı-ayrık manifest)
├── scripts/build_tokenizer.py         # kapısız (parçalardan vocab.json)
├── scripts/train.py                   # KAPILI: TRAINING (--synthetic kapısızdır)
├── scripts/evaluate.py                # kapısız (yereldeki kontrol noktası + veri → EvalReport)
└── queue.txt                          # tek-iş kilidinin açıklaması; kilidin kendisi queue.lock (git-ignore)
```

Betikler paket değildir; depo kökünden proje venv'i ile çağrılır. `src/` yolu betiklerin başında `sys.path`'e eklenir, bu yüzden editable kuruluma gerek yoktur:

```powershell
C:\Users\cagri\.venvs\ai-cls\Scripts\python.exe training\scripts\train.py --synthetic
```

## Kapılar (ADR-0012)

`download_*.py` ve `train.py` `aicls.core.gates.require_gate()` çağırır. Üç koşul **birlikte** aranır:

1. `--i-have-approval` bayrağı,
2. `AICLS_GATE_<KAPI>=approved` ortam değişkeni,
3. `docs/gelistirme/onay-defteri.md` içinde `| <KAPI> | approved | <tarih> | <kim> | <kapsam> |` satırı.

Biri eksikse **çıkış kodu 3** ve eksik koşulların tamamı Türkçe olarak yazdırılır. CI ortamında (`CI` veya `GITHUB_ACTIONS`) kapı üç koşul sağlansa bile reddeder. Her başarılı geçiş `AICLS_DATA_ROOT/gate_invocations.log` dosyasına yazılır.

**Her kapılı betikte `--dry-run` vardır ve kapıyı hiç sormaz.** Yaptığı tek şey eylemin maliyetini yazdırmaktır: katalog girdisi (URL, sürüm, lisans, beklenen boyut, sha256), hedef dizin, boş disk alanı ve kapının o anki durumu. Onay talebi hazırlamak bu yüzden kapısız bir iştir; `--json` aynı planı onay PR'ına eklenebilecek bir belge olarak verir.

Kapının kod tarafındaki ikinci yarısı statiktir: `tools/check_gates.py` `src`, `training`, `experiments`, `scripts`, `tools` ağaçlarında hub kimliği (`from_pretrained("org/ad")`), `load_dataset("org/ad")`, `hf_hub_download`/`snapshot_download`, `torch.hub`, `weights=` (None dışı), `pretrained=True`, `urllib`/`requests` ve HF/PyTorch URL değişmezlerini arar. İzinli tek istisna `training/scripts/download_*.py` ve `scripts/install_ml.*` dosyalarıdır; gözden geçirilmiş tek satırlık istisna `# gate-ok: <gerekçe>` ile işaretlenir. Depodaki **ağ kodu yalnızca iki dosyada** bulunur ve her ikisinde de tek bir işleve hapsedilmiştir (`download_archive`, `download_snapshot`), yalnızca `require_gate` döndükten sonra çağrılır. Bunu `tests/gated/test_gates.py` hem davranış hem de kaynak düzeyinde doğrular.

`poe_tasks.toml` içinde `train`/`download` görevi **yoktur**; bu betikler yalnızca açık çağrıyla çalışır.

## Tipik akış

```powershell
# 1) veri (KAPILI) — önce maliyeti gör
python training\scripts\download_mathwriting.py --dry-run
python training\scripts\download_mathwriting.py --i-have-approval      # + env + defter satırı

# 2) parçalar (kapısız)
python training\scripts\prepare_mathwriting.py --split train valid test --render

# 3) sözlük (kapısız; yalnızca train etiketlerinden)
python training\scripts\build_tokenizer.py --split train --min-freq 2

# 4) ağırlıklar (KAPILI, yalnızca ASR için)
python training\scripts\download_weights.py --model whisper-small --dry-run

# 5) ses manifesti (kapısız; konuşmacı-ayrık bölme)
python training\scripts\prepare_audio_commands.py --export <export.jsonl>

# 6) eğitim (KAPILI; sentetik yol kapısız)
python training\scripts\train.py --config training\configs\ocr\offline_resnet34_tfm.yaml --dry-run
python training\scripts\train.py --synthetic                            # CPU duman koşusu

# 7) ölçüm (kapısız)
python training\scripts\evaluate.py --component ocr --shards <kök> --split test --model-dir <kontrol noktası>
```

### Betikler tek tek

| Betik | Kapı | Ne yapar | Önemli bayraklar |
|---|---|---|---|
| `download_mathwriting.py` | `DATASET_DOWNLOAD` | katalog girdisini yazdırır, diski denetler, arşivi indirir, sha256 hesaplar, açar, `manifest.json` üretir | `--dry-run`, `--json`, `--dest`, `--no-extract`, `--keep-archive`, `--extract-factor` |
| `download_weights.py` | `WEIGHTS_DOWNLOAD` | `AICLS_MODELS_ROOT/pretrained/<ad>/` içine anlık görüntü indirir, dosya sha256'larını ve `manifest.json` yazar | `--list`, `--model <katalog anahtarı\|repo kimliği>`, `--dry-run`, `--json`, `--allow-patterns` |
| `prepare_mathwriting.py` | yok | InkML → `npz` parçaları + `shards_manifest.json` (`aicls.data.prepare_shards`) | `--src`, `--out`, `--split`, `--render`, `--image-size`, `--limit`, `--manifest`, `--dry-run` |
| `prepare_audio_commands.py` | yok | katkı dışa aktarımını doğrular (KVKK alanları **hata**), konuşmacı-ayrık tabakalı bölme üretir, manifesti ve `audio_commands_splits.json` dosyasını yazar | `--export`, `--out`, `--splits-path`, `--ratios`, `--seed`, `--no-freeze`, `--verify-files`, `--summary`, `--dry-run` |
| `build_tokenizer.py` | yok | parça etiketlerinden `vocab.json` (yalnızca `train` bölmesinden) | `--shards`, `--out`, `--split`, `--min-freq`, `--dry-run` |
| `train.py` | `TRAINING` (sentetikte yok) | `Trainer` (OCR) / `WhisperFineTuner` (ASR), kuyruk kilidi, manifest + kontrol noktası + izleme | `--config`, `--set k=v`, `--synthetic`, `--max-steps`, `--resume`, `--run-dir`, `--allow-dirty`, `--queue/--no-queue`, `--dry-run` |
| `evaluate.py` | yok | `aicls.eval.run_ocr_eval`/`run_asr_eval` → `reports/metrics/<ad>_<sha>.json` | `--component`, `--model-dir`/`--stub`, `--shards`/`--manifest`, `--split`, `--synthetic N`, `--max-n`, `--out-dir`, `--dry-run` |

Çıkış kodları her betikte aynıdır: `0` başarı veya kuru koşu, `1` tipli AI-CLS hatası (`AiclsError`), `3` kapı reddi.

### `--synthetic` neden güvenlidir

`train.py --synthetic` kapıyı **atlamaz**; yapılandırmayı yeniden yazar. `data.dataset` model ailesinin sentetik kardeşine (`synthetic_ink` / `synthetic_audio`) çevrilir ve kapısızlık kararını yine `TrainConfig.is_gate_free` verir — bu özellik yalnızca `data.dataset` adına bakar. Yani bir koşu, çağıranın iddiası yüzünden değil, gerçekten ne üzerinde eğitildiği için kapısızdır. Gerçek bir veri yapılandırması `--synthetic` ile çağrılırsa MathWriting'e dokunmaz, sentetik veriye geçer ve bunu plan çıktısında `note` satırı olarak söyler. `tests/gated/test_gates.py::test_gated_train_synthetic_flag_rewrites_the_dataset_instead_of_skipping_the_gate` bunu sabitler.

## `queue.lock` ve `queue.txt`

Tek RTX 5090 iki eğitim işini aynı anda taşıyamaz (docs/ml/egitim-altyapisi.md §8). `aicls.ml.queue.TrainingQueue` **`training/queue.lock`** dosyasını kilit olarak kullanır: kilidi alırken tek satırlık JSON yazar (`pid`, `host`, `started_at`, `job`, `user`), bırakırken siler. `nvidia-smi` başka bir sürecin GPU belleğini tuttuğunu gösterirse kuyruk yine reddeder (`TrainingQueueBusyError`, `ErrorCode.TRAINING_BUSY`, çıkış kodu 1). `training/queue.txt` ise yalnızca bu düzeneği anlatan, hiçbir kodun yazmadığı açıklama metnidir.

`train.py` kilidi gerçek veri koşularında varsayılan olarak alır, kapısız sentetik koşularda almaz (`--queue` ile zorlanır, `--no-queue` ile kapatılır): bir birim testinin `nvidia-smi` yoklaması gürültüdür, güvenlik değildir.

**Neden ayrı bir dosya (dalga 5 kusuru, kapatıldı):** kilit koşu SÜRERKEN duruyor ve `RunManifest` git durumunu tam o sırada kaydediyor; izlenen bir kilit dosyası her kuyruklu koşuyu kirli ağaç yüzünden `citable=false` yapardı (ADR-0007). Ayrıca eski kuyruk, okunamayan bir kilidi "bayat" sayıp **siliyordu**, yani depodaki `queue.txt` ilk gerçek koşuda yok oluyordu. Artık kilit `.gitignore` içindeki `training/queue.lock` dosyasıdır ve `TrainingQueue` hiçbir dosyayı silmez; sahibi olmayan bir kilidin yalnızca ÜZERİNE yazar (`ml.queue.unreadable_lock_ignored`).

## Dürüstlük notu

`--synthetic` koşuları, `--stub` ölçümleri ve `smoke_synthetic.yaml` yapılandırmaları **bağlantı denetimidir (wiring check), sonuç değildir**: rastgele başlatılmış ağırlıklar, üretilmiş fikstürler ve bölmesi olmayan bir "doğrulama" kümesi. Bu koşulardan çıkan her `fit_report.json` `wiring_check: true` ve `citable: false` taşır, her `EvalReport` `stub: true` ile işaretlenir ve renderer "kanıt değil" yazar. CER < %10 ve WER < %10 hedefleri için kanıt ancak `DATASET_DOWNLOAD`, `WEIGHTS_DOWNLOAD`, `CUDA_INSTALL` ve `TRAINING` kapıları açıldıktan sonra, gerçek veriyle üretilebilir.

## Donanım notu

RTX 5090 (Blackwell, sm_120) için CUDA yığını yalnızca `scripts/install_ml.ps1 -IHaveApproval` ile (cu128 pin, `get_device_capability() == (12, 0)` doğrulaması) kurulur. Bugünkü torch CPU derlemesidir; `train.py` bf16 isteyen bir yapılandırmada sessizce CPU fp32'ye düşer ve bunu `--dry-run` çıktısındaki `device` satırında söyler.
