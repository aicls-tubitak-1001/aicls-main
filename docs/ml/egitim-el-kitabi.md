# Eğitim El Kitabı (operatör)

Bu sayfa, **onay kapıları açıldığı gün açılacak** sayfadır. Tasarımı değil işletimi anlatır: hangi komut, hangi sırayla, ne bekleyerek, ne çıktığında ne yapılacak. Tasarım ve gerekçeler [egitim-altyapisi.md](egitim-altyapisi.md) içindedir; kapı kuralı [ADR-0012](../adr/0012-onay-kapilari.md) ve [kurulum.md §3](../gelistirme/kurulum.md#3-onay-kapıları) içindedir.

!!! danger "2026-09-08 itibarıyla bu el kitabındaki kapılı adımların hiçbiri koşturulmamıştır"

    Dört kapının dördü de kapalıdır. Bugüne kadar **hiçbir veri kümesi indirilmedi, hiçbir
    ağırlık indirilmedi ve hiçbir model gerçek veriyle eğitilmedi**. Projenin **OCR CER sayısı
    ve ASR WER sayısı yoktur**; `reports/metrics/` içinde model içeren her şey bir stub veya
    rastgele başlatılmış ağdan gelen **bağlantı denetimidir**.

    Kapılı komutlar yazıldı ve reddedilme davranışları test edildi; **başarı yolları hiç
    çalışmadı**. Her adımda önce `--dry-run` koşturun, çıktıyı okuyun, sonra devam edin. Bir
    adım beklenmedik davranırsa el kitabını düzeltmek de işin parçasıdır: bu sayfa ilk gerçek
    koşuda güncellenecek şekilde yazılmıştır.

!!! success "Değişen tek şey: GPU artık erişilebilir (kapısız, konteyner üzerinden)"

    2026-09-08'de §2 yeniden yazıldı. Windows sanal ortamındaki torch hâlâ `2.11.0+cpu` ve
    `CUDA_INSTALL` kapısı hâlâ kapalı; ama RTX 5090'a WSL2 + nvidia-docker üzerinden
    erişiliyor ve **sentetik duman koşusu ile profil ölçümü GPU'da gerçekten koştu**
    ([ADR-0020](../adr/0020-gpu-calisma-ortami-wsl-nvidia-docker.md)). Bu koşulardan çıkan
    hiçbir sayı sonuç değildir; §2.4'teki rakamlar bağlantı denetimidir.

---

## 1. Ön koşullar ve çağrı biçimi

| Ön koşul | Doğrulama |
|---|---|
| Depo temiz (kirli ağaç koşuyu reddeder) | `git status --short` boş çıkmalı |
| Sanal ortam | `C:\Users\cagri\.venvs\ai-cls\Scripts\python.exe` |
| `AICLS_DATA_ROOT` OneDrive dışında, NVMe üzerinde | `poe env-check` |
| Yeterli disk | MathWriting arşivi için ~3 GB (açılırsa +20 GB), ağırlıklar için ~1,5 GB, koşu dizinleri için ~20 GB |
| Onay defterinde ilgili satır | [onay-defteri.md](../gelistirme/onay-defteri.md) |

**`aicls` konsol komutu Windows sanal ortamında bugün kurulu değildir** (konteynerde vardır: [§2.1](#21-bugünkü-yol-kapısız-konteyner)). Windows tarafında iki geçerli çağrı biçimi vardır; el kitabı boyunca `$py` ve `aicls` şu anlama gelir:

```powershell
# PowerShell oturumunun başında bir kez:
$py = "C:\Users\cagri\.venvs\ai-cls\Scripts\python.exe"
cd "C:\Users\cagri\OneDrive\Masaüstü\fırat\PROJE\ABDULLAH_HOCA_1001\ai-cls"
$env:PYTHONPATH = "src"

# "aicls <komut>" yerine bugün:
& $py -m aicls.cli.main <komut> ...

# Kalıcı çözüm (paketleme işi, bir kez):
& $py -m pip install -e .        # sonra doğrudan: aicls <komut>
```

`training/scripts/*.py` betikleri paket değildir; `src/` yolunu kendileri ekler, `PYTHONPATH` olmadan da çalışırlar:

```powershell
& $py training\scripts\train.py --help
```

**Üçlü kilidi kurma kalıbı** (her kapı için aynı; `<KAPI>` yerine gerçek ad):

```powershell
# 1) Defter satırı — PR ile, inceleme sonrası (kurulum.md §3.3)
# 2) Ortam değişkeni — yalnızca bu oturum için, .env'e YAZMAYIN
$env:AICLS_GATE_<KAPI> = "approved"
# 3) Bayrak — komutun kendisine
... --i-have-approval
```

İş bitince ortam değişkenini düşürün: `Remove-Item Env:\AICLS_GATE_<KAPI>`.

---

## 2. GPU'yu kullanılabilir hâle getirmek

**Neden ilk bu?** Veri ve ağırlık indirmek GPU olmadan da mümkündür, ama profil ölçümü olmadan
hiçbir bütçe kurulamaz ve profil ölçümü GPU ister. Ayrıca `accelerate` ve `peft` GPU yığınıyla
gelir; ASR ince ayarı onlarsız *kurulamaz* bile.

İki yol vardır. **Bugün geçerli olan §2.1'dir**; §2.2 kapı açıldığı gün geçerli olacaktır.

### 2.1 Bugünkü yol: kapısız konteyner

Windows sanal ortamındaki torch CPU derlemesidir. GPU'ya WSL2 + nvidia-docker üzerinden
erişilir; hiçbir teker indirilmez, Windows sanal ortamı değişmez, hiçbir kapı açılmaz. Karar ve
kabul edilen bedeller: [ADR-0020](../adr/0020-gpu-calisma-ortami-wsl-nvidia-docker.md);
işletim ayrıntısı `scripts/gpu/README.md`; kurulum belgesi
[kurulum.md §7](../gelistirme/kurulum.md#7-gpu-wsl2--nvidia-docker).

```powershell
# 1) Ortamı hazırla ve donanımı doğrula (idempotent; istediğiniz kadar koşturun)
wsl -e bash scripts/gpu/setup-gpu-env.sh

# 2) Bundan sonra her aicls komutu bu sarmalayıcıdan geçer
.\scripts\gpu\aicls-gpu.ps1 --print train --config training\configs\ocr\smoke_synthetic.yaml
.\scripts\gpu\aicls-gpu.ps1 train --config training/configs/ocr/smoke_synthetic.yaml --synthetic --device cuda
```

`setup-gpu-env.sh` ayrıca `/mnt/d/aicls-venv` içine `pip install --no-deps
--no-build-isolation -e <depo>` uygular; **konteynerde `aicls` konsol komutu vardır**. §1'deki
`& $py -m aicls.cli.main ...` biçimi Windows tarafı için geçerliliğini korur.

!!! warning "Konteyner kapıyı zayıflatmaz"

    `AICLS_GATE_*` (dördü) ve `CI` / `GITHUB_ACTIONS` konteynere aktarılır, onay defteri
    bağlanan depodan okunur. Onaysız bir **gerçek-veri** koşusu konteynerde de çıkış kodu 3
    verir; 2026-09-08'de ölçüldü: `missing: ['cli_flag', 'env', 'ledger']`. Koşucu hiçbir kapı
    değişkeni ayarlamaz ve `--i-have-approval` eklemez.

### 2.2 Kapı `CUDA_INSTALL` — Windows'a gerçek kurulum (hâlâ koşturulmadı)

```powershell
# Önce kuru koşu: ne kurulacağını yazar, hiçbir şey kurmaz
pwsh scripts\install_ml.ps1 -DryRun

# Sonra gerçek kurulum (üçlü kilit)
$env:AICLS_GATE_CUDA_INSTALL = "approved"
pwsh scripts\install_ml.ps1 -IHaveApproval
```

Betik ne yapar: cu128 indeksini (`https://download.pytorch.org/whl/cu128`) sabitler,
`torch==2.11.0` + `torchvision==0.26.0` + `torchaudio==2.11.0` kurar, ardından `peft` ve
`accelerate` kurar, sonra **kurulum sonrası doğrulama** koşar ve
`AICLS_DATA_ROOT\gate_invocations.log` dosyasına bir satır ekler.

Bu adım, konteyner yolu varken **zorunlu değildir**. Gerekçesi tek ortam sadeliği ve
Windows servis dağıtımıdır; kararı ADR-0020 "Sonuçlar" 2'deki iş istasyonu notuyla birlikte
verin.

### 2.3 Doğrulama

```powershell
# Konteyner yolu — betik bunu kendisi yapar, elle de yapabilirsiniz
wsl -e bash scripts/gpu/setup-gpu-env.sh --verify

# Windows'a kurulum yapıldıysa
& $py -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability())"
nvidia-smi
```

`get_device_capability()` **(12, 0)** dönmelidir — Blackwell sm_120. Başka bir değer, yanlış
teker ya da yanlış kart demektir; `verify_gpu.py` ve `install_ml.ps1` bu durumda çıkış kodu 1
verir. Sürücü (bugün 610.88) ve araç takımı sürümünü defter satırının `kapsam` sütununa yazın.

### 2.4 Kapısız duman koşusu — 2026-09-08'de GPU'da gerçekten koştu

CUDA yolu wave 5'in sonuna kadar hiç çalışmamıştı. Aşağıdaki iki komut kapısızdır (sentetik
veri) ve GPU'yu gerçek veriye sokmadan önce yolun bütününü sınar:

```powershell
.\scripts\gpu\aicls-gpu.ps1 train --config training/configs/ocr/smoke_synthetic.yaml `
    --synthetic --device cuda --override train.amp=bf16

.\scripts\gpu\aicls-gpu.ps1 profile --config training/configs/ocr/smoke_synthetic.yaml `
    --synthetic --steps 20 --warmup 3 --device cuda --override train.amp=bf16
```

İlk koşuda ölçülenler (RTX 5090, torch `2.9.1+cu130`):

```text
device cuda, dtype bfloat16, autocast true, device_name NVIDIA GeForce RTX 5090
steps 2, train_loss 4.959
profile: 45.198 adım/s, 180.8 örnek/s, peak_vram_mb 66.97, dataloader_wait_pct %8.54
```

Beklenen: `device: cuda`, `dtype: bfloat16`, `peak_vram_mb` artık `null` **değil**. `null`
kalıyorsa CUDA seçilmemiştir; `--device cuda` ile zorlayın ve `select_device` günlüğündeki
`fallback_reason` alanını okuyun.

!!! danger "Bu rakamlar bağlantı denetimidir"

    Rastgele başlatılmış küçük bir ağın 64 sentetik iz üzerindeki kaybı ve CER'i, sözleşmedeki
    CER < %10 ölçütü hakkında **hiçbir şey** söylemez. `45,2 adım/s` de bir bütçe değildir:
    gerçek bütçe, kapılar açıldıktan sonra §5'teki 200 adımlık profilden türetilir.

### 2.5 GPU'da ilk koşuda çıkan bilinen kusurlar

Konteynerde `tests/unit/ml` + `tests/unit/ocr` koşturulduğunda (544 geçti / 10 kaldı) iki
**gerçek** kusur görüldü; ikisi de yalnızca CUDA'lı bir makinede görünür:

| Kusur | Belirti | Sahip |
|---|---|---|
| `Trainer.evaluate()` girdileri cihaza taşır ama **modeli taşımaz** (yalnızca `fit()` taşır) | `fit()` olmadan çağrılan değerlendirme: `Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same` | `aicls.ml` |
| `Trainer.resume()` durum yüklerken cihaz farkında değil | `Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu` | `aicls.ml` |

Kalan sekiz başarısızlık testlerin varsayımıdır (`Path("D:/aicls-data")` Linux'ta **görelidir**;
`peak_vram_mb is None` ve `torch.equal` iddiaları CPU varsayar) ve ürünü etkilemez.

---

## 3. Kapı `DATASET_DOWNLOAD` — MathWriting'i indirmek ve parçalara ayırmak

### 3.1 Maliyeti görün (kapısız)

```powershell
& $py training\scripts\download_mathwriting.py --dry-run
```

2026-09-08'de bu komutun verdiği plan: sürüm `mathwriting-2024.09`, lisans **CC-BY-NC-SA-4.0**, arşiv `mathwriting-2024.tgz` **2,88 GiB** (3 096 141 721 B), gerekli disk **2,88 GiB** (varsayılan: arşiv açılmaz), hedef `D:\aicls-data\mathwriting`, kapı durumu `closed — missing: cli_flag, env, ledger`. `--json` aynı planı onay PR'ına eklenecek belge olarak verir.

!!! danger "Onay talebine mutlaka yazılacak: lisans CC BY-NC-SA 4.0"
    MathWriting **ticari kullanıma kapalıdır** (NonCommercial) ve türev çalışma aynı lisansla
    dağıtılmak zorundadır (ShareAlike). Bu veriyle eğitilmiş ağırlıklar patent/ticarileşme
    aşamasına doğrudan taşınamaz. Kapı onayını isteyen kişi bu cümleyi onay defterindeki
    `kapsam` sütununa yazmalıdır. (Depoda önceden yanlışlıkla Apache-2.0 yazıyordu; 2026-09-08'de
    resmi kaynaktan doğrulanıp düzeltildi.)

### 3.2 Önce 1,5 MB'lık resmi örneklemi indirin

```powershell
$env:AICLS_GATE_DATASET_DOWNLOAD = "approved"
& $py training\scripts\download_mathwriting.py --excerpt --i-have-approval
```

`--excerpt`, resmi `mathwriting-2024-excerpt.tgz` (1 620 634 B) arşivini indirir. Aynı bölme
düzeni, aynı lisans. **Hattın tamamını (indirme → parçalama → sözlük → değerlendirme) bununla bir
kez uçtan uca koşturun**; 2,88 GiB ve saatler harcamadan önce her adımın çalıştığını görürsünüz.

### 3.3 Tam arşivi indirin

```powershell
& $py training\scripts\download_mathwriting.py --i-have-approval
```

Betik: URL'yi doğrular (yalnızca `https`, yalnızca `storage.googleapis.com`), diski denetler,
sunucunun bildirdiği `Content-Length` değerini **tek bir bayt yazmadan önce** katalogla
karşılaştırır, `.part` dosyasına indirir, `os.replace` ile yayınlar ve sha256 hesaplayıp
`manifest.json` üretir.

**Kesilirse aynı komutu tekrar çalıştırın:** `.part` dosyası korunur ve aktarım HTTP `Range`
isteğiyle kaldığı yerden devam eder; 3 GB'lık bir indirme sıfırdan başlamaz. Sıfırdan başlamak
isterseniz `--no-resume`.

Arşiv **varsayılan olarak açılmaz** — bir sonraki adım `.tgz` akışını doğrudan okur. Gerçek bir
dizin ağacı gerekiyorsa `--extract` ekleyin (~20 GB, tar-slip ve düz-dosya denetimiyle açılır;
`--keep-archive` arşivi de saklar).

```powershell
# Doğrulama: manifest ile disk uyuşuyor mu?
& $py -m aicls.cli.main data verify mathwriting
```

### 3.4 InkML → `npz` parçaları (kapısız, uzun sürer)

```powershell
& $py training\scripts\prepare_mathwriting.py --archive --dry-run
& $py training\scripts\prepare_mathwriting.py --archive --split train valid test symbols synthetic --shard-size 2048 --render --manifest
```

- `--archive`, InkML üyelerini **doğrudan `.tgz` içinden akıtır**; ~20 GB'lık açılmış ağaç ve
  ~630 bin küçük dosya yazımı hiç olmaz. `AICLS_DATA_ROOT` bir `drvfs` bağlama noktasındaysa
  (WSL2 → Windows `D:`) aradaki fark saatlerdir. Değersiz kullanıldığında
  (`--archive` tek başına) katalogdaki arşiv adı `AICLS_DATA_ROOT\mathwriting\` altında aranır;
  başka bir dosya için `--archive <yol>`.
- Bütün bölmeler **tek geçişte** üretilir: gzip akışında geri sarma yoktur, beş bölme için beş
  ayrı çağrı 2,88 GiB'ı beş kez açardı.
- Açılmış bir ağaçtan üretmek isterseniz eski yol duruyor: `--src <dizin>`.
- **Ölçülen kazanç (2026-09-08, konteyner, `drvfs` veri kökü, 20 bin dosyalık sentetik fikstür):**
  açma 166 s + dizinden parçalama 135 s = **301 s**; arşivden akıtma **6,5 s** ve diske tek bir
  InkML dosyası yazılmadı. Bu iki kod yolunun bu dosya sistemindeki G/Ç ölçümüdür; MathWriting'in
  içeriği hakkında hiçbir şey söylemez ve hiçbir metriğin kanıtı değildir.
- **Aynı külliyat, iki kaynak, aynı örnekler.** Dizin yolu dosyaları yola göre sıralı okur, arşiv
  yolu arşiv sırasını izler; `tar` sıralı bir dizin gezintisi yazdığı için ikisi pratikte çakışır
  ve parmak izleri de aynı çıkar (aynı ölçümde beş bölmenin beşinde). Garanti değildir: başka bir
  sırayla paketlenmiş bir arşiv parça sınırlarını kaydırır. Hangisinin kullanıldığı
  `shards_manifest.json` içindeki `source_root` alanında yazar (`tar:...!train` ya da bir dizin
  yolu); raporlar karşılaştırılırken bu alan da karşılaştırılır.
- `--render`, her örneğin rasterini parçaya yazar. **Çevrimdışı (offline) OCR yolu için şart değildir ama şiddetle önerilir**: eğitim sırasında her epoch'ta yeniden çizim yapılmaz, dataloader beklemesi düşer. Disk maliyeti artar.
- `--manifest`, parça kökünü hash'ler; bu parmak izi `run_manifest.json` ve her `EvalReport.dataset_fingerprint` alanına girer. **Atlamayın**: parmak izi olmayan bir koşu rapora giremez.
- `--skip-invalid`, bozuk bir InkML belgesinin ~630 bin dosyalık bir koşuyu düşürmesini engeller:
  belge atlanır, `n_skipped` sayacına yazılır ve günlüğe düşer. Varsayılan **kapalıdır** (bozuk
  külliyat sessiz kalmasın diye).
- Resmi bölmeler korunur; `test` bölmesi CER'in ölçüleceği bölmedir ve asla yeniden bölünmez.

### 3.5 Sözlük (kapısız)

```powershell
& $py training\scripts\build_tokenizer.py --split train --min-freq 2
```

Sözlük **yalnızca `train` etiketlerinden** kurulur. `valid`/`test` etiketlerinden kurmak, ölçüm kümesinin bilgisinin modele sızması demektir.

Çıktı yolunu yapılandırmadaki `model.vocab_path` ile karşılaştırın (`training/configs/ocr/offline_resnet34_tfm.yaml`).

---

## 4. Kapı `WEIGHTS_DOWNLOAD` — ön eğitimli ağırlıklar (yalnızca ASR)

OCR modelleri sıfırdan eğitilir (`weights=None`); bu kapı yalnızca Whisper/Wav2Vec2 içindir.

```powershell
& $py training\scripts\download_weights.py --list
& $py training\scripts\download_weights.py --model whisper-small --dry-run

$env:AICLS_GATE_WEIGHTS_DOWNLOAD = "approved"
& $py training\scripts\download_weights.py --model whisper-small --i-have-approval
```

Katalogdaki üç ağırlık ve boyutları: `whisper-small` **0,45 GiB** (MIT), `whisper-large-v3-turbo` **1,51 GiB** (MIT), `wav2vec2-xls-r-300m` **1,17 GiB** (Apache-2.0). Hedef `AICLS_MODELS_ROOT\pretrained\<ad>\`. Depo hiçbir yerde hub kimliği kullanmaz; `from_pretrained` yalnızca **yerel yol** alır ve `local_files_only=True` ile çağrılır.

Sonra ASR yapılandırmasındaki `model.base_dir` bu dizini göstermelidir:

```yaml
# training/configs/asr/whisper_small_ft.yaml
model:
  base_dir: ${AICLS_MODELS_ROOT}/pretrained/whisper-small
```

---

## 5. Kapı `TRAINING`, birinci kullanım — **ÖNCE PROFİL**

!!! tip "Kural: ölçmeden bütçe kurulmaz"

    [egitim-altyapisi.md §8](egitim-altyapisi.md#8-gpu-planı-tek-rtx-5090-32-gb-windows) bu
    projede elle yazılmış epoch süresi tahminini yasaklar. Kapı açıldıktan sonra her
    yapılandırma ailesinin **ilk kapılı eylemi** 200 adımlık profildir. Epoch ve bütçe
    yalnızca bu ölçümden türetilir.

```powershell
$env:AICLS_GATE_TRAINING = "approved"

& $py -m aicls.cli.main profile --config training\configs\ocr\profile.yaml --steps 200 --warmup 5 --i-have-approval
& $py -m aicls.cli.main profile --config training\configs\asr\profile.yaml  --steps 200 --warmup 5 --i-have-approval
```

Rapor `reports/metrics/profile_<ad>.json` dosyasına yazılır ve şu alanları taşır: `steps_per_s`, `samples_per_s`, `peak_vram_mb`, `dataloader_wait_pct`, `batch_size`, `grad_accum`, `device_name`, `dtype`, `amp`, `config_hash`, `git_sha` — ve 7. dalgadan beri **zaman bütçesinin bölümü** (`train_seconds`, `eval_seconds`, `checkpoint_seconds`, `dataloader_wait_s`) ile **doğrulama maliyeti** (`eval_measured`, `eval_samples`, `eval_batch_size`, `eval_beam`, `eval_decoder`, `seconds_per_val_sample`). Ölçüm en az bir tam mikro-batch genişliğinde yapılır: çözme çekirdek-başlatma sınırlı olduğu için 64 satırlık bir batch'in 8 satırı neredeyse 64 satır kadar sürer; kısmi batch ölçümü örnek başına maliyeti 5090'da 5 kat fazla gösterdi (0,139 s→ 0,018 s, beam 4).

!!! warning "Adım döngüsünden bütçe çıkmaz"

    Ölçülen gerçek koşuda (`runs/ocr_excerpt_wiring`, RTX 5090, 2026-09-08) 20 eğitim adımı **1,4 s**,
    her 16 örneklik doğrulama **~296 s** sürdü. Yalnız `steps_per_s`'ten türetilen bir epoch tahmini
    duvar saatinin %3'ünü ölçer. Bu yüzden `aicls profile` adım döngüsünden sonra bir de gerçek
    doğrulama geçişi ölçer ve raporun `projection` bloğunda tam takvimi fiyatlar. Doğrulama
    ölçülemediyse (örn. `valid` bölmesi hazır değilse) blok `projected: false`,
    `reason: eval_cost_not_measured` der ve **hiçbir saat değeri vermez**; bu bir hata değil, bir
    reddir.

### 5.1 Bütçeyi okumak

Önce raporun kendi projeksiyonunu okuyun — elle formül uygulanmaz:

```powershell
& $py -c @'
import json,sys; sys.path.insert(0,'src')
from aicls.ml.profile import ProfileReport
r = ProfileReport.model_validate_json(open('reports/metrics/profile_ocr_profile.json',encoding='utf-8').read())
pr = r.projection
if not pr.projected:
    print('projeksiyon yok:', pr.reason)           # ölçülmemiş terim varsa sayı üretilmez
else:
    print(f'{pr.total_steps} adim + {pr.eval_count} x {pr.n_val_per_eval} dogrulama')
    print(f'egitim {pr.train_seconds/3600:.2f} saat, dogrulama {pr.eval_seconds/3600:.2f} saat')
    print(f'toplam {pr.total_hours:.2f} saat, bunun %{pr.eval_share_pct:.1f} kadari dogrulama')
'@
```

`projection` bloğu profil sırasında geçerli olan `train.epochs`/`train.max_steps`,
`train.eval_every` ve `eval.max_n_val` değerlerini fiyatlar; `beam` alanı hangi ışın
genişliğiyle ölçüldüğünü söyler. **`training/configs/*/profile.yaml` `beam: 1` kullanır**, sevk
edilen OCR yapılandırması ise `beam: 4`: bir beam-1 ölçümü beam-4 takvimini fiyatlamaz. Gerçek
bütçe için profili asıl yapılandırma üzerinde (`-o train.max_steps=200` ile) koşturun.

### 5.1.1 Yalnız epoch süresi (doğrulama hariç)

Adım döngüsünün tek başına maliyeti (`aicls.ml.profile.estimate_epoch_seconds` ile aynı formül) —
**bu bir bütçe değildir, bütçenin bir terimidir**:

```text
adım/epoch    = ceil( ceil(n_örnek / batch_size) / grad_accum )
epoch_saniye  = adım/epoch ÷ steps_per_s
```

```powershell
& $py -c @'
import json,sys; sys.path.insert(0,'src')
from aicls.ml.profile import ProfileReport
r = ProfileReport.model_validate_json(open('reports/metrics/profile_ocr_profile.json',encoding='utf-8').read())
n = 230000                                  # train bölmesinin gerçek örnek sayısı: parça manifestinden okuyun
s = r.epoch_seconds(n)
print(f'{r.steps_per_s:.2f} adim/s, epoch {s/3600:.2f} saat, 30 epoch {30*s/3600:.1f} saat')
'@
```

`epoch_seconds` doğrulamayı **saymaz**; toplam için `projection.total_hours` kullanın.
Bu sayıyı `train.epochs` / `train.max_steps` değerine dönüştürün ve **yapılandırmaya yazın**. Sevk edilen değerler (`epochs: 30`, `max_steps: 4000`, batch 64/128/32/16) tasarım niyetidir, ölçüm değildir — profil sonrası hepsi gözden geçirilmelidir.

### 5.2 VRAM ve dataloader okuması

| Gözlem | Anlamı | Yapılacak |
|---|---|---|
| `peak_vram_mb` 32 GB'ın %85'inin üzerinde | OOM riski | `data.batch_size` yarıya, `train.grad_accum` iki katına (etkin batch sabit kalır) |
| `dataloader_wait_pct` > %20 | GPU veriyi bekliyor | `data.num_workers` artırın (bugünkü makine 6/8, iş istasyonu 12/16); parçalar NVMe'de mi; `--render` ile önceden çizin |
| `peak_vram_mb` çok düşük, `dataloader_wait_pct` düşük | GPU boşta | `batch_size` artırın, sonra profili **yeniden koşturun** |

Her değişiklikten sonra profil yeniden koşturulur. Değişmiş bir yapılandırmanın eski ölçümü geçersizdir; `config_hash` zaten farklı çıkar.

### 5.3 İlk gerçek koşunun planı — 7. dalga kapısının ÖLÇTÜĞÜ değerler (2026-09-08)

Bu bölümdeki her sayı, 7. dalga bütünleştirme kapısında **bu makinede, RTX 5090 üzerinde,
gerçek MathWriting örneklem parçalarına** (`/mnt/d/aicls-data/mathwriting-excerpt/shards`,
parmak izi `sha256:47a5962a…`, 100 gerçek `valid` örneği) ve **sevk edilen mimariye**
(`offline_resnet34_tfm`: ResNet34 + 6 katmanlı Transformer, d_model 256, 96×384, gerçek
198 simgelik sözlük, bf16) karşı ölçülmüştür.

!!! danger "Dürüstlük: ağırlıklar rastgele"

    Hiçbir kontrol noktası yok; bu ölçümlerdeki ağırlıklar **rastgele başlatılmıştır**. Bu
    yüzden buradaki hiçbir CER/TER değeri kanıt değildir ve hiçbiri yazılmamıştır. **Duvar
    saati sayıları kanıttır**: ÖNCE ve SONRA aynı süreç içinde, aynı veriye, aynı mimariye,
    aynı `max_len`e sürülmüştür. Rastgele ağırlıkla hemen hiçbir dizi `</s>` üretmediği için
    her çözme tam `max_len=160` bütçesini harcar — yani bu **en kötü durumdur**, bütçe
    kurmak için doğru olan da budur.

#### Doğrulama maliyeti: ÖNCE → SONRA

`Trainer.evaluate()`, aynı eğiticiyle, aynı 16 gerçek örnek üzerinde, `beam 4`,
`data.batch_size 8`. "ÖNCE" satırı yeniden yazım değildir: `beam_search`, `036266d`
(`runs/ocr_excerpt_wiring`'i üreten sürüm) git nesnesinden yüklenip yerine konmuştur.

| durum | 16 örnek toplam | örnek başına | çözücü |
|---|---:|---:|---|
| **ÖNCE** (6. dalga kodu, örnek-başına) | 255,61 s | **15,976 s** | `seq2seq.beam_search (per sample)` |
| **SONRA** (toplu + KV önbelleği) | 1,058 s | **0,0661 s** | `seq2seq.beam_search_batch` |

Ölçülen hızlanma **241,7×**. `runs/ocr_excerpt_wiring`'in kendi `metrics.jsonl` kaydı aynı işi
296 s / 16 = 18,5 s/örnek olarak yazmıştı; burada ölçülen 15,98 s/örnek onun %14 içindedir
(makinede aynı anda başka konteynerler boştaydı; fark ölçüm gürültüsüdür).

Daha büyük yığınlarda, hep **gerçek** örneklerle:

| yığın | ışın | örnek | toplam süre | örnek başına | tepe VRAM (ayrılmış) |
|---:|---:|---:|---:|---:|---:|
| 8 | 4 | 16 | 1,058 s | 0,0661 s | 206 MB |
| 16 | 4 | 100 | 3,935 s | 0,0394 s | 397 MB |
| 64 | 4 | 100 | 2,040 s | 0,0204 s | 914 MB |
| 64 | 1 | 100 | 1,634 s | 0,0163 s | 549 MB |
| 16 | 4 | **200** | **6,913 s** | 0,0346 s | 397 MB |
| 64 | 4 | **200** | **3,855 s** | **0,0193 s** | 914 MB |

**200 gerçek örneklik bir doğrulama geçişi, sevk edilen yığın boyunda 3,9 saniyedir.** (Resmi
örneklemin `valid` bölmesinde 100 örnek olduğu için 200'lük bölme, örneklemin `valid` + `test`
bölmelerinin gerçek mürekkepleri birleştirilerek kuruldu — hiçbir şey indirilmedi, hiçbir şey
üretilmedi; bkz. `/mnt/d/aicls-data/tmp/w8_val200`.)

#### Uçtan uca: kapıyı doğuran koşunun aynısı

`runs/ocr_excerpt_wiring`'in `params.json`'undaki her hiper-parametre kopyalanarak (yığın 8,
`num_workers` 2, `synthetic_ratio` 0,0, bf16, 20 adım, `eval_every` 10, `max_n_val` 16, ışın 4,
tohum 1234) `runs/w8_gate_e2e` koşuldu. İki koşu arasındaki tek fark **koddur**.

| | 6. dalga `ocr_excerpt_wiring` | 7. dalga `w8_gate_e2e` |
|---|---:|---:|
| `wall_time_s` | 613,36 | **21,85** (28,1×) |
| `steps_per_s` (eğitim) | 8,657 | **10,335** — gerileme yok |
| doğrulama örneği başına | 18,5 s | **0,102 s** (181×) |
| `dataloader_wait_pct` | 12,05 | 13,85 |
| doğrulamanın saatteki payı | ~%96,6 (metrics.jsonl'den) | **%14,95** (rapor artık yazıyor) |
| `samples_seen` / `epochs` | 156 / 1,538 | 156 / 1,538 |

`citable: false` — bu ağaç 7. dalga değişiklikleriyle kirlidir. Koşu bir **ölçümdür**, alıntılanabilir
bir sonuç değildir ve raporu bunu kendisi söyler.

#### Verim ve VRAM (`aicls profile`, gerçek yapılandırma, gerçek parçalar)

| | yığın 8 | yığın 64 (sevk edilen) |
|---|---:|---:|
| `steps_per_s` | 14,074 | 5,800 |
| `samples_per_s` | 112,6 | **371,2** |
| `peak_vram_mb` (eğitim + çözme) | 598,7 | **1 514,8** |
| `seconds_per_val_sample` | 0,1168 | **0,0176** |
| `dataloader_wait_pct` | 18,2 | 58,6 |

`dataloader_wait_pct` 58,6 örneklem eseridir: 100 örneklik `train` bölmesi yığın 64'te epoch
başına 2 adım verir, yükleyici sürekli yeniden başlar. Gerçek korpusta bu terim düşecektir;
**yine de gerçek parçalar hazır olur olmaz profil yeniden koşturulmalıdır.**

**VRAM boşluğu**: kart 32 607 MB. Sevk edilen ayarda tepe **1 514,8 MB — kartın %4,6'sı**.
Çözme tarafı ayrıca sınandı ve **hiçbirinde OOM olmadı**: yığın 128 × ışın 4 → 1 409 MB,
yığın 64 × ışın 8 → 1 700 MB. Toplu ışın araması bu mimaride bellek sorunu değildir.

#### 30 epoch mümkün mü? Evet — ve önce mümkün değildi

Ölçülen iki terim: eğitim **371,2 örnek/s**, doğrulama **0,0193 s/örnek** (yığın 64, ışın 4).
Aşağıdaki tablo bu ikisinin aritmetiğidir; `N` = `train` bölmesinin gerçek örnek sayısıdır ve
**hazırlama bitmeden bilinmez** (duyurulan ~230 000 ile hesaplandı, ölçülmedi):

| terim | ÖNCE (6. dalga kodu) | SONRA (bugün) |
|---|---:|---:|
| 30 epoch eğitim (230 000 gerçek örnek) | 5,2 sa | 5,2 sa (değişmedi) |
| doğrulama geçişi (2 000 örnek, ışın 4) | 8,9 sa | **38,5 s** |
| `eval_every: 2000` → ~54 doğrulama | **479 sa ≈ 20 gün** | **0,58 sa** |
| toplam | **imkânsız** | **≈ 5,8 saat** |

Yani: **evet, periyodik doğrulamalı 30 epoch artık mümkündür ve ölçülen bütçe ~6 saattir**;
6. dalga kodunda aynı takvimin doğrulaması tek başına 20 günden uzun sürerdi.

!!! warning "Bu bir projeksiyondur, ölçüm değil"

    5,2 saatlik eğitim terimi, **100 örneklik örneklem üzerinde** ölçülen 371,2 örnek/s'in
    230 000 örneğe uzatılmasıdır. Tam korpus (2,9 GiB arşiv indi, `npz` parçalarına
    **henüz dönüştürülmedi**) hazır olduğunda `aicls profile --config
    training/configs/ocr/offline_resnet34_tfm.yaml --steps 200` yeniden koşturulmalı ve bu
    tablo onun `projection` bloğuyla değiştirilmelidir. Doğrulama terimi (0,0193 s/örnek)
    gerçek `valid` mürekkepleriyle ölçülmüştür ve doğrudan taşınır.

#### Komut ve gerekçelendirilmiş ezmeler

```bash
# WSL içinden; kapı satırı defterde onaylı (2026-09-08, kapsam: OCR + MathWriting)
export AICLS_GATE_TRAINING=approved

scripts/gpu/aicls-gpu.sh train \
  --config training/configs/ocr/offline_resnet34_tfm.yaml \
  --i-have-approval \
  -o data.root=/mnt/d/aicls-data/mathwriting/shards \
  -o model.vocab_path=/mnt/d/aicls-data/mathwriting/vocab.json \
  -o data.synthetic_ratio=0.0 \
  -o data.batch_size=64 \
  -o data.num_workers=6 \
  -o eval.beam=4 \
  -o eval.max_n_val=2000 \
  -o train.eval_every=2000 \
  -o train.ckpt_every=2000 \
  -o train.epochs=30
```

| ezme | gerekçe (ölçüm) |
|---|---|
| `data.batch_size=64` | Tepe VRAM 1 514,8 MB = kartın %4,6'sı; sevk edilen değer güvenli. Çözme tarafı 128'de de (1 409 MB) sorunsuz — ama 128'de **eğitim adımı** ölçülmedi; yükseltmeden önce profil koşturun. |
| `eval.max_n_val=2000` | 2 000 × 0,0193 s = **38,5 s**. Bu değeri kısmak için sebep kalmadı; 2 000 örnek CER < %10 ile 2×%10'u ayırt etmeye yeter (`MIN_VAL_SAMPLES` gerekçesi). |
| `train.eval_every=2000` | 30 epoch ≈ 107 800 adım → ~54 doğrulama → 0,58 sa, toplam bütçenin ~%10'u. |
| `eval.beam=4` | Sözleşmedeki raporlama ayarı. Ölçülen fark: ışın 4 → 0,0204 s/örnek, açgözlü → 0,0163 s/örnek; ışın 4'ün maliyeti artık %25, eskiden 4×'ti. |
| `data.synthetic_ratio=0.0` | **Zorunlu, geçici**: `synthetic_ratio` bugün *her* bölmeye uygulanır. Sevk edilen 0,5 ile `build_dataset(cfg,'valid')` 100 gerçek + 100 sentetik = 200 örnek verir (bu makinede doğrulandı) ve `val_cer`'in yarısı sentetik çizimlerden gelir — üstelik `checkpoint.metric: val_cer` en iyi kontrol noktasını bu karışık sayıyla seçer. Bölme-başına düzeltme inene kadar 0,0 kullanın. Raporlanan test CER'i bundan **etkilenmez**: `training/scripts/evaluate.py` parçaları doğrudan okur. |
| `data.num_workers=6` | Bugünkü makine 8 mantıksal çekirdek; profil `dataloader_wait_pct` 58,6 gösteriyor ama bu örneklem eseridir. **Gerçek parçalarla ilk profilde yeniden bakın.** |

!!! warning "Yığın boyunu koşu boyunca DEĞİŞTİRMEYİN — CER ona bağlıdır"

    Toplu çözme kodlayıcıyı da toplu koşturur ve ResNet34 CUDA'da yığın-boyu değişmez değildir.
    7. dalga kapısında sevk edilen mimariyle, gerçek 16 örnekle ölçüldü:
    `encode(batch16)[i]` ile `encode(batch[i:i+1])[0]` arasında **bağıl 1,19e-3** fark var
    (147 449 / 147 456 eleman farklı; `torch.backends.cudnn.allow_tf32` varsayılan olarak
    `True`). Çözücünün kendisi *tam* eşdeğerdir — bellek sabit tutulduğunda toplu çözme 6. dalga
    çözücüsünün jetonlarını 16/16 greedy, 16/16 ışın listesi olarak birebir üretir — ama uçtan
    uca 16 satırın 2'si farklı jeton verir. **Sonuç: iki CER ancak aynı cihazda ve aynı
    `data.batch_size` ile karşılaştırılabilir.** Ablasyon tablosunun her satırı aynı yığınla
    koşturulmalı ve `data.batch_size` CER'in yanına yazılmalıdır.

#### Birinci, ikinci ve onuncu saatte neye bakılır

**1. saat — koşu ayakta mı, sayılar akıyor mu**

```bash
tail -f <run_dir>/metrics.jsonl
```

* `run_manifest.json`: `citable: true` ve `data_fingerprints` **dolu** (boş sözlük = 6. dalga
  kusuru, düzeltildi; boşsa koşuyu durdurun, sayı alıntılanamaz).
* İlk `valid` satırı geldi mi ve `eval_seconds` ≈ 38 s mi? On kat büyükse toplu çözücü devrede
  değildir: `fit_report.eval_decoder` **`seq2seq.beam_search_batch`** demeli. `... (per sample)`
  diyorsa koşuyu durdurun.
* `n_val` = 2 000, `n_val_total` = gerçek `valid` boyu, `val_truncated` beklediğiniz gibi mi.
* `train` satırlarında `loss` düşüyor, `grad_norm` patlamıyor, `lr` ısınma çizelgesini izliyor.
* `nvidia-smi`: kullanım sürekli %90+ değilse `dataloader_wait_pct`'e bakın.

**2. saat — öğreniyor mu, yoksa yalnızca koşuyor mu**

* İkinci ve üçüncü `val_cer` ilkinden **belirgin** düşük mü? Rastgele başlatılmış bir ağın ilk
  CER'i %100 civarındadır; 2. saatte hâlâ oradaysa öğrenme oranı/ısınma yanlıştır.
* `optim.warmup_steps: 2000` bitti mi (2 000 adım ≈ 6 dakika); bitmeden gelen hiçbir sayı
  yorumlanmaz — `fit_report.wiring_check` de bunu böyle işaretler.
* `checkpoints/best.json` gerçekten güncelleniyor mu (`checkpoint.metric: val_cer`, `mode: min`).
* Tepe VRAM 1,5 GB civarında kaldı mı; kartta yer varsa bir sonraki koşuda yığını büyütmek
  serbesttir — ama profil olmadan değil.

**10. saat — bütçe tuttu mu, sayı raporlanabilir mi**

* Duvar saati ölçülen ~5,8 saatlik projeksiyonun içinde mi? Değilse farkı **hangi terimin**
  yediğini `fit_report`'un `train_seconds` / `eval_seconds` / `checkpoint_seconds` /
  `other_seconds` bölümünden okuyun — 7. dalgadan beri bu bölüm var.
* `early_stop_patience: 25` devreye girdiyse `early_stopped: true` olur; bu bir başarısızlık
  değil, çizelgenin erken bitmesidir. Sabır 5 iken (9 Eylül'e kadar) bu **yanlış** biçimde
  tetiklendi: 2000 örneklik alt kümede ardışık CER farklarının std'si 0,0085, adım başına
  iyileşme 0,0013; şanslı bir rekordan sonra beş gürültülü doğrulama, kayıp hâlâ düşerken ve
  lr takvimin ortasındayken koşuyu bitirdi. `early_stopped: true` gördüğünüzde önce eğitim
  kaybının ve doğrulama kaybının (`val_loss`) o aralıkta ne yaptığına bakın; ikisi de
  düşüyorsa durma metrik gürültüsüdür ve `--resume` ile devam edilir (sayaç sıfırdan başlar).
* Koşu bittiğinde raporlanacak sayı **buradan değil**, `training/scripts/evaluate.py --component
  ocr --split test` çağrısından gelir (§9). Eğitim döngüsünün `val_cer`'i kontrol noktası
  seçimi içindir.
* `fit_report.wiring_check` **false** olmalı: aksi hâlde koşu ölçüt üretmemiştir ve
  `aicls report check` terfiyi reddeder.

---

## 6. Gerçek koşuyu başlatmak

```powershell
# 0) Ağaç temiz olmalı — kirli ağaç koşuyu reddeder (ADR-0007)
git status --short

# 1) Planı görün; kapı sorulmaz, hiçbir şey yazılmaz
& $py training\scripts\train.py --config training\configs\ocr\offline_resnet34_tfm.yaml --dry-run

# 2) Koşu
$env:AICLS_GATE_TRAINING = "approved"
& $py training\scripts\train.py --config training\configs\ocr\offline_resnet34_tfm.yaml --i-have-approval
```

Eşdeğer CLI biçimi:

```powershell
& $py -m aicls.cli.main train --config training\configs\ocr\offline_resnet34_tfm.yaml --i-have-approval
```

Sık kullanılan ek bayraklar: `--max-steps N` (**mutlak** bütçe, "N adım daha" değil), `--run-dir <yol>`, `--no-val`, `--json`; ayrıca yalnızca betikte `--queue` / `--no-queue` ve `--allow-dirty` vardır.

!!! note "Ezme bayrağının adı iki tarafta farklıdır"

    Betikte `--set optim.lr=1e-4`, CLI'da `--override optim.lr=1e-4` (`-o`). İkisi de
    tekrarlanabilir ve aynı noktalı yol sözdizimini kullanır.

### 6.1 Koşu dizini neye benzer

```text
<AICLS_DATA_ROOT>\runs\<run.name>-<ULID>\
├── run_manifest.json      # git SHA, kirli bayrağı, config hash, veri parmak izleri, GPU, citable
├── env.json               # tam paket listesi
├── params.json            # koşu kaydı; çözümlenmiş yapılandırma `params` altında
├── metrics.jsonl          # adım adım ölçümler (split: train | valid)
├── artifacts.jsonl        # kaydedilen artefaktlar
├── fit_report.json        # koşu özeti (adım, kayıp, en iyi, cihaz, wiring_check, citable)
└── checkpoints\
    ├── last.json          # işaretçi
    ├── best.json          # işaretçi (bkz. §12 "en iyi seçilmiyor")
    ├── step-00004000\     # 9 dosya: model.safetensors, optimizer.pt, scheduler.pt, rng_state.pt,
    │                      #          config.yaml, run_manifest.json, vocab.json, metrics.json, checkpoint.json
    └── interrupt\         # Ctrl+C ile yazılan kontrol noktası; asla budanmaz
```

### 6.2 Tek iş kuralı

32 GB VRAM iki eğitim işini taşımaz. Gerçek veri koşularında **her iki kapı da** `training/queue.lock` kilidini varsayılan olarak alır (`aicls train` 2026-09-09'a kadar almıyordu — o tarihe kadar koruma diğer komut kullanılarak atlanabiliyordu); `nvidia-smi` başka bir sürecin GPU belleğini tuttuğunu gösterirse de reddeder. İkinci bir iş başlatmayın; sıraya alın.

!!! warning "Ani ölümden sonra kilit takılı kalır — konteynerde otomatik devralma çalışmaz"

    Kuyruk, sahibi **aynı makinedeyse** ölü PID'i devralır. Ama eğitim konteyner içinde koşar
    (ADR-0020) ve konteynerin hostname'i her seferinde rastgele bir kimliktir, dolayısıyla her
    kilit "başka makineden" gelmiş sayılır ve devralma **hiç tetiklenemez**. Yeniden başlatma,
    `docker kill` veya OOM sonrası kalan kilit, sonraki her koşuyu bloke eder — üstelik hata
    mesajı var olmayan bir işin bitmesini beklemenizi söyler. 2026-09-09'da Windows Update
    makineyi eğitim ortasında yeniden başlattı ve tam bunu üretti (sahip: `20684ee4158c`, pid 1).

    Çıkış yolu **açık ve kasıtlıdır**, otomatik değil:

    ```powershell
    Get-Content training\queue.lock        # sahibi görün
    nvidia-smi                             # GPU'da hesap süreci VAR MI? varsa DURUN
    # yoksa:
    & $py training\scripts\train.py ... --force-queue
    # veya
    & $py -m aicls.cli.main train ... --force-queue
    ```

    `--force-queue`, GPU'da hesap süreci varken **reddeder** — bayrağın kendisi tek GPU'da iki
    eğitim durumunu üretemez — ve kimi devirdiğini `ml.queue.taken_over` olarak loglar. Otomatik
    süre aşımı bilerek eklenmedi: kuyruğun önlediği arıza sessiz ve geri dönüşsüzdür, kuyruğun
    ürettiği arıza ise gürültülü ve güvenlidir; ikincisini birincisinin ihtimaline değişmek kötü
    bir takastır.

---

## 7. Koşuyu izlemek

```powershell
# Canlı metrikler (son 5 doğrulama satırı)
Get-Content <run_dir>\metrics.jsonl -Wait -Tail 20

# Yalnızca doğrulama satırları, okunur biçimde
& $py -c @'
import json,sys; sys.path.insert(0,'src')
from aicls.ml.tracking import read_metrics
for r in read_metrics(r'<run_dir>'):
    if r['split']=='valid': print(r['step'], r['metrics'])
'@

# GPU
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv -l 5
```

Neye bakılır: `loss` düşüyor mu; `cer` (doğrulama) düşüyor mu; `grad_norm` patlamıyor mu; `lr` beklenen çizelgeyi izliyor mu; GPU kullanımı sürekli %90+ mı (değilse dataloader darboğazı).

---

## 8. Kesinti ve devam

Ctrl+C, `<run_dir>\checkpoints\interrupt` altına bir kontrol noktası yazar, izleyici koşusunu `killed` durumuyla kapatır ve `KeyboardInterrupt`'ı yeniden fırlatır. Beklenmedik bir istisna koşuyu `failed` durumuyla kapatır.

```powershell
& $py training\scripts\train.py `
    --config training\configs\ocr\offline_resnet34_tfm.yaml `
    --resume <run_dir>\checkpoints\interrupt `
    --run-dir <run_dir> `
    --max-steps 4000 `
    --i-have-approval
```

Devam ettirme model/optimizer/scheduler/adım sayacı ve RNG durumunu (python, numpy, torch, cuda) geri yükler.

2026-09-09'dan beri **`aicls train` de `--resume` alır**; artık devam etmek için CLI'dan betiğe geçmek gerekmiyor. İki kapı `--override` ve `--set`'in ikisini birden kabul eder.

!!! danger "`--max-steps` bir bütçeyi yalnızca **düşürebilir**"

    `Trainer` argüman, `train.max_steps` ve `train.epochs` arasından **en küçüğünü** alır. Yani
    yukarıdaki `--max-steps 4000`, adım 4000'den devam ederken **sıfır adım** koşturur; daha da
    kötüsü, yapılandırma 2000'de sınırlıysa `--max-steps 4000` yine 2000 verir. Bütçeyi
    yükseltmek isterseniz yapılandırmadan yükseltin: `--set train.max_steps=<yeni>`.

    Bu bilinçlidir: bir koşunun adım bütçesi onayın atıf yaptığı şeyin parçasıdır (ADR-0012), ve
    bütçeyi yükseltebilen bir bayrak, onaylananın ötesinde eğitim yapmanın yolu olurdu.

**Doğrulanmıştır (2026-09-09).** `runs/ocr_r34_tfm_full2` adım 18.000 kontrol noktasından devam
edildi: RNG geri yüklendi, koşu 215.520 adımlık bütçesine kaldığı yerden devam etti. Atlanan
batch sayısı `consumed % batches_per_epoch` olduğu için **en fazla bir epoch**tur (bu
yapılandırmada 7.184 batch, ölçülen ~15 dk), hangi adımdan devam edilirse edilsin.

---

## 9. Değerlendirme

Değerlendirme **kapısızdır** (yereldeki kontrol noktası + yereldeki veri).

!!! danger "`--use-git-status` olmadan rapor **terfi ettirilemez**"

    `git_info` git'i kendisi koşturmaz: ağacın durumu çağırandan gelmezse "bilinmiyor" sayılır ve
    koşu kirli işaretlenir, dolayısıyla `citable: false` olur. §11'deki `report check` ise
    `citable == true` arar — yani bu bayrak olmadan §9'dan §11'e giden yol çıkmaz sokaktır.
    Varsayılanın muhafazakâr olması bilinçlidir (kimsenin doğrulamadığı bir "temiz ağaç" iddiası,
    tekrarlanamayan bir sayının rapora girme yoludur); bayrak o doğrulamayı yapar.

    Ağaç gerçekten temiz olmalıdır. Kirliyken bayrağı vermek raporu yine `citable: false` yapar —
    ki doğrusu budur.

```powershell
# OCR — resmi test bölmesi
& $py training\scripts\evaluate.py --component ocr `
    --model-dir <run_dir>\checkpoints\step-00004000 `
    --shards D:\aicls-data\mathwriting\shards --split test --device cuda `
    --use-git-status

# ASR — konuşmacı-ayrık test bölmesi
& $py training\scripts\evaluate.py --component asr `
    --model-dir <ckpt> `
    --manifest D:\aicls-data\manifests\audio_commands.jsonl --split test --device cuda `
    --use-git-status
```

Çıktı `reports/metrics/<ad>_<gitsha7>.json`: metrikler, %95 güven aralığı, kırılımlar, en kötü 100 örnek, `dataset_fingerprint`, `model_version`, `model_sha256`, `config_hash`, `git_sha`, `dirty`, `citable`.

**Doğrulanmıştır (2026-09-09).** Bu yol gerçek bir kontrol noktasıyla uçtan uca koşturuldu (`step-00048000`, MathWriting test bölmesi, RTX 5090) ve `citable: true` bir rapor üretti. İki arıza o koşuda bulundu ve düzeltildi: `TorchRecognizer.load()` eğiticinin yazdığı **hiçbir** kontrol noktasını okuyamıyordu (`model.max_len: null` alanın varsayılanını eziyordu), ve bu betik `git status` bilgisini hiç geçirmediği için hiçbir raporu alıntılanabilir olamıyordu.

Ayrıca: doğrulama bölmesinden seçilen bir kontrol noktasının test bölmesinde daha kötü çıkması **beklenir** — 11 ölçüm içinden en iyiyi seçmek, o kümeye özgü şansı da seçmektir. Ayrılmış test kümesinin varlık sebebi budur; rapora giren sayı ondan gelmelidir.

### 9.1 Raporlanacak kontrol noktasını dürüst seçmek

Eğitim döngüsünün "en iyi"si, `eval.max_n_val` yüzünden doğrulama kümesinin bir **alt kümesinde**
ölçülmüş bir sayıya dayanır (sevk edilen OCR yapılandırmasında 15.674 örneğin 2.000'i). O sayı
gürültülüdür: 2026-09-09 koşusunda ardışık ölçümler 0,0926 ile 0,1271 arasında zıpladı ve
komşu ölçümler arasındaki fark çoğu zaman gürültüden büyük değildi. `checkpoint.keep_best: 1`
olduğu için yeni bir "en iyi" geldiğinde eskisi **silinir** — yani şansla seçilmiş bir kontrol
noktasıyla kalma ihtimali vardır ve sonradan düzeltilemez, çünkü alternatifler yok olmuştur.

Uzun bir koşuda adayları saklayın. Bir değerlendirme yalnızca ağırlıkları, yapılandırmayı ve
sözlüğü ister — `optimizer.pt`/`scheduler.pt`/`rng_state.pt` gerekmez, ki bunlar 324 MB'ın
211'idir. Kontrol noktası başına 113 MB'a adayları elde tutabilirsiniz.

Sonra üç adım:

```powershell
# 1) Adayları TAM doğrulama kümesinde yeniden sırala (--max-n vermeyin)
& $py training\scripts\evaluate.py --component ocr --split valid `
    --shards D:\aicls-data\mathwriting\shards --model-dir <aday> --device cuda `
    --no-write --json --use-git-status

# 2) Kazananı test bölmesinde ölç — rapora giren sayı budur
& $py training\scripts\evaluate.py --component ocr --split test `
    --shards D:\aicls-data\mathwriting\shards --model-dir <kazanan> --device cuda `
    --use-git-status

# 3) §10 ve §11: kaydet, etkinleştir, terfi ettir
```

Seçim doğrulamada, rapor testte. Aynı kümede hem seçip hem raporlamak, seçimin kendi şansını
raporlamaktır.

`--stub` (veya `--model-dir` vermemek) stub modelle koşar ve raporu `stub: true` işaretler. **Stub sayısı kanıt değildir**; `evaluate.py` örneklerin etiketini modele göstermeden önce siler (`blind_ink`), çünkü `StubRecognizer` etiketi görürse CER 0,0000 üretir ve bu bir sonuç gibi görünür.

---

## 10. Kontrol noktasını model kaydına almak

```powershell
# 1) Kontrol noktasını AICLS_MODELS_ROOT altına kopyalayın
Copy-Item -Recurse <run_dir>\checkpoints\step-00004000 "$env:AICLS_MODELS_ROOT\ocr-0.1.0"

# 2) models/registry.yaml'a bir girdi ekleyin (PR ile)
#    id / kind / path / checkpoint / sha256 / config / metrics_file / gate_refs

# 3) İçe aktarın — sha256 varsayılan olarak DOĞRULANIR
& $py -m aicls.cli.main models import models\registry.yaml

# 4) Listeleyin ve terfi ettirin
& $py -m aicls.cli.main models list
& $py -m aicls.cli.main models activate --component ocr --version 0.1.0
```

`sha256` değerini kontrol noktasının `checkpoint.json` dosyasından alın. `gate_refs` sütununa bu modeli mümkün kılan defter satırlarını yazın (`"DATASET_DOWNLOAD:2027-01-15"`, `"TRAINING:2027-02-01"`), böylece bir hakem "bu ağırlık hangi onayla üretildi" sorusuna tek bakışta yanıt bulur.

API yalnızca `model_versions` içindeki `is_active` satırını yükler ve bunun için `AICLS_MODELS_ENABLED=true` gerekir.

---

## 11. Sayıyı rapora terfi ettirmek

```powershell
& $py -m aicls.cli.main report check                                        # zinciri doğrular
& $py -m aicls.cli.main report promote reports\metrics\ocr_test_<sha>.json  # latest.json güncellenir
& $py -m aicls.cli.main report render                                       # docs/raporlama/sonuclar.md
& $py tools\check_docs_map.py
```

`report check` şunları arar: `git_sha` `HEAD`'in atası mı, `dirty == false` ve `citable == true` mi, `dataset_fingerprint` manifestlerle uyuşuyor mu, `normalizer_versions` güncel mi. Biri tutmazsa çıkış kodu 1 ve terfi yapılmaz. Ayrıntı: [degerlendirme.md §4.1](degerlendirme.md#41-koşudan-rapora-bir-eğitim-koşusu-nasıl-alıntılanabilir-sayıya-dönüşür).

Terfi bir **inceleme kararıdır**: `latest.json`'ı MkDocs makroları okur, yani terfi ettirdiğiniz an sayı belgelere girer.

---

## 12. Sorun giderme

| Belirti | Sebep | Çözüm |
|---|---|---|
| `GATE_BLOCKED: 'TRAINING' kapısı kapalı; eksik koşullar: ...` (çıkış 3) | Üç koşuldan biri eksik; mesaj hangisi olduğunu adıyla yazar | Eksik olanı sağlayın. `env` → `$env:AICLS_GATE_TRAINING = "approved"`; `ledger` → defterde `approved` satırı (PR); `cli_flag` → `--i-have-approval`. CI ortamında (`CI`/`GITHUB_ACTIONS`) kapı üçü de sağlansa **her zaman** reddeder |
| `DirtyWorkingTreeError` / koşu hemen duruyor | Çalışma ağacı kirli, `train.allow_dirty: false` | Değişiklikleri commit'leyin. Zorunluysa `--allow-dirty`, ama koşu `citable: false` işaretlenir ve **rapora giremez** |
| `torch.OutOfMemoryError` / CUDA OOM | Batch × dizi uzunluğu 32 GB'a sığmıyor | `data.batch_size` yarıya + `train.grad_accum` iki katına (etkin batch sabit); `data.max_label_len` düşürün; `WhisperFineTuner(gradient_checkpointing=True)`; son çare `train.amp: bf16` (zaten açıksa `data.image_size` küçültün). 8-bit optimizer (bitsandbytes) Windows'ta kullanılmaz. Değişiklik sonrası **profili yeniden koşturun** |
| GPU kullanımı dalgalı, `dataloader_wait_pct` > %20 | Veri yükleyici açlığı | `data.num_workers` artırın (6→8→12); parçaları NVMe'ye taşıyın; `prepare_mathwriting.py --render` ile rasterleri önceden yazın; `AICLS_DATA_ROOT` OneDrive altında olmasın |
| `TrainingQueueBusyError` (`TRAINING_BUSY`, "GPU meşgul") | Başka bir eğitim işi kilidi tutuyor veya `nvidia-smi` GPU'da başka süreç görüyor | `Get-Content training\queue.lock` ile sahibi okuyun. **`host` alanı bir konteyner kimliğiyse** (`20684ee4158c` gibi) otomatik devralma çalışmaz — aşağıya bakın. Aynı makinedeki ölü bir PID otomatik devralınır. Kesin bilgi `details["reason"] ∈ {training_queue_busy, gpu_busy}` içindedir |
| `training/queue.lock` `git status` çıktısında görünüyor | Kilit dosyası git-ignore edilmiş olmalı | `.gitignore` içindeki `training/queue.lock` satırının silinmediğini doğrulayın. İzlenen bir kilit dosyası koşuyu `citable=false` yapar (ADR-0007). `training/queue.txt` yalnızca belgedir; hiçbir kod ona yazmaz, dolayısıyla kaybolmaz |
| `CUDA error: no kernel image is available for execution on the device` | sm_120 uyumsuzluğu: CPU tekerleği ya da sm_120 içermeyen bir CUDA yapısı kurulu | `torch.version.cuda` ve `torch.cuda.get_device_capability()` yazdırın. `(12, 0)` gelmiyorsa `install_ml.ps1` cu128 indeksiyle **yeniden** koşturulmalıdır; `pip cache purge` sonrası tekrarlayın. `pip install torch` (indekssiz) **yanlış tekerleği** getirir |
| `torch.cuda.is_available() == False` ama `nvidia-smi` çalışıyor | CPU derlemesi kurulu (bugünkü durum) | Beklenen. `CUDA_INSTALL` kapısı açılmadan GPU yoktur; `run_manifest.json` yine de `gpu_name`/`driver` yazar çünkü onları `nvidia-smi`'den okur |
| `ModuleNotFoundError: accelerate` (ASR eğitiminde) | `Seq2SeqTrainingArguments` `accelerate>=0.26` şart koşar; kapılı `ml` extra'sında | `CUDA_INSTALL` kapısı; `install_ml.ps1` accelerate + peft kurar |
| `ModuleNotFoundError: peft` (`lora.enabled: true` iken) | Aynı | Aynı. LoRA yolu bugüne kadar hiç uygulanmadı |
| `weights_dir_missing` / `hub_id_rejected` | `model.base_dir` yok ya da hub kimliği yazılmış | `WEIGHTS_DOWNLOAD` kapısı ve **yerel dizin**. Kod hiçbir koşulda hub kimliğini kabul etmez |
| `fit_report.json`'da `best_checkpoint: null` | Yapılandırma `val_cer` diyor, eğitici `cer` yazıyor | Bilinen kusur ([§11.6](egitim-altyapisi.md#116-kontrol-noktası-düzeni-5e-göre-sapmalar)). Son kontrol noktası doğrudur; "en iyi" seçimi çalışmıyor. Düzeltilene kadar en iyi adımı `metrics.jsonl`'dan elle bulun |
| `report check` çıkış 1: "non-ancestor sha" | Rapor, `HEAD`'in atası olmayan bir commit'te üretilmiş | Doğru commit'e dönün ya da değerlendirmeyi yeniden koşturun. Terfi zorlanmaz |
| `metrics.jsonl` ile `run_manifest.json` `run_id` üzerinden birleşmiyor | İki ayrı ULID üretiliyor | Bilinen kusur ([§11.7](egitim-altyapisi.md#117-deney-takibi-6ya-göre-sapmalar)). Birleştirme anahtarı **koşu dizinidir** |
| Değerlendirmede `cer ≈ 0,97`, `exprate ≈ 0,02` | Stub sabit müfredat tablosundan tahmin ediyor; koşucu etiketi körleştirdiği için stub artık kopya çekemiyor | Beklenen. `--model-dir` verin; stub sayısı asla kanıt değildir |
| `latest.json`'da `ocr.cer: null` | Stub arkalı koşum terfi ederken sayı yayımlamaz (`measurement: "wiring_check"`) | Beklenen ve istenen. Stub'ın aldığı değer `wiring_check.metrics` altındadır; gerçek sayı için dört kapı da açılmalıdır |
| Değerlendirmede `cer = 0.0000` | Etiket modele sızıyor: körleştirme atlanmış ya da örnek tipi `blind_sample`'ın kopyalayamadığı bir şey | **Kırmızı bayrak.** `breakdown.blinded` alanına ve `label_leak_control`'e bakın; sayıyı hiçbir yere yazmayın |
| `mypy`/`ruff` torch tiplerinde şikâyet | `ocr/asr/data/ml` standart kipte | Beklenen; çekirdek paketler strict |

---

## 13. Kapıyı kapatmak (iptal) ve koşu sonrası hijyen

Onay **süresiz değildir**: kapsamı biten bir onay iptal edilir.

1. Deftere **yeni bir satır** ekleyin (eskisini silmeyin): `| DATASET_DOWNLOAD | revoked | 2027-03-01 | csahin | indirme tamamlandı, kapsam kapandı |`. En son satır geçerlidir.
2. Ortam değişkenini düşürün: `Remove-Item Env:\AICLS_GATE_DATASET_DOWNLOAD`; `.env` içinde kaldıysa silin.
3. `AICLS_DATA_ROOT\gate_invocations.log` dosyasını gözden geçirin: kaç kez, kim, hangi git SHA ile geçti. Bu dosya onayın **kullanıldığının** kaydıdır ve rapor ekine girer.

Koşu sonrası: `git status` temiz mi (kilit `training/queue.lock` git-ignore edilir; `training/queue.txt`'e hiçbir kod yazmaz), koşu dizini yedeklendi mi (`scripts/backup_db.ps1` `models/` ve veri kökünü kapsar), kontrol noktası kayda alındı mı, rapor terfi ettirildi mi.

---

## 14. Kontrol listesi

```text
[ ] git status temiz
[ ] poe env-check yeşil, AICLS_DATA_ROOT OneDrive dışında ve NVMe'de
[ ] GPU erişilebilir: setup-gpu-env.sh --verify yeşil, get_device_capability() == (12, 0)
    (ya da CUDA_INSTALL yolu: defter satırı + env + -IHaveApproval)
[ ] Sentetik GPU duman koşusu geçti (peak_vram_mb null değil)  -- 2026-09-08: geçti
[ ] DATASET_DOWNLOAD: --dry-run planı okundu; lisans kısıtı (CC BY-NC-SA 4.0) deftere yazıldı
[ ] Önce --excerpt (1,5 MB) indirildi ve tüm hat onunla uçtan uca koşturuldu
[ ] Tam arşiv indirildi, data verify mathwriting yeşil
[ ] prepare_mathwriting --archive --render --manifest koştu; parmak izi ve source_root var
[ ] build_tokenizer yalnızca train bölmesinden koştu
[ ] WEIGHTS_DOWNLOAD (ASR ise): yerel dizin var, model.base_dir onu gösteriyor
[ ] TRAINING: 200 adımlık profil koştu; steps_per_s / peak_vram_mb / dataloader_wait_pct /
    seconds_per_val_sample okundu ve projection.projected true
[ ] Bütçe profilden türetildi ve yapılandırmaya YAZILDI
[ ] Gerçek koşu başladı; run_manifest.json citable: true
[ ] Değerlendirme koştu; rapor stub: false ve citable: true
[ ] Kontrol noktası models/registry.yaml'a girdi, sha256 doğrulandı, activate edildi
[ ] report check yeşil, report promote yapıldı, check_docs_map temiz
[ ] Kapsamı biten kapılar deftere revoked satırıyla kapatıldı
```
