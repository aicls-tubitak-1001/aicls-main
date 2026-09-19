# ADR-0020: GPU çalışma ortamı: WSL2 + nvidia-docker; Windows'a CUDA torch **kurulmadı**

- **Durum:** kabul edildi
- **Tarih:** 2026-09-08
- **Karar vericiler:** Ç. Şahin (AI eş güdümü)
- **İlgili İP:** İP-1 (altyapı/GPU), İP-3 (OCR/ASR eğitimi)
- **Girdi:** ölçüm oturumu 2026-09-08; doğrulama çıktısı `scripts/gpu/README.md` §1

## Bağlam ve problem

Makinede **RTX 5090** var (Blackwell, sm_120, 32 GB, sürücü 610.88), ama Windows sanal
ortamındaki torch `2.11.0+cpu` derlemesidir ve kartı süremez. Windows'a CUDA torch kurmak
[ADR-0012](0012-onay-kapilari.md)'nin **3. kapısıdır** (`CUDA_INSTALL`) ve o kapı kapalıdır;
`scripts/install_ml.ps1` yazılmıştır ama hiç koşturulmamıştır.

Sonuç olarak, wave 5'in sonunda projedeki **her CUDA yolu yazılmış ama hiç çalışmamıştı**
(`DeviceSpec.autocast_context`, `GradScaler` dalları, `probe_device`'ın CUDA yarısı,
`peak_vram_mb`, `resume(map_location="cuda")`). Bunların ilk kez çalıştırılması gerekiyordu,
çünkü ölçülmemiş bir yol, ilk gerçek eğitim koşusunda patlayan bir yoldur.

Aynı makinede zaten kurulu ve çalışır durumda bulunanlar (2026-09-08 ölçümü):

- WSL2, dağıtım `Ubuntu-22.04`, `nvidia-smi` içeride çalışıyor;
- Docker `29.1.3`, `nvidia` çalışma zamanı kayıtlı;
- **`codex-research-cu130:latest` imajı** — 6,31 GB, 2026-04-30'da **yerel olarak** üretilmiş,
  *bizim üretmediğimiz* bir imaj: python `3.11.14`, torch `2.9.1+cu130`,
  torchvision `0.24.1+cu130`, torchaudio `2.9.1+cu130`, transformers `4.57.6`,
  accelerate `1.13.0`, numpy `2.3.4`, safetensors. `arch_list` sm_120 içeriyor.

Kısıtlar:

- **WSL ext4 diski %94 dolu** (62 GB boş). Oraya kalıcı bir şey yazılamaz, imaj üretilemez.
- `/mnt/d` (Windows D:) 3,0 TB boş; ama `drvfs` mount'udur: sıralı G/Ç iyidir, **yüz binlerce
  küçük dosya değildir** (bu, parçaların `npz` olarak tutulmasının nedenidir, ADR-0008).
- Bir koşunun rapora girebilmesi için `RunManifest` gerçek bir git SHA'sı ister (ADR-0007).

## Karar etkenleri

- Kapı 3 açılmadan GPU'ya dokunabilmek; hiçbir teker indirmemek.
- Kalıcı veriyi ve sanal ortamı dolu ext4 diskinden uzak, D:'de tutmak.
- Kapıların (ADR-0012) konteynerde **aynen** geçerli kalması; Docker bir kaçış yolu olmamalı.
- Koşunun kaynağının (git SHA, kirli bayrağı) korunması.

## Değerlendirilen seçenekler

1. **Windows'a cu128 torch (`install_ml.ps1`)** — artı: tek ortam, en yalın operasyon; eksi:
   kapı 3'ün açılmasını gerektirir, ~8 GB sanal ortam büyümesi, geri alması zahmetli. Bugün
   yapılamaz.
2. **WSL içine ikinci bir CUDA yığını (`pip install torch --index-url .../cu128`)** — eksi:
   ext4 diski %94 dolu; ~4 GB teker + ~8 GB kurulum yer bulamaz. Ayrıca kapı 3'ün ruhu.
3. **Var olan konteyner imajı + `/mnt/d` üzerinde ince bir sanal ortam** — artı: hiçbir indirme,
   hiçbir kurulum, GPU bugün kullanılabilir, venv 141 MB (`--system-site-packages`); eksi:
   imaj bizim değil, sürümleri seçemiyoruz, `git` içermiyor, digest'i yok.
4. **Bulut GPU** — eksi: bütçe, veri gizliliği (İP-9 etik onayı), ağ; öneri metninde yok.

## Karar

**Seçenek 3.** GPU'ya erişimin tek desteklenen yolu `scripts/gpu/` altındaki koşucudur:

```bash
scripts/gpu/setup-gpu-env.sh                    # hazırla + doğrula (idempotent)
scripts/gpu/aicls-gpu.sh <aicls komutu ...>     # WSL tarafı
.\scripts\gpu\aicls-gpu.ps1 <aicls komutu ...>  # Windows tarafı
```

Koşucunun kurduğu somut şeyler:

| Karar | Değer | Gerekçe |
|---|---|---|
| İmaj | `codex-research-cu130:latest` | Blackwell'i süren tek hazır yığın |
| İmaj pini | `sha256:a3359cc37244fe5cc7b1acb7eca12ccb63e44d9bda26b866cee510e747fd84d4` | Etiket taşınırsa koşucu çıkış 2 ile reddeder (`AICLS_GPU_ALLOW_IMAGE_DRIFT=1` tek seferlik istisna) |
| Sanal ortam | `/mnt/d/aicls-venv` (`--system-site-packages`) | ext4 dolu; torch imajdan gelir, venv 141 MB kalır |
| Depo bağlama | `<depo>:/workspace/ai-cls` (**`.git` dâhil**) | `RunManifest.capture()` gerçek `git` çağırır; commit'siz koşu `citable: false` |
| `git` | Ana makinenin `/usr/bin/git`'i salt-okunur bağlanır | **İmajda `git` yok**; ana makine ve imaj aynı Ubuntu 22.04 |
| Veri kökleri | `/mnt/d/aicls-data`, `/mnt/d/aicls-data/models` | Mutlak POSIX olmayan kök çıkış 2 ile reddedilir |
| Cihaz | `--gpus all` | |
| Paylaşımlı bellek | `--ipc=host`, `--ulimit memlock=-1`, `--ulimit stack=67108864` | 64 MB `/dev/shm` veri yükleyicileri aç bırakır (NGC uyarısı) |
| Kullanıcı | `--user $(id -u):$(id -g)` | `/mnt/d` altındaki dosyalar Windows'tan da kullanılabilir kalmalı |
| Kapılar | `AICLS_GATE_*` (4) + `CI` + `GITHUB_ACTIONS` aktarılır | Kapı Docker'da **zayıflamaz**; koşucu hiçbirini ayarlamaz |
| Git kirlilik paritesi | `core.excludesfile=scripts/gpu/container-excludes` | Windows junction'ı `web/node_modules` Linux git'te izlenmeyen sembolik bağdır ve her koşuyu kirli gösterirdi |
| Kuru koşu | `--print` | Tam docker komutunu yazar, çalıştırmaz |

Kurulum betiği ayrıca `pip install --no-deps --no-build-isolation -e /workspace/ai-cls`
uygulayarak **`aicls` konsol komutunu** üretir (el kitabının ilk komutu buydu ve kurulu değildi).

## Sonuçlar

### Olumlu

- GPU **bugün** kullanılabilir; kapı 3 kapalı kaldı, hiçbir teker indirilmedi, Windows sanal
  ortamı değişmedi.
- Daha önce hiç çalışmamış CUDA yolları ilk kez çalıştı (aşağıdaki "Doğrulama").
- Koşular git SHA'sı taşıyor; kapılar konteynerde de reddediyor.

### Olumsuz / kabul edilen bedeller

1. **Konteynerdeki torch 2.9.1, `pyproject.toml` ise `torch>=2.11,<3` istiyor.** Kurulum
   `--no-deps` ile yapıldığı için bu pini **hiçbir şey uygulamıyor**: ortam sessizce
   bildirilenden eski bir torch ile çalışıyor.
   **Öneri: pin gevşetilmelidir** (`ml` ve `dev` ekstralarında `torch>=2.7,<3`). Gerekçe:
   `>=2.11` işlevsel bir gereksinim değil, geliştirme makinesinde rastlantısal olarak kurulu
   olan CPU tekerinin sürümüdür; sm_120 desteği torch 2.7'den beri vardır ve 2.9.1+cu130 ile
   OCR eğitim döngüsünün tamamı bu kartta koştu (`tests/unit/ml` + `tests/unit/ocr`
   konteynerde **544 geçti / 10 kaldı**, kalanların hiçbiri sürüm kaynaklı değil — bkz.
   "Doğrulama"). Uygulanmayan ve gerçek ortamın ihlal ettiği bir pin, doğru olandan kötüdür.
   Bu değişiklik `pyproject.toml` sahibinindir; bu ADR yalnızca öneriyi kaydeder.
2. **İmaj bizim değil ve kayıt defteri digest'i yok.** `RepoDigests` boştur (yerel olarak
   üretilmiş), yani `image@sha256:...` biçiminde çekilebilir bir pin **mümkün değildir**.
   Bunun yerine yerel **image ID**'si pinlenmiştir
   (`sha256:a3359cc37244fe5cc7b1acb7eca12ccb63e44d9bda26b866cee510e747fd84d4`) ve koşucu
   uyuşmazlıkta reddeder. Bu pin **bu makineye özgüdür**: imaj silinirse yeniden üretilemez ve
   bu ortam kaybolur. Azaltım: iş istasyonu geldiğinde ya kendi `Dockerfile`'ımız (sürümleri
   biz seçeriz, `git` dâhil ederiz) ya da kapı 3 ile Windows'a gerçek kurulum.
3. **İmajda `git` yok.** Ana makinenin `git 2.34.1`'i salt-okunur bağlanıyor; bu, ana makine
   ile imajın **aynı Ubuntu 22.04** olmasına dayanan bir varsayımdır. Dağıtımlar ayrışırsa
   bağlama bozulur ve her koşu sessizce `citable: false` olur. Kendi imajımız `git` içermelidir.
4. **Kapı 3 ile gerilim, açıkça kaydedilir.** ADR-0012'nin 3. kapısı "CUDA/GPU PyTorch kurulumu"
   demektir. Burada **hiçbir kurulum yapılmadı**: teker indirilmedi, proje sanal ortamına CUDA
   torch girmedi, `install_ml.ps1` koşmadı. Ama kapının **korumak istediği etki** — GPU'nun
   kullanılabilir hâle gelmesi — gerçekleşti. Bu ADR bunu gizlemez ve kararı proje sahibine
   bırakır: (a) kapı 3'ün tanımı "GPU'yu kullanılabilir kılan her eylem" olarak genişletilir ve
   bu ortam için deftere bir satır yazılır; ya da (b) tanım "proje ortamına CUDA tekerleği
   kurmak" olarak daraltılır ve konteyner yolu kapısız kalır. Karar verilene kadar geçerli olan
   sınır şudur: koşucu hiçbir kapı değişkeni ayarlamaz, `--i-have-approval` eklemez ve gerçek
   veriyle eğitim konteynerde de çıkış kodu 3 ile reddedilir (testle sabitlendi).
5. **`drvfs` performansı.** `/mnt/d` sıralı G/Ç için yeterlidir, yüz binlerce küçük dosya için
   değildir. MathWriting hazırlığı `npz` parçaları üretmelidir (ADR-0008 ile uyumlu).
6. **Editable kurulum bağlama noktasına sabitlenir** (`/workspace/ai-cls`). `AICLS_GPU_WORKDIR`
   değiştirilirse `setup-gpu-env.sh` yeniden koşturulmalıdır; koşucu ayrıca
   `PYTHONPATH=/workspace/ai-cls/src` verdiği için bağlanan depo her hâlükârda kazanır.
7. **Windows sanal ortamında `aicls.exe` hâlâ yok.** Bu görev yalnızca konteyner tarafını
   kurdu. Windows tarafı `bootstrap.ps1`'in işidir (KNOWN_ISSUES "Orta öncelik" 1).

## Öneri metniyle uyum (zorunlu)

Öneri metni İP-1'de "gerekli donanım ve yazılım altyapısının kurulması"nı ve İP-3'te GPU
üzerinde model eğitimini öngörür; kullanılacak kütüphane olarak PyTorch'u adlandırır. Bir
konteyner çalışma zamanı öneri metninde adlandırılmamıştır, ama teknoloji seçimi değil
**kurulum biçimi** kararıdır: kullanılan yığın (Python + PyTorch + Hugging Face) öneri
metnindekinin aynısıdır. İP-1 risk planındaki "donanım/yazılım kurulum gecikmesi" maddesi,
alternatif kurulum yollarının denenmesini açıkça öngörür. Sapma yoktur; iş istasyonu teslim
alındığında ADR yeniden değerlendirilecektir (bkz. "Sonuçlar" 2).

## Doğrulama

Testler: `tests/unit/test_gpu_runner.py` —
`test_runner_mounts_the_repository_with_its_git`,
`test_runner_validates_that_roots_are_absolute_posix_paths`,
`test_runner_asks_for_the_gpu_host_ipc_and_raised_ulimits`,
`test_runner_runs_as_the_calling_user`,
`test_runner_forwards_every_gate_variable`,
`test_gpu_scripts_never_open_an_approval_gate`,
`test_runner_pins_the_image_and_the_pin_matches_the_adr`,
`test_runner_refuses_a_moved_image_tag`,
`test_gate_variables_reach_the_container`,
`test_real_data_training_inside_the_container_still_exits_3` (`gpu` işaretli).

Araç: `scripts/gpu/verify_gpu.py` (`setup-gpu-env.sh` içinden).

**2026-09-08'de ilk kez ölçülenler** (hepsi bağlantı denetimi, hiçbiri sözleşme metriği değil):

| Ne | Sonuç |
|---|---|
| `torch.cuda.get_device_capability()` | `(12, 0)` — Blackwell sm_120 |
| torch / CUDA | `2.9.1+cu130` / `13.0`, `NVIDIA GeForce RTX 5090`, 31,8 GiB |
| bf16 matmul (256×256), fp32 referansına göre | bağıl hata `0.0029` |
| `aicls train --config .../smoke_synthetic.yaml --synthetic --device cuda` | 2 adım, `train_loss 5.0337`, hata yok |
| Aynı koşu `--override train.amp=bf16` ile | `dtype bfloat16`, `autocast true` — `DeviceSpec.autocast_context` **ilk kez çalıştı** |
| `aicls profile ... --steps 20 --device cuda --override train.amp=bf16` | `45,198 adım/s`, `180,8 örnek/s`, **`peak_vram_mb 66,97`** (artık `null` değil), `dataloader_wait_pct %8,54` |
| Onaysız gerçek-veri `aicls train` (konteyner içinde) | çıkış **3**, `missing: ['cli_flag', 'env', 'ledger']` |
| Konteynerde `tests/unit/ml` + `tests/unit/ocr` | 544 geçti, 10 kaldı (6'sı testlerin CPU/`D:/` varsayımı, 2'si gerçek kusur — bkz. aşağıda) |

Konteynerdeki 10 başarısızlığın ikisi **gerçek kusurdur ve `aicls.ml` sahibine aittir**:
`Trainer.evaluate()` girdileri cihaza taşır ama **modeli taşımaz** (yalnızca `fit()` taşır), bu
yüzden CUDA'lı bir makinede `fit()` olmadan çağrılan her değerlendirme
`Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)` ile patlar; ve
`Trainer.resume()` cihaz farkındalığı olmadan durum yükler
(`found at least two devices, cuda:0 and cpu`). Kalan altısı testlerin varsayımıdır:
`Path("D:/aicls-data")` Linux'ta **görelidir** ve `peak_vram_mb is None` / `torch.equal` gibi
iddialar CPU varsayar.

## İlgili

[ADR-0006](0006-ml-altyapisi-pytorch-hf-yapilandirma.md),
[ADR-0007](0007-deney-takibi-jsonl-mlflow-manifest.md),
[ADR-0008](0008-veri-yonetimi-manifest-onedrive-disi.md),
[ADR-0011](0011-windows-onedrive-gelistirme-ortami.md),
[ADR-0012](0012-onay-kapilari.md),
[gelistirme/kurulum.md §7](../gelistirme/kurulum.md),
[ml/egitim-el-kitabi.md §2](../ml/egitim-el-kitabi.md),
`scripts/gpu/README.md`.
