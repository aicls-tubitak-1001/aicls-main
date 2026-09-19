# GPU çalışma ortamı (WSL2 + nvidia-docker)

Bu dizin, makinedeki **RTX 5090**'ı (Blackwell, sm_120, 32 GB, sürücü 610.88) AI-CLS için
kullanılabilir kılan tek desteklenen giriş noktasıdır. Karar ve gerekçeler:
[ADR-0020](../../docs/adr/0020-gpu-calisma-ortami-wsl-nvidia-docker.md).

> **Neden konteyner?** Windows sanal ortamındaki torch `2.11.0+cpu` derlemesidir ve 5090'ı
> süremez. Windows'a CUDA torch kurmak **onay kapısı 3'e** (`CUDA_INSTALL`, [ADR-0012]
> (../../docs/adr/0012-onay-kapilari.md)) bağlıdır ve o kapı kapalıdır. Bunun yerine makinede
> hazır duran, **bizim üretmediğimiz** `codex-research-cu130` imajındaki
> `torch 2.9.1+cu130` kullanılır. Hiçbir teker indirilmez, Windows sanal ortamına hiçbir şey
> kurulmaz.

## Dosyalar

| Dosya | Ne yapar |
|---|---|
| `aicls-gpu.sh` | Koşucu: verilen `aicls` komutunu konteynere iletir (WSL içinde çalışır) |
| `aicls-gpu.ps1` | Windows sarmalayıcı: PowerShell'den `.\aicls-gpu.ps1 train ...` |
| `setup-gpu-env.sh` | Fikirsiz (idempotent) kurulum + donanım doğrulaması |
| `verify_gpu.py` | Doğrulama adımları (yalnızca konteyner içinde çalışır) |
| `container-excludes` | Linux git'in sembolik bağ olarak gördüğü Windows junction'ları |

## 1. Kurulum ve doğrulama (bir kez, sonra istendiği kadar)

```powershell
# PowerShell'den
wsl -e bash scripts/gpu/setup-gpu-env.sh
```

```bash
# Ya da doğrudan WSL içinden
scripts/gpu/setup-gpu-env.sh            # hazırla + doğrula
scripts/gpu/setup-gpu-env.sh --verify   # yalnızca doğrula, hiçbir şeyi değiştirme
scripts/gpu/setup-gpu-env.sh --print    # docker komutlarını yaz, çalıştırma
```

Betik sırayla: docker + `nvidia` çalışma zamanı + imaj pinini denetler, `AICLS_DATA_ROOT`
altındaki dizinleri açar, `/mnt/d/aicls-venv` sanal ortamını (yoksa) `--system-site-packages`
ile kurar, `pip install --no-deps --no-build-isolation -e <depo>` ile **`aicls` konsol
komutunu** üretir ve `verify_gpu.py` ile donanımı doğrular.

2026-09-08'de ölçülen doğrulama çıktısı:

```text
GPU environment verification (ADR-0020)
  [PASS] interpreter              3.11.14 at /mnt/d/aicls-venv/bin/python
  [PASS] torch                    2.9.1+cu130 (cuda 13.0)
  [PASS] device                   NVIDIA GeForce RTX 5090, capability (12, 0), 31.8 GiB
  [PASS] bf16_matmul              256x256 bf16 matmul executed, relative error 0.0029 (< 0.05)
  [PASS] aicls_import             0.1.0.dev0 from /workspace/ai-cls/src/aicls
  [PASS] console_script           aicls 0.1.0.dev0
  [PASS] git                      HEAD 60aad42, working tree DIRTY
  [PASS] aicls_data_root          /mnt/d/aicls-data (absolute=True, exists=True)
  [PASS] aicls_models_root        /mnt/d/aicls-data/models (absolute=True, exists=True)
  [PASS] nvidia_smi               NVIDIA GeForce RTX 5090, 610.88, 32607 MiB
  ---- OK
```

## 2. Komut çalıştırmak

```powershell
# Windows tarafı — her argüman olduğu gibi iletilir
.\scripts\gpu\aicls-gpu.ps1 --print train --config training\configs\ocr\smoke_synthetic.yaml
.\scripts\gpu\aicls-gpu.ps1 train --config training/configs/ocr/smoke_synthetic.yaml --synthetic --device cuda
```

```bash
# WSL tarafı
scripts/gpu/aicls-gpu.sh train --config training/configs/ocr/smoke_synthetic.yaml --synthetic --device cuda
scripts/gpu/aicls-gpu.sh --raw python -m pytest -q tests/unit/ml     # aicls yerine başka bir komut
scripts/gpu/aicls-gpu.sh --shell                                     # konteynerde kabuk

# gpu işaretli testlerin GERÇEK yeri burasıdır (2026-09-08: 3 geçti, 1 atlandı)
AICLS_TEST_GPU=1 scripts/gpu/aicls-gpu.sh --raw python -m pytest -q -m gpu
```

> **`AICLS_TEST_GPU=1`'i Windows'ta ayarlamayın.** Windows sanal ortamındaki torch CPU
> derlemesidir; `gpu` işaretli testler orada `AssertionError: Torch not compiled with CUDA
> enabled` ile **kalır**, atlanmaz. `tests/conftest.py` yalnızca ortam değişkenine bakar,
> `torch.cuda.is_available()` değerine bakmaz.

Koşucunun **kendi** seçenekleri yalnızca `aicls` komutundan **önce** okunur; ilk tanınmayan
argümanda ayrıştırma durur. Bu yüzden `aicls train --dry-run` kendi bayrağını korur:

```bash
scripts/gpu/aicls-gpu.sh --print train --dry-run   # --print koşucunun, --dry-run aicls'in
```

| Koşucu seçeneği | Anlamı |
|---|---|
| `--print`, `--dry-run` | Tam docker komutunu yazar, çalıştırmaz |
| `--raw` | Argümanları `aicls` öneki olmadan çalıştırır |
| `--shell` | Konteynerde `bash` açar |
| `--repo YOL` | Bağlanacak depo (varsayılan: bu betiğin içinde bulunduğu depo) |
| `--data-root YOL` / `--models-root YOL` | Konteyner içi **mutlak POSIX** yollar |
| `--image REF` | İmaj etiketi |
| `--gpus SPEC` / `--no-gpu` | `docker --gpus` değeri / cihaz istememe |
| `--venv YOL` | Çalışma zamanı sanal ortamı |
| `--name AD` | Konteyner adı |

## 3. Koşucunun kurduğu altı şey (ve neden)

1. **`.git` ile birlikte deponun kendisi bağlanır.** `RunManifest.capture()` gerçek `git` alt
   süreçleri çalıştırır; commit'e bağlanamayan koşu `citable: false` işaretlenir ve TÜBİTAK
   raporuna giremez ([ADR-0007](../../docs/adr/0007-deney-takibi-jsonl-mlflow-manifest.md)).
   Depodan kopyalanmış, `.git`'i atılmış bir dizin **bağlanmaz** — koşucu reddeder.
   İmajda `git` **yoktur**; ana makinenin (aynı Ubuntu 22.04) `git`'i salt-okunur bağlanır.
2. **POSIX kökler.** `AICLS_DATA_ROOT` / `AICLS_MODELS_ROOT` mutlak POSIX yol olmak
   zorundadır. `D:\aicls-data` konteynerde **göreli** bir `PosixPath`'e dönüşür ve ağırlıklar
   sessizce `/workspace/ai-cls/D:/aicls-data/...` altına yazılır. Koşucu ters bölü, sürücü
   harfi ve göreli yolu çıkış kodu 2 ile reddeder; PowerShell sarmalayıcısı Windows biçimindeki
   bir kökü otomatik olarak `/mnt/<harf>/...` biçimine çevirir.
3. **`--gpus all`, `--ipc=host` ve yükseltilmiş ulimit'ler.** NGC imajlarının uyardığı gibi
   varsayılan 64 MB `/dev/shm` PyTorch veri yükleyicilerini aç bırakır;
   `--ulimit memlock=-1 --ulimit stack=67108864` bunun eşlik eden ayarıdır.
4. **UID/GID.** Konteyner çağıran kullanıcı olarak koşar (`--user $(id -u):$(id -g)`), böylece
   `/mnt/d` altına yazılan dosyalar hem Windows'tan hem WSL'den kullanılabilir kalır.
5. **Kapı ortam değişkenleri aktarılır.** `AICLS_GATE_*` (dördü) ve `CI` / `GITHUB_ACTIONS`
   konteynere geçirilir. **Kapı, Docker'da zayıflamaz**: onaysız bir gerçek-veri koşusu
   konteyner içinde de çıkış kodu 3 verir. Koşucu hiçbir kapı değişkeni **ayarlamaz** ve
   `--i-have-approval` **eklemez**; onay defteri (`docs/gelistirme/onay-defteri.md`) bağlı
   depodan okunur.
6. **`--print`.** Çalıştırılacak tam docker komutunu yazar ve çıkar. Tek yan etkisi, yazdırılan
   bağlama listesinin gerçek koşuyla aynı olması için veri kökü dizinini açmasıdır.

## 4. `aicls` konsol komutu

`setup-gpu-env.sh` konteyner içindeki `/mnt/d/aicls-venv` ortamına `pip install --no-deps
--no-build-isolation -e /workspace/ai-cls` uygular; sonrasında konteynerde doğrudan `aicls`
çalışır. Kurulum **bağlama noktasına** (`/workspace/ai-cls`) sabitlenir; koşucu ayrıca
`PYTHONPATH=/workspace/ai-cls/src` verdiği için bağlanan depo her koşulda kazanır.

**Windows tarafında durum ayrıdır ve bu betikler oraya dokunmaz.** 2026-09-08'de Windows sanal
ortamına da `pip install --no-deps --no-build-isolation --no-index -e .` uygulandı, ama üretilen
`aicls.exe` **çalışmadı**: `pip`, `site-packages/__editable__.aicls-*.pth` dosyasını UTF-8 yazar;
CPython 3.11'in `site.addpackage` işlevi `.pth` dosyalarını yerel kod sayfasıyla (`cp1254`)
çözer ve `Masaüstü` / `fırat` bozulduğu için satır sessizce atlanır (`PYTHONUTF8=1` çare
değildir: `locale.getencoding()` UTF-8 kipinden etkilenmez). Dosya `cp1254` ile yeniden yazılarak
komut çalışır hâle getirildi, ama bu **bir sonraki `pip install -e .`'de sessizce bozulur**.
Kalıcı çözüm deponun ASCII bir yola taşınmasıdır (`C:\src\ai-cls`, `kurulum.md` §1'de zaten
öneriliyor). Ayrıntı: `docs/gelistirme/kurulum.md` §6.

## 5. Sık sorunlar

| Belirti | Sebep / çözüm |
|---|---|
| `image ... is not present locally` | İmaj silinmiş. Bu imaj bizim üretimimiz değildir ve çekilecek bir kayıt defteri yoktur; ADR-0020'nin "Sonuçlar" bölümünü okuyun |
| `... but the pin says sha256:...` | Etiket başka bir imaja taşınmış. Pini bilerek güncelleyin (betik + ADR) ya da tek seferlik `AICLS_GPU_ALLOW_IMAGE_DRIFT=1` |
| `AICLS_DATA_ROOT='D:\...' contains a backslash` | Beklenen. POSIX yol verin ya da PowerShell sarmalayıcısını kullanın |
| `has no .git` | Deponun kopyasını değil kendisini bağlayın |
| `could not select device driver "" with capabilities: [[gpu]]` | `nvidia-container-toolkit` yok ya da WSL yeniden başlatılmalı |
| Koşu `dirty: true` çıkıyor | Çalışma ağacında commit edilmemiş değişiklik var. `web/node_modules` junction'ı `container-excludes` ile hariç tutulur; başka bir şey görünüyorsa gerçekten kirlidir |
| `torch.cuda.is_available()` False | Konteyner `--gpus` olmadan başlatılmış (`--no-gpu`) ya da sürücü görünmüyor |

## 6. Bu ortamın ölçtüğü sayılar kanıt değildir

Bu dizindeki her doğrulama ve her sentetik koşu bir **bağlantı denetimidir** (wiring check).
Rastgele başlatılmış bir ağın üretilmiş veri üzerindeki kaybı ya da CER'i, sözleşmedeki
CER < %10 / WER < %10 ölçütleri hakkında hiçbir şey söylemez. Gerçek sayı ancak
`DATASET_DOWNLOAD` ve `TRAINING` kapıları açıldıktan sonra üretilebilir; yol
[ml/egitim-el-kitabi.md](../../docs/ml/egitim-el-kitabi.md) içindedir.
