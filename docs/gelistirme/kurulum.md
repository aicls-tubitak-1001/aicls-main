# Geliştirici Kurulumu (Windows)

Hedef makine: Windows 11 Pro for Workstations, Python 3.11.9 (sistem), Node 24 / npm 11, git 2.52; `make` ve `uv` yok. Sanal ortam **OneDrive dışında**. Karar: [ADR-0011](../adr/0011-windows-onedrive-gelistirme-ortami.md), [ADR-0012](../adr/0012-onay-kapilari.md).

## 1. İlk 10 dakika

```powershell
# 1) Depo (OneDrive altında; uzak depo gelince C:\src\ai-cls'e taşınması önerilir)
cd "C:\Users\cagri\OneDrive\Masaüstü\fırat\PROJE\ABDULLAH_HOCA_1001\ai-cls"

# 2) Sanal ortam + bağımlılıklar + pre-commit + veri kökü
pwsh scripts/bootstrap.ps1          # C:\Users\cagri\.venvs\ai-cls, pip-sync requirements/base.txt dev.txt, pip install -e ., pre-commit install, D:\aicls-data (yoksa C:\aicls-data)
C:\Users\cagri\.venvs\ai-cls\Scripts\Activate.ps1

# 3) Ortam dosyası
Copy-Item .env.example .env         # AICLS_JWT_SECRET üret: python -c "import secrets;print(secrets.token_hex(32))"

# 4) Doğrulama
poe env-check                       # python/torch/cuda, OneDrive yolu uyarısı, disk, JWT sırrı
poe db-upgrade; poe db-seed         # SQLite dev DB + demo kullanıcılar + mini kitap
poe test-fast                       # birim testler
poe dev                             # http://127.0.0.1:8000/docs (OpenAPI)

# 5) Ön uç (ayrı terminal)
pwsh tools/onedrive_setup.ps1       # node_modules junction, PYTHONPYCACHEPREFIX, git ayarları
npm --prefix web ci; poe web        # http://localhost:5173 (proxy /api, /ws → 8000)

# 6) Belgeler
poe docs                            # http://127.0.0.1:8001
```

`bootstrap.ps1` mevcut sistem paketlerini (`sympy`, `torch` CPU, `transformers`, `numpy`, `pydantic`) `--system-site-packages` ile görür; yalnızca saf Python geliştirme bağımlılıkları kurar. CUDA torch, torchvision, peft **kurulmaz** (kapı 3).

## 2. Ortam değişkenleri (`.env`, `pydantic-settings`, önek `AICLS_`)

| Anahtar | Varsayılan (dev) | Açıklama |
|---|---|---|
| `AICLS_ENV` | `dev` | `dev` / `test` / `prod` |
| `AICLS_PROFILE` | `full` | `full` / `collect` |
| `AICLS_DATA_ROOT` | `D:\aicls-data` | veri kökü, OneDrive dışı (zorunlu) |
| `AICLS_MODELS_ROOT` | `%AICLS_DATA_ROOT%\models` | kontrol noktaları ve ön eğitimli ağırlıklar |
| `AICLS_DB_URL` | `sqlite+pysqlite:///D:/aicls-data/aicls-dev.db` | prod: `mysql+pymysql://aicls_app:***@127.0.0.1/aicls?charset=utf8mb4` |
| `AICLS_JWT_SECRET` | yok (zorunlu, ≥ 32 bayt hex) | `check_env.py` varsayılan/kısa ise hata |
| `AICLS_CORS_ORIGINS` | `["http://localhost:5173"]` | JSON listesi |
| `AICLS_MODELS_ENABLED` | `false` | `true` → `model_versions.is_active` modelleri yüklenir; yoksa stub |
| `AICLS_LLM_FALLBACK` | `false` | NLU niyet önerici (yalnızca niyet kimliği) |
| `AICLS_LLM_ENDPOINT` / `AICLS_LLM_API_KEY` | boş | yedek açıksa |
| `AICLS_CONTRIB_ENABLED` | `false` | toplama sekmesi |
| `AICLS_ETHICS_APPROVAL_REF` | boş | etik kurul karar no; boşsa toplama 503 |
| `AICLS_COMPUTE_TIMEOUT_S` | `2.5` | sandbox zaman aşımı |
| `AICLS_SANDBOX_WORKERS` | `2` | ısıtılmış SymPy işçisi |
| `AICLS_RATE_LIMIT_ENABLED` | `true` | testte kapatılır |
| `AICLS_LOG_LEVEL` | `INFO` | |
| `AICLS_GATE_DATASET_DOWNLOAD` | boş | `approved` (kapı 1) |
| `AICLS_GATE_WEIGHTS_DOWNLOAD` | boş | `approved` (kapı 2) |
| `AICLS_GATE_CUDA_INSTALL` | boş | `approved` (kapı 3) |
| `AICLS_GATE_TRAINING` | boş | `approved` (kapı 4) |
| `AICLS_TEST_MYSQL_URL` | boş | `mysql` işaretli testler |
| `AICLS_TEST_GPU` | boş | `1` → `gpu` işaretli testler |
| `PYTHONPYCACHEPREFIX` | `C:\Users\cagri\.cache\ai-cls\pycache` | OneDrive dışı bytecode |

`.env` git'e girmez; `.env.example` her anahtarı açıklamalı içerir. Üretimde sırlar Windows servis hesabının ortamında / docker `env_file` (600) içindedir; bkz. [guvenlik-gizlilik.md](../guvenlik-gizlilik.md#3-sırlar).

## 3. Onay kapıları

Dört eylem yalnızca **üç koşul birlikte** sağlandığında çalışır; biri eksikse betik/CLI çıkış kodu 3 ile durur ve eksik koşulları Türkçe olarak adıyla yazar. Karar: [ADR-0012](../adr/0012-onay-kapilari.md). Kapılar açıldığı gün izlenecek uçtan uca yol: [ml/egitim-el-kitabi.md](../ml/egitim-el-kitabi.md).

!!! info "2026-09-08 durumu: dört kapının dördü de KAPALI"

    Bugüne kadar hiçbir veri kümesi indirilmedi, hiçbir ağırlık indirilmedi, CUDA kurulmadı ve
    hiçbir model **gerçek veriyle** eğitilmedi. [onay-defteri.md](onay-defteri.md) tablosu
    boştur. Windows sanal ortamındaki torch hâlâ CPU derlemesidir (`2.11.0+cpu`). RTX 5090'a
    2026-09-08'den beri **konteyner üzerinden** erişiliyor ([§7](#7-gpu-wsl2--nvidia-docker),
    [ADR-0020](../adr/0020-gpu-calisma-ortami-wsl-nvidia-docker.md)); o yol da hiçbir kapı
    açmaz ve yalnızca sentetik, kapısız koşular için kullanıldı.

### 3.1 Kapılar, maliyetleri ve karşılığında ne elde edilir

Boyutlar `src/aicls/data/catalog.yaml` içindeki katalog girdilerinden gelir ve her betiğin `--dry-run` çıktısında **o anki disk durumuyla birlikte** yeniden hesaplanır. Süreler ağ hızına bağlıdır ve ölçülmemiştir; alt sınır olarak okunmalıdır.

| Kapı | Ne açar | İndirme | Disk (denetlenen) | Süre (tahmin) | Geri alınabilir mi |
|---|---|---|---|---|---|
| `DATASET_DOWNLOAD` | MathWriting 2024.09 (**CC BY-NC-SA 4.0** — ticari kullanım yasak), ~230 bin gerçek + ~400 bin sentetik ifade; `--excerpt` ile önce **1,55 MiB** resmi örneklem | **2,88 GiB** arşiv (`mathwriting-2024.tgz`) | **2,88 GiB** (varsayılan: arşiv açılmaz, parçalama doğrudan `.tgz` üzerinden akar); `--extract` eklenirse **23,05 GiB** | 100 Mbit'te ~4 dk indirme + parçalama **saatler** | Evet — dizin silinir, `revoked` satırı yazılır |
| `WEIGHTS_DOWNLOAD` | `whisper-small` **0,45 GiB** (MIT), `whisper-large-v3-turbo` **1,51 GiB** (MIT), `wav2vec2-xls-r-300m` **1,17 GiB** (Apache-2.0) | seçilen ağırlık kadar | indirme boyutu + pay | dakikalar | Evet — dizin silinir |
| `CUDA_INSTALL` | cu128 torch + torchvision + torchaudio, ardından `peft` ve `accelerate` | ~3–4 GiB teker | ~8 GiB venv büyümesi | 10–30 dk | Kısmen — venv'i CPU tekerleğine geri döndürmek gerekir |
| `TRAINING` | gerçek veriyle eğitim **ve** profil koşusu | yok | koşu dizini başına ~1–20 GB (kontrol noktası boyutu × `keep_last`) | günler | Evet — koşu durur, ağırlıklar kalır |

**Parçalama ve koşu dizinleri için ayrıca yer ayırın.** MathWriting'in `--render` ile hazırlanmış parçaları ham arşivden büyüktür; `AICLS_DATA_ROOT` için toplam **≥ 100 GB** boş alan planlayın ve OneDrive dışında, NVMe üzerinde tutun.

**Arşivi açmak zorunda değilsiniz.** `prepare_mathwriting.py --archive` InkML üyelerini doğrudan `.tgz` akışından okur; ~630 bin küçük dosyalık ağaç hiç oluşmaz. Bu, `AICLS_DATA_ROOT` bir `drvfs` bağlama noktasındaysa (WSL2 → Windows `D:`) saatlik fark yaratır: sıralı okuma hızlıdır, yüz binlerce küçük dosya yazmak değildir.

| Kapı adı | Betik / komut |
|---|---|
| `DATASET_DOWNLOAD` | `training/scripts/download_mathwriting.py [--dry-run] --i-have-approval` |
| `WEIGHTS_DOWNLOAD` | `training/scripts/download_weights.py --model whisper-small [--dry-run] --i-have-approval` |
| `CUDA_INSTALL` | `scripts/install_ml.ps1 -IHaveApproval` (`-Approve` takma adı; `-DryRun` vardır) |
| `TRAINING` | `aicls train --i-have-approval`, `aicls profile --i-have-approval`, `training/scripts/train.py --i-have-approval` |

Kapısız serbest olanlar: saf Python geliştirme bağımlılıkları, sentetik veriyle CPU testleri, `aicls train --synthetic` ve `aicls profile --synthetic`, `prepare_mathwriting.py` / `prepare_audio_commands.py` / `build_tokenizer.py` (yereldeki veri üzerinde), `training/scripts/evaluate.py`, ve **her kapılı betiğin `--dry-run` kipi**.

### 3.2 Talebi hazırlamak (kapısız)

Her kapılı betikte `--dry-run` vardır ve **kapıyı hiç sormaz**: katalog girdisini (URL, sürüm, lisans, beklenen boyut, sha256), hedef dizini, o anki boş/gerekli diski ve kapının durumunu yazıp 0 ile çıkar. `--json` aynı planı onay PR'ına eklenebilecek bir belge olarak verir.

```powershell
$py = "C:\Users\cagri\.venvs\ai-cls\Scripts\python.exe"
& $py training\scripts\download_mathwriting.py --dry-run --json > plan-mathwriting.json
& $py training\scripts\download_weights.py --model whisper-small --dry-run
pwsh scripts\install_ml.ps1 -DryRun
& $py training\scripts\train.py --config training\configs\ocr\offline_resnet34_tfm.yaml --dry-run
```

`TRAINING` için süre tahmini **verilmez**: bütçe ancak kapı açıldıktan sonra koşturulan 200 adımlık `aicls profile` ölçümünden türetilir ([ml/egitim-altyapisi.md §8](../ml/egitim-altyapisi.md#8-gpu-planı-tek-rtx-5090-32-gb-windows)).

### 3.3 Bir kapıyı açmanın tam üç adımı

```powershell
# ADIM 1 — Defter satırı (git'te, PR ile, inceleme sonrası).
#   docs/gelistirme/onay-defteri.md tablosunun SONUNA tek satır:
#   | DATASET_DOWNLOAD | approved | 2027-01-15 | csahin | MathWriting 2024.09, D:\aicls-data, e-posta 2027-01-14 |
#   `kim` = satırı ekleyen kişi; `kapsam` = onayın kaynağı VE sınırları.
#   Biçim testle korunur (test_gate_ledger_format); en son satır geçerlidir.

# ADIM 2 — Ortam değişkeni, YALNIZCA bu oturum için (.env'e yazmayın: orada unutulur).
$env:AICLS_GATE_DATASET_DOWNLOAD = "approved"

# ADIM 3 — Komutu bayrakla çalıştırın.
& $py training\scripts\download_mathwriting.py --i-have-approval

# İş bitince değişkeni düşürün.
Remove-Item Env:\AICLS_GATE_DATASET_DOWNLOAD
```

Üçü de sağlansa bile CI ortamında (`CI` veya `GITHUB_ACTIONS` ayarlıysa) kapı **her zaman** reddeder. Her başarılı geçiş `AICLS_DATA_ROOT/gate_invocations.log` dosyasına bir JSON satırı olarak eklenir: kapı adı, zaman, kullanıcı, git SHA, ayar hash'i, defter satırının kendisi. Bu dosya onayın *kullanıldığının* kaydıdır ve rapor ekine girer.

**`--ledger` diye bir bayrak yoktur ve olmayacaktır.** Defter, üçlü kilidin denetlenebilir yarısıdır: git geçmişinde durur ve bir hakemin "bunu kim onayladı" sorusuna baktığı yerdir. Onu işaret eden bir komut satırı bayrağı, operatörün kendi onayını üretmesine izin verirdi. Betiklerin `main()` işlevi yalnızca testlerin geçtiği anahtar-kelimeli bir `ledger_path` alır.

### 3.4 Bir onayı iptal etmek (revoke)

Onay süresiz değildir; kapsamı biten onay kapatılır.

1. Deftere **yeni bir satır** ekleyin, eskisini **silmeyin**:
   `| DATASET_DOWNLOAD | revoked | 2027-03-01 | csahin | indirme tamamlandı, kapsam kapandı |`
   `require_gate` yalnızca **en son** satıra bakar, yani bu satır kapıyı anında kapatır.
2. Ortam değişkenini düşürün (`Remove-Item Env:\AICLS_GATE_...`); `.env` içine yazıldıysa oradan da silin.
3. `gate_invocations.log` dosyasını gözden geçirin: kapı kaç kez, kim tarafından, hangi git SHA ile kullanıldı? Beklenmedik bir satır varsa olay olarak ele alın.
4. Gerekiyorsa eylemi geri alın: indirilen dizini silin, `CUDA_INSTALL` için venv'i CPU tekerleğine döndürün. İptal, *yeni* eylemleri engeller; geçmişte üretilmiş ağırlıkları veya raporları silmez — onlar `gate_refs` alanı üzerinden hangi onaya dayandıklarını taşımaya devam eder.

Kapının kod tarafındaki ikinci yarısı statiktir: `tools/check_gates.py` kaynak ağacında hub kimliği (`from_pretrained("org/ad")`), `load_dataset("org/ad")`, `hf_hub_download`/`snapshot_download`, `torch.hub`, `weights=` (None dışı), `pretrained=True`, `urllib`/`requests` ve HF/PyTorch URL değişmezlerini arar. İzinli tek istisna `training/scripts/download_*.py` ve `scripts/install_ml.*`; gözden geçirilmiş tek satırlık istisna `# gate-ok: <gerekçe>` ile işaretlenir. Depodaki ağ kodu **yalnızca iki dosyada** ve her birinde tek bir işlevde (`download_archive`, `download_snapshot`) hapsedilmiştir.

### RTX 5090 notu (Blackwell, sm_120)

`install_ml.ps1` yalnızca `--index-url https://download.pytorch.org/whl/cu128` (veya daha yenisi) ile sabitlenmiş torch sürümünü kurar; kurulum sonunda `torch.cuda.get_device_capability() == (12, 0)` ve bf16 matmul duman testi geçmezse başarısız sayılır; sürücü (bugün 610.88) ve araç takımı sürümünü onay defterine yazar.

!!! success "2026-09-08: GPU, kapı 3 açılmadan **konteyner üzerinden** kullanılabilir"

    Bu betik hâlâ koşturulmadı ve kapı 3 kapalı. Bunun yerine, makinede hazır duran WSL2 +
    nvidia-docker ortamı üzerinden GPU'ya erişiliyor: [§7](#7-gpu-wsl2--nvidia-docker) ve
    [ADR-0020](../adr/0020-gpu-calisma-ortami-wsl-nvidia-docker.md). Windows sanal ortamındaki
    torch **değişmedi** (`2.11.0+cpu`) ve hiçbir teker indirilmedi. Veri kökü NTFS'te kalır,
    konteynerden `/mnt/d/aicls-data` olarak görülür.

## 4. Sunucu / VM kurulumu (özet)

- İş istasyonu (İP-1 sunucusu): `deploy/workstation/install_service.ps1` → NSSM ile `aicls serve --profile full` Windows servisi, MySQL 8 servisi (`deploy/mysql/init.sql` ile üç kullanıcı), Caddy (üniversite sertifikası, `deploy/Caddyfile`), `web/dist` statik; yedek görevi (`deploy/backup/backup.ps1`, Görev Zamanlayıcı 02:00).
- Toplama VM'i (Ubuntu): `scripts/bootstrap.sh` + `deploy/collect-vm/docker-compose.yml` (`api` profile=collect, `mysql`, `caddy` Let's Encrypt); `.env` 600; iş istasyonundan gece `scripts/sync_recordings.py`.
- İşletme belgeleri (`docs/raporlama/ip-01-altyapi.md` altında sunucu konfigürasyon raporu şablonu): kurulum adımları, portlar, sertifika yenileme, yedekleme/geri yükleme tatbikatı, olay müdahale.

## 5. Yedekleme

`scripts/backup_db.ps1` gece 02:00: `mysqldump --single-transaction` (veya SQLite `.backup`), `AICLS_DATA_ROOT/audio/raw` ve `models/` artımlı kopya, 7-zip AES-256, 30 gün rotasyon, hedef 10 TB HDD + üniversite içi ikinci kopya. Geri yükleme tatbikatı: üç ayda bir, sonuç `docs/raporlama/ip-01-altyapi.md` günlüğüne.

## 6. Sık sorunlar

| Belirti | Çözüm |
|---|---|
| `git status` yavaş, OneDrive kilitleri | `core.fsmonitor=true`; depo klasörünü OneDrive senkronundan çıkar; büyük rebase sırasında OneDrive'ı duraklat |
| `OSError: [WinError 206]` uzun yol | `git config core.longpaths true` + `LongPathsEnabled` kayıt anahtarı (`onedrive_setup.ps1` ayarlar) |
| SQLite "database is locked" | dev DB OneDrive altında olmamalı; `poe env-check` |
| `aicls: command not found` / `Scripts\aicls.exe` yok | paket venv'e kurulmamış; testler `pythonpath = ["src"]` ile çalışır. Geçici: `$env:PYTHONPATH="src"; python -m aicls.cli.main ...` — kalıcı: `pip install -e .` (aşağıdaki satırı okuyun) |
| `aicls.exe` var ama `ModuleNotFoundError: No module named 'aicls'` | **Türkçe karakterli depo yolu.** `pip install -e .`, `site-packages/__editable__.aicls-*.pth` dosyasını **UTF-8** yazar; CPython 3.11'in `site.addpackage` işlevi `.pth` dosyalarını **yerel kod sayfasıyla** (burada `cp1254`) çözer. `Masaüstü` / `fırat` bozulur, yol bulunamaz ve satır sessizce atlanır — bu yüzden bugüne kadar konsol komutu hiç çalışmadı. `PYTHONUTF8=1` **çare değildir** (`locale.getencoding()` UTF-8 kipinden etkilenmez). Kalıcı çözüm: depoyu ASCII bir yola taşımak (`C:\src\ai-cls`, §1'de zaten öneriliyor). Geçici çözüm: `.pth` dosyasını `cp1254` ile yeniden yazmak — **bir sonraki `pip install -e .` bunu sessizce geri alır** |
| `torch.cuda.is_available() == False` | Windows'ta beklenen (CPU derlemesi); GPU için [§7](#7-gpu-wsl2--nvidia-docker) |
| `ModuleNotFoundError: torchvision` | beklenen; `aicls.ocr.models` yalnızca `install_ml.ps1` sonrası |
| `mypy` torch tiplerinde şikâyet | ocr/asr/data/ml standart modda; çekirdek strict |

## 7. GPU: WSL2 + nvidia-docker

Makinedeki **RTX 5090** (Blackwell, sm_120, 32 GB, sürücü 610.88) Windows sanal ortamından
kullanılamaz: oradaki torch `2.11.0+cpu` derlemesidir. Windows'a CUDA torch kurmak **kapı 3**
(`CUDA_INSTALL`) işidir ve o kapı kapalıdır. GPU'ya bugün erişmenin desteklenen tek yolu,
makinede hazır duran WSL2 + nvidia-docker ortamıdır. Karar ve kabul edilen bedeller:
[ADR-0020](../adr/0020-gpu-calisma-ortami-wsl-nvidia-docker.md); işletim ayrıntısı
`scripts/gpu/README.md`.

!!! warning "Bu yol hiçbir kapıyı açmaz"

    Konteyner yolu hiçbir teker indirmez, Windows sanal ortamına hiçbir şey kurmaz ve hiçbir
    `AICLS_GATE_*` değişkeni ayarlamaz. Kapı değişkenleri konteynere **aktarılır**, yani
    onaysız bir gerçek-veri koşusu konteynerde de çıkış kodu 3 verir. Kapı 3'ün tanımıyla
    ilgili açık soru ADR-0020 "Sonuçlar" 4'te kayıtlıdır ve proje sahibinin kararını bekler.

### 7.1 Kurulum ve doğrulama

```powershell
wsl -e bash scripts/gpu/setup-gpu-env.sh          # hazırla + doğrula (idempotent)
wsl -e bash scripts/gpu/setup-gpu-env.sh --verify # yalnızca doğrula
wsl -e bash scripts/gpu/setup-gpu-env.sh --print  # docker komutlarını yaz, çalıştırma
```

Betik: docker + `nvidia` çalışma zamanı + imaj pinini denetler → `AICLS_DATA_ROOT` dizinlerini
açar → `/mnt/d/aicls-venv` sanal ortamını (`--system-site-packages`, ~141 MB) hazırlar →
`pip install --no-deps --no-build-isolation -e <depo>` ile **`aicls` konsol komutunu** üretir →
donanımı doğrular. 2026-09-08 doğrulaması: capability `(12, 0)`, torch `2.9.1+cu130`,
bf16 matmul bağıl hata `0.0029`, `aicls 0.1.0.dev0`.

### 7.2 Komut çalıştırmak

```powershell
.\scripts\gpu\aicls-gpu.ps1 --print train --config training\configs\ocr\smoke_synthetic.yaml
.\scripts\gpu\aicls-gpu.ps1 train --config training/configs/ocr/smoke_synthetic.yaml --synthetic --device cuda
```

```bash
scripts/gpu/aicls-gpu.sh train --config training/configs/ocr/smoke_synthetic.yaml --synthetic --device cuda
scripts/gpu/aicls-gpu.sh --raw python -m pytest -q tests/unit/ml
scripts/gpu/aicls-gpu.sh --shell

# gpu işaretli testlerin gerçek yeri burasıdır; Windows'ta AICLS_TEST_GPU=1 ayarlamayın
AICLS_TEST_GPU=1 scripts/gpu/aicls-gpu.sh --raw python -m pytest -q -m gpu
```

Koşucunun kendi seçenekleri (`--print`, `--raw`, `--shell`, `--repo`, `--data-root`,
`--models-root`, `--image`, `--gpus`, `--no-gpu`, `--venv`, `--name`) yalnızca `aicls`
komutundan **önce** okunur; ilk tanınmayan argümanda ayrıştırma durur, bu yüzden
`aicls train --dry-run` kendi bayrağını korur.

### 7.3 Bilinmesi gereken dört tuzak

1. **Depo `.git`'iyle birlikte bağlanır.** `RunManifest` gerçek `git` çalıştırır; commit'e
   bağlanamayan koşu `citable: false` olur ve rapora giremez. `.git`'i olmayan bir kopya
   reddedilir. İmajda `git` yoktur; ana makinenin `git`'i salt-okunur bağlanır.
2. **Kökler mutlak POSIX olmalıdır.** `D:\aicls-data` konteynerde **göreli** bir `PosixPath`'e
   dönüşür ve ağırlıklar sessizce yanlış yere yazılır. Koşucu böyle bir kökü çıkış kodu 2 ile
   reddeder; PowerShell sarmalayıcısı `$env:AICLS_DATA_ROOT` Windows biçimindeyse onu
   `/mnt/<harf>/...` biçimine kendisi çevirir.
3. **`web/node_modules`** bir Windows junction'ıdır; Linux git onu izlenmeyen bir sembolik bağ
   olarak görür ve koşu kirli sayılır. Koşucu bunu `scripts/gpu/container-excludes` ile
   Windows davranışına eşitler. Kalıcı çözüm, depo `.gitignore`'unda `node_modules/` satırının
   yanına eğik çizgisiz `node_modules` eklemektir.
4. **Windows sanal ortamında `aicls.exe` hâlâ yoktur.** `setup-gpu-env.sh` yalnızca konteyner
   sanal ortamına kurar. Windows'ta çağrı biçimi hâlâ `python -m aicls.cli.main ...`'dir
   (§1'deki `bootstrap.ps1` işi).
