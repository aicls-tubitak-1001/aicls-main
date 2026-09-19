# Changelog

Bu dosya [Keep a Changelog](https://keepachangelog.com/tr/1.1.0/) biçimindedir; sürümleme [SemVer](https://semver.org/lang/tr/) `0.MINOR.PATCH` (bkz. `docs/gelistirme/is-akisi.md` §7). Her PR `Unreleased` altına bir satır ekler; bölüm başlıkları: Added, Changed, Deprecated, Removed, Fixed, Security. Girdi biçimi: `- <paket/alan>: açıklama (#PR, İP-N)`.

## [Unreleased]

### Added

- scaffold: repository skeleton — `pyproject.toml` (ruff/mypy/pytest/coverage/import-linter), `poe_tasks.toml` (no gated tasks), pre-commit, CI/nightly/docs workflows, CODEOWNERS, PR/issue templates, MkDocs site config, approval-gate tooling (`tools/check_gates.py`, `scripts/install_ml.ps1` gated), environment checks (`scripts/check_env.py`), docs-map check, coverage tier gate, ADR/experiment generators, empty `aicls` subpackages, shared test fixtures. (İP-2)

- tests: `tests/integration/test_runtime_contracts.py` — 6. dalganın beş ajanının kesiştiği dört sınır: aygıt yerleşimi, akış hâlinde hazırlanan shard'ların eğitim döngüsüne bağlanması, API'nin ürettiği gerçek sandbox havuzu ve `latest.json`'ı okuyan her tüketicinin `null` metrikle başa çıkması. İki test `gpu` işaretlidir ve RTX 5090'da gerçekten koştu. (İP-3)

- gelistirme: `poe test-linux` / `scripts/run_linux_suite.py` — süiti Linux CUDA konteynerinde koşturan tek komut; docker argv'si ve imaj sabitlemesi `scripts/gpu/aicls-gpu.sh` içinde kalır (ADR-0020). GPU'lu runner gerektireceği için GitHub workflow'u eklenmedi; gerekçesi `docs/gelistirme/test-stratejisi.md` §8. (İP-2)

### Changed

- mathengine.sandbox: bellek üst sınırı artık **ölçülmüş tabana göre** kurulur (`install_memory_cap`, ısınmadan sonra, taban + 1 GB baş boşluk) ve `WorkerProbe` gerçekten kurulan tavanı `memory_limit_bytes` ile bildirir. `PoolExecutor`/`worker_loop` parametresi `memory_limit_bytes` → `memory_headroom_bytes`; `DEFAULT_MEMORY_LIMIT_BYTES` eski ad olarak korunur. Gerekçe ve ölçümler: ADR-0016 Ek (2026-09-08). (İP-4)

### Deprecated

### Removed

### Fixed

- ml.trainer: `Trainer` ağırlıkları artık `__init__` içinde seçilen aygıta taşır, `fit()` içinde değil. RTX 5090'da ölçüldü (2026-09-08): `fit()` çağrılmadan yapılan `evaluate()`/`beam_search()` `Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same` ile ölüyor, `resume()` ise iyileştiriciyi CPU parametreleri üzerine kuruyordu — `training/scripts/train.py --resume` tam olarak bunu yapar. CPU'da hiçbir belirtisi yoktu. (İP-3)
- api.app: `build_executor(settings)` sandbox havuzunu kendisine verilen ayarlarla kurar; `pool_from_settings()` süreç genelinde önbelleklenmiş `get_settings()`'i okuduğu için `create_app(Settings(sandbox_workers=1))` sessizce iki işçilik bir havuz başlatıyordu. (İP-2)
- cli.train: ASR yolu artık OCR yolunun her zaman aldığı `JsonlTracker`'ı alır — bir ASR koşu dizini yalnızca `run_manifest.json` + `env.json` içeriyordu, `metrics.jsonl`/`params.json` yoktu (ADR-0007 kaydı iki döngüden yalnızca birinde vardı). `--device` whisper yolunda uygulanamıyor; artık sessizce yutulmak yerine `cli.train.device_ignored` ile bildirilir. (İP-3)
- gitignore: `node_modules` (ayraçsız) eklendi. `web/node_modules` bir Windows junction'ıdır; Linux git onu sembolik bağ olarak görür ve `node_modules/` kalıbı eşleşmez, bu yüzden kapsayıcıdaki her koşu `dirty` → `citable=false` oluyordu. (İP-2)
- eval: bir taslak (stub) arka uç artık doğru cevabı göremez ve bir bağlantı denetimi koşusu metrik yerine `null` yayımlar (KNOWN_ISSUES P1, İP-3). (İP-3)
- mathengine.sandbox: mutlak 1 GB'lık `RLIMIT_AS` Linux'ta işçinin tabanının altındaydı (NumPy'nin OpenBLAS'ı import anında CPU başına 128 MB adres alanı ayırır); ısınma `MemoryError` alıyor, `PoolExecutor.start()` 90 s zaman aşımına düşüyor ve **her CAS isteği `INTERNAL` yanıtlıyordu**. `tests/unit/mathengine` konteynerde artık tam yeşil. (İP-4)
- tests/ml: `Path("D:/aicls-models")` Windows'ta mutlak, Linux'ta göreli bir yoldur; `test_config.py` ve `test_configs_validate.py` kökleri platforma göre kurar ve mutlaklığı ayrı bir testle sabitler. İddia zayıflatılmadı. (İP-3)
- tools/check_env: `is_under_onedrive` yol bileşenlerini her iki ayraçla böler, böylece POSIX yorumlayıcısına verilen bir Windows yolundaki OneDrive klasörünü de görür. (İP-2)

### Security

- api.security: JWT doğrulaması artık `CLOCK_SKEW_LEEWAY_S = 30` saniyelik bir tolerans kullanır. Geriye adım atan bir duvar saati, az önce üretilmiş bir belirteci "gelecekte üretilmiş" gösteriyor ve PyJWT >= 2.10 bunu reddediyordu; WSL2 kapsayıcısında ölçüldü (2026-09-08: `05:04:58.600Z` → `05:04:57.698Z`), sonuç taze belirteçte sahte 401'dir. NTP sanal makinelerde bunu yapar; okul pilotunda kullanıcılar rastgele oturumdan düşerdi. Bedeli açıkça yazılmıştır: süresi dolmuş bir belirteç de bu kadar süre kabul edilir kalır (900 s ömrün %3,3'ü). İmza doğrulaması etkilenmez. (İP-2)
- mathengine.sandbox: Windows Job Object bellek sınırı **hiç kurulmuyordu** — `ctypes` prototipi bildirilmediği için `GetCurrentProcess()`'in 64-bit sözde tanıtıcısı budanıyor ve `AssignProcessToJobObject` her seferinde `ERROR_INVALID_HANDLE` veriyordu. Sınır en-iyi-çaba sayıldığı ve fonksiyon yalnızca `False` döndürdüğü için sessiz kalmıştı. Prototipler bildirildi; `test_the_memory_cap_stops_a_runaway_allocation` kaçak ayırmanın iki platformda da durdurulduğunu kanıtlar (ADR-0016). (İP-4)

[Unreleased]: https://github.com/ai-cls/ai-cls/compare/main...HEAD
