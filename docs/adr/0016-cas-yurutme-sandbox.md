# ADR-0016: CAS yürütme — ısıtılmış spawn işçi havuzu, zaman aşımında işçiyi öldürüp yeniden başlatma, ölçülmüş tabana göre bellek sınırı

- **Durum:** kabul edildi
- **Tarih:** 2026-09-07 (bellek sınırı bölümü 2026-09-08'de değiştirildi — aşağıdaki ek)
- **Karar vericiler:** M. M. Turhan, Ç. Şahin
- **İlgili İP:** İP-4 (SymPy yavaşlığı risk planı), İP-2 (< 3 s)

## Bağlam ve problem

`sympy.integrate`/`simplify` bazı girdilerde saniyeler-dakikalar sürebilir. Bir Python iş parçacığı çalışan SymPy hesabını durduramaz; iş parçacığı zaman aşımı çağırana döner ama çekirdek yanmaya devam eder, sınıf yükünde uvicorn işçisi zombi hesaplarla dolar. İstek başına yeni süreç Windows'ta `spawn` ile 1–2 s SymPy import maliyeti taşır (bütçenin yarısı). `resource.setrlimit` Windows'ta yoktur. Hakemler bu üç noktayı ayrı ayrı işaret etti.

## Değerlendirilen seçenekler

1. **İş parçacığı zaman aşımı** — basit; hesabı durduramaz, CPU sızıntısı.
2. **İstek başına süreç** — durdurabilir; spawn maliyeti bütçeyi yer.
3. **Isıtılmış `multiprocessing` havuzu (`spawn` bağlamı, `AICLS_SANDBOX_WORKERS=2`), işçiler sympy + dilbilgisini önceden import eder; `apply_async(...).get(timeout)`; zaman aşımında o işçi `kill()`, eşzamansız yeniden başlatma, `CAS_TIMEOUT` yanıtı; Windows'ta Job Object (`ctypes`, `JOBOBJECT_EXTENDED_LIMIT_INFORMATION`, 1 GB) / Linux'ta `RLIMIT_AS` en-iyi-çaba; girdi ≤ 2048 karakter; yeniden başlatma sayısı metrik** — durdurulabilir, ısınmış, ölçülür.

## Karar

Seçenek 3. `aicls.mathengine.sandbox.PoolExecutor` FastAPI lifespan'de başlar/durur; komut LaTeX dizesi + tipli komut olarak işçiye gider (pickle edilebilir küçük yük), sonuç `Result`; işçi başına istek sayısı sınırı (500) sonrası döngüsel yeniden başlatma (bellek büyümesine karşı); `InProcessExecutor` yalnızca testler ve CLI için. `restarts` sayacı `GET /health/ready` ve gecikme raporunda; `aicls eval latency` yeniden başlatma maliyetini dahil eder. Bellek sınırının en-iyi-çaba olduğu belgelenir; asıl koruma zaman aşımı + girdi sınırı + işçi döngüsüdür. **(Sınırın mutlak 1 GB olan tanımı 2026-09-08'de değişti; aşağıdaki "Ek" bölümü bağlayıcıdır.)** Sonuç önbelleği (kanonik LaTeX + komut anahtarı, LRU 4096) İP-4 risk planının "önbellekleme" maddesidir.

## Sonuçlar

- Olumlu: 2,5 s'de kesin durdurma; ısınmış işçi ile CAS aşaması ortanca problemlerde < 150 ms; havuz toparlanması test edilir.
- Olumsuz: süreçler arası seri hale getirme; `Result.plot_png` baytları havuzdan geçer (≤ 300 kB); Windows Job Object kodu `ctypes` ile bakım ister; havuz boyutu CPU sayısına göre ayarlanır.

## Öneri metniyle uyum

İP-4 risk planı: "SymPy ... yavaş kalması ... önbellekleme (caching) mekanizmaları ... Bir ifadenin SymPy tarafından çözülemediği durumlarda ... 'Bu problem tipi şu an için desteklenmemektedir.' gibi bilgilendirici bir mesaj" → `UnsupportedProblem` ve zaman aşımında aynı Türkçe mesaj ailesi; "sunucu kaynakları (CPU, RAM) artırılacaktır" → havuz boyutu yapılandırma anahtarı.

## Ek (2026-09-08): mutlak 1 GB sınırı yanlış araçtı — sınır artık ölçülmüş tabana görelidir

Bu ADR'nin ilk hâli işçi başına **mutlak** 1 GB'lık bir tavan koyuyordu: Linux'ta `RLIMIT_AS`, Windows'ta Job Object `ProcessMemoryLimit`. Depo ilk kez bir Linux konteynerinde (`codex-research-cu130`, 8 CPU) koşturulduğunda bu tavanın **taban değerin altında** olduğu ölçüldü ve karar değiştirildi.

### Ölçüm (2026-09-08, konteyner içinde, numpy 2.3.4 + OpenBLAS 0.3.30 pthread yapısı)

| Adım | `VmSize` | `VmRSS` |
|---|---|---|
| yorumlayıcı açılışı | 12 MB | 8 MB |
| `import numpy` | **1152 MB** | 25 MB |
| `+ import sympy` | 1189 MB | 61 MB |
| `+ Lark dilbilgisi` | 1229 MB | 95 MB |
| `+ Matplotlib (Agg)` | **1329 MB** | 119 MB |

Fark tek bir eşlemeden geliyor: `libopenblasp-r0.3.30.so` import anında **CPU başına 128 MB** anonim adres alanı ayırıyor (8 × 128 MB = 1032 MB; hiçbiri yerleşik değil). Sandbox işçisi NumPy'yi `aicls.mathengine.plotting` üzerinden zorunlu olarak import eder, `spawn` ile çocuk süreç `aicls.mathengine` paketini yeniden import ettiği için de bu ayırma sınır kurulmadan **önce** olur.

Sonuçlar ölçüldü, tahmin edilmedi:

- `RLIMIT_AS = 1 GiB`: ısınma `MemoryError` ile ölür, `grammar_ready=False` döner ve **her istek `INTERNAL` yanıtlar**. Sınır NumPy'den önce kurulduğunda ise OpenBLAS başlatıcısı hata bile vermeden **sonsuza kadar döner**; ebeveyn durumu ancak `WORKER_START_TIMEOUT_S` (90 s) sonunda fark eder. Yani sınır, korumak için var olduğu servisi kendisi durduruyordu.
- `RLIMIT_DATA` da çözüm değil: aynı ısınmanın `VmData` değeri 1190 MB; 256/512/1024 MiB'de ısınma başarısız, 2048 MiB'de başarılı. Araç değil, **sabit sayı** yanlıştı.
- `RLIMIT_AS = 2 GiB` ve `4 GiB`: ısınma başarılı, 3 GiB'lik kaçak ayırma `MemoryError` ile durduruldu.
- Sayı çekirdek sayısıyla büyür: tedarik listesindeki 16 çekirdekli iş istasyonunda taban ≈ 2 GB olur, yani 2 GiB'lik bir sabit de yarın yetmezdi.

### Karar

Sınır **ısınmadan sonra** ve **o anki kullanımın üstüne baş boşluk** olarak kurulur: `install_memory_cap(headroom)` = `process_memory_bytes() + headroom`, `headroom` varsayılanı 1 GB (`DEFAULT_MEMORY_HEADROOM_BYTES`; eski ad `DEFAULT_MEMORY_LIMIT_BYTES` korunur, sayı değişmedi). Ölçülen büyüklük her platformda o platformun sınırladığı büyüklüktür: POSIX'te `/proc/self/statm` (adres alanı, `RLIMIT_AS` bunu sınırlar), Windows'ta `PROCESS_MEMORY_COUNTERS_EX.PrivateUsage` (commit, Job Object bunu sınırlar). Taban ölçülemiyorsa sınır **kurulmaz** ve `WorkerProbe.memory_limit_applied=False` ile bu söylenir — tahmini bir tabandan hesaplanan sınır tam olarak yukarıdaki arızadır.

Sırayı tersine çevirmek güvenliği zayıflatmaz: ısınma yalnızca kendi modüllerimizi import eder, işçiye giden yük doğrulanmış bir `CasCommand`'dir, kod değildir. Sınırsız pencere kullanıcı girdisinden önce kapanır.

POSIX tarafında yalnızca **soft** sınır düşürülür (hard sınır ebeveynin bıraktığı yerde kalır), böylece `docker run --ulimit as=...` gibi dıştaki bir kum havuzu son sözü söyler ve sınır süreç içinde geri alınabilir.

### Bunu ararken bulunan ikinci kusur: Windows'ta sınır hiç kurulmuyordu

`AssignProcessToJobObject` her çağrıda `ERROR_INVALID_HANDLE (6)` ile başarısız oluyordu, çünkü `ctypes` prototip verilmemiş fonksiyonlarda dönüş tipini `c_int` sayar ve `GetCurrentProcess()`'in 64-bit sözde tanıtıcısı budanıyordu. ADR sınırı "en-iyi-çaba" dediği ve fonksiyon yalnızca `False` döndürdüğü için bu sessizce kaldı. Prototipler bildirildi (`_declare`); ölçüm 2026-09-08: aynı üç çağrı artık başarılı ve `test_the_memory_cap_stops_a_runaway_allocation` Windows'ta da kaçak ayırmayı durduruyor.

Yani bu ek öncesinde bellek üst sınırı **iki platformda da** gerçek bir koruma değildi: Windows'ta hiç kurulmuyordu, Linux'ta işçiyi öldürüyordu.

### İşletme notu (koruma değil, kaynak notu)

1 GB baş boşluk, tabanın üstünedir; Linux'ta taban çekirdek başına 128 MB OpenBLAS rezervasyonunu içerir. Bu rezervasyon istenirse dağıtım ortamında `OPENBLAS_NUM_THREADS=1` ile 128 MB'a indirilebilir (ölçüldü: taban 1329 MB → 377 MB) — CAS işçisi 400 noktalık bir grafik dışında doğrusal cebir yapmadığı için bu ayrıca CPU aşırı-abonesini de kaldırır. Kod bunu kendi ayarlamaz: `spawn` çocuğu ortam değişkenini kuramadan önce NumPy'yi import eder.

## Doğrulama

`test_execute_timeout_kills_worker_and_pool_recovers`, `test_pool_executor_prewarms_sympy_import`, `test_execute_timeout_returns_cas_timeout_within_budget`, `test_cache_hit_on_repeated_command_under_5ms`, `test_input_over_2048_chars_rejected`, `test_latency_eval_includes_sandbox_respawn_cost`.

Bellek sınırı için (2026-09-08): `test_install_memory_cap_is_relative_to_the_measured_baseline`, `test_install_memory_cap_declines_when_the_baseline_is_unknown`, `test_process_memory_bytes_reports_this_process`, `test_a_warm_worker_carries_a_memory_cap_above_its_own_baseline`, `test_the_memory_cap_stops_a_runaway_allocation` (asıl kanıt: sınır kurulduğunu bildirmeyen bir makinede kendini atlar), `test_rlimit_is_really_installed_on_posix`, `test_a_stricter_outer_limit_is_left_alone`.

## İlgili

[mimari/moduller.md](../mimari/moduller.md#aiclsmathengine--sahibi-mmt--kapsam--95-), ADR-0014, deneme `2026-10-sandbox-spawn-cost/`.
