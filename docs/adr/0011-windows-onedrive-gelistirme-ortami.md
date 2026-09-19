# ADR-0011: Windows/OneDrive geliştirme ortamı — OneDrive dışı venv ve veri, poethepoet, pip-tools, junction'lar

- **Durum:** kabul edildi
- **Tarih:** 2026-09-07
- **Karar vericiler:** Ç. Şahin, Ş. Güler
- **İlgili İP:** İP-1 (iş istasyonu), İP-2 (CI pratikleri)

## Bağlam ve problem

Depo OneDrive senkronlu klasördedir; makinede Python 3.11.9, Node 24, git 2.52 var; `make` ve `uv` yok; venv `C:\Users\cagri\.venvs\ai-cls` olarak zaten OneDrive dışında kuruludur. OneDrive; venv, `node_modules`, SQLite ve kontrol noktası dosyalarında kilit/bozulma ve senkron gürültüsü üretir; uzun yollar Windows sınırına takılır.

## Değerlendirilen seçenekler

1. **Makefile** — `make` yok, Windows'ta yabancı.
2. **just / nox / invoke** — ek ikili ya da farklı sözdizimi.
3. **poethepoet (`poe_tasks.toml`) + PowerShell bootstrap + pip-tools kilit dosyaları + junction'lar** — saf Python, CI ile aynı komutlar, kabuktan bağımsız.

## Karar

Seçenek 3. venv `C:\Users\cagri\.venvs\ai-cls` (`--system-site-packages`; sistemdeki sympy/torch-cpu/transformers kullanılır); `PYTHONPYCACHEPREFIX=C:\Users\cagri\.cache\ai-cls\pycache`; `AICLS_DATA_ROOT=D:\aicls-data` (yoksa `C:\aicls-data`), modeller/koşular/mlruns/SQLite orada; `web/node_modules` → `C:\Users\cagri\.cache\ai-cls\node_modules` junction (`tools/onedrive_setup.ps1`); git `core.autocrlf=false` + `.gitattributes eol=lf`, `core.longpaths=true`, `core.fsmonitor=true`, Windows `LongPathsEnabled`; pytest `basetemp=%TEMP%`; `scripts/check_env.py` OneDrive altındaki veri/model yolunu hata sayar; `attrib +S` gibi etkisiz yöntemler kullanılmaz; uzak depo kurulunca klon `C:\src\ai-cls`'e taşınır, OneDrive yalnızca belge yedeği olur. Görevler `poe_tasks.toml`; kapılı komutlar görev listesinde yoktur. Bağımlılıklar `requirements/{base,dev,ml}.in` → `pip-compile --generate-hashes` (`poe lock`), `ml.txt` kapılı. CI matrisi `windows-latest` + `ubuntu-latest`.

## Sonuçlar

- Olumlu: OneDrive'a bağlı kilit/bozulma riski sıfıra yakın; geliştirici ve CI aynı `poe` komutlarını çalıştırır; hash'li kilit dosyaları `run_manifest` ile tutarlı.
- Olumsuz: `--system-site-packages` `pip freeze`'i şişirir (manifest bunu not eder); junction'lar yeni klonda yeniden kurulmalı (`bootstrap.ps1` çağırır).

## Öneri metniyle uyum

İP-1 sunucu/iş istasyonu; İP-2 risk planı "sürekli entegrasyon (CI) pratikleri". Öneri metni geliştirme ortamı aracı adlandırmaz.

## Doğrulama

`poe env-check` (`test_check_env_warns_on_onedrive_path`), `test_poe_tasks_contain_no_gated_commands`, CI Windows koşusu.

## İlgili

[gelistirme/kurulum.md](../gelistirme/kurulum.md), [gelistirme/is-akisi.md §8–9](../gelistirme/is-akisi.md), ADR-0008.
