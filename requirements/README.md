# requirements/ — hash'li kilit dosyaları (pip-tools)

Karar: [ADR-0011](../docs/adr/0011-windows-onedrive-gelistirme-ortami.md). Tek bağımlılık kaynağı `pyproject.toml`'dur; bu dizin **üretilmiş** kilit dosyalarını tutar. Kilit dosyaları henüz üretilmedi (ağ gerektirir); `bootstrap.ps1`/`bootstrap.sh` `base.txt` yoksa `pip install -e .[dev]` ile kurar.

## Akış

```text
pyproject.toml ──(pip-compile)──> requirements/base.txt   (proje bağımlılıkları)
                                  requirements/dev.txt    (+ dev extras)
                                  requirements/ml.txt     (+ ml extras, cu128 index — KAPILI)
```

Girdi dosyaları (`*.in`) pyproject'i referans alır:

```text
# base.in
-e .            # [project].dependencies
# dev.in
-c base.txt
-e .[dev]
# ml.in  (GATED — CUDA_INSTALL)
-c base.txt
--index-url https://download.pytorch.org/whl/cu128
-e .[ml]
```

Yenileme yalnızca `poe lock` ile (ayrı `chore(build)` PR'ı):

```powershell
pip-compile --generate-hashes --strip-extras -o requirements/base.txt requirements/base.in
pip-compile --generate-hashes --strip-extras -o requirements/dev.txt  requirements/dev.in
# ml.txt yalnızca onay defterinde CUDA_INSTALL satırı varken ve install_ml.ps1 içinden
```

Kurulum: `pip install --require-hashes -r requirements/base.txt -r requirements/dev.txt` + `pip install --no-deps -e .`. CI hash uyuşmazlığında kırılır; `pip-audit` gece koşar. `--system-site-packages` venv `pip freeze` çıktısını şişirir; `run_manifest` bunu not eder.

`poe lock` görevi kilit dosyaları ilk kez üretilirken `poe_tasks.toml`'a eklenecektir (şu an ağ gerektiren görev listede yoktur).
