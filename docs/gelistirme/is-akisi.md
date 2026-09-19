# Geliştirme İş Akışı

Kararlar: [ADR-0002](../adr/0002-monorepo-katmanli-paketler.md), [ADR-0010](../adr/0010-test-stratejisi.md), [ADR-0011](../adr/0011-windows-onedrive-gelistirme-ortami.md). İlke: her PR küçük, testli, belgeli; kalite kapıları yerelde (`poe ci`) ve uzakta (GitHub Actions) aynı komutlarla çalışır.

## 1. Dallanma

- Gövde tabanlı: `main` korumalı (uzak depo kurulunca PR + yeşil CI + 1 inceleme; kendi PR'ını birleştirmek yasak). Uzak depo henüz yok: o zamana kadar yerel dallar + `poe ci` + PR şablonu checklist'i ile öz-inceleme; ilk uzak depo GitHub'da özel organizasyon deposu (üniversite GitLab aynası gerekirse ADR ile).
- Kısa ömürlü dallar: `feat/<ip>-<slug>` (`feat/ip4-latex-limit-direction`), `fix/<slug>`, `docs/<slug>`, `exp/<yyyy-mm>-<slug>` (deneme alanı), `chore/<slug>`. Uzun ömürlü `develop` yok.
- Birleştirme: rebase sonrası squash; başlık Conventional Commit biçiminde.
- Sürüm etiketleri `vX.Y.Z`; rapor anlık görüntüleri `report-YYYY-MM` (`reports/metrics/`, `datasets/`, `models/registry.yaml` bu etikette dondurulur).

## 2. Commit kuralı

Conventional Commits, İngilizce: `type(scope): summary`. `type ∈ {feat, fix, docs, test, refactor, perf, chore, ci, build, exp}`; `scope` = paket adı (`core, db, latex2sympy, mathengine, nlu, ocr, asr, book, collect, data, eval, ml, api, cli`) veya `web, docs, training, datasets, deploy`. Gövde "neden"i anlatır. Altbilgi: `IP: 4`, `ADR: 0004`, `Refs: #123`. `conventional-pre-commit` kancası `commit-msg` aşamasında denetler.

```
feat(latex2sympy): add one-sided limit direction to grammar

Curriculum cards lim-0031..0045 need x -> a^+ and a^-. Direction is carried
on the Limit node and mapped to sympy.limit(dir=...).

IP: 4
ADR: 0004
```

## 3. Kod incelemesi

- `CODEOWNERS`: `src/aicls/{latex2sympy,mathengine,book,collect,db}` → mmturhan; `src/aicls/{core,nlu,ocr,asr,data,eval,ml,cli}`, `training/`, `datasets/nlu`, `datasets/latex` → csahin; `web/`, `src/aicls/api` → sguler (İP-2 kapsamı arka ucu da içerir; `api/services/` için ikinci gözden geçiren csahin); `docs/` herkes. Yazar kendi PR'ını birleştiremez.
- PR şablonu checklist'i: testler önce yazıldı mı; docs güncellendi mi; ADR gerekli mi (bkz. [adr/README.md](../adr/README.md)); hangi İP; `openapi.json` yenilendi mi; migrasyon elle incelendi mi; kapılı bir komut çalıştırıldı mı (hayır olmalı); metrik dosyası değiştiyse `poe report-check` geçti mi.
- Hedef: PR < 400 satır; büyükse önce tasarım notu (`experiments/` ya da ADR taslağı).

## 4. CI işleri

`.github/workflows/ci.yml` — matris `windows-latest` (iş istasyonuyla aynı) + `ubuntu-latest` (VM ile aynı), Python 3.11; her iş `poe` görevini çağırır:

| Adım | Komut | Kırılma koşulu |
|---|---|---|
| install | `pip install -r requirements/base.txt -r requirements/dev.txt -e .` | hash uyuşmazlığı |
| lint | `poe lint` = `ruff check .` + `ruff format --check .` | |
| type | `poe type` = `mypy src` (strict: core, db, latex2sympy, mathengine, nlu, book, collect, eval, api; standart + `ignore_missing_imports`: ocr, asr, data, ml) | |
| imports | `poe imports` = `lint-imports` (`[tool.importlinter]` katman + yasak sözleşmeleri) | |
| gates | `python tools/check_gates.py` + `python tools/lint_no_raw_sql.py` | kapı dışı indirme/ham SQL |
| test | `poe test` = `pytest -m "not gpu and not data and not mysql and not slow and not browser" -n auto --cov=aicls --cov-report=xml --timeout=60` | |
| coverage | `python scripts/coverage_gate.py coverage.xml` | paket eşikleri ([test-stratejisi](test-stratejisi.md)) |
| db | `alembic upgrade head` (SQLite) + `pytest tests/integration/db -m "not mysql"` | |
| openapi | `python scripts/export_openapi.py && git diff --exit-code docs/api/openapi.json web/src/api/generated/schema.d.ts` | bayat şema |
| docs-map | `python tools/check_docs_map.py` | İP eşleme tablosunda olmayan dosya |
| docs | `mkdocs build --strict` | kırık bağlantı |
| report-check | `poe report-check` | terfi edilmiş metrik dosyası HEAD'in atası değil / parmak izi uyuşmaz |

`web.yml`: `npm ci`, `eslint`, `prettier --check`, `tsc --noEmit`, `vitest run --coverage`, `vite build`, Playwright chromium (PR). `nightly.yml` (cron 02:00, elle de tetiklenir): `mysql:8` servisi ile `-m mysql` (head→base→head + entegrasyon), `-m "slow or perf"`, hypothesis 1000 örnek, Playwright chromium+firefox+webkit, `pip-audit`, `npm audit`, müfredat çıktılarının KaTeX render denetimi (Node betiği), `poe report-check`. `docs.yml`: `main`'de site yayını (özel GitHub Pages) + iş istasyonuna kopya.

CI hiçbir zaman veri/ağırlık indirmez; `gpu`/`data` işaretli testler CI'da koşmaz.

## 5. Kalite kapıları

- **ruff:** kurallar `E,F,W,I,N,UP,B,SIM,PL,RUF,S,PT`, satır 100, `D` yalnızca genel API'de; `ruff format`.
- **mypy 1.x:** yukarıdaki bölünmüş sıkılık; `warn_unreachable`, `disallow_any_generics` çekirdekte.
- **import-linter:** katman sözleşmesi + 5 yasak sözleşmesi ([genel-bakis.md §3](../mimari/genel-bakis.md#3-modül-sınırları)).
- **pytest 9:** `-W error::aicls` (paket uyarıları hata), `pytest-timeout` (birim 5 s, entegrasyon 60 s), hypothesis `deadline=200 ms`, `max_examples=50` PR / `1000` gece.
- **Kapsam:** genel ≥ 85; latex2sympy/mathengine/nlu ≥ 95; eval ≥ 95 (metrics/normalizers ≥ 98); core ≥ 90; book/collect ≥ 85; api/db ≥ 80; data/ml ≥ 75; ocr/asr ≥ 70.
- **Bağımlılık:** `pip-audit`, `npm audit` gece; `requirements/*.txt` hash'li; `poe lock` (`pip-compile --generate-hashes`) tek yenileme yolu, yeniden derleme ayrı PR.
- **Büyük dosya:** `scripts/check_no_large_files.py` > 1 MB engeller (`datasets/`, `docs/assets/` izinli, yine de < 5 MB).

## 6. pre-commit

`.pre-commit-config.yaml`: `ruff` (fix), `ruff-format`, `mypy` (yalnızca `src/`, önbellekli), `check-yaml`, `check-json`, `check-toml`, `end-of-file-fixer`, `trailing-whitespace`, `mixed-line-ending --fix=lf`, `check-added-large-files --maxkb=1024`, `detect-private-key`, `conventional-pre-commit` (commit-msg), yerel kancalar: `tools/lint_no_raw_sql.py`, `scripts/check_no_large_files.py`, `web/` için `prettier` + `eslint` (yalnızca değişen dosyalar). `pre-push` aşamasında `pytest tests/unit -x -q -m unit`.

## 7. Sürümleme ve yayın

- SemVer `0.MINOR.PATCH` proje boyunca: `v0.0.1` (Faz 0), `v0.1.0` (8. hafta deterministik çekirdek), `v0.2.0` (gerçek zamanlı girdi + toplama), `v0.3.0` (ilk modeller), `v0.5.0` (model dondurma, 2027-08), `v1.0.0` (pilot hazır). `aicls.__version__` tek kaynak (`pyproject` dinamik).
- `CHANGELOG.md` Keep a Changelog; her PR `Unreleased` altına satır ekler.
- `python scripts/release.py minor` → sürüm artır, CHANGELOG bölümü, etiket, `web/dist` + wheel derleme; `deploy/` compose dosyaları etiketi referans alır.
- Rapor anlık görüntüsü: `git tag report-2027-05` + `scripts/reproduce_report.ps1` temiz klondan geçmeli.

## 8. Windows görev çalıştırıcı (poethepoet)

`poe_tasks.toml` (kapılı görevler **yoktur**; `train`, `download`, `install_ml` yalnızca açık betik çağrısıyla):

| Görev | Komut |
|---|---|
| `poe bootstrap` | `pwsh scripts/bootstrap.ps1` |
| `poe env-check` | `python scripts/check_env.py` |
| `poe lint` / `poe fmt` / `poe type` / `poe imports` | ruff / ruff format / mypy / lint-imports |
| `poe test` | PR seçimi (yukarıda) |
| `poe test-fast` | `pytest -m unit -x -q` |
| `poe test-all` | `+ slow, perf, e2e` |
| `poe cov` | `pytest --cov --cov-report=html` + `coverage_gate.py` |
| `poe ci` | lint → type → imports → gates → test → cov → db → openapi → docs-map → docs → report-check |
| `poe dev` | `python scripts/dev_server.ps1` eşdeğeri: `uvicorn aicls.api.app:create_app --factory --reload --port 8000` |
| `poe web` / `poe web-test` / `poe web-e2e` | `npm --prefix web run dev` / `vitest` / `playwright test` |
| `poe docs` / `poe docs-build` | `mkdocs serve` / `mkdocs build --strict` |
| `poe db-upgrade` / `poe db-revision -- -m "msg"` / `poe db-seed` | alembic / `aicls db seed` |
| `poe openapi` | `python scripts/export_openapi.py` |
| `poe erd` | `python tools/gen_erd.py` |
| `poe eval -- cas` | `aicls eval cas` |
| `poe metrics-promote -- <file>` / `poe report-check` / `poe report-render` | `aicls report promote|check|render` |
| `poe fixtures` | `python scripts/make_synthetic_fixtures.py` |
| `poe curriculum` | `python scripts/build_curriculum_testset.py` |
| `poe lock` | `pip-compile --generate-hashes requirements/{base,dev,ml}.in` |
| `poe new-adr -- <slug>` / `poe new-exp -- <slug>` | `scripts/new_adr.py` / `scripts/new_experiment.py` |

## 9. Sanal ortam ve OneDrive

- venv: `C:\Users\cagri\.venvs\ai-cls` (`py -3.11 -m venv --system-site-packages` — mevcut sistem paketleri sympy/torch-cpu/transformers kullanılır; `--system-site-packages` `pip freeze` çıktısında sistem paketlerini de listeler, `run_manifest` bunu not eder). Etkinleştirme `C:\Users\cagri\.venvs\ai-cls\Scripts\Activate.ps1`; `.vscode/settings.json` yorumlayıcıyı buraya bağlar; `.python-version` = 3.11.
- `PYTHONPYCACHEPREFIX=C:\Users\cagri\.cache\ai-cls\pycache` (`__pycache__` OneDrive dışına).
- `AICLS_DATA_ROOT`, `AICLS_MODELS_ROOT`, `runs/`, `mlruns/`, SQLite dev DB OneDrive dışı; `scripts/check_env.py` denetler.
- `web/node_modules` → `tools/onedrive_setup.ps1` ile `C:\Users\cagri\.cache\ai-cls\node_modules` junction'ı.
- git: `core.autocrlf=false` (`.gitattributes eol=lf`), `core.longpaths=true`, `core.fsmonitor=true`; Windows `LongPathsEnabled`.
- OneDrive: depo klasörü OneDrive'ın "klasör seçimi"nden çıkarılır ya da uzak depo kurulunca klon `C:\src\ai-cls`'e taşınır (ADR-0011'de önerilir); `attrib +S` gibi etkisiz yöntemler kullanılmaz. pytest `basetemp` `%TEMP%`.

## 10. Uzun vadeli süreklilik

Her paketin Türkçe belgesi, ADR'ler, `experiments/` notları, üretilen rapor tabloları ve `reproduce_report.ps1`; herhangi bir geliştirici ayrılsa da 2028 raporu temiz klondan yeniden üretilebilir. Bu, hakemlerin "tek kişiye bağımlılık" riskine verilen yanıttır.

## 11. Uygulama notları (iskele aşaması, 2026-09)

- `poe_tasks.toml` §8'deki listenin iskele aşamasında çalışabilen alt kümesini içerir: `env-check, lint, fmt, type, imports, gates, test, test-fast, test-all, cov, coverage-gate, docs-map, ci, dev, docs, docs-build, db-upgrade, db-revision, openapi, eval, report-check, fixtures, new-adr, new-exp`. `web*`, `db-seed`, `erd`, `metrics-promote`, `report-render`, `curriculum` ilgili paketlerle; `lock` (`pip-compile`, ağ gerektirir) kilit dosyaları ilk üretildiğinde eklenir (`requirements/README.md`).
- `poe ci` sırası şimdilik `lint → type → imports → gates → test → coverage-gate → openapi → docs-map → docs-build`; `db` ve `report-check` adımları `aicls.db` ve `aicls report` geldiğinde eklenir. `scripts/export_openapi.py` API paketi yokken çıkış 0 verir (`--require` ile zorlanır).
- `tools/check_docs_map.py` varsayılan kipte satır bazında denetler: satır `(planlı: …)` işaretliyse ya da listelediği depo yollarından en az biri varsa geçer; eksik her yol uyarı olarak yazılır. `--strict` (rapor etiketinde) her yolun varlığını ister ve kalan `planlı` işaretini hata sayar.
- Kapsam eşikleri `pyproject.toml [tool.aicls.coverage]` tablosundadır; `scripts/coverage_gate.py` henüz ölçülen satırı olmayan paketleri `n/a` sayar (`--require-all` rapor etiketinde).
- `mkdocs.yml` içinde `macros` eklentisi `render_by_default: false` ile kurulmuştur; düzyazı sayfalar `{{ metrics.<kind>.<metric> }}` örneğini harfiyen alıntıladığı için makro kullanan rapor sayfaları ön bilgide `render_macros: true` ile açılır. Başlık çapaları `pymdownx.slugs.slugify(case=lower)` ile Unicode korunarak üretilir (ör. `#3-onay-kapıları`).
- mypy: `strict` yalnızca `core, latex2sympy, mathengine, nlu, eval`; diğer paketlerde `disallow_untyped_defs`; `ocr, asr, data, ml` için `ignore_missing_imports`. `db, book, collect, api` için strict'e geçiş ilgili paketler yazıldığında ADR'siz yapılabilir (sıkılaştırma).
- `scripts/check_env.py` ince bir sarmalayıcıdır; uygulama `tools/check_env.py` içindedir (testler doğrudan onu yükler).
