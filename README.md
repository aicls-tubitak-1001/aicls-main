# AI-CLS

**TÜBİTAK 1001 — 125E748** · *Temel Analiz Öğretimine Yönelik Yapay Zeka Destekli-Etkileşimli Bir Platformun Tasarımı ve Geliştirilmesi (AI-CLS)* · 01.05.2026 → 01.05.2028

## Ne?

Türkiye'deki üniversitelerde Temel Analiz (Calculus) öğretimi için yapay zeka destekli, etkileşimli bir web platformu. Tarayıcıda **el yazısı** (Canvas; iz verisi/InkML veya PNG) ve **ses** (mikrofon) alınır; el yazısı OCR modeliyle LaTeX'e, ses ASR modeliyle Türkçe metne çevrilir; Türkçe metin kural tabanlı **metin→komut** katmanıyla tipli komuta dönüşür ("x karenin türevini al" → `diff(x**2, x)`); LaTeX kendi Lark dilbilgimizle SymPy'a çevrilir; sonuç yalnızca **SymPy** (deterministik CAS) tarafından üretilir ve KaTeX ile çizilir. "Türev tanımını getir" gibi komutlar ders kitabı bilgi grafiğinden (JSON-LD/RDF, MySQL) içerik çeker.

```text
el yazısı ─→ OCR ─→ LaTeX ─┐
ses ─→ ASR ─→ Türkçe metin ─→ metin→komut ─┤─→ LaTeX→SymPy ─→ SymPy ─→ LaTeX sonucu ─→ web arayüzü (React, KaTeX, Matplotlib)
```

## Neden bu depo böyle?

- **Deterministik çekirdek, olasılıksal kenarlar.** Kullanıcının gördüğü her sonuç SymPy'dan gelir; OCR/ASR yalnızca aday üretir. Bu, öneri metnindeki "halüsinasyon üretmez" iddiasının teknik karşılığıdır.
- **Her rapor sayısı yeniden hesaplanabilir.** `reports/metrics/*.json` git SHA'sı, veri parmak izi ve normalizasyon sürümüyle yazılır; belgeler sayıları makroyla alır.
- **Onay kapıları koddadır** (aşağıda).
- **Windows-önce, OneDrive-dışı.** venv, veri kökü, model ağırlıkları ve koşu kayıtları OneDrive dışında; görev çalıştırıcı `poethepoet` (`make` yok).
- **Test-önce, katmanlı bağımlılık.** `import-linter` torch'u çekirdekten uzak tutar; stub modellerle API ve değerlendirme veri/GPU olmadan test edilir.

Ayrıntılar: [docs/index.md](docs/index.md) (MkDocs sitesi: `poe docs`).

## Hızlı başlangıç (Windows)

```powershell
# 1) Sanal ortam (OneDrive dışı), bağımlılıklar, pre-commit, veri kökü
pwsh scripts/bootstrap.ps1
C:\Users\cagri\.venvs\ai-cls\Scripts\Activate.ps1

# 2) Ortam dosyası
Copy-Item .env.example .env      # AICLS_JWT_SECRET: python -c "import secrets;print(secrets.token_hex(32))"

# 3) Doğrulama ve ilk koşu
poe env-check
poe test-fast
poe ci                           # lint → type → imports → gates → test → coverage → docs
poe docs                         # http://127.0.0.1:8001
```

Linux/WSL2: `bash scripts/bootstrap.sh` — aynı `poe` görevleri.

`poe`, `python`'ı PATH'ten çözer; bu yüzden görevler **etkin venv** içinden çalıştırılır. Etkinleştirmeden çalıştırmak için: `poe -e virtualenv -X location=C:\Users\cagri\.venvs\ai-cls ci`.

| Görev | Ne yapar |
|---|---|
| `poe lint` / `poe fmt` / `poe type` / `poe imports` | ruff / ruff format / mypy / import-linter |
| `poe test` / `poe test-fast` / `poe test-all` / `poe cov` | pytest seçimleri ve kapsam kapısı |
| `poe gates` | `tools/check_gates.py` + `tools/lint_no_raw_sql.py` |
| `poe docs` / `poe docs-build` | MkDocs (Türkçe) |
| `poe db-upgrade` / `poe db-revision -- -m "msg"` | Alembic |
| `poe new-adr -- <slug>` / `poe new-exp -- <slug>` | ADR ve deneme şablonları |

Görev listesinde **kapılı komut yoktur** (`poe_tasks.toml`).

## Onay kapıları (ADR-0012)

Dört eylem yalnızca üç koşul birlikte sağlandığında çalışır; biri eksikse çıkış kodu 3:

| Kapı | Eylem | Koşullar |
|---|---|---|
| `DATASET_DOWNLOAD` | MathWriting, Common Voice vb. indirme | `--i-have-approval` + `AICLS_GATE_DATASET_DOWNLOAD=approved` + [onay defteri](docs/gelistirme/onay-defteri.md) satırı |
| `WEIGHTS_DOWNLOAD` | Whisper/HF ağırlıkları | aynı üçlü |
| `CUDA_INSTALL` | CUDA torch (cu128, RTX 5090 sm_120), torchvision, peft | `scripts/install_ml.ps1 -IHaveApproval` + aynı üçlü |
| `TRAINING` | gerçek veriyle eğitim/profil koşusu | aynı üçlü (`--synthetic` kapısızdır) |

CI hiçbirini ayarlamaz ve reddi test eder; `tools/check_gates.py` kaynak ağacında örtük indirme desenlerini (hub kimlikli `from_pretrained`, `weights=` ≠ `None`, HF/PyTorch URL'leri) yakalar.

## Depo düzeni

```text
src/aicls/      core · db · latex2sympy · mathengine · nlu · ocr · asr · book · collect · data · eval · ml · api · cli
web/            React arayüz (ayrı npm projesi)
training/       yapılandırmalar ve kapılı betikler
datasets/       küçük, sürümlü test setleri (müfredat, LaTeX, NLU, kitap örneği, manifestler)
reports/        değerlendirme çıktıları (metrics/*.json, latest.json)
experiments/    deneme alanı (1 sayfalık notlar)
docs/           Türkçe MkDocs sitesi (mimari, ML, geliştirme, raporlama, ADR)
scripts/ tools/ bootstrap, kapı betikleri, denetim araçları
deploy/         iş istasyonu servisi, toplama VM'i, yedekleme
```

Katman sözleşmesi: `cli > api|ml > eval > data|collect|book > nlu > mathengine > latex2sympy > ocr|asr > db > core` (ADR-0002).

## Ekip

Çağrı Şahin (AI koordinasyonu, modeller, metin→komut, ortam) · Mehmet Murat Turhan (matematik motoru İP-4, veri boru hatları, kitap şeması/API İP-8) · Şahiner Güler (İP-2 web). Belgeler Türkçe; kod, docstring ve commit mesajları İngilizce.

## Lisans

Karar proje yürütücüsündedir; bkz. [LICENSE](LICENSE) ve ADR-0002.
