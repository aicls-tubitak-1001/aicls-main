# ADR-0010: Test stratejisi — pytest işaretleri, hypothesis, stub modeller, paket bazlı kapsam eşikleri

- **Durum:** kabul edildi
- **Tarih:** 2026-09-07
- **Karar vericiler:** Ç. Şahin, M. M. Turhan, Ş. Güler
- **İlgili İP:** İP-2 (test), İP-9 (sistem testleri), tüm paketler

## Bağlam ve problem

Proje sahibi "adım adım, test-önce" ister. Veri seti, ağırlık, CUDA ve eğitim kapılıdır; yine de API, arayüz, değerlendirme koşumu ve eğitim döngüsü bugün test edilmelidir. Hakemler tüm ağaçta `mypy --strict` ve torch paketlerinde %95 kapsamın sürdürülemez olduğunu belirtti.

## Değerlendirilen seçenekler

1. **Tek düzey kapsam (%95 her yerde) + strict mypy her yerde** — torch/transformers kodunda `type: ignore` duvarı ve `--no-cov` alışkanlığı.
2. **Katmanlı sıkılık: deterministik çekirdek %95 + strict; model kenarları %70 + standart mypy; işaretli testler; stub modeller; sentetik fikstürler; hypothesis** — gerçekçi ve savunulabilir.

## Karar

Seçenek 2. İşaretler: `unit, property, integration, ws, e2e, perf, slow, gated, gpu, data, mysql, browser`; PR seçimi `not gpu and not data and not mysql and not slow and not browser`; gece geri kalanı. Kapsam (`scripts/coverage_gate.py`): genel 85; latex2sympy/mathengine/nlu 95; eval 95 (metrics/normalizers 98); core 90; book/collect 85; api/db 80; data/ml 75; ocr/asr 70; web (vitest) 70. mypy strict: core, db, latex2sympy, mathengine, nlu, book, collect, eval, api; standart + `ignore_missing_imports`: ocr, asr, data, ml. Fikstürler: `settings` (test), Alembic head SQLite `:memory:`, `client` stub modellerle, sentetik iz/görüntü/ses (seed 1234, `scripts/make_synthetic_fixtures.py`, < 2 MB), `mini_book`. Özellik testleri: ayrıştırıcı gidiş-dönüşü, normalizasyon idempotansı, metrik değişmezleri, fuzz. Öneri metnindeki örnekler test adı olur. Playwright üç tarayıcı (Chrome/Firefox/Safari karşılığı chromium/firefox/webkit). Kapı testleri her PR'da; gerçek eğitim hiçbir testte yok.

## Sonuçlar

- Olumlu: `poe test` verisiz/GPU'suz dakikalar içinde; çekirdek yüksek güvence; İP-9 test raporu CI artefaktlarından derlenir.
- Olumsuz: model kenarlarında düşük kapsam gerçek veriyle işaretli testlere ve profil koşularına bırakılır; kapsam düşürmek ADR ister.

## Öneri metniyle uyum

İP-2 görevi "Test: Uyumluluk, performans, güvenlik ve kullanılabilirlik testleri"; A.1.2 "farklı tarayıcılarda (Chrome, Firefox, Safari) uyumluluk testleri periyodik olarak"; İP-9 "kapsamlı işlevsellik testleri", "performans testlerinde ... 3 saniyenin altında"; İP-2 risk planı "sürekli entegrasyon (CI)".

## Doğrulama

`scripts/coverage_gate.py` CI'da; `test_poe_tasks_contain_no_gated_commands`; `e2e/browsers.spec.ts`.

## İlgili

[gelistirme/test-stratejisi.md](../gelistirme/test-stratejisi.md), ADR-0002, ADR-0012.
