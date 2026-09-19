# ADR-0009: Dokümantasyon — MkDocs Material (Türkçe), mkdocstrings, metrik makroları, sözlük yok

- **Durum:** kabul edildi
- **Tarih:** 2026-09-07
- **Karar vericiler:** Ç. Şahin, M. M. Turhan, Ş. Güler
- **İlgili İP:** İP-2 (API dokümantasyonu), İP-3 (geliştirici dokümanları, teknik raporlar), İP-4 (teknik rapor), İP-8 (arayüz dokümantasyonu), gelişme raporları

## Bağlam ve problem

Proje "kapsamlı dokümantasyon" ister; belgeler Türkçe, kod İngilizce; raporlardaki sayılar elle yazılmamalı; `mkdocs-material` ve `mkdocstrings` geliştirme ortamında kuruludur. Kullanıcı tercihi: kısaltmalar metin içinde açılır, ayrı sözlük sayfası istenmez.

## Değerlendirilen seçenekler

1. **Sphinx** — güçlü ama reST, Türkçe tema desteği zayıf, öğrenme eğrisi.
2. **Docusaurus** — Node tabanlı; Python API referansı zor.
3. **MkDocs Material + mkdocstrings-python + mkdocs-macros-plugin + pymdownx.superfences (mermaid)** — saf Python, `language: tr`, strict derleme, makro ile metrik enjeksiyonu.

## Karar

Seçenek 3. `mkdocs.yml` `theme.language: tr`, `plugins: [search, mkdocstrings, macros]`, `markdown_extensions: [pymdownx.superfences (mermaid), tables, admonition, toc]`, `strict: true`. Ağaç: `index.md`, `mimari/`, `ml/`, `gelistirme/`, `guvenlik-gizlilik.md`, `raporlama/` (İP başına rapor sayfaları + otomatik `sonuclar.md`), `deneme-alani.md`, `adr/`, planlı `literatur/`, `referans/` (mkdocstrings, İngilizce docstring'ler "Kod Referansı" altında), `deneyler/` (experiments notları otomatik dizin). Makrolar `reports/metrics/latest.json`'dan `metrics.<kind>.<metric>` sunar; rapor sayfalarında elle sayı yok. `tools/check_docs_map.py` İP eşleme tablosunu doğrular. Sözlük sayfası yok; kısaltma ilk geçişte parantez içinde açılır. ADR biçimi MADR + zorunlu "Öneri metniyle uyum". CI `mkdocs build --strict`; `main`'de özel GitHub Pages + iş istasyonu kopyası.

## Sonuçlar

- Olumlu: TÜBİTAK hakemi İP numarasıyla gezinir; sayılar tek kaynaktan; kırık bağlantı CI'ı kırar.
- Olumsuz: makro/eklenti bağımlılıkları; rapor PDF'i için `mkdocs-print-site` veya pandoc ayrıca (gerektiğinde).

## Öneri metniyle uyum

İP-2 "API Dokümantasyonu"; İP-3 "Modellerin nasıl kullanılacağını açıklayan geliştirici dokümanları" ve "2 adet teknik rapor"; İP-4 "teknik rapor" ve "analiz raporu"; İP-8 "yazılım arayüzü ve dokümantasyonu"; İP-9 "kapsamlı raporlar". Belgeler bu çıktıların doğrudan kaynağıdır ([raporlama/ip-eslestirme.md](../raporlama/ip-eslestirme.md)).

## Doğrulama

CI `docs` ve `docs-map` adımları; `test_metrics_macro_reads_latest_json`; `test_report_pages_have_no_hand_typed_percentages` (uyarı düzeyi).

## İlgili

[index.md](../index.md), ADR-0007, ADR-0015.
