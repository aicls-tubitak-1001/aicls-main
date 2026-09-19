# ADR-0019: Kitap bilgi grafiği — JSON-LD belge sütunu + üçlü tablosu (RDF) + takma ad tablosu, MySQL; üçlü deposu yok

- **Durum:** kabul edildi
- **Tarih:** 2026-09-07
- **Karar vericiler:** M. M. Turhan, Ç. Şahin
- **İlgili İP:** İP-8

## Bağlam ve problem

A.3.9 "JSON-LD formatında metadata şemaları", "RDF tripleri" ve "MySQL tabanlı bir API" der. İP-8 ölçütleri: içeriğin %100 etiketlenmesi, komutla getirme > %95, hatasız görselleştirme; risk planı: "Tanım 1.1:" gibi kalıpları tanıyan otomatik etiketleme betikleri, indeksleme ve önbellekleme. Sorgu yükü "türev tanımını getir" gibi takma ad aramalarıdır, SPARQL değil.

## Değerlendirilen seçenekler

1. **Üçlü deposu (Jena/Fuseki, Blazegraph)** — ikinci veritabanı, JVM, işletme; SPARQL gerekmez.
2. **rdflib bellek içi graf** — kalıcılık ve eşzamanlılık yok.
3. **İlişkisel: `book_units` (normalize sütunlar + tam JSON-LD belge), `book_triples` (subject, predicate, object = RDF üçlüsü, iki yönlü indeks), `book_aliases` (Türkçe ek tablosuyla üretilmiş), `book_assets`; JSON-LD `@graph` ve Turtle dışa aktarım (rdflib)** — tek veritabanı, indeksli takma ad araması, öneri metnindeki biçimler somut dosya olarak.

## Karar

Seçenek 3. Şema [mimari/veri-modeli.md §5](../mimari/veri-modeli.md#5-kitap-içerik-şeması-json-ld--rdf); `@context` `src/aicls/book/context.jsonld`, sözlük `vocab.ttl`; birim türleri Definition..Section; yüklemler `aicls:partOf/requires/illustrates/proves/next/hasFigure`, `skos:related`. Yazım: Markdown `::: tanim 2.4 "Türev" ...` blokları ve düz "Tanım 2.4:" desenleri (`ingest.markdown_tagger`), LaTeX ortamları (`ingest.latex_tagger`); `validators.coverage()` etiketsiz başlıkta yüklemeyi reddeder → İP-8 %100 ölçütü ölçülür. Çözümleme: `tr_casefold` + ek soyma → alias tam eşleşme (indeks) → MySQL FULLTEXT / SQLite LIKE → difflib ≥ 0.8; LRU 2048. Graf yürüyüşü tek atlama birleşimi; `requires` kapanışı ≤ 3 derinlik uygulama döngüsüyle (taşınabilirlik). Dışa aktarım İP-8 "RDF tripleri" teslimatının somut dosyasıdır.

## Sonuçlar

- Olumlu: tek veritabanı/yedek; getirme p50 < 20 ms; kapsama ve doğruluk otomatik ölçülür; kitap ekibi Markdown yazar.
- Olumsuz: SPARQL yok (gerekmiyor); FULLTEXT lehçeye özgü (ADR-0003 kuralı); takma ad üretimi Türkçe ek tablosunun kapsamıyla sınırlı, eksikler `fetch_test.jsonl` hatalarıyla büyütülür.

## Öneri metniyle uyum

A.3.9 birebir: JSON-LD metadata, RDF üçlüleri, MySQL tabanlı API. İP-8 görevleri: etiketli/sorgulanabilir biçim, veritabanı, API, "x tanımını ekrana getir" komutu, hatasız görselleştirme (KaTeX + `sanitize_latex`). İP-8 risk planı: otomatik etiketleme betikleri, indeksleme, önbellekleme.

## Doğrulama

`test_markdown_tagger_recognises_tanim_1_1`, `test_coverage_is_one_for_sample_chapter`, `test_loader_rejects_untagged_heading_with_list`, `test_book_resolve_turevin_tanimi_and_turev_tanimi_same_unit`, `test_book_fetch_test_set_accuracy_above_0_95`, `test_book_export_turtle_reparses_with_rdflib_triple_count_equals_edges`, `test_book_resolve_p50_under_20ms`.

## İlgili

ADR-0003, [mimari/api.md §3.7](../mimari/api.md#37-kitap-ip-8).
