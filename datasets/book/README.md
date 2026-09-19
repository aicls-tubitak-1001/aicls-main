# datasets/book — İP-8 kitap etiketleme ve getirme test seti

Bu dizin `aicls.book` paketinin ölçüm setidir. Tümü **elle yazılmış, sentetik** içeriktir;
gerçek *Temel Analiz* kitabından alıntı **yoktur** (telif). Boyut birkaç on KB'dir.

## İçerik

| Yol | İçerik | Kullanan test |
|---|---|---|
| `sample/bolum-01-limit.md` | Düz `## Tanım 1.1:` / `## Teorem 1.2 (…):` desenli örnek bölüm (8 birim) | `test_markdown_tagger_recognises_tanim_1_1`, `test_coverage_is_one_for_sample_chapter` |
| `sample/bolum-02-turev.md` | `::: tanim 2.4 "Türev" aliases=[…] requires=[…]` bloklu örnek bölüm (12 birim) | `test_markdown_tagger_recognises_block_syntax_with_aliases` |
| `sample/bolum-04-integral.tex` | `\begin{tanim}[…]\label{tanim:4.1}` ortamlı LaTeX bölümü (5 birim) | `test_latex_tagger_recognises_labelled_environments` |
| `invalid/bolum-03-etiketsiz.md` | **Kasıtlı hatalı**: iki başlık numarasız → kapsama 0.5 | `test_loader_rejects_untagged_heading_with_list` |
| `expected_units.json` | Yukarıdaki dört kaynak için beklenen birim/üçlü listesi (altın standart) | `test_sample_chapter_matches_expected_units` |
| `fetch_test.jsonl` | 72 Türkçe getirme ifadesi (ekli biçimler, ASCII yazım, yazım hatası, komut fiilleri) | `test_book_fetch_test_set_accuracy_above_0_95` |

## `fetch_test.jsonl` biçimi

Her satır bir sorgudur:

```json
{"id": "fetch-0002", "query": "türevin tanımını getir", "expect_unit": "tanim-2.4",
 "unit_type": "Definition", "kind": "suffix_gen"}
```

- `unit_type` `null` ise çözümleyiciye tür süzgeci verilmez; türü ifadeden çıkarması beklenir.
- `kind` yalnızca hata çözümlemesi içindir: `command`, `question`, `bare`, `title`, `suffix_gen`,
  `suffix_acc`, `head_noun`, `authored_alias`, `synonym`, `ascii`, `typo`, `partial`, `section`.
- Ölçüt: ilk sonucun `expect_unit` ile eşleşme oranı > %95 (İP-8 "komutla getirme" ölçütü).
  Ölçülen değer `reports/metrics/` altına yazılır ve ilerleme raporunda bu dosyanın sürümüyle
  birlikte anılır.

## Kurallar

1. Satır silinmez; yeni ifade yeni satırdır (`fetch-00NN` artan).
2. Bir ifade başarısız olursa **önce takma ad üretimi** (`aicls.book.aliases`) veya çözümleyici
   düzeltilir; sorgu setten çıkarılmaz — set bu yüzden vardır (ADR-0019, "Sonuçlar").
3. Kişisel veri yoktur; içerik yazarın kendi metnidir.
4. Lisans: proje lisansı (bkz. depo kökündeki `LICENSE`).

## Ölçüm notu (rapor için)

Bu set **bağımsız bir derlemden örneklenmemiştir**; proje ekibi tarafından yazılmıştır. İlerleme
raporunda "%100 etiketleme" ve "> %95 getirme" rakamları verilirken bu köken açıkça belirtilmeli,
pilot çalışmada gerçek kitap bölümleriyle yeniden ölçülmelidir
(bkz. `docs/gelistirme/test-stratejisi.md` ve KNOWN_ISSUES "Metric provenance").
