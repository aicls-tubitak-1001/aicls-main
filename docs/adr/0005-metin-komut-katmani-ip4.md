# ADR-0005: Kural tabanlı metin→komut katmanı; İP-4 altında raporlama; LLM yalnızca niyet önericisi

- **Durum:** kabul edildi
- **Tarih:** 2026-09-07
- **Karar vericiler:** Ç. Şahin, M. M. Turhan
- **İlgili İP:** İP-4 (raporlama), İP-3 (ASR çıktısının tüketicisi), İP-2 (komut iletimi > %95)

## Bağlam ve problem

Öneri metninde ASR çıktısı ("x karenin türevini al") ile CAS çağrısı (`diff(x**2, x)`) arasındaki katman adlandırılmamıştır, ancak zorunludur. Ekip kararı (memory: bekleyen kararlar #2) katmanın İP-4 altında raporlanması ve sahibinin Ç. Şahin olmasıdır. "Halüsinasyon üretmez" iddiası (A.2.2) sonuçların yalnızca SymPy'dan gelmesini gerektirir.

## Değerlendirilen seçenekler

1. **LLM-önce (işlev çağırma)** — esnek; çevrimdışı değil, deterministik değil, test edilemez, KVKK/ağ bağımlılığı, iddiaya aykırı.
2. **İnce ayarlı sınıflandırıcı** — veri gerektirir (yok), açıklanamaz.
3. **Kural tabanlı: YAML komut envanteri + Türkçe normalizasyon + konuşulan matematik Lark dilbilgisi + niyet/yuva çözümleme; LLM yalnızca niyet kimliği önerebilir, kapalı gelir** — deterministik, çevrimdışı, izlenebilir, altın küme ile ölçülebilir.

## Karar

Seçenek 3. `aicls.nlu`; envanter paket kaynağı `inventory/commands.yaml` (17 niyet v1); `Matched | Clarification | NoMatch` çıktısı; ses modalitesinde arayüz her zaman `CommandPreview` onayı ister; `IntentFallback.suggest()` yalnızca envanter niyet kimliği döndürebilir, ifade/sonuç üretemez, `AICLS_LLM_FALLBACK=false` varsayılan; LLM kaynaklı çözümler `nlu_source=llm` ile ayrı sütunda raporlanır. Katman `mathengine`'in tipli komut modellerini üretir; bilinmeyen niyet SymPy'a ulaşamaz. NLU normalizasyonu (sözlük katlama, ASR varyantları) yalnızca bu katmandadır; WER normalizasyonundan ayrıdır (ADR-0015).

## Sonuçlar

- Olumlu: iddia korunur; `command_intent_accuracy` İP-3/İP-4 sınırını ölçer; İP-2 "komut iletimi > %95" ölçütü `intent_accuracy` ile somutlaşır; envanter büyüdükçe okuma senaryoları kütüphanesi ile tek kaynak.
- Olumsuz: Türkçe ek/sözdizimi çeşitliliği kural bakımı ister; ASR hataları kapsam dışına düşerse açıklama isteği oranı artar (sınıfta yine kullanılabilir).

## Öneri metniyle uyum

A.2.3: "Ses verisinden elde edilen metnin uygun komut karşılığı, Bilgisayarlı Cebir Sistemi (CAS) tarafından icra edilecektir." İP-4 tanımı: "OCR ve ses tanıma modüllerinden gelen matematiksel ifadelerin ... CAS'a entegre edilerek hatasız çözümlenmesi"; İP-4 görevi "ara yazılım modülü". İP-2 ölçütü "Komutların >%95 başarıyla iletilmesi". İP-3 risk planı B: "komut seti basitleştirilerek en temel ve en sık kullanılan komutlara odaklanılacaktır" → envanterde `core` etiketi ile öncelikli niyet alt kümesi.

## Doğrulama

`test_nlu_command_test_set_intent_accuracy_above_0_95`, `test_inventory_every_example_resolves_to_own_intent`, `test_llm_fallback_disabled_by_default`, `test_llm_fallback_can_only_return_inventory_intent_id`, `test_asr_eval_reports_command_intent_accuracy`.

## İlgili

[mimari/metin-komut-katmani.md](../mimari/metin-komut-katmani.md), ADR-0004, ADR-0015.
