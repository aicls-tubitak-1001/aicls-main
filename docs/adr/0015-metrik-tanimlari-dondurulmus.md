# ADR-0015: Metrik tanımları dondurulmuş, normalizasyon sürümlü; WER normalizasyonu sözlük katlamasız; güven aralıkları

- **Durum:** kabul edildi
- **Tarih:** 2026-09-07
- **Karar vericiler:** Ç. Şahin, M. M. Turhan
- **İlgili İP:** İP-2, İP-3, İP-4, İP-8, İP-9 (tüm ölçütler)

## Bağlam ve problem

Sözleşme yedi sayısal ölçüt verir; 2027 ve 2028 raporlarında aynı sayının aynı anlama gelmesi, hakemin tanımı görebilmesi ve küçük test kümelerinde belirsizliğin bildirilmesi gerekir. Hakem değerlendirmesi ilk tasarımda WER normalizasyonunun ASR hata tablosu ve sözlük birleştirme içermesini (metriği test setine göre eniyileme) ve CER'in karakter/token belirsizliğini eleştirdi.

## Değerlendirilen seçenekler

1. **Kütüphane varsayılanları (jiwer, torchmetrics), tanım belgesiz** — sürüm değişince sayı değişir.
2. **Tek uygulama `aicls.eval.metrics`, sürümlü normalizasyon (`latex_v1`, `tr_v1`) her raporda yazılı, tanımlar belgede formülle, jiwer yalnızca çapraz denetim; bootstrap %95 GA; üç durumlu sembolik eşdeğerlik** — denetlenebilir ve kararlı.

## Karar

Seçenek 2. Tanımlar [ml/degerlendirme.md §2](../ml/degerlendirme.md#2-metrikler) içinde dondurulmuştur; özet:
- **CER:** korpus düzeyi karakter Levenshtein, `latex_v1` sonrası, boşluksuz, MathWriting `normalized_label` üzerinde; TER ve ExpRate ikincil.
- **WER:** korpus düzeyi, konuşmacı-ayrık test; **ham** ve **`tr_v1`** (NFC, Türkçe küçük harf, noktalama, rakam→sayı sözcüğü, boşluk — başka kural yok) yan yana. ASR varyant katlama yalnızca `aicls.nlu.normalizer` içinde kalır; etkisi `command_intent_accuracy` ile ayrı ölçülür.
- **Ayrıştırma başarısı:** parse + to_sympy + gidiş-dönüş `ok` (sessiz yanlış ayrıştırma başarısızlık).
- **CAS doğruluğu:** `checker` tabanlı; `unsupported` ayrı sütun, paydada.
- **Sembolik eşdeğerlik:** `equal | different | unparsable`.
- **Gecikme:** istemci ölçümlü ortalama + p50/p95/n, modalite bazında.
- **SUS:** standart puanlama.
- Tanım değişikliği yeni sürüm (`latex_v2`) açar; eski sürüm silinmez; `EvalReport.normalizer_versions` ve `definitions_ref` alanları zorunlu; `poe report-check` sürüm uyuşmazlığında kırılır.
- **Arkalık cevabı görmez ve stub sayı yayımlamaz** (2026-09-08 eki). (a) Değerlendirme koşucusu örneği modele vermeden önce `blind_sample` ile körleştirir — `Ink.label` kopyadan silinir; körleştirilemeyen bir örnek tipinde koşum sayı üretmek yerine durur. (b) Öznesi öğrenilmiş model olan türlerde (`ocr`, `asr`) stub arkalı koşum `latest.json`'a ölçüt değil `null` + `measurement: "wiring_check"` + gerekçe yazar; gözlenen değerler `wiring_check.metrics` altında saklanır. Gerekçe: stub tanıyıcı kendisine verilen etiketi geri verdiği için terfi eden dosya `ocr.cer = 0.0` / `ocr.exprate = 1.0` yayımlıyordu; sözleşmedeki `CER < %10` hedefinin yanında bu savunulamaz bir satırdır.

## Sonuçlar

- Olumlu: literatürle karşılaştırılabilir WER; hakem tanımı formülle görür; GA ile "<%10" iddiası savunulabilir.
- Olumsuz: ham WER daha yüksek görünür (bilinçli); her yeni normalizasyon kuralı sürüm ve ADR ister.

## Öneri metniyle uyum

A.2.1 "Karakter Hata Oranı (CER) ... (Michael ve ark., 2019)"; A.2.3 ve İP-3 "Kelime Hata Oranı (WER)"; İP-4 ölçütleri (%99/%99); İP-2/İP-4/İP-9 "ortalama 3 saniye"; İP-2/İP-9 SUS "> 75/100"; İP-8 "%100 etiketleme", "%95 doğruluk"; İP-2 "komutların >%95 iletilmesi".

## Doğrulama

`test_cer_reference_values_hand_computed`, `test_cer_uses_normalized_label_not_raw`, `test_wer_corpus_matches_jiwer_reference`, `test_tr_v1_normalizer_changes_word_count_only_by_digit_rule`, `test_symbolic_equivalence_returns_unparsable_not_different`, `test_parse_success_rate_counts_silent_misparse_as_failure`, `test_eval_report_has_fingerprint_git_sha_and_normalizer_versions`, `test_bootstrap_ci_contains_point_estimate`, `test_the_ocr_runner_never_lets_the_backend_see_the_label`, `test_the_asr_runner_hands_the_backend_a_waveform_and_nothing_else`, `test_a_stub_backed_run_publishes_no_number_only_a_reason`, `test_promote_publishes_null_for_a_stub_backed_model_run`, `test_promote_still_publishes_a_stub_run_of_a_kind_that_measured_real_code`.

## İlgili

[ml/degerlendirme.md](../ml/degerlendirme.md), ADR-0005, ADR-0007.
