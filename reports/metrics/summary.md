| Koşum | Ölçüt | Değer | %95 GA | n | Hedef | Durum |
|---|---|---:|---:|---:|---|---|
| parser | failures | 0.0000 | — | 3709 | — | — |
| parser | success_rate | 1.0000 | [1.0000, 1.0000] | 3709 | > 0.99 | tuttu |
| cas | correct_rate | 1.0000 | [1.0000, 1.0000] | 1204 | > 0.99 | tuttu |
| cas | error_rate | 0.0000 | — | 1204 | — | — |
| cas | mean_ms | 5.5859 | — | 1204 | — | — |
| cas | p50_ms | 2.3840 | — | 1204 | — | — |
| cas | p95_ms | 17.4659 | — | 1204 | — | — |
| cas | parse_n | 1134 | — | 1204 | — | — |
| cas | parse_success_rate | 1.0000 | — | 1204 | — | — |
| cas | unsupported_rate | 0.0000 | — | 1204 | — | — |
| nlu | clarification_rate | 0.0000 | — | 351 | — | — |
| nlu | command_exact | 0.9972 | — | 351 | — | — |
| nlu | intent_accuracy | 0.9972 | [0.9915, 1.0000] | 351 | >= 0.95 | tuttu |
| nlu | no_match_rate | 0.0684 | — | 351 | — | — |
| nlu | slot_f1 | 0.7064 | — | 351 | — | — |
| nlu | slot_f1_declared_only | 1.0000 | — | 351 | — | — |
| ocr | cer | 0.9679 | [0.9001, 1.0383] | 128 | < 0.10 | stub · kanıt değil |
| ocr | exprate | 0.0156 | — | 128 | — | — |
| ocr | mean_ms | 0.1147 | — | 128 | — | — |
| ocr | p95_ms | 0.1759 | — | 128 | — | — |
| ocr | symeq_different | 0.6953 | — | 128 | — | — |
| ocr | symeq_equal | 0.0156 | — | 128 | — | — |
| ocr | symeq_unparsable | 0.2891 | — | 128 | — | — |
| ocr | ter | 1.0231 | — | 128 | — | — |
| asr | cer | 0.8922 | — | 25 | — | — |
| asr | command_exact | 0.0000 | — | 25 | — | — |
| asr | command_intent_accuracy | 0.0000 | — | 25 | > 0.95 | stub · kanıt değil |
| asr | mean_ms | 0.0837 | — | 25 | — | — |
| asr | p95_ms | 0.1506 | — | 25 | — | — |
| asr | wer_raw | 1.1118 | — | 25 | — | — |
| asr | wer_tr_v1 | 1.1118 | [1.0000, 1.2618] | 25 | < 0.10 | stub · kanıt değil |
| book | fetch_top1 | 1.0000 | [1.0000, 1.0000] | 72 | > 0.95 | tuttu |
| book | fetch_top3 | 1.0000 | — | 72 | — | — |
| book | tagging_coverage | 1.0000 | — | 72 | == 1.00 | tuttu |
| latency | mean_ms | 5.2382 | [3.4363, 6.9565] | 200 | < 3000 | stub · kanıt değil |
| latency | p50_ms | 1.8555 | — | 200 | — | — |
| latency | p95_ms | 41.5618 | — | 200 | — | — |

Kaynak: parser_valid · commit 60aad42 · veri parmak izi sha256:9ff283b2c0d163194867f05716d73078c899b95cbda41e5574f3b6707c152ed0 · 2026-09-08T02:46:01Z
UYARI: bu koşum alıntılanabilir değil (kirli ağaç veya commit'siz çalışma); rapora yazılamaz.

Kaynak: cas_curriculum · commit 60aad42 · veri parmak izi sha256:a84a2e1ee92baf12a534f5a69e293164e7955f0c283e5fba0405c740872e20e5 · 2026-09-08T02:46:11Z
UYARI: bu koşum alıntılanabilir değil (kirli ağaç veya commit'siz çalışma); rapora yazılamaz.
PROVENANCE: this curriculum set was authored inside the project (machine-expanded from hand-written templates). It is not an independent test set and no row is copied from a textbook. Expected answers come from the curriculum rule, not from the CAS; the compiler compares the engine against them. Any number measured here must be reported with this sentence.

Kaynak: nlu_commands · commit 60aad42 · veri parmak izi sha256:bc4a60ac1665c0044f751a03cf56406ca3b61c0fac54cf034896c39b31d81a74 · 2026-09-08T02:46:12Z
UYARI: bu koşum alıntılanabilir değil (kirli ağaç veya commit'siz çalışma); rapora yazılamaz.

Kaynak: ocr_test · commit 60aad42 · veri parmak izi sha256:6317985ce7728e72f609ff63eb436dea1bd68af73c87f71a51e44630295133bb · 2026-09-08T02:46:15Z
UYARI: bu koşum alıntılanabilir değil (kirli ağaç veya commit'siz çalışma); rapora yazılamaz.
UYARI: bu sayı stub (taklit) model ile üretilmiştir; bağlantı denetimidir, sözleşme ölçütü için kanıt değildir.
WIRING CHECK, NOT A RESULT: the stub recogniser guesses from a fixed curriculum table and every sample is blinded before it reaches the backend, so this CER is the score of a lookup and NOT evidence for the < 10 % CER target (that needs MathWriting and the training gate). The promoted latest.json therefore publishes null for it.

Kaynak: asr_test · commit 60aad42 · veri parmak izi sha256:83edf842c3dbde2b1d96a7704c5ec91fea90cf66469e0d7d2061466b2f0c0579 · 2026-09-08T02:46:15Z
UYARI: bu koşum alıntılanabilir değil (kirli ağaç veya commit'siz çalışma); rapora yazılamaz.
UYARI: bu sayı stub (taklit) model ile üretilmiştir; bağlantı denetimidir, sözleşme ölçütü için kanıt değildir.
WIRING CHECK, NOT A RESULT: the stub transcriber picks a sentence by hashing the waveform (the runner hands it the waveform and nothing else), so its WER is the WER of a random draw from the sentence pool. It proves the speaker-disjoint split, the two WER columns and the command-intent path are wired; it is NOT evidence for the < 10 % WER target, and the promoted latest.json therefore publishes null for it.

Kaynak: book_fetch · commit 60aad42 · veri parmak izi sha256:168e60b283d99255e758a03fa59d61d40457a44636e8b7daca34e87b098076d9 · 2026-09-08T02:46:16Z
UYARI: bu koşum alıntılanabilir değil (kirli ağaç veya commit'siz çalışma); rapora yazılamaz.

Kaynak: latency_local · commit 60aad42 · veri parmak izi sha256:bc4a60ac1665c0044f751a03cf56406ca3b61c0fac54cf034896c39b31d81a74 · 2026-09-08T02:46:17Z
UYARI: bu koşum alıntılanabilir değil (kirli ağaç veya commit'siz çalışma); rapora yazılamaz.
UYARI: bu sayı stub (taklit) model ile üretilmiştir; bağlantı denetimidir, sözleşme ölçütü için kanıt değildir.

Tanımlar: docs/ml/degerlendirme.md#2-metrikler
