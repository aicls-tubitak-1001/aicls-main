# datasets/ — git'te tutulan küçük, sürümlü test setleri

Bu dizin **yalnızca küçük, elle incelenmiş, sürümlü** veri setlerini içerir (dosya başına < 5 MB, toplam birkaç MB). Büyük veri (MathWriting, ses kayıtları, model ağırlıkları) `AICLS_DATA_ROOT` altında, OneDrive dışında durur ve yalnızca SHA-256 manifestleriyle temsil edilir ([ADR-0008](../docs/adr/0008-veri-yonetimi-manifest-onedrive-disi.md), [docs/ml/veri-yonetimi.md](../docs/ml/veri-yonetimi.md)).

## Planlanan düzen

| Dizin | İçerik | Sahip |
|---|---|---|
| `curriculum/` | müfredat kartları (`cards/<konu>.yaml`), `problems.jsonl` (≥ 1000 problem), `FROZEN.sha256` | mmturhan (İP-4) |
| `latex/` | `valid_inputs.jsonl` (ayrıştırıcı başarı seti), `unsupported.jsonl` | csahin / mmturhan |
| `nlu/` | `commands_test.jsonl`, `asr_noisy_tr.jsonl`, `reading_scenarios.yaml` | csahin (İP-4 / İP-3) |
| `book/` | `sample/*.md` örnek bölüm, `expected_units.json`, `fetch_test.jsonl` | mmturhan (İP-8) |
| `manifests/` | `mathwriting.json`, `audio_commands.json`, `audio_commands_splits.json` (yalnızca parmak izi ve bölme listesi) | csahin |

## Kurallar

1. Her JSONL satırı `id` alanı taşır; testler bu kimliği pytest kimliği olarak kullanır.
2. Bir set dondurulduğunda (`FROZEN.sha256`) satır silinmez/değişmez; yeni satır yeni sürümdür (`test_frozen_curriculum_rows_unchanged`).
3. Kişisel veri yok: ses/kimlik bilgisi bu dizine girmez; manifestler yalnızca hash, süre ve bölme bilgisi taşır.
4. Lisans: her alt dizinde kaynak ve lisans notu (`LICENSE.md` veya README bölümü); MathWriting'in kendisi burada değildir. **MathWriting CC BY-NC-SA 4.0'dır** (ticari kullanım yasak, türev aynı lisansla, atıf zorunlu); ondan türetilmiş hiçbir etiket veya örnek bu dizine bu not olmadan girmemelidir ([veri-yonetimi.md §1](../docs/ml/veri-yonetimi.md)).
5. Boyut kapısı: `scripts/check_no_large_files.py` (`datasets/` için 5 MB).
