# reports/ — değerlendirme çıktıları ve rapor kaynağı

Her `aicls eval <kind>` koşusu `reports/metrics/<kind>_<split>_<gitsha>.json` yazar; PR incelemesiyle terfi edilen dosyalar `reports/metrics/latest.json` altında birleştirilir ve MkDocs makroları (`{{ metrics.<kind>.<metric> }}`) sayıları oradan alır. Elle yazılmış sayı yoktur ([ADR-0009](../docs/adr/0009-dokumantasyon-mkdocs-turkce.md), [ADR-0015](../docs/adr/0015-metrik-tanimlari-dondurulmus.md), [docs/raporlama/ip-eslestirme.md §2](../docs/raporlama/ip-eslestirme.md)).

## `metrics/latest.json` şeması

Terfi edilmiş biçim:

```json
{
  "parser": {
    "success_rate": 0.996,
    "n": 2143,
    "ci95": [0.993, 0.998],
    "git_sha": "abc1234",
    "dataset_fingerprint": "sha256:...",
    "normalizer": "latex_v1",
    "dirty": false,
    "citable": true,
    "generated_at": "2027-03-01T10:00:00Z",
    "source_file": "parser_valid_abc1234.json"
  },
  "cas": { "accuracy": 0.993, "n": 1000, "...": "..." },
  "nlu": { "intent_accuracy": 0.97, "slot_f1": 0.95, "...": "..." },
  "ocr": { "cer": 0.083, "exprate": 0.61, "measurement": "measured", "...": "..." },
  "asr": { "wer_tr_v1": 0.091, "wer_raw": 0.14, "measurement": "measured", "...": "..." },
  "book": { "coverage": 1.0, "fetch_accuracy": 0.97, "...": "..." },
  "latency": { "mean_ms": 1240, "p50_ms": 980, "p95_ms": 2600, "...": "..." }
}
```

- Üst anahtar = değerlendirme türü (`kind`); alt anahtarlar metrik adları [docs/ml/degerlendirme.md](../docs/ml/degerlendirme.md) ile birebir.
- Her tür için zorunlu meta alanlar: `git_sha`, `dataset_fingerprint`, `normalizer`, `dirty`, `citable`, `generated_at`, `source_file`, `n`.
- `poe report-check` (CI) her terfi edilmiş dosyanın SHA'sının HEAD'in atası olduğunu, `dirty=false`, parmak izinin manifestle eşleştiğini doğrular.
- `poe metrics-promote -- <dosya>` bu dosyayı günceller; doğrudan elle düzenlenmez.

### Stub arkalı koşumlar sayı yayımlamaz

`ocr` ve `asr` için ölçütün öznesi öğrenilmiş bir modeldir. Arkada stub varsa ölçülecek model yoktur, dolayısıyla `promote()` o girdinin **her ölçütünü `null`** yapar ve nedenini yazar. Bugünkü dosya (2026-09-08, `60aad42`) tam olarak böyledir:

```json
"ocr": {
  "cer": null, "exprate": null, "ter": null, "ci95": {},
  "stub": true, "measurement": "wiring_check",
  "wiring_check": {
    "reason": "WIRING CHECK, NOT A MEASUREMENT: ...",
    "metrics": { "cer": 0.9679, "exprate": 0.0156, "...": "..." }
  }
}
```

- Ölçüt **anahtarları** kalır: MkDocs makroları `on_undefined: strict` ile çalışır, `{{ metrics.ocr.cer }}` çözülür ve sayfada boş/`None` görünür — alıntılanacak bir sayı olmadığı okuyucuya görünür.
- Stub'ın gerçekte aldığı değerler kaybolmaz: `wiring_check.metrics` altında ve `ocr_test_<sha>.json` koşum dosyasında dururlar.
- `latency` de `stub: true` taşır ama `measurement: "measured"`: orada `stub`, "tarayıcı değil sunucu tarafı" demektir ve ölçülen NLU → CAS hattı gerçek koddur.
- 2026-09-08 öncesinde bu dosya `ocr.cer = 0.0` / `ocr.exprate = 1.0` yayımlıyordu. Stub tanıyıcı kendisine verilen etiketi geri verdiği, koşucu da örneği etiketiyle birlikte verdiği için ortaya çıkan bir orakl skorudur; koşucu artık körleştiriyor (`aicls.eval.runners.blind_sample`).

`reports/tmp/` git dışıdır (geçici çıktılar).
