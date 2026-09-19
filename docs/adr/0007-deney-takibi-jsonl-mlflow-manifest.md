# ADR-0007: Deney takibi — JSONL izleyici varsayılan, MLflow opsiyonel, zorunlu koşu manifesti

- **Durum:** kabul edildi
- **Tarih:** 2026-09-07
- **Karar vericiler:** Ç. Şahin
- **İlgili İP:** İP-3 (teknik raporlar), tüm metrik raporlaması

## Bağlam ve problem

Gelişme raporundaki her sayı bir koşuya, bir git durumuna ve bir veri parmak izine bağlanmalı; koşular çevrimdışı üniversite iş istasyonunda, KVKK'ya uygun (veri kurum dışına çıkmaz) izlenmeli. Hakem değerlendirmesi MLflow'un kapılı ML katmanına konmasını ve üç örtüşen izleme çıktısını (TensorBoard + MLflow + manifest) eleştirdi.

## Değerlendirilen seçenekler

1. **Weights & Biases** — bulut hesabı, veri kurum dışına → KVKK ve çevrimdışı gereksinimiyle uyumsuz.
2. **Yalnızca MLflow** — saf Python ama CI ve ilk haftalarda gereksiz bağımlılık; kaldırılırsa geçmiş kaybolur.
3. **`Tracker` protokolü: `JsonlTracker` (bağımlılıksız, varsayılan) + `MlflowTracker` (opsiyonel, `requirements/dev.txt`, yerel dosya deposu) + her koşuda zorunlu `run_manifest.json`; TensorBoard yok** — manifest birincil alıntı anahtarı, izleyici değiştirilebilir.

## Karar

Seçenek 3. `aicls.ml.tracking.{base,jsonl,mlflow_}`; `run_manifest.json` ilk optimizer adımından önce yazılır (git SHA, kirli bayrağı, diff hash, config hash, veri parmak izleri, pip freeze hash, torch/CUDA/cuDNN, GPU/sürücü, hostname, adım hızı); kirli ağaç `train.allow_dirty=false` iken reddedilir, `--allow-dirty` ile `citable=false`. Rapor alıntısı: `run_manifest.git_sha + config_hash` (+ varsa MLflow `run_id`). `reports/metrics/*.json` git'te; `poe metrics-promote`, `poe report-check` (SHA atası, kirli değil, parmak izi tutuyor), `report-YYYY-MM` etiketi, `scripts/reproduce_report.ps1`.

## Sonuçlar

- Olumlu: CI'da izleyici testi bağımlılıksız; kayıt kurum içinde; her rapor sayısı temiz klondan yeniden üretilebilir.
- Olumsuz: MLflow arayüzü kurulmadan koşu karşılaştırması JSONL'den elle; iki izleyici arasında alan adları `Tracker` protokolüyle sabitlenir.

## Öneri metniyle uyum

İP-3 ara çıktısı "eğitim sürecini ... detaylandıran teknik rapor" ve B.1.2 "tüm verilerin gizliliği en üst düzeyde" — çevrimdışı, kurum içi izleme. Öneri metni izleme aracı adlandırmaz.

## Doğrulama

`test_jsonl_tracker_schema`, `test_run_manifest_captures_git_sha_and_dirty_flag`, `test_run_manifest_refuses_dirty_tree_without_allow_dirty`, `test_report_check_rejects_non_ancestor_sha`, `test_report_check_rejects_dirty_run`.

## İlgili

[ml/egitim-altyapisi.md §6–7](../ml/egitim-altyapisi.md), [ml/degerlendirme.md §4](../ml/degerlendirme.md), ADR-0015.
