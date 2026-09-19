# experiments/ — deneme alanı

Kurallar ve not şablonu: [docs/deneme-alani.md](../docs/deneme-alani.md). Özet:

1. Yeni deneme: `poe new-exp -- <slug>` → `experiments/YYYY-MM-DD-<slug>/` (`_template/` kopyası).
2. Her dizinde `NOTE.md` (1 sayfa), isteğe bağlı `run.py`, `config.yaml`, küçük `results/`; büyük çıktılar `AICLS_DATA_ROOT/runs/`.
3. `aicls` paketi import edilebilir; `src/` asla `experiments/` import etmez (`import-linter` `experiments-isolated`).
4. Ölçüm `aicls.eval` metrikleri ve `datasets/` setleriyle yapılır; kendi metrik tanımı yazılmaz.
5. Onay kapıları burada da geçerlidir: veri/ağırlık indirme, CUDA, gerçek veriyle eğitim yok.
6. Dal `exp/<slug>`; ruff/format zorunlu, test zorunlu değil; `src/`'a taşınırken test yazılır.

Planlı ilk denemeler için tablo: docs/deneme-alani.md §3.
