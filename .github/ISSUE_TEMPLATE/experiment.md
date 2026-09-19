---
name: Deneme (experiment)
about: experiments/ altında küçük, kayıtlı bir deneme önerisi (docs/deneme-alani.md)
title: "exp: <slug>"
labels: ["experiment"]
assignees: []
---

## Soru

<!-- Tek cümle. Hangi kararı/okumayı besliyor? (ADR-00NN, okuma başlığı) -->

## Kurulum

- Veri: sentetik / fikstür / `datasets/...` (parmak izi)
- Model / stub:
- Yapılandırma: `config.yaml`
- Donanım: CPU (GPU yalnızca kapı 3 sonrası)
- Tahmini süre:

## Ölçüm

- `aicls.eval` metrikleri:
- n / güven aralığı:

## Kapılar

- [ ] Veri seti indirme gerekmiyor
- [ ] Ağırlık indirme gerekmiyor
- [ ] Gerçek veriyle eğitim gerekmiyor

## Çıktı

- Dizin: `experiments/YYYY-MM-DD-<slug>/` (`poe new-exp -- <slug>`)
- Dal: `exp/<slug>`
- NOTE.md 1 sayfa; sonuç "gerekçeli öneri" olarak ekibe
