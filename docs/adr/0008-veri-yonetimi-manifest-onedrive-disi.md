# ADR-0008: Veri yönetimi — SHA-256 manifestleri, OneDrive dışı veri kökü, git'te yalnızca küçük değerlendirme setleri

- **Durum:** kabul edildi
- **Tarih:** 2026-09-07
- **Karar vericiler:** Ç. Şahin, M. M. Turhan
- **İlgili İP:** İP-3 (veri seti), İP-4 (test seti), İP-8 (örnek bölüm)

## Bağlam ve problem

MathWriting ~20 GB, ses 50+ saat; depo OneDrive altında; git'e giren değerlendirme setleri (müfredat, geçerli LaTeX, NLU, kitap) sürümlenmeli ve raporlarda parmak iziyle anılmalı. Hakem, ilk tasarımdaki `.gitignore data/` satırının git'e girmesi gereken setleri dışladığını tespit etti.

## Değerlendirilen seçenekler

1. **DVC / Git LFS** — ek araç ve sunucu; ekip küçük; LFS OneDrive'da kötü.
2. **Manifest tabanlı: büyük veri `AICLS_DATA_ROOT` (OneDrive dışı) + `manifest.json` (dosya/boyut/sha256/lisans/kaynak/tarih), küçük setler `datasets/` altında git'te, JSONL parmak izleri raporlarda** — araçsız, denetlenebilir; DVC'ye sonradan geçiş manifestleri bozmaz.

## Karar

Seçenek 2. `datasets/` (git; `curriculum/`, `latex/`, `nlu/`, `book/`, `manifests/`) ve `AICLS_DATA_ROOT` (`D:\aicls-data`, git/OneDrive dışı; `scripts/check_env.py` OneDrive altındaysa hata). `/data/` deposu **yoktur**. `aicls data manifest|verify`; parça manifestleri; ses dışa aktarım manifesti yalnızca takma ad; konuşmacı-ayrık, tabakalı, önceki test konuşmacılarını donduran split dosyası; `FROZEN.sha256`; `report-YYYY-MM` etiketi tüm setleri dondurur; sürüm adları `<ad>-<YYYY.MM>`. Katalog `src/aicls/data/catalog.yaml` (URL, lisans, boyut, kapı adı).

## Sonuçlar

- Olumlu: her rapor sayısının veri parmak izi var; OneDrive çakışması yok; onay taleplerinde boyut/lisans hazır.
- Olumsuz: manifest üretimi disiplini gerektirir (`tests/data` işaretli testler doğrulanmamış manifestle koşmaz); büyük dosya paylaşımı ekip içinde manuel (iş istasyonu ortak disk).

## Öneri metniyle uyum

A.2.1 MathWriting (Gervais ve ark., 2024); A.2.3 üç üniversiteden 50 saat, 200+ saate artırma; İP-3 ara çıktısı "toplanan ve etiketlenen ses dosyaları"; İP-4 "test veri seti"; B.1.2 gizlilik.

## Doğrulama

`test_data_manifest_verify_detects_tampered_sha`, `test_check_env_warns_on_onedrive_path`, `test_split_freezes_previous_test_speakers`, `test_export_manifest_has_no_identity_fields`, `test_frozen_curriculum_rows_unchanged`.

## İlgili

[ml/veri-yonetimi.md](../ml/veri-yonetimi.md), ADR-0011, ADR-0012.
