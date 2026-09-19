# tests/fixtures/ — küçük, sentetik, deterministik fikstürler

Kural: toplam < 2 MB; her şey kodla ve sabit tohumla (1234) üretilir; kişisel veri yoktur ([ADR-0010](../../docs/adr/0010-test-stratejisi.md), [docs/gelistirme/test-stratejisi.md §4](../../docs/gelistirme/test-stratejisi.md)).

| Alt dizin | İçerik | Üretici |
|---|---|---|
| `synthetic/ink/` | 8 InkML (Bezier izleri, bilinen LaTeX etiketleri) | `scripts/make_synthetic_fixtures.py` |
| `synthetic/png/` | 8 PNG (matplotlib mathtext; TeX kurulumu gerekmez) | aynı |
| `synthetic/wav/` | 8 WAV (16 kHz PCM16, sinüs + gürültü, bilinen SNR/kırpılma) | aynı |
| `synthetic/manifests/` | 2 manifest (parmak izi + bölme) | aynı |
| `book/` | (isteğe bağlı) mini kitap bölümü kopyası; birincil kaynak `datasets/book/sample/` | — |

Paylaşılan fikstür fonksiyonları `tests/conftest.py` içindedir (`test_env`, `settings`, `data_root`, `seed`, `rng`, `freeze_time`, `load_script`); paket düzeyinde olanlar `tests/unit/<paket>/conftest.py` içinde tanımlanır. `poe fixtures` yeniden üretir; üretim çıktısı değişirse PR'da nedeni açıklanır.
