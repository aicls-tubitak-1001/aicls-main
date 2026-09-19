# Deneme Alanı (experiments/)

Ekip literatür-önce çalışır; deneme alanı okumaya paralel, küçük, kayıtlı ve düşük riskli denemelerin yeridir. Amaç eleştiri değil katkıdır: işe yarayan sonuç "gerekçeli öneri" olarak ekibe sunulur, normal PR ile `src/`'a taşınır.

## 1. Kurallar

1. Her deneme bir dizin: `experiments/YYYY-MM-DD-<slug>/` (`python scripts/new_experiment.py <slug>` şablondan açar).
2. Her dizinde `NOTE.md` (1 sayfa, aşağıdaki şablon) ve isteğe bağlı `run.py`, `config.yaml`, `results/` (küçük; büyük çıktılar `AICLS_DATA_ROOT/runs/`).
3. Deneme `aicls` paketini import edebilir; `src/` asla `experiments/` import edemez (`import-linter` `experiments-isolated`).
4. Ölçüm yapan her deneme `aicls.eval` metriklerini ve paylaşılan `datasets/` setlerini kullanır; kendi metrik tanımı yazmaz — böylece not ile rapor aynı dili konuşur.
5. Onay kapıları denemelerde de geçerlidir: veri/ağırlık indirme, CUDA, gerçek veriyle eğitim yok. Sentetik veri, stub modeller, CPU ve küçük yapılandırmalar serbest.
6. Denemeler `exp/<slug>` dalında; `NOTE.md` `docs/deneyler/` dizinine (planlı) otomatik listelenir (`mkdocs` makrosu `experiments/*/NOTE.md` başlıklarını toplar).
7. Kod kalitesi: ruff/format zorunlu, test zorunlu değil; `src/`'a taşınırken test yazılır.
8. Bir deneme bir kararı etkiliyorsa (örn. çevrimiçi/çevrimdışı el yazısı) ADR'ye `Girdi:` satırıyla bağlanır.

## 2. Not şablonu (`experiments/_template/NOTE.md`)

```markdown
# <Deneme başlığı>

- **Tarih:** YYYY-MM-DD   **Kişi:** <ad>   **Dal:** exp/<slug>   **Git SHA:** <7 hane>
- **Soru:** Tek cümle. Hangi kararı/okumayı besliyor? (ADR-00NN, okuma başlığı)
- **Kurulum:** veri (sentetik/fikstür/`datasets/...` + parmak izi), model/stub, yapılandırma (`config.yaml`), donanım (CPU), süre.
- **Ölçüm:** `aicls.eval` metrikleri; n; güven aralığı varsa.

## Sonuç
| Koşul | Metrik | Değer (GA) | n |
|---|---|---|---|

## Yorum
3–5 cümle: ne öğrendik, neyi göremedik, sınırlamalar.

## Karar / öneri
- [ ] `src/`'a taşı (PR: ...)  - [ ] ADR güncelle (...)  - [ ] Bırak (neden)
- Okuma programına dönüş: hangi başlık, hangi kaynak.
```

## 3. İlk planlı denemeler

| Dizin | Soru | Beslediği karar |
|---|---|---|
| `2026-09-online-vs-offline-ink/` | Sentetik iz + iki tablet üzerinde iz (InkML) ve raster (PNG) girdileri: tarayıcıdan aktarım boyutu, `StubRecognizer` ile tur süresi, öğretmen deneyimi | ADR-0013 (6. hafta) |
| `2026-10-nl2cmd-coverage/` | Okuma senaryoları kütüphanesindeki cümlelerin envanter v1 ile kapsama oranı; en sık `NoMatch` nedenleri | envanter v1.1 |
| `2026-10-sandbox-spawn-cost/` | Windows'ta ısıtılmış havuz vs. istek başına spawn: SymPy import süresi, zaman aşımında toparlanma, p95 | ADR-0016 |
| `2026-11-wav-in-browser/` | `wavEncoder.ts` ile Chrome/Firefox/WebKit'te 16 kHz PCM16 üretimi; boyut ve QC tutarlılığı | ADR-0017 |
| `2026-11-whisper-zero-shot-tr-math/` (kapılı: ağırlık) | Ekip kayıtlarında (ayrı rıza) sıfır-atış Türkçe matematik komutu WER'i ve en sık hatalar | envanter ASR varyant tablosu, İP-3 taban çizgisi |
| `2027-01-tts-augmentation-pilot/` (kapılı: TTS motoru) | TTS ile üretilen cümlelerin ince ayara katkısı (sentetik/gerçek oranı) | `data.augment.tts_ratio` |

## 4. Okuma programıyla bağ

Dört okuma başlığı (el yazısı matematik tanıma; Türkçe ASR ve alan uyarlama; doğal dil → sembolik hesap; teknoloji destekli matematik öğretimi) için notlar `docs/literatur/<baslik>.md` (planlı) altında birikir; her deneme notu ilgili okuma sayfasına bağlantı verir. Bu birikim gelişme raporunun ve makalelerin "ilgili çalışmalar" bölümüne dönüşür.
