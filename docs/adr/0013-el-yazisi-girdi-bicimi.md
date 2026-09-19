# ADR-0013: El yazısı girdi biçimi — çevrimiçi iz (InkML) birincil aday, çevrimdışı görüntü (PNG) desteklenir; karar 6. hafta deneyiyle

- **Durum:** önerildi (karar tarihi: 6. hafta, ~2026-10-30; girdi `experiments/2026-09-online-vs-offline-ink/NOTE.md`)
- **Tarih:** 2026-09-07
- **Karar vericiler:** Ç. Şahin, M. M. Turhan, Ş. Güler
- **İlgili İP:** İP-2 (Canvas API, aktarım biçimi), İP-3 (OCR modeli)

## Bağlam ve problem

Öneri A.1.2 el yazısının "PNG, JPEG veya INKML formatında" gönderilebileceğini söyler; A.2.1 raster ön işleme adımlarını (Gauss, Laplace, gri ton, [0,1]) anlatır. MathWriting iz-doğaldır (InkML) ve raster'a deterministik biçimde çizilebilir; tarayıcı Canvas'ından izler zaten alınır; etkileşimli tahta/tablet hedefi iz sağlar. Ekip kararı (memory: bekleyen kararlar #1) iki tabletle 6. haftada karşılaştırmaktır. Bekleyen karar OCR işini durdurmamalıdır.

## Değerlendirilen seçenekler

1. **Yalnızca raster (Pix2Tex tarzı)** — öneri metnindeki ön işleme birebir; iz zamanlaması kaybolur; MathWriting'in iz bilgisi kullanılmaz.
2. **Yalnızca iz** — MathWriting ile doğal; yalnızca görüntü veren cihaz/akış (fotoğraf, PDF) desteklenmez.
3. **İkisi de tek `Recognizer` arayüzü arkasında; `model.family: seq2seq_stroke | seq2seq_image` yapılandırma anahtarı; aynı parçalar ve etiketler; raster ön işleme ablasyonlu; ilk model kararı 6. hafta deneyiyle** — karar yapılandırma anahtarı olur, yeniden yazım gerekmez.

## Karar (öneri)

Seçenek 3. `StrokeBuffer.toInkML()`/`toCanvasJson()` ve `toPNG()` ikisi de arayüzde; `/ws/v1/ink` iz gönderir; `POST /ocr/recognize` her iki biçimi kabul eder; `aicls.data.MathWritingDataset(mode="online"|"offline")` aynı parçalardan; `PreprocessConfig.steps` ablasyon listesi. 6. hafta deneyi ölçer: aktarım boyutu ve tur süresi (stub ile), iki tablette iz kalitesi (örnekleme, basınç), öğretmen tercihi; kapı açılınca CER karşılaştırması (`ablation_preproc.yaml`, `online_stroke_tfm.yaml` vs `offline_resnet34_tfm.yaml`) eklenir. Karar ADR'yi `kabul edildi` yapar ve birincil aileyi `models/registry.yaml` tohumuna yazar.

## Sonuçlar

- Olumlu: OCR altyapısı beklemez; ablasyon dürüst; her iki cihaz sınıfı desteklenir.
- Olumsuz: iki model ailesi bakımı; raporda iki tablo. Karar sonrası ikinci aile "destekli" olarak kalır, ana geliştirme birincil ailede.

## Öneri metniyle uyum

A.1.2: "Alınan el yazısı verileri (PNG, JPEG veya INKML formatında) sunucuya gönderilecektir." A.2.1: raster ön işleme adımları (ablasyonlu yapılandırma anahtarı olarak korunur), encoder-decoder, MathWriting, CER. İP-3 görevi "Görüntü ön işleme adımlarının ... uygulanması" çevrimdışı yolda birebir uygulanır.

## Doğrulama

`test_preprocess_steps_are_ablatable_and_ordered`, `test_render_ink_shape_and_range_0_1`, `test_inkml_read_write_roundtrip_synthetic`, `InkCanvas.test.tsx`, `test_registry_builds_every_model_at_tiny_size` (her iki aile).

## İlgili

[deneme-alani.md §3](../deneme-alani.md), ADR-0006, ADR-0014.
