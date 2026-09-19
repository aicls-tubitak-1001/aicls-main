# ADR-0014: Gecikme mimarisi — WebSocket akışı ile girdiyle örtüşen tanıma, Server-Timing, istemci tarafı gecikme telemetrisi, aşama bütçeleri

- **Durum:** kabul edildi
- **Tarih:** 2026-09-07
- **Karar vericiler:** Ş. Güler, Ç. Şahin
- **İlgili İP:** İP-2 (< 3 s, WebSocket), İP-3 (< 3 s model yanıtı), İP-4 (komut → ekran < 3 s), İP-9 (performans testleri)

## Bağlam ve problem

Sözleşme ölçütü "kullanıcının bir komut girmesi ile sonucun web arayüzünde görüntülenmesi arasındaki süre ... ortalama 3 saniyenin altında". Whisper-sınıfı çıkarım + ışın aramalı OCR + SymPy, girdi bittikten sonra sırayla başlarsa 3 s'ye sığmaz. Sunucu tarafı kronometre sözleşmenin ölçtüğü şeyi (ekranda görüntülenme) ölçmez; sözleşme istatistiği **ortalama**dır.

## Değerlendirilen seçenekler

1. **Yalnızca REST, sunucu zamanlaması** — basit; ölçüm eksik, örtüşme yok.
2. **Socket.IO / WebRTC** — ek sunucu kütüphanesi ya da aşırı karmaşıklık.
3. **REST (komut/sonuç) + düz WebSocket (`/ws/v1/ink`, `/ws/v1/audio`) tipli zarflarla; kalem-boşta 400 ms sonra kısmi tanıma, enerji tabanlı ses bitişi; `Server-Timing` aşama başlıkları; istemci `performance.now()` ile gönderim→render süresini `POST /telemetry/latency`'e yazar; `latency_samples` tablosu; `tests/perf/test_latency_budget.py` aşama bütçeleri** — sözleşmeye uygun ölçüm ve gerçekçi mühendislik yolu.

## Karar

Seçenek 3. Bilet tabanlı WS kimlik doğrulama (`POST /auth/ws-ticket`); bağlantı başına tanıma semaforu ve `slow_down` geri basıncı; `PipelineService` `StageTimer` ile `nlu/parse/cas/render` aşamaları; her yanıtta `Server-Timing`; `latency_samples(server_ms, stage_json, client_total_ms, render_ms, modality)`; rapor `mean`, `p50`, `p95`, `n` modalite bazında (`GET /admin/metrics/latency`, `aicls eval latency --from-db`); CI'da stub modellerle sentetik bütçe testi (metin yolu p50 ≤ 300 ms, CAS ≤ 150 ms, kitap çözümleme ≤ 20 ms); gerçek modellerle bütçe OCR ≤ 800 ms, ASR ≤ 1200 ms (5 s ses), toplam ortalama < 3000 ms. CAS önbelleği (İP-4 risk planı), kitap LRU (İP-8 risk planı), model ısıtma, sandbox ısıtılmış havuz (ADR-0016) bütçenin parçalarıdır.

## Sonuçlar

- Olumlu: metrik sözleşmedeki tanımla ölçülür; regresyonlar CI'da; tanıma girdiyle örtüşür.
- Olumsuz: istemci telemetrisi ek uç ve tablo; WS protokolü bakımı; Whisper akışlı kısmi sonuç üretmez (yalnızca > 5 s kayıtlarda parça bazlı kısmi).

## Öneri metniyle uyum

İP-2 risk planı: "WebSocket gibi teknolojilerle kalıcı bağlantılar kurulacaktır. Sunucu tarafında kod optimizasyonu yapılacak ... yük testleri (load testing)". İP-4 ölçütü komut→görüntülenme süresi; İP-3 ölçütü "gerçek zamanlı olarak kabul edilebilir bir hızda (ortalama < 3 saniye)"; İP-9 "performans testlerinde ... ortalama yanıt süresinin 3 saniyenin altında".

## Doğrulama

`test_api_server_timing_header_present`, `test_api_telemetry_latency_stores_client_and_server_ms`, `test_ws_ink_stream_partial_then_final`, `test_ws_audio_backpressure`, `test_latency_budget_text_path_p50_under_300ms`, `test_latency_stats_mean_p50_p95`.

## İlgili

[mimari/api.md §4](../mimari/api.md#4-websocket-akışları), [ml/degerlendirme.md](../ml/degerlendirme.md), ADR-0001, ADR-0016.
