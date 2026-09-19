# ADR-0001: Web çatısı olarak Flask yerine FastAPI

- **Durum:** kabul edildi
- **Tarih:** 2026-09-07
- **Karar vericiler:** Ç. Şahin, M. M. Turhan, Ş. Güler
- **İlgili İP:** İP-2 (arka uç), İP-3 (model uç noktaları), İP-4 (CAS uç noktaları)

## Bağlam ve problem

Öneri metni A.1.2 arka uç için "Python, Node.js ve Flask" der. Platformun <3 s hedefi, gerçek zamanlı iz/ses aktarımı (İP-2 risk planı WebSocket'i şart koşar), tipli istek/yanıt sözleşmeleri (İP-2 "API Dokümantasyonu" ara çıktısı) ve üç kişilik ekibin bakım yükü çatı seçimini belirler. `fastapi` ve `uvicorn` geliştirme ortamında zaten kuruludur; `pydantic` v2 hem API şemalarında hem ML yapılandırmasında kullanılır.

## Karar etkenleri

- Yerel (native) asenkron WebSocket (Starlette) — Flask için Flask-SocketIO + eventlet/gevent katmanı gerekir.
- pydantic v2 ile otomatik OpenAPI → `openapi-typescript` ile tipli TypeScript istemcisi; İP-2 API dokümantasyonu üretilir, elle yazılmaz.
- `Depends` ile rol/oturum bağımlılıkları test edilebilir.
- Ekip kapasitesi: tek çatı, tek şema dili.

## Değerlendirilen seçenekler

1. **Flask 3 + Flask-SocketIO + marshmallow** — öneri metnine birebir; WebSocket ve OpenAPI için ek katman, senkron model.
2. **FastAPI + uvicorn** — tipli, asenkron, OpenAPI dahili; öneri metninden sapma ama risk planında açıkça izinli.
3. **Django + Channels + DRF** — ağır; üç kişilik ekip için fazla.

## Karar

FastAPI 0.11x + uvicorn; `aicls.api.app.create_app()` fabrikası; REST `/api/v1` + WebSocket `/ws/v1/ink`, `/ws/v1/audio`; hata modeli RFC 9457; OpenAPI `docs/api/openapi.json` git'te ve CI'da farksız olmalı.

## Sonuçlar

- Olumlu: akışlı girdi yolu, tipli sözleşme, otomatik belge, test edilebilir bağımlılıklar.
- Olumsuz: öneri metnindeki teknoloji adı değişti → gelişme raporunda açık gerekçe; asenkron kodda SymPy gibi CPU-bağımlı işler ayrı süreçte koşmalı (ADR-0016).

## Öneri metniyle uyum

A.1.2 Flask'ı adlandırır. İP-2 Risk Yönetimi B planı: "Bir teknolojide aşılamayan bir sorunla karşılaşılırsa, projenin temel hedeflerini etkilemeyecek şekilde o modül için alternatif bir teknolojiye (örn: **Flask yerine FastAPI**) geçiş değerlendirilecektir." Aynı risk planı "iyi tanımlanmış RESTful API'ler veya WebSocket" ve "WebSocket gibi teknolojilerle kalıcı bağlantılar" ister; FastAPI bunları yerleşik sunar. Gelişme raporunda bu cümle alıntılanarak sapma açıklanır.

## Doğrulama

`test_api_problem_json_shape_on_parse_error_with_position`, `test_ws_ink_stream_partial_then_final`, CI `openapi-drift` adımı.

## İlgili

ADR-0014, ADR-0016, [mimari/api.md](../mimari/api.md).
