# API Tasarımı

REST (JSON) + WebSocket. Uygulama `aicls.api.app.create_app()`; OpenAPI belgesi `scripts/export_openapi.py` ile `docs/api/openapi.json` ve `web/src/api/generated/schema.d.ts` olarak üretilir, CI'da bayat kalırsa iş başarısız olur (`openapi-drift`). Bu belge İP-2'nin "API Dokümantasyonu" ara çıktısının insan tarafından okunan yüzüdür; makine tarafından okunan yüzü OpenAPI dosyasıdır.

## 1. Kurallar

| Konu | Karar |
|---|---|
| Taban yol | `/api/v1` (URI sürümleme). Kırıcı değişiklik → `/api/v2` yönlendiricisi, eski sürüm bir küçük (minor) sürüm boyunca korunur. `info.version` = paket sürümü. |
| Gövde | JSON, `snake_case` alanlar, kimlikler ULID (26 karakter), zamanlar UTC ISO-8601, sayfalama `{items, next_cursor}`. |
| Kimlik doğrulama | `Authorization: Bearer <access>` (15 dk, HS256). Yenileme belirteci `aicls_refresh` httpOnly, Secure, SameSite=Strict çerezi (7 gün, kullanımda döndürülür, aile iptali). WebSocket için `POST /auth/ws-ticket` ile 60 s'lik tek kullanımlık bilet (URL'de JWT yok). |
| Roller | `student`, `instructor`, `contributor`, `admin` (admin hepsini kapsar). `require_role(*roles)` bağımlılığı; çalışma alanı erişimi depo katmanında `(workspace_id, user_id)` ile denetlenir, istemci kimliğine güvenilmez. |
| Profil kapılama | `AICLS_PROFILE=full` tüm yönlendiriciler; `collect` yalnızca `auth`, `users`, `contribute`, `feedback`, `health`, `admin` (QC kuyruğu). |
| Yanıt başlıkları | `X-Request-Id`; `Server-Timing: nlu;dur=12.3, parse;dur=1.1, cas;dur=41.0, render;dur=0.4, app;dur=58.2` (RFC 9211). |
| Hız sınırı | `/auth/*` 5/dk/IP ve 20/saat/hesap; `/compute/*` 60/dk/kullanıcı; yüklemeler (`/ocr`, `/asr`, `/contribute/recordings`) 20/dk/kullanıcı; WS mesaj hızı aşımında `slow_down` zarfı, ısrarda bağlantı kapatma. Bellek içi token kovası (Redis sonradan, ADR ile). |
| Yükleme sınırları | PNG/JPEG ≤ 2 MB ve ≤ 4096 px; InkML/Canvas JSON ≤ 1 MB ve ≤ 20 000 nokta; WAV ≤ 10 MB ve ≤ 30 s. Caddy `request_body max 12MB`. |

## 2. Hata modeli

RFC 9457 `application/problem+json`:

```json
{
  "type": "https://ai-cls.example/errors/PARSE_ERROR",
  "title": "LaTeX ifadesi ayrıştırılamadı",
  "status": 422,
  "detail_tr": "Konum 8: ifade tamamlanmamış, } bekleniyor",
  "code": "PARSE_ERROR",
  "instance": "01J8R2ZQ7G4V0X9K3M2N5P6Q7R",
  "errors": [{"loc": ["body", "latex"], "msg": "..."}],
  "position": 8
}
```

| `code` | HTTP | Ne zaman |
|---|---|---|
| `VALIDATION_ERROR` | 422 | pydantic doğrulaması (`errors[]`) |
| `PARSE_ERROR` | 422 | `latex2sympy.ParseError` (`position`, `expected`) |
| `UNSUPPORTED_CONSTRUCT` | 422 | dilbilgisi dışı komut (`construct`) |
| `UNSUPPORTED_PROBLEM` | 422 | SymPy çözemedi; `detail_tr` = "Bu problem tipi şu an için desteklenmemektedir." (İP-4 risk planı metni, `messages_tr.MSG_UNSUPPORTED_PROBLEM`) |
| `CAS_TIMEOUT` | 504 | sandbox zaman aşımı (2,5 s); işçi öldürülüp yeniden başlatıldı |
| `NLU_NO_INTENT` | 422 | hiçbir niyet eşleşmedi (`suggestions_tr[]`) |
| `OCR_INPUT_INVALID` / `ASR_INPUT_INVALID` / `UPLOAD_INVALID` | 400 | sihirli bayt, boyut, örnekleme hızı, nokta sayısı |
| `AUTH_INVALID` | 401 | belirteç yok/geçersiz/süresi dolmuş |
| `FORBIDDEN` | 403 | rol veya sahiplik |
| `CONSENT_REQUIRED` | 403 | güncel rıza sürümü kabul edilmemiş |
| `NOT_FOUND` | 404 | |
| `RATE_LIMITED` | 429 | `Retry-After` başlığı |
| `TRAINING_BUSY` | 503 | tek GPU başka bir eğitim işinde (`aicls.ml.queue`); CLI tarafı, API bu kodu bugün üretmez |
| `GATE_BLOCKED` | 503 | `AICLS_ETHICS_APPROVAL_REF` boşken contribute; model eğitimdeyken hazır değil |
| `INTERNAL` | 500 | günlükte `request_id` ile |

Hata **olmayan** durumlar (200): NLU açıklama isteği (`kind:"clarification"`), kayıt QC reddi (`accepted:false, qc.reasons_tr[]`).

## 3. Uç noktalar

Gösterim: `YÖNTEM yol — istek → yanıt — rol — amaç`.

### 3.1 Sağlık ve sürüm (profil: hepsi, açık)

- `GET /health/live` — → `{status:"ok"}` — açık — süreç ayakta.
- `GET /health/ready` — → `{status, db:"ok"|"error", models:{ocr:"stub"|"loaded"|"absent"|"training", asr:...}, sandbox:{workers, restarts}}` — açık — 503 hazır değilse.
- `GET /health/version` — → `{version, git_sha, models:{ocr:"ocr-0.3.0"|null, asr:...}, profile}` — açık.

### 3.2 Kimlik ve kullanıcı

- `POST /auth/register` — `RegisterIn{email, password(≥10), display_name, invite_code?}` → `UserOut{id, email, display_name, role, created_at}` — açık — `invite_code` yalnızca `instructor` rolü için; `admin` yalnızca `aicls db seed`/admin ile.
- `POST /auth/login` — `LoginIn{email, password}` → `TokenOut{access_token, token_type:"bearer", expires_in:900}` + `aicls_refresh` çerezi — açık — 10 başarısız girişte 15 dk kilit.
- `POST /auth/refresh` — çerez → `TokenOut` (yenileme döndürülür; eskisi tekrar kullanılırsa aile iptal edilir) — açık(çerez).
- `POST /auth/logout` — → 204 — herkes — yenileme ailesi iptal.
- `POST /auth/ws-ticket` — → `{ticket, expires_in:60}` — herkes.
- `GET /users/me` — → `UserOut` — herkes. `PATCH /users/me` — `UserPatch{display_name?, password?}` → `UserOut`.

### 3.3 Çalışma alanları ve geçmiş (A.1.2: kayıt, çalışma alanı, geçmiş işlemler)

- `GET /workspaces` — → `Page[WorkspaceOut{id, title, created_at, updated_at, n_operations}]` — herkes (kendi).
- `POST /workspaces` — `WorkspaceIn{title}` → `WorkspaceOut` — herkes.
- `GET | PATCH | DELETE /workspaces/{id}` — sahip veya admin.
- `GET /workspaces/{id}/operations?cursor&limit=50` — → `Page[OperationOut{id, input_kind:"text"|"latex"|"ink"|"audio", input_summary, input_latex, command, result_latex, result_kind, plot_url, error_code, timings, created_at}]` — sahip.
- `DELETE /workspaces/{id}/operations/{op_id}` — → 204 — sahip.
- `POST /workspaces/{id}/operations/{op_id}/export?format=tex|pdf` — → dosya (`text/x-tex` veya `application/pdf`; PDF için sunucuda `matplotlib` mathtext ile tek sayfa) — sahip — A.2.2 "PDF veya TeX olarak kaydedilebilecek".

### 3.4 Hesaplama (İP-4)

- `POST /compute/text` — `ComputeTextIn{text, workspace_id?, context?:{last_latex?, active_var?}, confirm_intent?}` → `ComputeOut` — herkes — tam metin yolu: NLU → CAS (veya kitap/UI). `confirm_intent` verilirse açıklama adaylarından biri seçilmiş sayılır.
- `POST /compute/latex` — `ComputeLatexIn{command: CasCommand (op ayırıcılı), workspace_id?}` → `ComputeOut` — herkes — deterministik yol; NLU yok.
- `POST /compute/parse` — `{latex}` → `ParseOut{ok, ast?, srepr?, canonical_latex?, repairs[], error?}` — herkes — arayüz canlı önizlemesi ve hata ayıklama; hesaplama yapmaz.
- `GET /compute/plots/{asset_id}.png` — → `image/png` — sahip.

`ComputeOut`:

```json
{
  "kind": "result" | "clarification" | "book" | "ui",
  "operation_id": "01J...",
  "input_latex": "x^{2}",
  "command": {"op": "diff", "expr": "x^{2}", "var": "x", "order": 1},
  "result": {"kind": "expr", "latex": "2 x", "plain": "2*x", "srepr": "Mul(Integer(2), Symbol('x'))", "steps": [], "plot_url": null, "cache_hit": false},
  "clarification": null,
  "book_units": [],
  "nlu": {"intent": "differentiate", "score": 0.93, "trace": ["normalize: ...", "pattern: ..."], "source": "rules"},
  "timings": {"nlu": 12.3, "parse": 1.1, "cas": 41.0, "render": 0.4, "total": 58.2},
  "model_versions": {"ocr": null, "asr": null}
}
```

`kind:"clarification"` durumunda `clarification:{question_tr, missing_slot, candidates:[{intent, command, latex_preview, score}]}` dolu, `result` boştur; istemci `CommandPreview` gösterir ve `confirm_intent` ile yeniden çağırır.

### 3.5 El yazısı (İP-3)

- `POST /ocr/recognize` — JSON `{ink: CanvasInkJson, workspace_id?}` **veya** multipart `image=png|jpeg` → `RecognizeOut{recognition_id, latex, confidence, alternatives:[[latex, score]], timings, model_version}` — herkes — tek atış.
- `POST /ocr/feedback` — `{recognition_id, corrected_latex}` → 204 — herkes — `recognitions.corrected_latex` (İP-3 risk planı: ek örnek toplama).

### 3.6 Ses (İP-3)

- `POST /asr/transcribe` — multipart `audio=wav` (16 kHz mono PCM16 tercih; 44,1/48 kHz sunucuda yeniden örneklenir; ≤ 30 s) → `TranscribeOut{transcript_id, text, confidence, timings, model_version}` — herkes.
- `POST /asr/command` — multipart `audio=wav` + `workspace_id?` + `context?` → `ComputeOut` — herkes — transkript → NLU → CAS; `nlu.source` = `"asr+rules"`; ses yolunda `confidence < 0.8` veya modality voice ise arayüz her zaman `CommandPreview` onayı ister.
- `POST /asr/feedback` — `{transcript_id, corrected_text}` → 204.

### 3.7 Kitap (İP-8)

- `GET /book/units/{id}` — → `ContentUnitOut` (JSON-LD belgesi + `rendered:{body_html, latex_blocks, assets[]}`) — herkes.
- `GET /book/search?q&type&limit=5` — → `[HitOut{unit: ContentUnitOut, score, matched_alias}]` — herkes.
- `POST /book/resolve` — `{text}` → `[HitOut]` — herkes — `"x tanımını getir"` yolu; NLU'suz doğrudan çözümleme.
- `GET /book/units/{id}/related?predicate` — → `[ContentUnitOut]` — herkes.
- `GET /book/chapters` — → bölüm ağacı — herkes.
- `GET /book/assets/{asset_id}` — → `image/svg+xml | image/png` — herkes.
- `POST /admin/book/load` — multipart kaynak dizini zip → `LoadReport{units, triples, aliases, tagged_ratio}` — admin — `tagged_ratio < 1.0` ise 422 ve etiketsiz başlıklar listelenir.
- `GET /book/export.jsonld`, `GET /book/export.ttl` — instructor/admin.

### 3.8 Ses verisi toplama ("Model Gelişimine Katkıda Bulun"; profil: full ve collect)

- `GET /contribute/consent/current` — → `ConsentTextOut{version, text_tr, sha256}` — açık.
- `POST /contribute/consent` — `ConsentIn{version, accepted:true, demographics:{age_band:"18-24"|"25-34"|"35-44"|"45+", gender:"f"|"m"|"na", university:"mus"|"mersin"|"amasya"|"other", accent_region?}}` → `SpeakerOut{speaker_pseudonym:"P00012", token}` — giriş yapılmış (contributor/student/instructor) veya anonim katılımcı belirteci — `AICLS_ETHICS_APPROVAL_REF` boşsa 503 `GATE_BLOCKED`.
- `GET /contribute/prompts/next?n=5` — → `[PromptOut{id, text_tr, scenario_tag, difficulty}]` — katılımcı — en az kaydedilmiş cümle dengeli.
- `POST /contribute/recordings` — multipart `audio=wav` + `prompt_id` → `RecordingOut{id, accepted, qc:{passed, checks:[{name, value, threshold, passed}], reasons_tr[]}, duration_s}` — katılımcı — 200; QC eşzamanlı (< 300 ms); reddedilen dosya yazılmaz.
- `GET /contribute/recordings/{id}` — sahibi. `DELETE /contribute/recordings/{id}` — sahibi → 204.
- `GET /contribute/me/progress` — → `{n_accepted, seconds_accepted, target_seconds}`.
- `POST /contribute/withdraw` — → 204 — katılımcı — KVKK geri çekme: dosyalar ve türev özellikler silinir, rıza denetim satırı kalır.
- `GET /contribute/stats` — → `{hours_accepted, utterances, speakers, by_university{}, by_gender{}, by_age_band{}, qc_pass_rate}` — instructor/admin.

### 3.9 Geri bildirim ve SUS (İP-9)

- `POST /feedback` — `FeedbackIn{category:"critical"|"usability"|"suggestion", text, page, screenshot?}` → `FeedbackOut{id, status:"open"}` — herkes.
- `POST /feedback/sus` — `SusIn{answers:int[10] (1-5), cohort, collected_week?}` → `{score}` — herkes — puan `aicls.eval.metrics.sus_score` ile.
- `GET /feedback?status&category&cursor` — → `Page[FeedbackOut]` — instructor/admin. `PATCH /admin/feedback/{id}` — `{status:"triaged"|"fixed"|"deferred"}`.

### 3.10 Telemetri (gecikme, sözleşmeye uygun ölçüm)

- `POST /telemetry/latency` — `LatencySampleIn{operation_id, modality:"text"|"latex"|"ink"|"audio", client_submit_ts, client_render_ts, render_ms}` → 204 — herkes — sunucu `Server-Timing` aşamalarını `operations.timings` ile birleştirip `latency_samples` tablosuna yazar: `server_ms`, `client_total_ms`, `render_ms`.

### 3.11 Yönetim (admin)

- `GET /admin/models` — → `[ModelVersionOut{component, version, path, sha256, metrics, run_id, is_active, promoted_at}]`.
- `POST /admin/models/activate` — `{component:"ocr"|"asr", version}` → 204 — `ModelRegistry.reload()`; denetim günlüğüne yazılır.
- `GET /admin/eval-runs?kind&since` — → `[EvalRunOut]`; `GET /admin/eval-runs/{id}`.
- `GET /admin/metrics/latency?since&modality` — → `{mean_ms, p50_ms, p95_ms, n, by_modality{}}` — `latency_samples`'tan.
- `GET /admin/contributions/stats`; `POST /admin/contributions/export` — → `{job_id}` — manifest + split dosyası yazar (`collect.export_manifest`).
- `GET /admin/qc/queue?status=pending`; `POST /admin/qc/{recording_id}/review` — `{verdict:"accepted"|"rejected"|"fixed", transcript_fix?}` — admin/instructor.
- `GET /admin/users?cursor`; `PATCH /admin/users/{id}/role` — `{role}`.
- `GET /admin/audit?cursor` — denetim günlüğü.

## 4. WebSocket akışları

Bağlantı: `wss://<host>/ws/v1/ink?ticket=<ticket>` ve `/ws/v1/audio?ticket=<ticket>`. Zarflar `aicls.api.ws.protocol` içinde pydantic ayrımlı birlik; her zarf `{type, seq, payload}`; `seq` istemci tarafından artan, sunucu `ack` yerine sonuç zarfında `seq` geri döndürür.

### 4.1 `/ws/v1/ink`

```
istemci → sunucu:  stroke{points:[{x,y,t}], canvas:{w,h}} | pen_up{} | clear{} | finalize{} | ping{}
sunucu → istemci:  partial{latex, confidence, seq} | final{latex, alternatives, recognition_id, seq} | error{problem} | slow_down{retry_after_ms} | pong{}
```

- Sunucu her `pen_up` sonrası 400 ms boşta bekler (debounce), ardından `Recognizer.recognize()` çalıştırır ve `partial` gönderir; `finalize` veya 1,5 s boşta → `final`.
- Tanıma bir `asyncio.Semaphore(1)` ile bağlantı başına sıralanır; kuyrukta 2'den fazla istek birikirse eski istek düşürülür ve `slow_down` gönderilir.
- `final` sonrası istemci `POST /compute/text` ile devam eder (`context.last_latex` = final latex).

### 4.2 `/ws/v1/audio`

```
istemci → sunucu:  start{sample_rate:16000, format:"pcm16"} | ikili çerçeve (250 ms PCM16) | stop{} | cancel{}
sunucu → istemci:  partial{text} | final{text, confidence, transcript_id} | command{outcome: ComputeOut} | error{problem} | slow_down{}
```

- Sunucu tamponu 30 s ile sınırlar; enerji tabanlı bitiş tespiti (600 ms sessizlik) `final` üretir; `stop` de üretir.
- `final` sonrası aynı bağlantıda NLU + CAS koşar ve `command` zarfı `ComputeOut` taşır; ses yolunda arayüz her zaman onay ister.
- Whisper akışlı kısmi sonuç üretmediği için `partial` yalnızca 5 s'den uzun kayıtlarda parça tabanlı olarak gönderilir; gecikme ölçümünde `final`→`command` süresi `Server-Timing` eşdeğeri olarak zarfa eklenir.

### 4.3 Testler

`tests/integration/ws/test_ink_stream.py::test_ws_ink_stream_partial_then_final`, `test_ws_ink_slow_down_when_flooded`; `tests/integration/ws/test_audio_stream.py::test_ws_audio_backpressure`, `test_ws_audio_final_then_command_with_stub`; `test_ws_rejects_expired_ticket`.

## 5. Sürümleme ve uyumluluk

- OpenAPI `docs/api/openapi.json` git'te; PR'da `scripts/export_openapi.py` yeniden üretir, fark varsa CI kırmızı.
- Şema alan ekleme geriye uyumludur; alan silme/anlam değiştirme `v2` yönlendiricisi gerektirir ve ADR ile kayda alınır.
- Arayüz istemcisi `openapi-typescript` ile üretildiğinden uyumsuzluk derleme zamanında (`tsc --noEmit`) yakalanır.

## 6. Uygulama notları (`aicls.api`)

Bu bölüm, tasarımdan sapan veya tasarımın açık bırakıp uygulamanın karara bağladığı noktaları
kaydeder. Her madde bir testle bağlıdır.

### 6.1 Sözleşme davranışları

- **"Tanındı ama hesaplanamıyor" yolu.** `\forall`, `\binom` gibi v1 dilbilgisi dışı yapılar
  ayrıştırıcıdan `UNSUPPORTED_CONSTRUCT` olarak döner. *Cevap veren* uçlarda
  (`POST /compute/text`, `POST /compute/latex`, `/ws/v1/audio` komut zarfı) bu hata
  `aicls.api.pipeline.as_unsupported` ile `UNSUPPORTED_PROBLEM` (422) yapılır: `detail_tr`
  dondurulmuş İP-4 cümlesidir (`MSG_UNSUPPORTED_PROBLEM`) ve `construct` alanı hangi simgeden
  kaçınılacağını söyler. *Tanı koyan* uç `POST /compute/parse` ayrıştırıcının kendi kodunu
  (`UNSUPPORTED_CONSTRUCT`) 200 gövdesi içinde `ok:false` ile bildirir; canlı önizlemenin kesin
  olması gerekir. KNOWN_ISSUES 2. maddesinin API'ye bıraktığı açık yol budur.
  `test_api_unsupported_returns_422_with_contract_message`,
  `test_api_compute_parse_reports_the_construct_without_computing`.
- **Açıklama isteği 200'dür.** `Clarification` çıktısı `kind:"clarification"` gövdesiyle döner;
  her aday, kural tabanlı çözümleyici yeniden çalıştırılarak üretilmiş bir `command` ve
  `latex_preview` taşır (LLM değil; `confirm_intent` de aynı yolu kullanır).
- **`GET /workspaces`** şekil tutarlılığı için `Page[WorkspaceOut]` döner ama her zaman tek
  sayfadır: `WorkspaceRepository.list_for_user` imleç almaz.

### 6.2 Yönlendirme ve şema

- **`POST /ocr/recognize` tek yol, iki gövde.** api.md §3.5 JSON *veya* multipart der; FastAPI
  bunu parametre imzasıyla ifade edemediğinden istek `Content-Type` ile ayrıştırılır ve iki gövde
  OpenAPI'ye `openapi_extra` ile yazılır. Desteklenmeyen bir tür `OCR_INPUT_INVALID` (400) verir.
- **`POST /admin/book/load`** zip'i `file` (veya `archive`) multipart parçasında bekler; üye
  yolları çıkarma kökü dışına taşarsa (`zip slip`) veya arşivde Markdown yoksa istek 422 olur.
- **`GET /contribute/prompts/next?n=5`** sorgu parametresiyle çalışır; katılımcı kimliği
  `X-Participant-Token` başlığından ya da oturum açmış hesaptan çözülür.
- **`ComputeOut`** ses yolunda iki alan daha taşır: `transcript` ve `confidence`. Arayüz sesli
  komutta her zaman onay istediğinden (api.md §3.6) bu iki alan aynı belgede bulunur.
- **`ProblemOut.construct`** şemada bu adla görünür; pydantic alanı `construct_name`'dir çünkü
  `construct` bir `BaseModel` üyesini gölgeler.
- Kayıt, çalışma alanı, geri bildirim, SUS ve rıza uçları **201** döner; `logout`, `telemetry`,
  `ocr/asr feedback`, `qc review`, `role`, `withdraw`, `recording delete` **204** döner.

### 6.3 Depo katmanı

`aicls.db` bu iki tablo için sorgu yüzeyi tanımlamadığından (`modules/db.json` kapsamı dışı),
API kendi depolarını taşır ve ikisi de `aicls.db.repositories.base.Repository` alt sınıfıdır, yani
ADR-0003'ün "sorgu yalnızca depoda kurulur" kuralı korunur:

- `aicls.api.security.RefreshTokenRepository` — `refresh_tokens` (aile, döndürme, iptal);
- `aicls.api.participants.ParticipantRepository` — konuşmacı belirteci, ilerleme, kayıt sahipliği.

İleride `aicls.db` bunları devralırsa API tarafındaki sınıflar ince birer sarmalayıcıya iner.

### 6.4 Çalışma zamanı seçimleri

- **Yürütücü.** `AICLS_ENV=test` ise `InProcessExecutor`, aksi hâlde ADR-0016'nın `PoolExecutor`
  havuzu kullanılır: her uygulama nesnesi için süreç doğurmak test paketini dakikalara çıkarırdı.
  `GET /health/ready` in-process yürütücüde `sandbox.workers = 1` bildirir.
- **argon2id.** Üretim parametreleri docs/guvenlik-gizlilik.md §1'deki değerlerdir
  (`3 / 64 MiB / 2`); yalnızca `AICLS_ENV=test` altında ucuz bir profil (`1 / 8 MiB / 1`)
  kullanılır. Doğrulama parametreleri hash'in içinden okuduğu için davranış değişmez;
  `test_password_hasher_uses_the_documented_production_cost` üretim değerlerini sabitler.
- **`AICLS_INVITE_CODE`.** `instructor` rolüyle kayıt için gereken davet kodu bu ortam
  değişkeninden okunur (boşsa özellik kapalıdır). `Settings` alanı değil; `aicls.core.settings`
  başka bir modülün mülkü olduğu için bir sonraki `core` dalgasında oraya taşınmalıdır.
- **Katmanlama.** `aicls.ocr.factory` ve `aicls.asr.interfaces` **statik** olarak ``torch``a
  ulaşır; `no-torch-in-core` sözleşmesi `sys.modules`'e değil içe aktarma çizgesine baktığı için
  `ModelRegistry` bu iki modülü ancak eğitilmiş bir kontrol noktası etkinleştirildiğinde
  `importlib.import_module` ile çözer, saplamaları ise doğrudan torch'suz alt modüllerden
  (`aicls.ocr.stub`, `aicls.asr.stub`) alır.

### 6.5 WebSocket geri basıncı

api.md §4.1 "kuyrukta 2'den fazla istek birikirse `slow_down`" der. Saplama tanıyıcıyla kuyruk
pratikte hiç oluşmadığından geri basınç, api.md §1'in "WS mesaj hızı aşımında `slow_down`,
ısrarda bağlantı kapatma" kuralıyla uygulanır: bağlantı başına kayan pencerede 30 mesaj/sn;
aşımda `slow_down{retry_after_ms:250}`, üst üste 5 aşımda 1013 ile kapatma. Tanıma ayrıca
bağlantı başına `asyncio.Semaphore(1)` ile sıralanır. Akış bütçeleri (`WsTimings`: 400 ms kalem
boşta, 1,5 s final, 600 ms sessizlik, 30 s tampon) `app.state.ws_timings` üzerinden
değiştirilebilir; testler kısa değerlerle çalışır, varsayılanlar ADR-0014'ün değerleridir.

### 6.6 OpenAPI dışa aktarımı

`scripts/export_openapi.py` iki dosya yazar: `docs/api/openapi.json` ve
`web/src/api/generated/openapi.json`. İkincisi `openapi-typescript`in girdisidir
(`web/` içinde `schema.d.ts` üretir); Node aracı bu depodan çalıştırılmaz. Belge her zaman
`profile=full` ile üretilir, böylece `.env` dosyasında `collect` yazan bir geliştirici kırpılmış
bir şemayı sessizce işleyemez.
