# Güvenlik ve Gizlilik

Öneri A.1.2: SSL ile şifreli iletişim, SQL enjeksiyonuna karşı önlem, periyodik yedekleme, rol tabanlı yetkilendirme; A.2.3/B.1.2: rıza, KVKK (Kişisel Verilerin Korunması Kanunu) ve etik kurul. Bu belge bunların koddaki karşılığını ve İP-9 güvenlik testi kontrol listesini tanımlar.

## 1. Kimlik doğrulama ve roller

| Konu | Uygulama |
|---|---|
| Parola | argon2id (`argon2-cffi`; `time_cost=3, memory_cost=64 MiB, parallelism=2`), en az 10 karakter; 10 başarısız girişte 15 dk kilit (`users.failed_logins`, `locked_until`) |
| Erişim belirteci | PyJWT HS256, 32 bayt sır (`AICLS_JWT_SECRET`; iki anahtar kabul penceresiyle döndürme), 15 dk, `sub`, `role`, `jti` |
| Yenileme belirteci | `aicls_refresh` httpOnly Secure SameSite=Strict çerezi, 7 gün, DB'de hash, kullanımda döndürme, yeniden kullanımda aile iptali |
| WebSocket | `POST /auth/ws-ticket` 60 s tek kullanımlık bilet; URL'de JWT yok (sunucu günlüklerine sızmaz) |
| Roller | `student`, `instructor`, `contributor`, `admin`; `require_role()` bağımlılığı; her yönlendirici × rol matrisi `test_api_rbac_matrix_every_router_every_role` ile |
| Sahiplik | çalışma alanı/geçmiş/kayıt erişimi depo katmanında `(kaynak_id, user_id)` ile; IDOR testi |
| Yönetici işlemleri | `audit_log` (rol değişikliği, model etkinleştirme, kitap yükleme, dışa aktarım, QC kararı, rıza olayları) |
| Üniversite SSO | sonradan ADR ile; `require_role` değişmez |

## 2. KVKK ve rıza (ses verisi)

Etik kurul ve KVKK süreci başka ekibin sorumluluğu (dış bağımlılık); bu depo süreci zorlar. Ayrıntı [ml/veri-yonetimi.md §4](ml/veri-yonetimi.md#4-kişisel-veri-kvkk-ve-rıza). Özet: sürümlü rıza metni ve kabul kaydı → takma adlı konuşmacı (`P00001`) → yalnızca bant demografi → ses dosyaları `AICLS_DATA_ROOT/audio/raw` NTFS izinli, halka açık sunulmaz → geri çekme uç noktası dosya + türev silme, rıza izi kalır → dışa aktarımda kimlik alanı yok → toplama VM'inde gece senkronizasyonundan sonra silme → `AICLS_ETHICS_APPROVAL_REF` boşsa sekme kapalı (503).

Günlükler: `request_id`, `user_id` (e-posta değil), rota, süre; ses/iz gövdeleri, LaTeX girdileri **günlüğe yazılmaz**; saklama 30 gün.

### Uygulama notları (aicls.collect)

§2'deki zincirin koddaki karşılığı — her madde bir testle bağlıdır:

- **Sürümlü rıza.** `aicls.collect.current_consent()` paket kaynağından sürümlü, SHA-256'lı metni verir; `accept_recording` yalnızca **yürürlükteki** sürümü kabul etmiş konuşmacıyı geçirir. Bilinmeyen takma ad, geri çekilmiş konuşmacı ve eski sürüm aynı `403 CONSENT_REQUIRED` yanıtını üretir; uç nokta bir takma adın kayıtlı olup olmadığını doğrulamaz (numaralandırma sızıntısı yok). `test_consent_required_before_recording`, `test_stale_consent_version_must_be_re_accepted`.
- **Veri asgariliği.** `Demographics` pydantic modeli `extra="forbid"` ile tanımlıdır: `name`, `email`, `birth_date` gibi bir alan taşıyan istek **doğrulamada** düşer, sessizce yok sayılmaz. `test_rejects_identity_fields`.
- **IP/tarayıcı.** Ham adres hiçbir yolla veritabanına ulaşamaz: `accept_consent`, `ip_hash`/`user_agent_hash` alanında IPv4/IPv6 adresi veya hex olmayan bir değer görürse `VALIDATION_ERROR` verir; hash üretimi `hash_ip()` ile yapılır (tuz varsayılan olarak `AICLS_JWT_SECRET`). `test_rejects_a_raw_ip_address`.
- **Yol geçişi.** Ses dosyası dizinleri yalnızca `^P\d{5}$` desenine uyan takma addan ve üretilen ULID'den kurulur; istemciden gelen hiçbir dize dosya adına veya dizin bileşenine dönüşmez. `test_check_pseudonym_rejects_anything_unexpected`.
- **Reddedilen yükleme diske yazılmaz.** QC başarısızsa ne dosya ne satır oluşur; kabul edilen kayıt `.part` geçici adı üzerinden atomik olarak taşınır, yarım WAV kalmaz. `test_recording_rejected_by_qc_writes_no_file`.
- **Kapı.** `AICLS_ETHICS_APPROVAL_REF` boşken toplama uçları `503 GATE_BLOCKED` verir; **geri çekme kapıya tabi değildir**. `test_gate_blocks_recording_without_an_ethics_reference`.
- **Geri çekme.** Ses ve türev öznitelik dosyaları silinir, `recordings` satırları `deleted_at` alır, `ip_hash`/`user_agent_hash` NULL'lanır, rıza satırı ve takma ad denetim kanıtı olarak kalır; sonraki dışa aktarımlar konuşmacıyı içermez. `test_withdraw_deletes_files_keeps_consent_audit`.
- **Dışa aktarımda kimlik yok.** Manifest satırı yalnızca `MANIFEST_FIELDS` alanlarını taşır; testte kullanıcı ULID'si, e-posta, konuşmacı ULID'si ve IP hash'inin dosyanın hiçbir yerinde geçmediği doğrulanır. `test_export_manifest_has_no_identity_fields`.
- **Günlükler.** `aicls.collect` günlüklerinde takma ad, kayıt kimliği ve süre bulunur; ses gövdesi, dosya içeriği veya rıza metni **günlüğe yazılmaz**.

Kalan (bu depo dışındaki) sorumluluklar: `audio/raw` NTFS izinleri, gece senkronizasyonu sonrası toplama VM'inde silme, IP hash'lerinin 90 gün sonra NULL'lanması (zamanlanmış görev) ve yönetici dinlemesi için 5 dakikalık imzalı URL — bunlar dağıtım/işletim adımlarıdır ve `docs/gelistirme/kurulum.md` ile İP-9 kontrol listesinde izlenir.

## 3. Sırlar

- `.env` git dışı (`.gitignore`), `pydantic-settings` ile okunur; `.env.example` yalnızca anahtar adları.
- Üretim: Windows servis hesabının ortam değişkenleri (NSSM `AppEnvironmentExtra`) veya docker `env_file` (600); `configs`/YAML içinde sır yok.
- CI: yalnızca MySQL test kimlik bilgisi GitHub encrypted secrets'ta.
- `pre-commit` `detect-private-key` + `AICLS_JWT_SECRET=` deseni tarama; `scripts/check_env.py` sır varsayılan/kısa ise hata.
- Günlük filtresi `Authorization`, çerez ve `AICLS_*SECRET*` değerlerini maskeler.

## 4. SSL / taşıma

- Caddy TLS sonlandırma (iş istasyonunda üniversite sertifikası veya iç CA, VM'de Let's Encrypt), TLS 1.2+, HSTS, HTTP→HTTPS; uvicorn yalnızca `127.0.0.1`; WebSocket `wss`.
- Güvenlik başlıkları (`api.middleware.security_headers`): `Content-Security-Policy: default-src 'self'; img-src 'self' data:; font-src 'self'; style-src 'self' 'unsafe-inline'` (KaTeX satır içi stilleri), `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, `Permissions-Policy: microphone=(self)`.
- CORS: yalnızca `AICLS_CORS_ORIGINS`.

## 5. SQL enjeksiyonu

- Tüm erişim SQLAlchemy ORM/Core bağlı parametrelerle; ham `text()` yalnızca migrasyonlarda; `tools/lint_no_raw_sql.py` (`text(` + f-string / `%` / `.format`) CI'da.
- FULLTEXT sorgusunda kullanıcı metni bağlı parametre, boolean-mode operatörleri (`+ - * " ( ) < > ~`) soyulur; uzunluk ≤ 200.
- En az yetkili MySQL kullanıcıları; `aicls_app` DDL yapamaz.
- `test_api_sql_injection_string_in_search_is_parameterized` (`' OR 1=1 --`, `x"; DROP TABLE`).

## 6. Yükleme ve girdi doğrulama

| Girdi | Kontroller |
|---|---|
| PNG/JPEG | sihirli bayt (`\x89PNG`, `\xFF\xD8`), Pillow `verify()` + yeniden kodlama, `MAX_IMAGE_PIXELS = 20e6`, ≤ 2 MB, ≤ 4096 px, dosya adı ULID |
| InkML / Canvas JSON | `defusedxml`, ≤ 1 MB, ≤ 20 000 nokta, pydantic şema, koordinat aralığı |
| WAV | `RIFF`/`WAVE` başlığı, PCM16, mono/stereo, 16/44,1/48 kHz, ≤ 30 s, ≤ 10 MB; başka konteyner yok (ffmpeg yok); yeniden örnekleme sunucuda numpy |
| LaTeX (kullanıcı, OCR, ASR, CAS çıktısı) | ≤ 2048 karakter; `sanitize_latex` izin listesi; KaTeX `trust: false`, `maxExpand: 1000`, `maxSize: 50`; `\href \url \includegraphics \input \write \def` yasak |
| Metin komutu | ≤ 500 karakter, NFC |
| Çok parçalı gövde | Caddy `request_body max 12MB`, yönlendiricide akış sayacı |
| CAS yürütme | tipli AST'den kurulur, `sympify`/`eval` yok; ayrı süreçte 2,5 s zaman aşımı, işçi öldürme; Windows Job Object ile bellek üst sınırı en-iyi-çaba (ADR-0016) |
| Grafik | Matplotlib Agg, sabit örnek sayısı (400), ≤ 300 kB PNG |

## 7. Hız sınırı ve kötüye kullanım

`/auth/*` 5/dk/IP, 20/saat/hesap; `/compute/*` 60/dk/kullanıcı; yüklemeler 20/dk/kullanıcı; `/contribute/recordings` 60/saat/katılımcı; WS mesaj hızı aşımında `slow_down`, ısrarda kapatma; `429` + `Retry-After`. Bellek içi token kovası (tek süreç); çok süreçte Redis (ADR ile).

## 8. Bağımlılık hijyeni ve yedekleme

`pip-audit` ve `npm audit` gece; hash'li `requirements/*.txt`; Dependabot yalnızca `web/`. Yedekleme: gece şifreli `mysqldump` + ses + modeller → 10 TB HDD + üniversite içi kopya, 30 gün; geri yükleme tatbikatı üç ayda bir ([gelistirme/kurulum.md §5](gelistirme/kurulum.md#5-yedekleme)).

## 9. İP-9 güvenlik testi kontrol listesi (rapor şablonu)

1. Kimlik: parola politikası, kilitleme, belirteç süresi, yenileme döndürme, bilet süresi — testler yeşil.
2. Yetki: rol matrisi, IDOR, yönetici uçları.
3. Taşıma: TLS yapılandırması (Caddy), HSTS, başlıklar — `curl -I` çıktısı rapora.
4. Girdi: sihirli bayt, boyut, XML varlık, LaTeX izin listesi, SQL enjeksiyon dizeleri.
5. Kaynak tüketimi: sandbox zaman aşımı, WS geri basınç, hız sınırı.
6. Gizlilik: günlükte gövde yok, dışa aktarımda kimlik yok, geri çekme çalışıyor.
7. Bağımlılık: `pip-audit`/`npm audit` temiz veya gerekçeli istisna.
8. Yedek/geri yükleme tatbikatı tarihi.

Sonuçlar `docs/raporlama/ip-09-test-ve-geri-bildirim.md` içinde, ilgili test adlarıyla.
