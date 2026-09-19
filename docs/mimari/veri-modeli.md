# Veri Modeli

SQLAlchemy 2.0 (tipli `Mapped[]` bildirimsel modeller) + Alembic. Geliştirme ve testlerde SQLite, üretimde MySQL 8 (öneri A.1.2 ve A.3.9: "MySQL tercih edilecektir"). Karar: [ADR-0003](../adr/0003-veritabani-sqlite-dev-mysql-prod.md), kitap depolama: [ADR-0019](../adr/0019-kitap-bilgi-grafigi-iliskisel-depolama.md).

## 1. Motor stratejisi ve lehçe tarafsızlığı

| Ortam | URL | Notlar |
|---|---|---|
| dev | `sqlite+pysqlite:///D:/aicls-data/aicls-dev.db` | WAL modu, `PRAGMA foreign_keys=ON`, OneDrive dışı |
| test | `sqlite+pysqlite:///:memory:` | `StaticPool`; şema Alembic head'den kurulur (migrasyonlar her testte sınanır) |
| prod | `mysql+pymysql://aicls_app:...@127.0.0.1/aicls?charset=utf8mb4` | InnoDB, `utf8mb4_0900_ai_ci`, `pool_pre_ping`, `sql_mode=STRICT_TRANS_TABLES`; üç kullanıcı: `aicls_migrate` (DDL), `aicls_app` (DML), `aicls_ro` (rapor) |

Lehçe tarafsızlığı kuralları (`tests/integration/db/test_schema_rules.py` ile denetlenir):

1. Tüm birincil anahtarlar `CHAR(26)` ULID dizesi (`db.base.UlidPk`). Otomatik artan tam sayı yok.
2. Her `String` sütunun uzunluğu belirtilir (MySQL indeks sınırı için ≤ 255 indekslenen sütunlarda).
3. Enum'lar `sa.Enum(..., native_enum=False, length=32)` + uygulama tarafında `StrEnum`; CHECK kısıtı adlandırma kuralıyla.
4. JSON alanları `sa.JSON` (SQLite'ta TEXT, MySQL'de yerel JSON).
5. Zamanlar saat dilimi bilgisi olmayan UTC `DateTime`; sunucu tarafı fonksiyon yok.
6. Boolean `sa.Boolean` (MySQL TINYINT(1)).
7. MySQL'e özgü tek şey `FULLTEXT` indeksidir; yalnızca Alembic migrasyonunda `if op.get_bind().dialect.name == "mysql":` koruması altında oluşturulur ve `aicls.book.query.search()` SQLite'ta `LIKE` + `difflib` yoluna düşer.
8. Ham SQL yok: `sqlalchemy.text()` yalnızca migrasyonlarda; `tools/lint_no_raw_sql.py` CI'da denetler.

Alembic: `src/aicls/db/migrations/`, `render_as_batch=True` (SQLite ALTER desteği), PR başına en fazla bir revizyon, autogenerate çıktısı elle incelenir, `MetaData(naming_convention=...)` ile `ix_%(column_0_label)s`, `fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s`, `uq_%(table_name)s_%(column_0_name)s`, `ck_%(table_name)s_%(constraint_name)s`. CI: SQLite'ta `upgrade head` → `downgrade base` → `upgrade head`, autogenerate diff boş, tüm migrasyonların MySQL lehçesiyle derlenmesi (`test_alembic_upgrade_head_on_sqlite_and_mysql_dialect_compile`); gece işi `mysql:8` servisiyle aynı döngü (`marker mysql`).

## 2. Varlık-ilişki diyagramı

`tools/gen_erd.py` metadata'dan üretir; aşağıdaki blok elle taslak olup ilk migrasyonla birlikte üretilenle değiştirilecektir.

```mermaid
erDiagram
  users ||--o{ refresh_tokens : has
  users ||--o{ workspaces : owns
  workspaces ||--o{ operations : contains
  operations ||--o| assets : plot
  users ||--o{ recognitions : made
  users ||--o{ transcripts : made
  users ||--o| speakers : "links (admin-only)"
  consent_versions ||--o{ speakers : accepted
  speakers ||--o{ recordings : recorded
  prompt_sentences ||--o{ recordings : read
  recordings ||--o| assets : wav
  book_units ||--o{ book_aliases : alias
  book_units ||--o{ book_triples : subject
  book_units ||--o{ book_triples : object
  book_units ||--o{ book_assets : figure
  model_versions ||--o{ eval_runs : evaluated
  operations ||--o| latency_samples : measured
  users ||--o{ feedback : wrote
  users ||--o{ sus_responses : answered
  users ||--o{ audit_log : actor
```

## 3. Tablolar

Ortak: `id CHAR(26) PK`, `created_at`, `updated_at` (TimestampMixin). Dizinler `idx(...)`, tekillik `uq(...)`.

### 3.1 Kimlik ve çalışma alanı

| Tablo | Sütunlar | Dizinler |
|---|---|---|
| `users` | `email VARCHAR(255) uq`, `password_hash VARCHAR(255)` (argon2id), `display_name VARCHAR(120)`, `role ENUM(student,instructor,contributor,admin)`, `university ENUM(mus,mersin,amasya,other) NULL`, `is_active BOOL`, `failed_logins INT`, `locked_until DATETIME NULL`, `last_login_at` | `uq(email)` |
| `refresh_tokens` | `user_id FK`, `token_hash CHAR(64) uq`, `family_id CHAR(26)`, `expires_at`, `revoked_at NULL`, `replaced_by CHAR(26) NULL`, `ip_hash CHAR(64)` | `idx(user_id)`, `idx(family_id)` |
| `workspaces` | `owner_id FK users`, `title VARCHAR(200)`, `settings_json JSON`, `is_shared BOOL` | `idx(owner_id)` |
| `workspace_members` | `workspace_id FK`, `user_id FK`, `role_in_ws ENUM(viewer,editor)` | `uq(workspace_id,user_id)` |
| `operations` | `workspace_id FK`, `user_id FK`, `input_kind ENUM(text,latex,ink,audio)`, `source_text TEXT NULL`, `input_latex TEXT NULL`, `command_json JSON`, `result_kind ENUM(expr,number,set,bool,plot,steps) NULL`, `result_latex TEXT NULL`, `result_srepr TEXT NULL`, `steps_json JSON NULL`, `plot_asset_id FK assets NULL`, `error_code VARCHAR(32) NULL`, `timings_json JSON`, `cache_hit BOOL`, `model_versions_json JSON`, `nlu_source ENUM(rules,asr+rules,llm) NULL` | `idx(workspace_id, created_at)` |
| `assets` | `kind ENUM(plot_png,ink_json,ink_png,audio_wav,book_figure,screenshot)`, `path VARCHAR(400)` (AICLS_DATA_ROOT'a göreli), `sha256 CHAR(64)`, `bytes INT`, `mime VARCHAR(64)`, `owner_id FK NULL`, `retention_until DATETIME NULL` | `idx(kind)`, `idx(retention_until)` |

### 3.2 Model çıktıları ve düzeltmeler (İP-3 geri besleme)

| Tablo | Sütunlar | Dizinler |
|---|---|---|
| `recognitions` | `user_id FK NULL`, `operation_id FK NULL`, `ink_asset_id FK NULL`, `image_asset_id FK NULL`, `latex TEXT`, `confidence FLOAT`, `alternatives_json JSON`, `model_version VARCHAR(40)`, `elapsed_ms INT`, `corrected_latex TEXT NULL`, `corrected_at NULL` | `idx(model_version)`, `idx(corrected_at)` |
| `transcripts` | `user_id FK NULL`, `operation_id FK NULL`, `audio_asset_id FK NULL`, `text TEXT`, `confidence FLOAT`, `model_version VARCHAR(40)`, `elapsed_ms INT`, `corrected_text TEXT NULL` | `idx(model_version)` |

### 3.3 Ses verisi toplama (KVKK)

| Tablo | Sütunlar | Dizinler |
|---|---|---|
| `consent_versions` | `version VARCHAR(16) uq`, `text_tr TEXT`, `sha256 CHAR(64)`, `effective_from DATE`, `ethics_approval_ref VARCHAR(80)` | |
| `speakers` | `user_id FK users NULL uq` (yalnızca admin dışa aktarımı dışında; eğitim manifestine asla girmez), `pseudonym CHAR(6) uq` (P00001), `consent_version_id FK`, `consented_at`, `ip_hash CHAR(64)`, `user_agent_hash CHAR(64)`, `age_band ENUM(18-24,25-34,35-44,45+)`, `gender ENUM(f,m,na)`, `university ENUM(mus,mersin,amasya,other)`, `accent_region VARCHAR(40) NULL`, `token_hash CHAR(64) NULL` (anonim katılımcı), `withdrawn_at NULL` | `uq(pseudonym)`, `idx(university, gender)` |
| `prompt_sentences` | `text_tr VARCHAR(500)`, `text_normalized VARCHAR(500)`, `scenario_tag VARCHAR(40)` (limit, turev, integral, denklem, grafik, kitap, genel), `intent VARCHAR(40) NULL` (envanter niyeti), `difficulty TINYINT`, `is_active BOOL`, `target_count INT` | `idx(scenario_tag, is_active)` |
| `recordings` | `speaker_id FK`, `prompt_id FK`, `asset_id FK`, `duration_s FLOAT`, `sample_rate INT`, `sha256 CHAR(64)`, `qc_passed BOOL`, `qc_json JSON`, `snr_db FLOAT`, `clipping_ratio FLOAT`, `silence_ratio FLOAT`, `review_status ENUM(pending,accepted,rejected,fixed)`, `reviewed_by FK NULL`, `transcript_final VARCHAR(500)`, `split ENUM(unassigned,train,dev,test)`, `deleted_at NULL` | `idx(speaker_id)`, `idx(prompt_id, qc_passed)`, `idx(split)`, `idx(review_status)` |

### 3.4 Kitap bilgi grafiği (İP-8)

| Tablo | Sütunlar | Dizinler |
|---|---|---|
| `book_units` | `id VARCHAR(64) PK` ("tanim-2.4"), `unit_type ENUM(Definition,Theorem,Lemma,Corollary,Proof,Example,Exercise,Figure,Table,Remark,Section)`, `chapter VARCHAR(64)`, `number VARCHAR(16)`, `order_in_chapter INT`, `title_tr VARCHAR(255)`, `title_normalized VARCHAR(255)`, `body_md MEDIUMTEXT`, `latex_json JSON`, `search_text TEXT` (normalize edilmiş başlık + gövde), `jsonld JSON` (tam belge), `source_page INT NULL`, `tag_status ENUM(auto,reviewed)`, `version INT` | `idx(unit_type, chapter)`, `idx(title_normalized)`, MySQL: `FULLTEXT(search_text)` |
| `book_aliases` | `unit_id FK`, `alias_normalized VARCHAR(255)`, `weight FLOAT` (1.0 elle, 0.8 ek tablosu) | `uq(unit_id, alias_normalized)`, `idx(alias_normalized)` |
| `book_triples` | `subject_id FK book_units`, `predicate VARCHAR(32)` (aicls:partOf, aicls:requires, aicls:illustrates, aicls:proves, aicls:next, aicls:hasFigure, skos:related), `object_id FK book_units`, `provenance VARCHAR(64)` | `uq(subject_id,predicate,object_id)`, `idx(subject_id,predicate)`, `idx(object_id,predicate)` |
| `book_assets` | `unit_id FK`, `asset_id FK assets`, `caption_tr VARCHAR(500)`, `alt_tr VARCHAR(500)` | `idx(unit_id)` |

### 3.5 Model kaydı, değerlendirme, telemetri, geri bildirim, denetim

| Tablo | Sütunlar | Dizinler |
|---|---|---|
| `model_versions` | `component ENUM(ocr,asr)`, `version VARCHAR(40)` ("ocr-0.3.0"), `path VARCHAR(400)` (AICLS_MODELS_ROOT'a göreli), `sha256 CHAR(64)`, `config_hash CHAR(64)`, `git_sha CHAR(40)`, `run_id VARCHAR(64) NULL`, `metrics_json JSON`, `is_active BOOL`, `promoted_at NULL`, `promoted_by FK NULL` | `uq(component, version)`, `idx(component, is_active)` |
| `eval_runs` | `kind ENUM(parser,cas,nlu,ocr,asr,book,latency)`, `name VARCHAR(120)`, `git_sha CHAR(40)`, `dirty BOOL`, `config_hash CHAR(64) NULL`, `dataset_fingerprint CHAR(64)`, `dataset_version VARCHAR(40)`, `model_version VARCHAR(40) NULL`, `normalizer_versions_json JSON`, `n INT`, `metrics_json JSON`, `ci95_json JSON`, `report_path VARCHAR(400)`, `is_reportable BOOL`, `created_by FK NULL` | `idx(kind, created_at)`, `idx(is_reportable)` |
| `latency_samples` | `operation_id FK NULL`, `request_id CHAR(26)`, `modality ENUM(text,latex,ink,audio)`, `endpoint VARCHAR(80)`, `server_ms FLOAT`, `stage_json JSON`, `client_total_ms FLOAT NULL`, `render_ms FLOAT NULL`, `model_versions_json JSON` | `idx(modality, created_at)` |
| `feedback` | `user_id FK NULL`, `category ENUM(critical,usability,suggestion)`, `text TEXT`, `page VARCHAR(120)`, `screenshot_asset_id FK NULL`, `status ENUM(open,triaged,fixed,deferred)`, `collected_week INT NULL` | `idx(status, category)` |
| `sus_responses` | `user_id FK NULL`, `cohort VARCHAR(40)`, `answers_json JSON` (10 × 1–5), `score FLOAT`, `collected_week INT NULL` | `idx(cohort)` |
| `audit_log` | `actor_id FK NULL`, `action VARCHAR(64)`, `target_table VARCHAR(40)`, `target_id VARCHAR(64)`, `detail_json JSON`, `ip_hash CHAR(64)` | `idx(action, created_at)` |

Yazılan denetim olayları: rol değişikliği, rıza kabulü/geri çekme, kitap yükleme, model etkinleştirme, dışa aktarım, QC inceleme kararı.

## 4. Saklama ve yedekleme

- `assets.retention_until`: iz/grafik dosyaları 180 gün sonra temizlenir (incelenmiş bir `recognitions.corrected_latex` kaydına bağlıysa saklanır).
- Ses kayıtları: rıza aktif olduğu sürece; geri çekmede `deleted_at` + dosya silme.
- `speakers.ip_hash` / `user_agent_hash` 90 gün sonra NULL'lanır (kötüye kullanım denetimi dışında gereksiz).
- `scripts/backup_db.ps1` gece `mysqldump` (veya SQLite kopyası) + `AICLS_DATA_ROOT/audio` → 10 TB HDD, 30 gün rotasyon, 7-zip AES; geri yükleme tatbikatı üç ayda bir (`docs/gelistirme/kurulum.md#yedekleme`).

## 5. Kitap içerik şeması (JSON-LD / RDF)

### 5.1 Bağlam

`src/aicls/book/context.jsonld`:

```json
{
  "@context": {
    "aicls": "https://ai-cls.example/ns#",
    "schema": "https://schema.org/",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "dcterms": "http://purl.org/dc/terms/",
    "name": "schema:name",
    "description": "schema:description",
    "inLanguage": "schema:inLanguage",
    "identifier": "dcterms:identifier",
    "source": "dcterms:source",
    "altLabel": {"@id": "skos:altLabel", "@container": "@set"},
    "related": {"@id": "skos:related", "@type": "@id", "@container": "@set"},
    "chapter": "aicls:chapter",
    "number": "aicls:number",
    "bodyMarkdown": "aicls:bodyMarkdown",
    "latex": {"@id": "aicls:latex", "@container": "@list"},
    "figure": {"@id": "aicls:hasFigure", "@type": "@id", "@container": "@set"},
    "partOf": {"@id": "aicls:partOf", "@type": "@id"},
    "requires": {"@id": "aicls:requires", "@type": "@id", "@container": "@set"},
    "illustrates": {"@id": "aicls:illustrates", "@type": "@id", "@container": "@set"},
    "proves": {"@id": "aicls:proves", "@type": "@id"},
    "next": {"@id": "aicls:next", "@type": "@id"}
  }
}
```

### 5.2 Örnek birim

```json
{
  "@context": "https://ai-cls.example/context.jsonld",
  "@id": "aicls:tanim-2.4",
  "@type": "aicls:Definition",
  "identifier": "2.4",
  "chapter": "aicls:bolum-2",
  "number": "2.4",
  "name": "Türev",
  "inLanguage": "tr",
  "altLabel": ["türev", "türevin tanımı", "türev tanımı", "türevi", "diferansiyel katsayı"],
  "bodyMarkdown": "$f$ fonksiyonu $a$ noktasında ... limiti varsa ...",
  "latex": ["f'(a) = \\lim_{h \\to 0} \\frac{f(a+h)-f(a)}{h}"],
  "figure": ["aicls:sekil-2.3"],
  "partOf": "aicls:bolum-2",
  "requires": ["aicls:tanim-2.1"],
  "next": "aicls:ornek-2.7",
  "source": "Temel Analiz, s. 58"
}
```

`@type` sözlüğü `aicls:Definition|Theorem|Lemma|Corollary|Proof|Example|Exercise|Figure|Table|Remark|Section` (`src/aicls/book/vocab.ttl` içinde RDF Schema sınıfları olarak tanımlı). İlişki özellikleri yükleyici tarafından `book_triples` satırlarına düzleştirilir; tam belge `book_units.jsonld` sütununda saklanır; sık sorgulanan alanlar (`unit_type`, `chapter`, `number`, `title_normalized`, `search_text`) normalize sütunlara kopyalanır.

### 5.3 Yazım ve etiketleme

Kitap ekibi Markdown yazar; yapılandırılmış bloklar için:

```markdown
::: tanim 2.4 "Türev" aliases=["türevin tanımı","diferansiyel katsayı"] requires=[tanim-2.1]
$f$ fonksiyonu ...
:::
```

`aicls.book.ingest.markdown_tagger` hem bu blokları hem de düz "Tanım 2.4:" / "Teorem 3.1 (Ortalama Değer Teoremi):" desenlerini tanır (İP-8 risk planı: otomatik etiketleme betiği). `ingest.validators.coverage()` bölümdeki her Tanım/Teorem/Lemma/Sonuç/Örnek/Alıştırma/Şekil/Tablo başlığının bir birime dönüştüğünü doğrular; etiketsiz başlık varsa `aicls book load` reddeder ve listeyi yazar — İP-8'in "%100 etiketleme" ölçütü bu şekilde ölçülür (`aicls book coverage <dir>` oranı `reports/metrics/book_coverage_<sha>.json`'a yazar).

### 5.4 Sorgulama

| Sorgu | Yol |
|---|---|
| Kimlikle | `BookRepository.get("tanim-2.4")` → `book_units` PK |
| Komutla ("türev tanımını getir") | `nlu` → `BookFetch(unit_type=Definition, query="türev")` → `resolver.resolve()`: (1) `tr_casefold` + ek soyma ile `book_aliases.alias_normalized` tam eşleşme (indeksli), tür filtresi; (2) yoksa `search_text` üzerinde MySQL `MATCH ... AGAINST (? IN BOOLEAN MODE)` / SQLite `LIKE %?%`; (3) adayları `difflib.SequenceMatcher` oranı ≥ 0.8 ile yeniden sırala; (4) en iyi + adaylar. LRU (2048) `alias → unit_id` (İP-8 risk planı: önbellekleme). |
| Graf yürüyüşü | `related(unit_id, predicate)` → `book_triples` tek atlama birleşimi; `requires` kapanışı için en fazla 3 derinlik (özyinelemeli CTE değil, uygulama döngüsü; taşınabilirlik). |
| Bölüm ağacı | `unit_type=Section` + `partOf` üçlüleri |
| Dışa aktarım | `export_jsonld()` tek `@graph`; `export_turtle()` rdflib ile; İP-8 "RDF tripleri" çıktısının somut dosyası |

Performans ölçütü: `tests/perf/test_latency_budget.py::test_book_resolve_p50_under_20ms` (örnek bölüm, SQLite) ve gece MySQL koşusunda aynı test.

### 5.5 Uygulama notları

Bu bölümdeki tasarım `src/aicls/book/` altında uygulandı (`schema.py`, `vocab.py` + `context.jsonld`/`vocab.ttl`, `aliases.py`, `ingest/{common,markdown_tagger,latex_tagger,validators}.py`, `loader.py`, `query.py`, `resolver.py`, `export.py`). Yazma yalnızca `BookRepository` üzerinden yapılır; pakette tek satır SQL yoktur. Aşağıdakiler tasarımı somutlaştıran ya da bilinçli olarak ondan ayrılan noktalardır.

#### 5.5.1 Etiketleme kuralları

- **Etiketsiz başlığın tanımı.** Bir başlık, `tr_casefold` sonrası bir birim anahtar sözcüğüyle (Tanım, Teorem, Lemma, Önerme, Sonuç, Kanıt/İspat, Örnek, Alıştırma, Şekil, Tablo, Not/Uyarı, Bölüm) **başlıyor** ama ardından **numara gelmiyorsa** etiketsizdir: kararlı kimlik (`tanim-2.4`) üretilemez. LaTeX tarafında karşılığı `\label` taşımayan ortamdır (`kind="missing_label"`). Anahtar sözcükle başlamayan başlıklar (`## Giriş`) ne etiketlenir ne de sayılır — kapsama oranını bozmazlar. "Tanımlayıcı" gibi anahtar sözcükle *başlayan ama devam eden* kelimeler anahtar sözcük sayılmaz.
- **`coverage()` payı/paydası.** `len(nodes) / (len(nodes) + len(untagged_spans))`; kaynak boşsa 1.0. Böylece "etiketsiz başlık varsa < 1.0" kuralı tanım gereği sağlanır.
- **Bölüm birimi.** `# Bölüm 2: Türev` başlığı, kimliği **çağıranın verdiği `chapter` değeri olan** `Section` birimine dönüşür. Böylece `aicls:partOf` hedefi her zaman aynı yüklemede yazılan bir satırdır ve `book_triples` yabancı anahtarı hiç boşta kalmaz. Bölüm biriminin `order_in_chapter` değeri 0, gövde birimleri 1'den başlar.
- **`next` zinciri** belge sırasına göre ardışık gövde birimleri arasında kurulur; bölüm birimi zincire girmez.
- **Blok öznitelikleri.** `::: tanim 2.4 "Türev" aliases=[...] requires=[...] figure=[...] proves=... illustrates=... related=... page=58 id=...`. `id=` verilirse basılan kimliğin yerine geçer. LaTeX'te aynı öznitelikler `\begin` satırının sonundaki `%` yorumuna yazılır.
- **LaTeX numarası** `\label{tanim:4.1}` veya `\label{tanim-4.1}` etiketinin son sayı öbeğinden okunur.

#### 5.5.2 Takma ad üretimi ve ağırlıklar

- Üretim `title_tr` üzerinden yapılır: (a) başlığın çekimli biçimleri, (b) tür sözcüğü sonda tekrarlanıyorsa ("Ortalama Değer **Teoremi**") atılmış çekirdek öbeğin biçimleri, (c) başlık zaten iyelik taşıyorsa ("… Hız**ı**") gövdeden yeniden çekimlenmiş biçimler — `hızını`, `hızının` ancak böyle üretilir —, (d) çekirdek tek sözcükse çıplak baş adın biçimleri, (e) `{çekirdek, çekirdeğin, baş, başın} × {tanımı, tanımını, tanımının, tanımına}` birleşimleri, (f) hepsinin ASCII karşılığı. Ünlü uyumu, `y/n/s` kaynaştırma ünsüzü, `d→t` benzeşmesi ve `k→ğ, p→b, ç→c` yumuşaması uygulanır; `şekil→şekl`, `resim→resm` gibi ünlü düşmesi küçük bir düzensizler tablosundadır. `t` bilerek yumuşatılmaz (`kanıt` → `kanıtı`).
- **Ağırlıklar** (`book_aliases.weight`, indeks sıralamasını belirler): başlık 1.0, başlığın çekimi 0.9, çekirdek 0.85, birleşim 0.8, çıplak baş 0.7, bölüm başlığı 0.4; ASCII karşılığı kaynağının yarısı. Bir bölüm ("Bölüm 2: **Türev**") ile aynı adlı tanımın ("Tanım 2.4: **Türev**") çakışmasını bu sıralama çözer: tanım kazanır. Bölüm birimleri çıplak çekimli biçim üretmez.
- **Değişmez.** Aynı takma ada sahip iki birim olabilir (bir teoremle onun kanıtı), ama **en yüksek ağırlıkta beraberlik olamaz**; aksi hâlde sonuç kimliğe göre sıralanır ve keyfîleşir. `test_sample_chapter_aliases_have_no_tie_at_the_top` bunu örnek kitap üzerinde doğrular. Örnek kitapta 20 birim için 1575 takma ad üretilir (birim başına ~79).

#### 5.5.3 Çözümleme merdiveni

`BookResolver.resolve()` sırası: (1) normalleştirme (`tr_casefold` + noktalama temizliği), (2) komut fiili/dolgu sözcüğü atma (`getir`, `göster`, `nedir`, `lütfen`, `ekrana`… ve bunların ASCII biçimleri), (3) tür sözcüğünü ekleriyle birlikte tanıma (`tanımını` → gövde `tanım` → `Definition`) ve öbekten çıkarma, (4) takma ad indeksinde **tam eşleşme** — önce tam öbek, sonra türsüz öbek, sonra son sözcüğü gövdelenmiş biçimler, en son **bitişik alt öbekler** (uzundan kısaya). Alt öbek adımı, kitapta geçmeyen fazladan sözcük eklenmiş komutları ("türev tanımının **kendisini** yaz") tam eşleşme kuralını gevşetmeden kurtarır. (5) Hepsi boşsa `search()`: MySQL `MATCH … AGAINST` / SQLite `LIKE`, ardından `difflib ≥ 0.8` bulanık geçiş. Tür süzgeciyle sonuç bulunamazsa süzgeçsiz bir kez daha denenir (geri çağırma > kesinlik; İP-8 ölçütü ilk sonucun doğruluğudur).

LRU (öntanımlı 2048) `(normalleştirilmiş sorgu, tür, k)` anahtarıyla **sonucun kendisini** saklar; saklanan nesneler ORM'e bağlı olmayan pydantic modelleridir, bu yüzden oturum kapandıktan sonra da okunabilirler. `cache_size=0` önbelleği kapatır.

#### 5.5.4 Yükleme, dışa aktarım ve dosyalar

- **Boşta kalan kenarlar.** `book_triples` gerçek yabancı anahtar taşıdığı için, uçlarından biri ne yüklenen kümede ne veritabanında olan kenarlar `load_units` tarafından **düşürülür ve uyarı olarak günlüklenir** (yükleme iptal edilmez). Bölümler arası `requires` kenarları bu yüzden `load_markdown_dir` ile birlikte yüklendiğinde çözülür; ayrı ayrı yüklemede ikinci bölüm birinciyi veritabanında bulur.
- **Yeniden yükleme fikirdeştir** (idempotent): birim yerinde güncellenir (`version` artar), takma adları ve giden kenarları yeniden yazılır.
- `book_units.jsonld` tam JSON-LD belgesini, `search_text` ise normalleştirilmiş başlık + takma adlar + gövdeyi tutar (LIKE geçişi takma adları da tarar).
- **`export_jsonld()` `@context`'i gömer**, URL vermez: hiçbir tüketici (rdflib dâhil) ağa çıkmaz. `export_turtle()` yalnızca yerel dizgi ayrıştırır. Turtle çıktısı deterministiktir (aynı graf → aynı bayt), böylece raporda özetiyle anılabilir.
- Türkçe kullanıcı metni üretilmez: `UntaggedHeadingError` `aicls.core.messages_tr`'deki `VALIDATION_ERROR` iletisini kullanır, etiketsiz başlık listesi `details["untagged_headings"]` içinde taşınır (api.md §3.7: 422).
- **Performans testinin yeri.** `test_book_resolve_p50_under_20ms` bu belgede `tests/perf/test_latency_budget.py` altında öngörülmüştü; o dosya henüz yok, test `tests/integration/book/test_book_perf.py` içinde `perf` işaretiyle duruyor. `tests/perf/` oluşturulduğunda oraya taşınmalıdır.

#### 5.5.5 Ölçüm kökeni (rapor için)

`datasets/book/` içeriği **proje ekibi tarafından yazılmıştır**; gerçek *Temel Analiz* kitabından örneklenmemiştir. "%100 etiketleme" ve "%95 üzeri getirme" rakamları ilerleme raporunda bu kökenle birlikte verilmeli, pilot öncesi gerçek bölümlerle yeniden ölçülmelidir (bkz. `datasets/book/README.md`).

## 6. Uygulama notları

Şema `src/aicls/db/` altında hayata geçirildi; ilk revizyon `0001_initial_schema.py` 22 tabloyu, tüm dizinleri ve CHECK kısıtlarını kurar. Aşağıdaki noktalar bu belgedeki tasarımı ya somutlaştırır ya da bilinçli olarak ondan ayrılır.

### 6.1 Tasarımdan sapmalar

| Konu | Belgedeki tasarım | Uygulama | Gerekçe |
|---|---|---|---|
| Özet (hash) sütunları | `CHAR(64)` | `VARCHAR(80)` (`sha256`, `*_hash`, `config_hash`, `dataset_fingerprint`) | `aicls.core.hashing` özetleri `"sha256:<hex>"` önekiyle döndürür (71 karakter). Her iki biçim de kayıpsız saklanır; MySQL STRICT modunda sessiz kırpma yerine hata riski ortadan kalkar. Birincil anahtarlar `CHAR(26)` olarak kaldı. |
| Yabancı anahtar tipi | — | Birincil ve yabancı anahtar sütunları aynı `CHAR(26)` tipini kullanır | MySQL, `CHAR`/`VARCHAR` karışımında yabancı anahtarı reddeder. |
| `prompt_sentences.difficulty` | `TINYINT` | `SmallInteger` (`SMALLINT`) | `TINYINT` MySQL'e özgüdür; `SMALLINT` her iki lehçede de aynıdır. |
| `book_units.body_md` | `MEDIUMTEXT` | `Text().with_variant(mysql.MEDIUMTEXT, "mysql")` | SQLite'ta `TEXT`, MySQL'de `MEDIUMTEXT`; tek bildirimle lehçe tarafsızlığı. |
| FULLTEXT indeksi | Yalnızca migrasyonda, lehçe korumalı | Aynı; ayrıca **ORM metadata'sında yer almaz** (`ftx_book_units_search_text`) | Metadata'da olsaydı SQLite'ta autogenerate farkı "eksik indeks" olarak görünür ve `test_autogenerate_diff_is_empty` kırılırdı. Koruma `op.get_bind()` yerine `op.get_context().dialect.name` okur; böylece çevrimdışı (`--sql`) derleme de bu dalı üretir. |
| Enum depolama | `sa.Enum(..., native_enum=False, length=32)` | Aynı; ek olarak `values_callable` ile **üye değerleri** yazılır | SQLAlchemy öntanımlı olarak üye *adını* saklar; `asr+rules`, `45+`, `Definition` gibi değerler ancak böyle korunur. |

### 6.2 Eklenen davranış

- **Motor:** `make_engine` SQLite'ta `foreign_keys=ON`, `busy_timeout=5000`, dosya tabanlıysa `journal_mode=WAL` + `synchronous=NORMAL` uygular; `:memory:` için `StaticPool` seçer. MySQL'de `charset=utf8mb4`, `sql_mode=STRICT_TRANS_TABLES`, `pool_pre_ping`, `pool_recycle=3600`. Tabloların hepsi `ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_0900_ai_ci` ile üretilir.
- **Migrasyon yardımcıları:** `upgrade_head` URL'nin yanı sıra `Engine`/`Connection` alır (paylaşılan `:memory:` test veritabanı aksi hâlde boş kalırdı); `downgrade`, `current_revision`, `head_revision`, `diff_metadata` ve `offline_sql` aynı modülde. `offline_sql("mysql+pymysql://...")` MySQL sunucusu olmadan tüm migrasyonları MySQL lehçesiyle derler — `test_alembic_upgrade_head_on_sqlite_and_mysql_dialect_compile` bunu kullanır.
- **Şema yalnızca migrasyonlardan kurulur;** testlerde de `metadata.create_all` kullanılmaz, böylece her koşu migrasyonları sınar.
- **Sayfalama:** `{items, next_cursor}` ULID anahtarına dayalıdır (`id < cursor`, azalan sıra); araya yeni satır girse bile sayfa kaymaz. Sayfa boyutu 1–200 arasına sıkıştırılır.
- **Aramada puan:** `search_text` MySQL'de `MATCH ... AGAINST` (SQLAlchemy `Column.match`, ham SQL yok), SQLite'ta `LIKE`; her iki yolda da adaylar `difflib.SequenceMatcher` oranıyla yeniden sıralanır ve `(birim, puan)` çiftleri döner.
- **Girdi esnekliği:** `book`/`collect`/`api` paketleri pydantic modeli, dataclass veya düz sözlük geçirebilir; `repositories.base.field_of` üçünü de okur, `enums.coerce_enum` üye/değer/ad biçimlerini kabul eder. Bilinmeyen değer `InputValidationError` üretir (asla sessiz kabul).
- **KVKK:** `mark_withdrawn` konuşmacının `withdrawn_at` alanını damgalar, `ip_hash`/`user_agent_hash` alanlarını `NULL`'lar, kayıtları `deleted_at` ile yumuşak siler ve rıza satırını (denetim kaydı) korur; `exportable()` yalnızca QC'den geçmiş, silinmemiş ve geri çekilmemiş kayıtları döndürür.
- **SUS puanı:** `FeedbackRepository.add_sus` puan verilmediğinde ADR-0015 formülünü uygular (`aicls.eval.sus_score` tanımın sahibidir; `db` katman kuralı gereği `eval`'i içe aktaramaz). Gecikme yüzdelikleri "nearest-rank" yöntemiyle hesaplanır.

### 6.3 Doğrulama

`tests/integration/db/` (migrasyonlar, şema kuralları, motor, depo davranışları) ve `tests/unit/db/` (genel API sözleşmesi, enum/kayıt yardımcıları, `tools/lint_no_raw_sql.py`): 118 test, 1 atlanan (`mysql` işaretli gece koşusu), `aicls.db` satır kapsamı %99.
