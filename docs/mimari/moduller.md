# Modüller

Her paket için: sorumluluk, açık API (imzalar), bağımlılıklar, sahibi ve zorunlu test kapsamı. Katman düzeni ve import yasakları [genel-bakis.md](genel-bakis.md#3-modül-sınırları) içindedir. Sahipler: **cs** = Çağrı Şahin, **mmt** = Mehmet Murat Turhan, **sg** = Şahiner Güler. Kapsam eşikleri `scripts/coverage_gate.py` ile zorlanır.

Ortak kurallar:

- Genel API'ler `__init__.py` içinde yeniden dışa aktarılır; iç modüller alt çizgi ile başlamaz ama belgelenmemiş olanlar özel sayılır.
- Kullanıcıya görünen her Türkçe metin `aicls.core.messages_tr` içinden gelir; paketler kendi Türkçe dizesini tutmaz.
- Her paket `py.typed`; `mypy --strict` deterministik çekirdekte (core, db, latex2sympy, mathengine, nlu, book, collect, eval, api), standart mod + `ignore_missing_imports` torch komşusu paketlerde (ocr, asr, data, ml).

---

## aicls.core — sahibi cs — kapsam ≥ 90 %

**Sorumluluk.** Sıfır proje-içi bağımlılık: ayarlar ve profiller, yollar, hata hiyerarşisi ve kodları, Türkçe kullanıcı mesajları, yapılandırılmış günlük, ULID, hash, aşama zamanlayıcı, Türkçe metin yardımcıları, onay kapısı, LaTeX temizleyici.

```python
class Settings(BaseSettings):                      # env prefix AICLS_, .env dosyası
    env: Literal["dev", "test", "prod"] = "dev"
    profile: Literal["full", "collect"] = "full"
    data_root: Path                                 # AICLS_DATA_ROOT, OneDrive dışı
    models_root: Path                               # AICLS_MODELS_ROOT
    db_url: str                                     # sqlite+pysqlite:///... | mysql+pymysql://...
    jwt_secret: SecretStr
    cors_origins: list[str] = []
    models_enabled: bool = False                    # False → StubRecognizer/StubTranscriber
    llm_fallback_enabled: bool = False
    contrib_enabled: bool = False
    ethics_approval_ref: str = ""                   # boşsa contribute yönlendiricisi 503 GATE_BLOCKED
    compute_timeout_s: float = 2.5
    sandbox_workers: int = 2
    gate_dataset_download: str = ""                 # "approved" | ""
    gate_weights_download: str = ""
    gate_cuda_install: str = ""
    gate_training: str = ""
    log_level: str = "INFO"

def get_settings() -> Settings                      # lru_cache; testlerde override
def data_root() -> Path; def models_root() -> Path; def runs_root() -> Path; def repo_root() -> Path
def package_resource(pkg: str, name: str) -> Path   # importlib.resources ile paket içi YAML/lark
def long_path(path: Path | str) -> Path             # Windows MAX_PATH kaçışı (\?\ öneki)

class ErrorCode(StrEnum):
    PARSE_ERROR; UNSUPPORTED_CONSTRUCT; UNSUPPORTED_PROBLEM; CAS_TIMEOUT; NLU_NO_INTENT
    OCR_INPUT_INVALID; ASR_INPUT_INVALID; UPLOAD_INVALID; CONSENT_REQUIRED; AUTH_INVALID
    FORBIDDEN; NOT_FOUND; RATE_LIMITED; TRAINING_BUSY; VALIDATION_ERROR; GATE_BLOCKED; INTERNAL
class AiclsError(Exception):
    code: ErrorCode; message_tr: str; details: dict[str, Any]; http_status: int
def msg(code: ErrorCode | str, **kwargs: Any) -> str   # messages_tr sözlüğünden biçimlendirir
MSG_UNSUPPORTED_PROBLEM = "Bu problem tipi şu an için desteklenmemektedir."   # sözleşme metni, test edilir

def ulid(*, timestamp_ms: int | None = None) -> str; def short_hash(s: str, n: int = 8) -> str
class UlidGenerator                                 # süreç içi monoton üretici; testler kendi örneğini kurar
def sha256_file(p: Path) -> str; def sha256_json(obj: Any) -> str; def fingerprint_jsonl(p: Path) -> str   # "sha256:<hex>"
class StageTimer:                                   # with timer.stage("nlu"): ...; timer.stages -> dict[str, float] (ms)
    def add(name, ms); def server_timing_header(*, total_name="app") -> str; .total_ms; .counts; .depth
def tr_casefold(s: str) -> str                      # İ→i, I→ı, sonra casefold
def tr_lower(s: str) -> str; def strip_diacritics(s: str) -> str; def strip_suffix(word, *, families=None, min_stem=3) -> str
TR_SUFFIXES: dict[str, list[str]]                   # ünlü uyumuna göre -i/-ı/-u/-ü, -in/-ın, -ini/-ını ...
def require_gate(name: Literal["DATASET_DOWNLOAD","WEIGHTS_DOWNLOAD","CUDA_INSTALL","TRAINING"], *, cli_flag: bool,
                 ledger_path: Path | None = None, settings: Settings | None = None) -> None
class GateError(AiclsError)                         # exit code 3; mesajda üç koşul listelenir; details["missing"]
def read_gate_ledger(path: Path | None = None) -> list[GateEntry]   # docs/gelistirme/onay-defteri.md tablosu
def sanitize_latex(s: str, *, max_len: int | None = None) -> str   # izin listesi dışı komutları kaldırır, \href \url \includegraphics \input \write yasak
def find_disallowed_commands(s: str) -> list[str]  # tanı amaçlı: kaldırılacak komut adları
def get_logger(name) -> FilteringBoundLogger; def configure_logging(level=None, *, json_output=True, stream=None)
def request_context(request_id=None); def bind_request_id(request_id); def log_error(logger, err, *, request_id=None)
```

**Bağımlılıklar.** yok (pydantic, pydantic-settings, structlog).
**Zorunlu testler.** `tests/unit/core/`: settings yükleme ve override, `tr_casefold` Türkçe İ/ı, `require_gate` üçlü kilit (bayrak yok / env yok / defter satırı yok → GateError), `sanitize_latex` izin listesi, `StageTimer` iç içe aşamalar, `fingerprint_jsonl` satır sırasına duyarsız mı (**hayır**, sıra dahil; test bunu belgeler).

**Uygulama notları (2026-09-07).**

- `Settings`: `models_root` ve `db_url` verilmezse `data_root`'tan türetilir (`<data_root>/models`, `sqlite+pysqlite:///<data_root>/aicls-dev.db`); `jwt_secret` yalnızca `env=prod` iken ≥ 32 karakter zorunludur. `.env.example`/conftest ile uyum için ek alanlar: `llm_fallback` (kısa takma ad `AICLS_LLM_FALLBACK`, `llm_fallback_enabled` ile OR'lanır), `llm_endpoint`, `llm_api_key`, `rate_limit_enabled`. `settings.gate_value(name)` / `settings.is_gate_env_approved(name)` kapı alanlarını okur; `settings_override(**alanlar)` bağlam yöneticisi testler içindir. Varsayılan `data_root`: Windows'ta `D:\aicls-data` (D: yoksa `C:\aicls-data`), diğerlerinde `~/aicls-data`.
- `AiclsError(code, message_tr=None, details=None, http_status=None)`: `message_tr` ve `http_status` koddan türetilir; `to_problem(instance=...)` RFC 9457 sözlüğü üretir (`type`, `title`, `status`, `detail_tr`, `code`, `instance` + `details` alanları; çakışan anahtarlar `details` altına gider); `exit_code` sınıf niteliği (1; `GateError` 3). Bilinmeyen kod dizesi `INTERNAL`'a düşer (`details["original_code"]`). Alt sınıflar: `InputValidationError` (422), `NotFoundError` (404), `InternalError` (500), `GateError` (503), `GateLedgerError` (422).
- `msg(code, **kw)`: bilinmeyen anahtar veya eksik yer tutucu **KeyError** verir (programlama hatası, testte yakalanır). Kod başına varsayılan metinler `MESSAGES[<KOD>]`, kısa başlıklar `TITLES[<KOD>]`.
- `require_gate`: `CI` veya `GITHUB_ACTIONS` ortam değişkeni doğruyken üç koşul sağlansa da reddeder (`missing=["ci"]`, ADR-0012 "CI reddi test eder"). Env koşulu `Settings` üzerinden okunur (`.env` sayılır). Başarılı geçiş `AICLS_DATA_ROOT/gate_invocations.log` dosyasına JSON satırı ekler (`gate`, `at`, `who`, `git_sha` (`.git/HEAD` dosyasından, git çalıştırmadan), `config_hash`, `ledger_entry`). Defter en son satır kuralıyla okunur (`revoked` sonuncuysa onay yok); biçim hatası `GateLedgerError`.
- `fingerprint_jsonl` `"sha256:<hex>"` döner (rapor alanı biçimi); satır sırası dahil, `\r\n`/`\n` ve son satır sonu farkı dahil değil. `sha256_json` kanonik JSON (sıralı anahtar, sıkı ayraç, `ensure_ascii=False`; `Path`→POSIX, `set`→sıralı liste, `bytes`→hex).
- `StageTimer`: aşamalar ilk görülme sırasında (dış aşama içtekilerden önce); aynı ad tekrar edince süreler toplanır; `server_timing_header()` `"nlu;dur=12.3, ..., app;dur=58.2"` üretir (`app` = zamanlayıcının kuruluşundan bu yana geçen süre). Saat enjekte edilebilir (`StageTimer(clock=...)`).
- `ulid()` süreç genelinde monoton (aynı milisaniyede veya saat geri gidince rastgele kısım artırılır); bu yüzden açık `timestamp_ms` en son üretilen kimlikten geriye gidemez — sabit zaman damgası isteyen testler `UlidGenerator()` kurar.
- `sanitize_latex`: yasaklı komutlar (`\href \url \includegraphics \input \include \write \def \let \newcommand \csname \catcode \usepackage \html* \rule ...`) **argümanlarıyla** silinir; izin listesi dışı diğer komutlar yalnızca komut belirteci silinerek (`\foo{x}` → `{x}`), izinsiz ortamlar sarmalı açılarak geçer; `$`/`#` başıboş karakterleri, kontrol karakterleri ve sondaki tek `\` atılır; küme parantezleri dengelenir; `$...$`, `\(...\)`, `\[...\]` sarmalayıcıları soyulur. Fonksiyon idempotenttir (hypothesis ile test edilir). Uzunluk sınırı varsayılan olarak uygulanmaz; API katmanı `max_len=MAX_LATEX_LENGTH` (2048) verir.
- Günlük: `get_logger` ilk çağrıda `AICLS_LOG_LEVEL` ile JSON yapılandırması kurar; çekirdek modüller içe aktarma anında logger oluşturmaz (yapılandırma sonradan değiştirilebilsin diye).

---

## aicls.db — sahibi mmt — kapsam ≥ 80 %

**Sorumluluk.** SQLAlchemy 2.0 tipli modeller, motor/oturum fabrikası (SQLite dev/test, MySQL prod), sorgu üreten tek yer olan depo (repository) sınıfları, Alembic migrasyonları.

```python
class Base(DeclarativeBase)                         # naming_convention, type_annotation_map {dict: JSON}
class TimestampMixin: created_at: Mapped[datetime]; updated_at: Mapped[datetime]
def make_engine(url: str, *, echo: bool = False) -> Engine   # sqlite: WAL, PRAGMA foreign_keys=ON, check_same_thread=False; mysql: utf8mb4, pool_pre_ping, STRICT_TRANS_TABLES
def session_scope(engine: Engine) -> ContextManager[Session]
def get_session() -> Iterator[Session]              # FastAPI bağımlılığı
def upgrade_head(url: str) -> None                  # programatik alembic

class UserRepository:      get_by_email(email) -> User | None; create(email, password_hash, display_name, role) -> User; set_role(user_id, role) -> None; get(user_id); set_active(user_id, *, is_active); page(cursor, limit) -> Page[User]
class WorkspaceRepository: list_for_user(user_id) -> list[Workspace]; create(owner_id, title) -> Workspace; get_owned(ws_id, user_id) -> Workspace | None; add_member(ws_id, user_id, role_in_ws); delete(ws_id, user_id) -> bool
class OperationRepository: add(ws_id, user_id, op: OperationCreate | dict) -> Operation; page(ws_id, cursor, limit) -> Page[Operation]; delete(ws_id, op_id) -> bool; get(op_id)
class AssetRepository:     add(kind, path, *, sha256, size_bytes, mime, owner_id, retention_until) -> Asset; get(asset_id); expired(*, now, limit) -> list[Asset]
class RecognitionRepository: add(latex, *, user_id, operation_id, ink_asset_id, image_asset_id, confidence, alternatives, model_version, elapsed_ms) -> Recognition; set_correction(rec_id, latex) -> None; corrections(); add_transcript(text, ...) -> Transcript; set_transcript_correction(id, text)
class RecordingRepository: add(speaker_id, prompt_id, path, qc: dict, duration_s, sha256) -> Recording; next_prompts(speaker_id, n) -> list[PromptSentence]; stats() -> ContribStats; mark_withdrawn(speaker_id) -> int; upsert_consent_version(...); create_speaker(...); get_speaker(id|pseudonym); speaker_for_user(user_id); add_prompts(rows) -> int; set_review(...); set_split(...); qc_queue(status); exportable() -> list[tuple[Recording, Speaker, PromptSentence]]
class BookRepository:      upsert_units(units, triples, aliases, *, tagged_ratio) -> LoadReport; get(unit_id) -> BookUnit | None; search_alias(alias_norm, unit_type) -> list[BookUnit]; search_text(q, unit_type, limit) -> list[tuple[BookUnit, float]]; related(unit_id, predicate) -> list[BookUnit]; units(); triples(); chapters(); count()
class ModelVersionRepository: register(component, version, path, sha256, metrics, run_id) -> ModelVersion; active(component) -> ModelVersion | None; activate(component, version, *, promoted_by) -> ModelVersion; list_versions(component)
class EvalRunRepository:   add(report: dict, *, is_reportable: bool) -> EvalRun; list(kind, since, *, limit) -> list[EvalRun]; reportable(kind)
class LatencyRepository:   add(sample: dict) -> None; stats(since, modality) -> LatencyStats
class FeedbackRepository:  add_feedback(user_id, category, text, page, *, screenshot_asset_id, collected_week) -> Feedback; add_sus(user_id, cohort, answers, *, score, collected_week) -> SusResponse; page(status, category, cursor, limit) -> Page[Feedback]; set_status(id, status); sus_average(cohort)
class AuditRepository:     log(actor_id, action, target, detail, *, ip_hash) -> AuditLog; page(action, cursor, limit) -> Page[AuditLog]
```

**Bağımlılıklar.** core. (Depo sınıfları `dict`/pydantic girdisi alır; asla `aicls.ml` veya `aicls.eval` tipi almaz.)
**Zorunlu testler.** `tests/integration/db/`: `alembic upgrade head` SQLite'ta, head→base→head, autogenerate diff boş, tüm migrasyonların MySQL lehçesiyle derlenmesi, `mysql` işaretli gece koşusu.

**Uygulama notları.**

- Modüller: `base.py` (Base, `TimestampMixin`, `UlidPk`/`Digest`/`GitSha` sütun takma adları, `MYSQL_TABLE_KWARGS`), `enums.py` (`coerce_enum` ile üye/değer/ad kabulü), `models/` (identity, compute, audio, book, registry), `engine.py`, `repositories/`, `migrate.py`, `migrations/`.
- Sözleşmeyi genişleten (bozmayan) noktalar: `upgrade_head` bir URL'nin yanı sıra `Engine`/`Connection` da kabul eder (paylaşılan `:memory:` testleri için zorunlu); `session_scope(engine=None)` varsayılan motoru kullanır; `activate` ve `log` `None` yerine yazdıkları satırı döndürür; ayrıca `AssetRepository`, `set_default_engine`, `dispose_engines`, `engine_options`, `diff_metadata`, `offline_sql` eklendi.
- Depo sınıfları hiç `commit` etmez; işlemi `session_scope` veya `get_session` sahiplenir. Sayfalama ULID anahtarına dayalıdır (`id < cursor`, azalan), böylece yeni satırlar sayfaları kaydırmaz.
- `aicls.db` yalnızca `aicls.core` ve üçüncü taraf paketleri içe aktarır; `tests/unit/db/test_public_api.py` bunu AST ile denetler.

---

## aicls.latex2sympy — sahibi mmt — kapsam ≥ 95 %

**Sorumluluk.** LaTeX matematik dizesi → tipli AST → SymPy; AST → kanonik LaTeX; SymPy → KaTeX uyumlu LaTeX; normalizasyon; gidiş-dönüş doğrulama. Ayrıntı [latex-ayristirici.md](latex-ayristirici.md).

```python
def parse_latex(src: str, *, strict: bool = True) -> Node
def to_sympy(node: Node, symbols: SymbolTable | None = None, *,
             config: ParserConfig | None = None, evaluate: bool = True) -> sympy.Basic
def latex_to_sympy(src: str, *, symbols: SymbolTable | None = None, config: ParserConfig | None = None,
                   strict: bool = True, evaluate: bool = True) -> sympy.Basic
def to_latex(node: Node) -> str                            # kanonik AST yazıcısı
def sympy_to_latex(expr: sympy.Basic, *, style: PrintStyle = PrintStyle.KATEX) -> str
def normalize_latex(src: str) -> NormalizeResult            # .text, .repairs: list[str] (yapılan onarımlar izlenir)
def roundtrip_check(src: str, *, numeric_samples: int = 5) -> RoundTripResult   # ok, reason, canonical
def expressions_equivalent(a, b, *, numeric_samples: int = 5) -> bool   # gidiş-dönüş 3. seviye
def numerically_equal(a, b, *, samples: int = 5) -> bool                # tohumlu (1234) örnekleme
class LatexParser: parse(src, *, strict=True) -> Node; parse_with_repairs(src) -> tuple[Node, list[str]]
class SymbolTable: declare(name: str, **assumptions) -> None; symbol(name) -> sympy.Symbol; reserved: frozenset[str]
                   is_declared(name) -> bool; bound(name, **assumptions)   # bağlam yöneticisi (Σ/Π indisi)
class ParseError(AiclsError): line: int; col: int; expected: list[str]; snippet: str
class UnsupportedConstruct(AiclsError): construct: str
class SemanticError(AiclsError): reason: str               # details["reason"] makine okunur
class ParserConfig: log_base: Literal["e", "10"] = "e"; imaginary_i: bool = False; real_symbols: bool = True
class PrintStyle(StrEnum): KATEX; RAW
MAX_INPUT_CHARS = 2048
# AST (ast.py, dondurulmuş dataclass, span=(start,end) eşitlikte yok sayılır):
# Num Sym Const MathSet Neg Add Mul Div Pow Root Abs Factorial PlusMinus Func Log Call Prime
# Integral Derivative Limit Sum Product Relation Interval Matrix + Visitor[T] (tam kapsamlı)
# commands.py: command_table(), classify(), scan_commands(), report_rows(), SUPPORTED_COMMANDS,
#              UNSUPPORTED_CONSTRUCTS  (kaynak: grammar/commands.yaml)
```

**Bağımlılıklar.** core (lark, sympy, pyyaml).
**Zorunlu testler.** birim (dilbilgisi kuralı başına), özellik tabanlı gidiş-dönüş, `datasets/latex/valid_inputs.jsonl` ayrıştırma oranı ≥ 0.99, `sympy.latex()` çıktısının dilbilgisi içinde kalması, `sympy.parsing.latex` ile farksal test (antlr yoksa atlanır).
**Durum (2026-09-07).** Uygulandı; 282 test geçiyor, kapsam %98.5, `valid_inputs.jsonl` (3599 satır) üzerinde sıkı başarı oranı %100. Sapmalar ve ek davranışlar: [latex-ayristirici.md §9](latex-ayristirici.md#9-uygulama-notları).

---

## aicls.mathengine — sahibi mmt — kapsam ≥ 95 %

**Sorumluluk.** Tipli komut modelleri, sandbox'lı SymPy yürütme (zaman aşımı, bellek en-iyi-çaba), cache, adımlar, deterministik grafik, sembolik eşdeğerlik, desteklenmeyen problem politikası.

```python
class Diff(BaseModel):      op: Literal["diff"]; expr: LatexStr; var: str = "x"; order: int = 1
class Integrate(BaseModel): op: Literal["integrate"]; expr: LatexStr; var: str = "x"; lower: LatexStr | None = None; upper: LatexStr | None = None
class Limit(BaseModel):     op: Literal["limit"]; expr: LatexStr; var: str = "x"; to: LatexStr; direction: Literal["+", "-", "+-"] = "+-"
class Solve(BaseModel):     op: Literal["solve"]; equation: LatexStr; var: str = "x"; domain: Literal["reals", "complexes"] = "reals"
class Simplify / Expand / Factor(BaseModel): op: Literal[...]; expr: LatexStr
class Series(BaseModel):    op: Literal["series"]; expr: LatexStr; var: str = "x"; point: LatexStr = "0"; order: int = 6
class Evaluate(BaseModel):  op: Literal["evaluate"]; expr: LatexStr; subs: dict[str, LatexStr]
class Plot(BaseModel):      op: Literal["plot"]; expr: LatexStr; var: str = "x"; xmin: float = -10.0; xmax: float = 10.0
CasCommand = Annotated[Diff | Integrate | Limit | Solve | Simplify | Expand | Factor | Series | Evaluate | Plot, Field(discriminator="op")]

class Step(BaseModel):      title_tr: str; latex: str; rule: str = ""; detail_tr: str = ""
class Result(BaseModel):    kind: Literal["expr","number","set","bool","plot","steps"]; latex: str; plain: str; srepr: str; steps: list[Step]; plot_png: bytes | None; timings: dict[str, float]; cache_hit: bool; elapsed_ms: float

class MathEngine:
    def __init__(self, executor: Executor | None = None, cache: LruCache | None = None, timeout_s: float | None = None)
    def execute(self, cmd: CasCommand | Mapping[str, Any]) -> Result   # doğrulama + LRU cache + StageTimer
    def validate(self, cmd) -> CasCommand;  def clear_cache(self) -> None
class Executor(Protocol):   def run(self, cmd: CasCommand, timeout_s: float) -> Result
class InProcessExecutor(Executor)                  # testler ve CLI; iş parçacığı zaman aşımı (yalnızca ölçüm)
class PoolExecutor(Executor):                      # üretim: spawn bağlamı, N ısıtılmış işçi (sympy + dilbilgisi + Agg ön-import)
    def start(self) -> None; def stop(self) -> None; def probe(self) -> WorkerProbe
    restarts: int; recycles: int; prewarm_diagnostics: list[WorkerProbe]  # zaman aşımında Process.kill() + eşzamansız yeniden başlatma
SandboxPoolExecutor = PoolExecutor                 # İP-4 notlarındaki ad; aynı sınıf
def pool_from_settings(**overrides) -> PoolExecutor                      # FastAPI lifespan yardımcısı
def compute(cmd: CasCommand) -> Result                                   # saf gönderim: cache yok, süreç yok
def symbolic_equivalent(a, b, *, numeric_probes: int = 5, tol: float = 1e-6) -> Literal["equal","different","unparsable"]
def render_plot(expr: sympy.Expr, var: sympy.Symbol, xmin: float, xmax: float, *, fmt: Literal["png","svg"] = "png", samples: int = 400) -> bytes
def canonical_key(cmd) -> str;  def command_input_chars(cmd) -> int;  def check_input_size(cmd) -> None
class LruCache:             get / put / clear / stats;  maxsize = DEFAULT_CACHE_SIZE = 4096
class UnsupportedProblem(AiclsError)               # message_tr = MSG_UNSUPPORTED_PROBLEM
class CasTimeout(AiclsError)
MAX_INPUT_CHARS = 2048
```

**Bağımlılıklar.** core, latex2sympy (sympy, numpy, matplotlib — matplotlib tembel yüklenir).
**Zorunlu testler.** öneri metnindeki örnekler (`diff(x**2)`, `∫₀¹x²`, `x²−4=0`, `lim sin x/x`), zaman aşımının işçiyi öldürüp havuzu toparladığı, ön ısıtma, cache isabeti, grafik PNG hash'inin deterministik olduğu, desteklenmeyen tipte sözleşme mesajı, eşdeğerliğin üç durumu, hypothesis ile diff/integrate tutarlılığı.
**Durum (2026-09-07).** Uygulandı; 231 test geçiyor, kapsam %98,3, `mypy --strict` ve `ruff` temiz. Sapmalar, ek davranışlar ve açık noktalar: [Uygulama notları](#aiclsmathengine--uygulama-notları).

### aicls.mathengine — Uygulama notları

**Sözleşmeye eklenenler (hiçbir ad değiştirilmedi).**

- `Result.elapsed_ms` ve `Result.timings` birlikte döner: `timings = {"sympy": işçideki saf SymPy süresi, "cas": motorun beklediği süre (süreç borusu dahil), "total": ...}`; cache isabetinde ayrıca `"cache"` anahtarı gelir. `elapsed_ms` daima `timings["total"]` ile aynıdır.
- `Step` alanları `{title_tr, latex, rule, detail_tr}` olarak sabitlendi. `web/` tarafının varsaydığı `{title_tr, latex}` ikilisi korunuyor; `rule` makine tarafı içindir (`product_rule`, `chain_rule`, `result`), `detail_tr` isteğe bağlıdır. (KNOWN_ISSUES'taki "Step'in gerçek şekli mathengine'e ait" maddesi kapandı.)
- `MathEngine.__init__` hem konumsal hem anahtar sözcüklü çağrıyı kabul eder (`MathEngine(executor, cache, timeout_s)` ve `MathEngine(executor=...)`); yukarıdaki yalnız-anahtar imza bunun alt kümesidir. Üç argüman da isteğe bağlıdır; varsayılanlar `InProcessExecutor()`, yeni bir `LruCache()` ve `settings.compute_timeout_s`.
- `MathEngine.execute` ham `{"op": ...}` sözlüğünü de kabul eder (API katmanı için); pydantic doğrulama hatası `InputValidationError`'a çevrilir.
- `PoolExecutor(target="modül:fonksiyon")`: işçinin görev başına çağırdığı giriş noktası dışarıdan verilebilir. Zaman aşımı yolu bu şekilde test edilir — "yavaş" bir SymPy ifadesine bel bağlamak, SymPy sürümü değiştiğinde yanlış nedenle kırılan bir test demektir. `PoolExecutor.probe()` ve `prewarm_diagnostics`, işçinin ısınma durumunu (`sympy_imported`, `grammar_ready`, `memory_limit_applied`, `memory_limit_bytes`, `warm_ms`, `tasks_done`) döndürür; `GET /health/ready` bunları kullanabilir. `memory_limit_bytes` gerçekten kurulan tavanı (bayt) taşır, 0 ise tavan yok — "sandbox'ın bellek sınırı var" cümlesi böylece belgede değil, ölçümde doğrulanır.
- `recycles` sayacı `restarts`'ın yanına eklendi: `max_tasks_per_worker` (500) dolduğunda işçi öldürülmez, düzgünce durdurulup yenisi açılır; bu bir çökme değildir ve `restarts`'ı artırmaz.
- Süreç sınırından geçen hatalar için `error_to_payload` / `payload_to_error`. İstisnalarımızın `__init__` imzası `Exception.__reduce__` varsayımına uymadığı için pickle ile güvenli taşınamazlar; JSON uyumlu sözlük olarak taşınıp ebeveynde aynı sınıf olarak yeniden kurulurlar.

**Türkçe metin borcu (core'a taşınacak).** Adım başlıkları ve grafik etiketleri için gereken sekiz dize `aicls.core.messages_tr` içinde yoktu ve bu paket `core`'u düzenleyemez. Geçici çözüm `aicls.mathengine.messages.tr()`: anahtarı önce `messages_tr.MESSAGES` içinde arar, bulamazsa `PENDING_CORE_MESSAGES` tablosuna düşer. `MSG_STEP_DERIVATIVE`, `MSG_STEP_PRODUCT_RULE`, `MSG_STEP_CHAIN_RULE`, `MSG_STEP_POWER_RULE`, `MSG_STEP_RESULT`, `MSG_PLOT_TITLE`, `MSG_PLOT_X_LABEL`, `MSG_PLOT_Y_LABEL` core'a eklendiğinde bu paket kod değişikliği olmadan core'un metnini kullanmaya başlar ve tablo silinebilir. (`aicls.asr.qc.QC_REASONS_TR` ile aynı desen.)

**Karar verilen davranışlar.**

- **Belirsiz integral `+ C` içermez.** SymPy'nin ters türevi ailenin bir temsilcisidir; sembolik bir `C` eklemek `aicls.eval`'in "türevini al, aynısı geri gelsin" denetimini bozar. "+ C" arayüzde eklenir.
- **İki taraflı limiti olmayan ifade** (`|x|/x`, x→0) `UnsupportedProblem(reason="limit_does_not_exist")` verir. Uydurma bir değer döndürmemek, projenin "halüsinasyon yok" kuralının limit hâlidir.
- **`solveset` `ConditionSet` döndürürse** problem desteklenmiyor sayılır: soruyu yeniden ifade eden bir küme cevap değildir.
- **Karmaşık alanda çözerken** değişken `real=True` varsayımı olmadan tanımlanır; aksi hâlde `x²+1=0` boş küme döner.
- **Adımlar yalnızca çarpım ve zincir kuralında** üretilir; başka durumlarda anlatı yoktur ve boş liste bir hata değil, belgelenmiş normal sonuçtur. Anlatı yapısaldır (kullanıcının yazdığı ağaç üzerinden gider), `simplify` çağırmaz, bu yüzden gecikme bütçesinin içinde kalır.
- **Grafikler `pyplot` kullanmaz** (küresel durum tutar, sandbox işçisinde güvenli değildir); `Figure` + `FigureCanvasAgg` nesne API'si, `rc_context` ile sabitlenmiş parametreler ve PNG `Software` etiketinin silinmesi sayesinde aynı ifade her seferinde aynı baytlara hash'lenir.
- **matplotlib tembel yüklenir** (`plotting.load_backend()`, `lru_cache`). `aicls.nlu` her istek yolunda komut modellerini import eder; bir türev için ~200 ms matplotlib bedeli ödemek 2,5 s bütçenin onda birini yer. Sandbox işçisi ısınırken `load_backend()` çağırır, böylece grafik isteği yine hazır bir arka uç bulur.
- **`symbolic_equivalent` kesilmiş seriyi ayrı ele alır**: `O(x⁶) − O(x⁶) = O(x⁶)` olduğundan iki serinin farkı hiçbir zaman sıfıra sadeleşmez, sayısal sonda da `O` terimini değerlendiremez. Kural: aynı mertebede kesilmiş iki serinin polinom kısımları eşdeğerse `equal`; kesme mertebesi farklıysa veya bir taraf çıplak polinomsa `different`.
- **Bellek sınırı ölçülmüş tabana göre kurulur, mutlak değildir** (ADR-0016 Ek, 2026-09-08). İşçi ısındıktan *sonra* `install_memory_cap(headroom)` çağrılır: tavan = o anki kullanım + 1 GB baş boşluk. Mutlak 1 GB'lık eski tavan Linux'ta tabanın altındaydı — NumPy'nin OpenBLAS'ı import anında CPU başına 128 MB adres alanı ayırır (8 çekirdekte 1032 MB, yerleşik değil), ısınma `MemoryError` alır ve her istek `INTERNAL` yanıtlar. Ölçülen büyüklük her platformda o platformun sınırladığı büyüklüktür: POSIX'te adres alanı (`/proc/self/statm` → `RLIMIT_AS`), Windows'ta commit (`PrivateUsage` → Job Object). Taban ölçülemezse tavan kurulmaz ve `memory_limit_applied=False` bunu söyler. Sınır hâlâ en-iyi-çabadır (kilitli bir makine iki mekanizmayı da reddedebilir); asıl koruma zaman aşımı, 2048 karakter girdi sınırı ve işçi döngüsüdür. Artık gerçek bir koruma olduğu `test_the_memory_cap_stops_a_runaway_allocation` ile iki platformda da kanıtlanır.

**Açık nokta — küme sonuçları yalnızca görüntülenebilir.** `sympy_to_latex` bir çözüm kümesini `\left\{-2, 2\right\}` olarak yazar; v1 Lark dilbilgisinde küme değişmezi yoktur, dolayısıyla küme cevabı ifade cevapları gibi girdi olarak geri verilemez. Sonuçları: (1) `aicls.api` / `web`, `kind == "set"` için "bu sonucu girdi yap" seçeneğini sunmamalı; (2) `aicls.eval` küme cevaplarını `latex`i yeniden ayrıştırarak değil `srepr` üzerinden karşılaştırmalı. Kapatmak için dilbilgisine küme değişmezi eklenmelidir (sahibi: latex2sympy). `tests/integration/test_mathengine_contracts.py::test_set_valued_results_are_display_only_in_v1` bu boşluğu açıkça sabitler.

**Müfredat testinin kaynağı.** `tests/unit/mathengine/test_curriculum.py` ~210 problemi kapalı formüllü ailelerden kod içinde üretir; bu bir gerileme (regression) takımıdır, İP-4'ün ≥1000 problemlik müfredat test kümesi değildir. Sözleşmedeki "> %99 doğruluk" metriği yalnızca o küme üzerinden raporlanabilir (bkz. KNOWN_ISSUES "Metrik kaynağı").

---

## aicls.nlu — sahibi cs — kapsam ≥ 95 %

**Sorumluluk.** Türkçe (yazılı veya ASR çıktısı) metin → normalizasyon → niyet/yuva → tipli komut; konuşulan matematik → LaTeX; açıklama isteme; niyet-yalnız LLM yedeği (kapalı). Ayrıntı [metin-komut-katmani.md](metin-komut-katmani.md).

```python
def load_inventory(path: Path | None = None) -> Inventory       # varsayılan paket kaynağı
class Inventory(BaseModel): version: str; intents: list[IntentSpec]; slot_types: dict[str, SlotType]; def validate_examples(self) -> list[str]
def normalize_text(text: str) -> NormalizedText                 # .tokens, .original, .spans
def spoken_to_latex(text: str) -> str
class ParseContext: last_latex: str | None = None; active_var: str | None = None; modality: Literal["text","voice","ink"] = "text"
def parse_command(text: str, *, context: ParseContext | None = None, inventory: Inventory | None = None, fallback: IntentFallback | None = None) -> ParseOutcome
class Matched:        command: CasCommand | BookFetch | UiAction; intent: str; slots: dict[str, Any]; score: float; alternatives: list[Alternative]; trace: list[str]
class Clarification:  question_tr: str; missing_slot: str | None; candidates: list[Alternative]
class NoMatch:        suggestions_tr: list[str]; reason: str
ParseOutcome = Matched | Clarification | NoMatch
class BookFetch(BaseModel): op: Literal["book_fetch"]; unit_type: UnitType | None; query: str
class UiAction(BaseModel):  op: Literal["ui"]; action: Literal["clear","undo","repeat_last","export"]
class IntentFallback(Protocol): def suggest(self, text: str, intents: Sequence[str]) -> str | None   # yalnızca niyet kimliği
```

**Bağımlılıklar.** core, latex2sympy, mathengine (komut modelleri). **db ve book yasak.**
**Zorunlu testler.** altın küme doğruluğu ≥ 0.95, her envanter örneğinin kendi niyetine çözülmesi, normalizasyon idempotans (hypothesis), ASR-gürültülü küme ≥ 0.90, LLM yedeğinin varsayılan olarak hiç çağrılmaması.

---

## aicls.ocr — sahibi cs — kapsam ≥ 70 % (ölçülen: 97,7 % satır)

**Sorumluluk.** El yazısı tanıma kenarı: `Recognizer` arayüzü, InkML/Canvas JSON okuma, iz özellikleri, raster, ablasyonlu ön işleme, LaTeX tokenizer, son işleme, model tanımları (torch tembel), çözme, stub ve torch uygulamaları.

```python
class OcrInputError(AiclsError)               # kod OCR_INPUT_INVALID; her geçersiz girdi bunu atar
class Stroke:  points: np.ndarray             # (N,3) float32 x,y,t (kopyalanır, salt okunur)
               @classmethod from_xyt(xs, ys, ts=None); n_points; xy; t; duration_ms; bbox
class Ink:     strokes: list[Stroke]; label: str | None; writer_id: str | None; meta: dict
               n_strokes; n_points; is_empty; bbox; duration_ms; t0_ms; all_points()
class Image:   array: np.ndarray              # (H,W) veya (H,W,C) uint8; @classmethod from_bytes(data, *, max_bytes=2 MB, grayscale=True); to_float01()
class Recognition: latex: str; tokens: list[str]; confidence: float; alternatives: list[tuple[str, float]]; elapsed_ms: float; model_version: str; to_dict()
class Recognizer(Protocol):
    def recognize(self, inp: Ink | Image, *, beam: int = 4) -> Recognition
    def warmup(self) -> None
def get_recognizer(name: str = "stub", *, model_dir: Path | None = None, device: str = "cpu") -> Recognizer
def available_recognizers() -> tuple[str, ...]                                   # ("stub", "torch")
def read_inkml(src: Path | str | bytes) -> Ink; def write_inkml(ink: Ink) -> bytes   # defusedxml, DTD/varlık/dış başvuru yasak
def ink_from_canvas_json(obj: dict) -> Ink; def ink_to_canvas_json(ink: Ink) -> dict  # mutlak zaman damgası meta["t0_ms"] + göreli t
def normalize_strokes(ink: Ink, *, resample_px: float = 2.0) -> np.ndarray       # (T, 12) delta+kalem+Bezier özellikleri
def resample_polyline(points, spacing) -> np.ndarray; def fit_cubic_bezier(points) -> np.ndarray  # (4,2) kontrol noktası
STROKE_FEATURE_NAMES: tuple[str, ...]; STROKE_FEATURE_DIM = 12
def render_ink(ink: Ink, size: tuple[int, int] = (96, 384), *, stroke_width: int = 2) -> np.ndarray   # float32, kağıt 1.0 / mürekkep 0.0
def render_ink_to_image(ink, size=(96, 384), *, stroke_width=2) -> Image
class PreprocessConfig: steps: Sequence[Literal["gaussian_blur","laplacian_sharpen","grayscale","normalize01","resize_pad"]] = (); size: tuple[int,int] = (96, 384); blur_sigma=1.0; sharpen_amount=1.0; pad_value=1.0; as_dict()
def preprocess_image(img: np.ndarray | Image, cfg: PreprocessConfig) -> np.ndarray    # steps boşsa birebir aynı dizi
class LatexTokenizer: tokenize(latex) -> list[str]; encode(latex, *, add_special=False, max_len=None) -> list[int]; decode(ids, *, skip_special=True) -> str; normalize(latex) -> str; vocab; vocab_size; token_to_id/id_to_token; save(path); @classmethod from_file(path); @classmethod build(labels, *, min_freq=1, extra_tokens=())
def default_tokenizer() -> LatexTokenizer     # müfredat tablosundan (önbellekli); SPECIAL_TOKENS = <pad> <s> </s> <unk>
def postprocess_latex(latex: str, *, sanitize: bool = True) -> str               # boşluk + süslü parantez onarımı + core.sanitize_latex
def repair_braces(latex) -> str; def canonicalize_whitespace(latex) -> str; def grammar_precheck(latex) -> PrecheckReport(ok, reasons)
class OcrModelConfig(BaseModel): family: "seq2seq_image"|"seq2seq_stroke"; encoder: cnn_small|resnet18|resnet34|vgg16|efficientnet_b0|stroke_transformer|stroke_bilstm; decoder: transformer|lstm_attn|gru_attn; d_model=256; layers=6; heads=8; ff_dim; dropout=0.1; max_len=160; image_size=(96,384); stroke_feature_dim=12
def build_model(cfg: OcrModelConfig, vocab_size: int) -> "torch.nn.Module"       # torchvision kodlayıcıları weights=None; tembel import
def greedy_decode(model, inputs, *, max_len=160) -> tuple[list[int], float]      # aicls.ocr.models.seq2seq; tek örnek
def beam_search(model, inputs, *, beam=4, max_len=160, length_penalty=1.0) -> list[tuple[list[int], float]]   # tek örnek
def greedy_decode_batch(model, inputs, *, max_len=160, bos_id=1, eos_id=2, incremental=True) -> list[tuple[list[int], float]]
def beam_search_batch(model, inputs, *, beam=4, max_len=160, bos_id=1, eos_id=2, length_penalty=1.0, incremental=True) -> list[list[tuple[list[int], float]]]
class DecoderState: reorder_beams(index) -> None    # aicls.ocr.models.decoders; artımlı çözme durumu
# çözücüler ayrıca: init_state(memory, *, max_len) -> DecoderState; step(state, token_ids) -> (rows, V)
class StubRecognizer(Recognizer)              # deterministik: ink.label varsa onu, yoksa hash'e bağlı müfredat LaTeX'i
class TorchRecognizer(Recognizer):  @classmethod load(model_dir: Path, device: str = "cpu") -> TorchRecognizer
```

**Bağımlılıklar.** core (numpy, Pillow, defusedxml; torch/torchvision yalnızca `models/`, `torch_recognizer.py` içinde tembel).
**Zorunlu testler.** InkML gidiş-dönüş, Canvas JSON zaman damgaları, XXE reddi, ön işleme adımlarının ablasyonlu ve sıralı olması, tokenizer müfredat LaTeX'inde gidiş-dönüş, stub determinizmi, küçük modelin (d_model=32) sentetik yığında ileri geçiş şekli, ışın aramanın k sıralı sonuç vermesi, torchvision kodlayıcılarının `weights=None` ile kurulması, `import aicls.ocr`'ın torch'u yüklememesi.

### Uygulama notları

- **Zaman damgaları.** Tarayıcı mutlak epoch ms gönderir; float32 bunu tam saklayamaz. `ink_from_canvas_json` en küçük damgayı `meta["t0_ms"]` içine alır, noktalarda göreli ms tutar (`t0_ms + t` özgün damgayı verir). `write_inkml` `t0_ms`'i ve diğer `meta` alanlarını `annotation` olarak yazar; koordinatlar 9 anlamlı basamakla yazıldığından gidiş-dönüş float32'de birebirdir.
- **Girdi bütçeleri** (docs/mimari/api.md §1) kodda sabittir: InkML ≤ 1 MB, ≤ 20 000 nokta, ≤ 2 000 iz; raster ≤ 2 MB ve ≤ 4096 px. Aşım `OcrInputError` (`OCR_INPUT_INVALID`, 400).
- **Tuval boyutu (entegrasyon kapısı, 2026-09-07).** İstemci yüzey boyutunu iç içe `{"canvas": {"w": 800, "h": 300}}` olarak gönderir (api.md §5 ve `web/src/lib/inkml.ts`); `ink_from_canvas_json` başlangıçta yalnızca düz `width`/`height` anahtarlarını okuduğu için boyut sessizce düşüyordu. Artık her iki biçim de kabul edilir ve `meta["canvas_width"] / meta["canvas_height"]` alanlarına yazılır (düz anahtarlar öncelikli); `ink_to_canvas_json` boyut biliniyorsa her ikisini de üretir. Sözleşme `tests/integration/test_web_payload_contracts.py` ile kilitlenmiştir.
- **XXE savunması.** `read_inkml` `defusedxml` ile `forbid_dtd=True, forbid_entities=True, forbid_external=True` kullanır; DOCTYPE taşıyan belge ayrıştırılmadan reddedilir (`details.reason = "xml_dtd_forbidden"`).
- **Raster kutbu.** `render_ink` kağıdı 1.0, mürekkebi 0.0 verir (taranmış sayfayla aynı kutup), 2× süper örnekleme + BOX küçültme ile deterministiktir; ölçek/konum bbox'tan türetildiği için öteleme ve büyütmeye karşı değişmezdir.
- **Ön işleme.** Adımlar **verilen sırayla** uygulanır ve her biri tek başına ölçülebilir; `steps=[]` girdiyi (dtype dahil) aynen döndürür. `normalize01` tamsayı girdiyi 255'e böler, `resize_pad` en-boy oranını koruyup `pad_value` ile ortalar. Gauss/Laplace çekirdekleri saf numpy'dır (scipy bağımlılığı yok).
- **Tokenizer kanonik biçimi.** Boşluk token değildir; `decode` yalnızca gerekli yere (komut sözcüğü + harf) boşluk koyar. Bu yüzden `decode(encode(s))` bir *normal biçim*dir ve müfredat tablosundaki 32 ifade için birebir `s`'e eşittir. `normalize()` sözlükten bağımsız aynı biçimi verir.
- **Müfredat tablosu.** Gerçek sözlük MathWriting etiketlerinden üretilecektir (`aicls.data.prepare_shards` → `vocab.json`); paketteki `curriculum.py` yalnızca torch'suz yolu (stub, API duman testleri, değerlendirme koşum takımı) çevrimdışı ayakta tutar.
- **Tembel torch.** `aicls.ocr` içe aktarımı torch'a dokunmaz; `aicls.ocr.models.encoders/decoders/seq2seq` modülleri torch'u modül düzeyinde içe aktarır ve bunlara tek kapı `build_model` (önce `require_torch()`, torch yoksa Türkçe mesajlı `TorchUnavailableError`). `test_importing_aicls_ocr_does_not_import_torch` bunu ayrı bir süreçte doğrular.
- **Kontrol noktası düzeni.** `TorchRecognizer.load` yalnızca yerel dizinden okur: `model.safetensors | model.pt`, `config.json | config.yaml` (`model:` bloğu da kabul edilir), `vocab.json`, isteğe bağlı `model_version.txt` (yoksa dizin adı sürüm olur). Hiçbir yerde hub kimliği kullanılmaz (ADR-0012).
- **Yığın hâlinde ve artımlı çözme (2026-09-08, dalga 7).** `greedy_decode`/`beam_search` her adımda çözücüyü **tüm önek** üzerinde yeniden koşuyordu (`O(T^2)`) ve her örneği — ışın aramada her örneğin her ışınını — ayrı bir `batch == 1` ileri geçişte çözüyordu. RTX 5090'da `ocr_excerpt_wiring` koşusunda ölçülen bedel: **16 doğrulama örneği için 296 s, örnek başına 18,5 s**, aynı koşuda 20 eğitim adımı toplam 1,4 s. Sevk edilen `eval.max_n_val: 2000` ile bu, 4 dakikalık bir eğitim aralığına karşılık 10,3 saatlik bir doğrulama geçişi demektir. Çözüm iki parçalıdır: (1) `greedy_decode_batch` / `beam_search_batch` tüm yığını (ışın aramada `memory.repeat_interleave(beam, 0)` ile tüm ışınları) tek geçişte çözer; (2) her iki çözücü artık `init_state`/`step` ile anahtar/değer önbelleği (Transformer) ya da RNN gizli durumu (LSTM/GRU) taşır, böylece adım başına iş `O(1)`'dir. Eski tek örneklik imzalar korunur ve yığın sarmalayıcısıdır. **Artımlı yol yalnızca `eval()` kipinde kullanılır**: tek konumluk bir adım, tüm önek üzerinden yapılan bir ileri geçişin dropout çekilişlerini yeniden üretemez; eğitim kipinde kod sessizce tam yeniden hesaplamaya döner. `encode`/`decode` dışında bir şey sunmayan modeller (testlerdeki vekiller) de yığınlanır, yalnızca önbelleksiz.
- **Eşdeğerlik sözleşmesi.** Çözücü değişikliği **çıktıyı değiştirmez**: `tests/unit/ocr/_reference_decoding.py` dalga 6 uygulamasının dondurulmuş kopyasıdır ve `tests/unit/ocr/test_decoding_equivalence.py` üç çözücü ailesinin de aynı jetonları ürettiğini token token doğrular (CPU'da birebir). RTX 5090'da gerçek MathWriting alıntı parçalarıyla, **aynı bellek verildiğinde 16/16 örnekte tüm ışın listeleri ve tüm greedy çıktıları birebir aynıdır**. Kalan tek fark kodlayıcıdan gelir: ResNet34 CUDA'da yığın boyutuna duyarlıdır (16'lık yığın ile tek tek kodlama arasında bağıl fark 1,2e-3, 147 449/147 456 eleman farklı), çünkü TF32 evrişim çekirdekleri yığın şekline göre değişir. Yığın hâlinde çözme kodlayıcıyı da yığınlar; bu, eğitim döngüsünün ve `_forward_loss`'un zaten yaptığı şeydir, dolayısıyla eski `batch == 1` çözme yolundan **daha** tutarlıdır — ama eski koşuların CER'leriyle rakam rakam karşılaştırılamaz.
- **Ek genel API** (sözleşmeye eklenenler, hiçbiri yeniden adlandırma değildir): `OcrInputError`, `available_recognizers`, `ink_to_canvas_json`, `render_ink_to_image`, `resample_polyline`, `fit_cubic_bezier`, `repair_braces`, `canonicalize_whitespace`, `grammar_precheck`, `default_tokenizer`, `CURRICULUM_LATEX`, `STROKE_FEATURE_NAMES/DIM`, `PREPROCESS_STEPS`.

---

## aicls.asr — sahibi cs — kapsam ≥ 70 %

**Sorumluluk.** Ses kenarı: `Transcriber` arayüzü, WAV okuma/doğrulama/yeniden örnekleme (stdlib + numpy), log-Mel, artırma, kalite denetimi (toplama sekmesi), stub ve Whisper/Wav2Vec2 uygulamaları.

```python
# types.py  (torch'a hiçbir yoldan ulaşmayan saf değer tipleri; alt modüller ve aicls.collect buradan alır)
class Audio:      samples: np.ndarray (float32, mono); sr: int; duration_s: float   # duration_s türetilir
class Segment:    start_s: float; end_s: float; text: str; confidence: float = 1.0
class Transcript: text: str; confidence: float = 1.0; segments: list[Segment] = []; elapsed_ms: float = 0.0; model_version: str = "unknown"
class Transcriber(Protocol):
    def transcribe(self, audio: Audio) -> Transcript
    def warmup(self) -> None
class TtsAugmenter(Protocol): def synthesize(self, text: str, voice: str) -> Audio   # motor sonra, kapılı
def concat(audios: Sequence[Audio]) -> Audio
TARGET_SAMPLE_RATE = 16000; TRANSCRIBER_NAMES = ("stub", "whisper", "wav2vec2")

# interfaces.py  (types.py'nin tamamını yeniden dışa verir + ağır yığına dokunabilen iki işlev)
def get_transcriber(name: str = "stub", *, model_dir: Path | None = None, device: str = "cpu") -> Transcriber
def resolve_device(device: str) -> str          # cuda yoksa cpu'ya düşer

# audio_io.py  (stdlib wave + numpy; ffmpeg/torchaudio/soundfile/librosa yok — ADR-0017)
MAX_WAV_BYTES = 10 * 1024 * 1024; MAX_SECONDS = 30.0; ACCEPTED_SAMPLE_RATES = (16000, 44100, 48000)
def read_wav_pcm16(data: bytes | bytearray | memoryview, *, max_seconds: float = 30.0) -> Audio  # RIFF/WAVE doğrulama; 16k/44.1k/48k mono/stereo kabul
def write_wav_pcm16(audio: Audio, *, channels: int = 1) -> bytes                # read_wav_pcm16'nın tersi
def resample_to_16k(audio: Audio) -> Audio                                      # windowed-sinc, numpy
def resample(samples: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray
def resample_by_ratio(samples: np.ndarray, ratio: Fraction) -> np.ndarray       # artırma için kesirli oran

# features.py
def log_mel(audio: Audio, *, n_mels: int = 80, n_fft: int = 400, hop: int = 160, normalize: bool = True) -> np.ndarray  # (n_mels, T)
def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None) -> np.ndarray        # lru_cache'li
def power_spectrogram(audio, *, n_fft=400, hop=160, center=True) -> np.ndarray
def frame_rms(samples, *, frame_len, hop) -> np.ndarray
def hz_to_mel(hz); def mel_to_hz(mel)

# qc.py
class QcThresholds: min_duration_s: float = 0.8; max_duration_s: float = 20.0; max_clipping_ratio: float = 0.001; min_snr_db: float = 15.0; max_silence_ratio: float = 0.6; max_dc_offset: float = 0.02
class QcCheck:  name: str; value: float; threshold: float; passed: bool;  def as_dict() -> dict
class QcReport: passed: bool; checks: list[QcCheck]; reasons_tr: list[str];  .failed_checks, .value(name), .as_dict()
QcResult = QcReport                                                             # takma ad
def run_qc(audio: Audio, thresholds: QcThresholds | None = None) -> QcReport    # duration_min/duration_max/clipping_ratio/snr_db/silence_ratio/dc_offset
def clipping_ratio(audio); def estimate_snr_db(audio); def silence_ratio(audio); def dc_offset(audio)

# augment.py
def spec_augment(mel, *, time_masks=2, freq_masks=2, rng=None, time_mask_max=30, freq_mask_max=13, fill=None) -> np.ndarray
def pitch_shift(audio, semitones, rng=None) -> Audio            # süre birebir korunur
def add_noise(audio, snr_db, noise_bank=None, rng=None, *, clip=False) -> Audio  # bank yoksa beyaz gürültü
def speed_perturb(audio, factor) -> Audio
def measured_snr_db(clean: Audio, noisy: Audio) -> float

# synth.py  (deterministik sentetik fikstürler; depoda gerçek kayıt yok)
def make_tone(...); def make_speech_like(...); def make_noise(...) -> Audio

# stub.py / whisper_transcriber.py / wav2vec2_transcriber.py
class StubTranscriber(Transcriber)            # ses parmak izine bağlı sabit Türkçe cümle (STUB_SENTENCES)
def fingerprint(audio: Audio) -> str          # "sha256:<hex>", PCM16'ya nicelenmiş
class WhisperTranscriber(Transcriber):  @classmethod load(model_dir: Path, device: str = "cpu", *, language="tr", task="transcribe") -> WhisperTranscriber   # from_pretrained(local_dir, local_files_only=True) — asla hub kimliği
class Wav2Vec2Transcriber(Transcriber): @classmethod load(model_dir: Path, device: str = "cpu") -> Wav2Vec2Transcriber
class AsrInputError(AiclsError)   # ASR_INPUT_INVALID, 400
class AsrModelError(AiclsError)   # INTERNAL, 500 — model hazır değil
```

**Bağımlılıklar.** core (numpy; transformers/torch tembel).
**Zorunlu testler.** WAV başlık reddi (PNG baytı ile .wav), yeniden örnekleme uzunluğu, QC kırpılma/SNR/süre/sessizlik, log-Mel şekli (80, T), perde kaydırmanın süreyi koruması, `test_whisper_ft_builds_from_tiny_random_config_without_download`.

### Uygulama notları

- **`Audio.duration_s` türetilmiştir.** Kurucuya verilse bile `len(samples) / sr` ile yeniden hesaplanır; örnekler her zaman bitişik 1-B `float32`'ye çevrilir. Böylece süre ile örnek sayısı hiçbir zaman çelişemez.
- **`QcReport` ve `QcResult`.** Rapor sınıfının adı `QcReport`'tur (belge ve modül şartnamesi böyle diyor); erken taslaklarda geçen `QcResult` adı takma ad olarak korunmuştur.
- **Süre denetimi iki ayrı kontroldür** (`duration_min`, `duration_max`), çünkü `QcCheck` tek bir `threshold` alanı taşır ve red gerekçesinin hangi sınırın aşıldığını söylemesi gerekir. API yanıtındaki `checks` dizisi bu altı adı sabit sırayla içerir.
- **SNR kestirimi duraklama ister.** Çerçeve RMS dağılımının %90'lık dilimi / %10'luk dilimi oranı kullanılır; bu yüzden *tek düze* bir sinyal (saf ton, sürekli uğultu) tasarım gereği düşük SNR okur. Gerçek okuma istemleri baş/son/kelime arası sessizlik içerdiğinden kestirim amacına uygundur; `synth.make_speech_like()` bu şekli üreten ve QC'den geçen kanonik fikstürdür.
- **Türkçe red gerekçeleri `aicls.core.messages_tr` içindedir** (KNOWN_ISSUES kaydı kapandı). `aicls.asr.qc.QC_REASONS_TR` artık yalnızca `QcCheck.name` → `MSG_QC_*` bağlayan bir eşlemedir; çeviri metni tek dosyada tutulur, modül ise kararlı İngilizce anahtarlarını korur.
- **`types.py` / `interfaces.py` ayrımı (entegrasyon kapısı, Dalga 2).** `aicls.collect` kayıt kabulünde `audio_io` + `qc` kullanır; bu modüller `Audio`'yu `interfaces`'ten aldığı sürece `aicls.collect → aicls.asr.interfaces → (tembel) whisper_transcriber → torch` şeklinde **statik** bir içe aktarım yolu oluşuyor ve ADR-0002'nin `no-torch-in-core` sözleşmesi kırılıyordu. Yol hiçbir zaman çalışmadığı için çalışma zamanı testi (`sys.modules`'te torch yok) bunu göremedi. Değer tipleri bu nedenle torch'a hiçbir yoldan ulaşmayan `aicls.asr.types` modülüne alınmıştır; `interfaces` hepsini yeniden dışa verdiği için `from aicls.asr.interfaces import Audio` dahil tüm mevcut içe aktarımlar aynen çalışır. Saf modüller (`audio_io`, `features`, `qc`, `augment`, `synth`, `stub`) artık `types`'tan alır. Sözleşme `tests/integration/test_layering_contracts.py` ile pytest içinde de kilitlenmiştir.
- **Eklenen genel API.** Şartnamedeki adların tümü birebir uygulanmıştır; ek olarak `write_wav_pcm16`, `resample`, `resample_by_ratio`, `concat`, `resolve_device`, `measured_snr_db`, `fingerprint`, `synth.*` ve `AsrInputError`/`AsrModelError` dışa verilmiştir. `aicls.collect` kayıt kabulünde `read_wav_pcm16` + `run_qc` + `write_wav_pcm16` üçlüsünü kullanabilir.
- **Tembel dışa aktarım.** `aicls.asr` içe aktarıldığında `torch`/`transformers` yüklenmez; `WhisperTranscriber` ve `Wav2Vec2Transcriber` modül düzeyinde `__getattr__` ile geç bağlanır.
- **Ağırlık indirme yok.** Her iki sarmalayıcı da yalnızca yerel dizin alır (`from_pretrained(local_dir, local_files_only=True)`), dizin/ağırlık eksikse `NotFoundError`/`AsrModelError` verir; testler `save_pretrained` ile üretilmiş **rastgele başlatılmış küçük** kontrol noktalarını kullanır (ADR-0012).
- **Whisper çözümlemesi.** `language`/`task` yalnızca kontrol noktasının üretim yapılandırması bu belirteçleri tanıyorsa zorlanır; `max_new_tokens` modelin `max_target_positions` değerine göre kısıtlanır. Güven skoru üretilen belirteçlerin ortalama olasılığıdır (CTC yolunda çerçeve başına en yüksek softmax olasılığının ortalaması).

---

## aicls.book — sahibi mmt — kapsam ≥ 85 %

**Sorumluluk.** İP-8 kitap bilgi grafiği: JSON-LD şeması ve RDF sözlüğü, Markdown/LaTeX etiketleyiciler, %100 kapsama doğrulayıcı, yükleyici, Türkçe takma ad üretimi, sorgu ve çözümleyici, JSON-LD/Turtle dışa aktarım. Ayrıntı [veri-modeli.md](veri-modeli.md#5-kitap-içerik-şeması-json-ld--rdf).

```python
class UnitType(StrEnum): DEFINITION, THEOREM, LEMMA, COROLLARY, PROOF, EXAMPLE, EXERCISE, FIGURE, TABLE, REMARK, SECTION
class Predicate(StrEnum): PART_OF, REQUIRES, ILLUSTRATES, PROVES, NEXT, HAS_FIGURE, RELATED
class ContentUnit(BaseModel): id: str ("tanim-2.4"); type: UnitType; chapter: str; number: str; title_tr: str; aliases: list[str]; body_md: str; latex_blocks: list[str]; assets: list[AssetRef]; source_page: int | None; jsonld: dict
class Triple(BaseModel): subject: str; predicate: Predicate; object: str; provenance: str
def tag_markdown(src: str, *, chapter: str) -> TagResult      # nodes, triples, untagged_spans
def tag_latex(src: str, *, chapter: str) -> TagResult
def coverage(result: TagResult) -> float                        # 1.0 olmalı; untagged_spans boş değilse < 1.0
def generate_aliases(title_tr: str, unit_type: UnitType) -> list[str]   # ek tablosu: türev → türevi, türevin, türevini, türev tanımı ...
def load_units(units: list[ContentUnit], triples: list[Triple], repo: BookRepository) -> LoadReport   # units, triples, aliases, tagged_ratio
def search(repo, q: str, *, unit_type: UnitType | None = None, limit: int = 5) -> list[Hit]         # alias tam → LIKE/FULLTEXT → difflib
class BookResolver: def __init__(self, repo: BookRepository, *, cache_size: int = 2048); def resolve(self, query: str, *, unit_type: UnitType | None = None, k: int = 5) -> list[ResolvedUnit]
def export_jsonld(units, triples) -> dict; def export_turtle(units, triples) -> str    # rdflib
```

**Bağımlılıklar.** core, db (rdflib).
**Zorunlu testler.** "Tanım 1.1:" tanıma, örnek bölümde kapsama = 1.0, etiketsiz başlıkta kapsama < 1.0 ve yükleyicinin reddetmesi, "türevin tanımı"/"türev tanımı" aynı birime, `fetch_test.jsonl` ≥ 0.95, Turtle çıktısının rdflib ile yeniden ayrışıp üçlü sayısının eşleşmesi.

### Uygulama notları

Sözleşmedeki adlar birebir uygulandı; aşağıdakiler **eklenen** yüzeydir (hiçbiri var olan bir adı değiştirmez). Ayrıntı ve gerekçeler [veri-modeli.md §5.5](veri-modeli.md#55-uygulama-notları).

```python
# schema.py — kayıtlar
class UntaggedSpan: line: int; text: str; kind: str; source: str      # etiketsiz başlık
class TagResult: nodes; triples; untagged_spans; chapter; source
class Hit / ResolvedUnit: unit; score; matched_alias; stage           # ResolvedUnit ayrıca query_normalized, unit_type
class ContentUnit: ... + order_in_chapter: int = 0                     # bölüm içi sıra (book_units.order_in_chapter)
def unit_id(unit_type, number) -> str                                  # "tanim-2.4"

# etiketleyiciler ve yükleyici
def tag_markdown(src, *, chapter, source="") -> TagResult              # `source` eklendi (hata mesajında dosya adı)
def tag_latex(src, *, chapter, source="") -> TagResult
def load_units(units, triples, repo, *, tagged_ratio=1.0, untagged=()) -> LoadReport
def load_tag_result(result, repo) -> LoadReport                        # kapsama < 1.0 ise UntaggedHeadingError
def load_markdown_dir(dir, repo) -> LoadReport                         # `aicls book load <dir>`
def tag_markdown_dir(dir) -> list[TagResult]; def chapter_id_for(path, i) -> str
def require_full_coverage(*results) -> float; def untagged_headings(*results) -> list[UntaggedSpan]
class UntaggedHeadingError(InputValidationError)                       # .headings, .tagged_ratio, details["untagged"]

# sorgu, çözümleme, dışa aktarım
def unit_from_row(row) -> ContentUnit; def get_unit(repo, id); def related(repo, id, *, predicate=None)
def units_from_repo(repo, *, unit_type=None); def triples_from_repo(repo)
class BookResolver: .resolve_one(query, *, unit_type=None); .cache_clear(); .cache_info()
def detect_unit_type(tokens); def resolution_plan(query, unit_type); def normalize_query(text)
def generate_alias_entries(title_tr, unit_type) -> list[AliasEntry(alias, weight, kind)]
def unit_to_jsonld(unit, triples=()) / unit_from_jsonld(doc) / triples_from_jsonld(doc) / units_from_jsonld(doc)
def load_context() / context_document() / load_vocab_graph() / predicate_uri() / unit_type_uri()
```

**Veri kümesi.** `datasets/book/`: `sample/bolum-01-limit.md` (düz desen, 8 birim), `sample/bolum-02-turev.md` (`:::` blok, 12 birim), `sample/bolum-04-integral.tex` (LaTeX, 5 birim), `invalid/bolum-03-etiketsiz.md` (kasıtlı hatalı), `expected_units.json` (altın standart), `fetch_test.jsonl` (72 ifade). Ölçülen: kapsama 1.0, getirme doğruluğu 72/72, çözümleme p50 ≪ 20 ms, birim testi kapsamı %97.

---

## aicls.collect — sahibi mmt (sg ile) — kapsam ≥ 85 %

**Sorumluluk.** "Model Gelişimine Katkıda Bulun" alan mantığı: rıza sürümleri ve kabul kaydı, cümle dengeleme, kayıt kabul politikası, KVKK geri çekme, eğitim manifesti dışa aktarımı.

```python
class ConsentText: version: str; text_tr: str; sha256: str; effective_from: date   # frozen dataclass; as_dict()
def consent_texts() -> tuple[ConsentText, ...]                                     # resources/consent_texts.yaml, lru_cache
def current_consent(*, on: date | None = None) -> ConsentText
def get_consent_text(version: str) -> ConsentText
class Demographics(BaseModel): age_band; gender; university; accent_region         # extra="forbid" (yalnızca bant)
def hash_ip(value: str, *, salt: str | None = None) -> str | None                  # tuzlu SHA-256 (varsayılan tuz: JWT sırrı)
def contribution_enabled() -> bool; def require_contribution_enabled() -> None     # AICLS_ETHICS_APPROVAL_REF boşsa 503
def accept_consent(repo, *, user_id: str | None = None, version: str, demographics: Demographics | Mapping, ip_hash: str | None = None, user_agent_hash: str | None = None) -> Speaker   # pseudonym P + 5 hane
def next_prompts(repo, speaker_id: str, n: int = 5) -> list[PromptSentence]     # en az kaydedilmiş, senaryo dengeli (1 ≤ n ≤ MAX_PROMPTS = 50)
def require_consent(repo, speaker_id: str) -> Speaker                            # 403: kayıt yok / geri çekilmiş / rıza sürümü eski
def accept_recording(repo, *, speaker_id: str, prompt_id: str, wav: bytes, thresholds: QcThresholds | None = None, data_root: Path | None = None) -> RecordingDecision
class RecordingDecision: accepted; qc: QcReport; recording_id; path; rel_path; sha256; duration_s; reasons_tr; as_dict()
def withdraw(repo, speaker_id: str, *, data_root: Path | None = None) -> WithdrawReport   # silinen dosya/satır sayısı; rıza denetim izi kalır
def export_manifest(repo, *, out: Path, split_seed: int = 1234, data_root=None, splits_path=None, ratios=(0.8, 0.1, 0.1), source="contrib", verify_files=True) -> ManifestReport
    # jsonl: path, text, speaker_pseudonym, university, gender, age_band, split, duration_s, sample_rate,
    #        sha256, scenario_tag, consent_version, source, recording_id  (MANIFEST_FIELDS); yalnızca qc_passed AND withdrawn_at IS NULL
class SplitFile: by_speaker; seed; ratios; updated_at
def load_splits(path) -> SplitFile; def save_splits(path, splits) -> None
def assign_splits(strata, *, seed=1234, ratios=(0.8, 0.1, 0.1), frozen=None) -> dict[str, str]   # konuşmacı-ayrık, tabakalı, donmuş
def speaker_audio_dir(pseudonym, *, data_root=None) -> Path; def new_recording_path(pseudonym, *, data_root=None) -> tuple[Path, str]
class ConsentRequiredError(AiclsError)   # CONSENT_REQUIRED, 403
class ContributionGateError(AiclsError)  # GATE_BLOCKED, 503
```

**Bağımlılıklar.** core, db, asr (qc).
**Zorunlu testler.** rıza olmadan kayıt reddi, QC başarısızlığında dosya yazılmaması, geri çekmenin dosyaları silip rıza satırını koruması, dışa aktarımda gerçek kimlik alanı bulunmaması.

**Uygulama notları.**

- **Rıza metni paket kaynağı.** Sürümlü metin `src/aicls/collect/resources/consent_texts.yaml` içindedir; `text_tr`'nin SHA-256'sı yükleme sırasında hesaplanır. Türkçe kullanıcı dizelerinin yalnızca `aicls.core.messages_tr` içinde bulunması kuralının bilinçli istisnasıdır: bu metin çevrilebilir bir arayüz dizesi değil, sürümlenen ve özeti denetim kanıtı olan bir *veri* nesnesidir (aynı metin `consent_versions` tablosuna da kopyalanır). Hata iletileri yine `messages_tr`'den gelir.
- **Rıza sürümü zorunluluğu.** `accept_recording`, konuşmacının **yürürlükteki** rıza sürümünü kabul etmiş olmasını arar; yeni bir sürüm yayımlandığında katılımcı yeniden onam vermelidir. Bilinmeyen takma ad, geri çekilmiş konuşmacı ve eski sürüm aynı 403 `CONSENT_REQUIRED` yanıtını üretir (uç nokta kimin kayıtlı olduğunu doğrulamaz).
- **QC belgesi düzleştirmesi.** `QcReport.as_dict()` ölçümleri `checks[]` altında yuvalar, `RecordingRepository.add` ise `snr_db`/`clipping_ratio`/`silence_ratio` alanlarını **üst düzeyden** sütunlara taşır. `service.qc_document()` her iki sözleşmeyi de karşılar; aksi hâlde `recordings.snr_db` sessizce 0.0 kalırdı (`tests/integration/test_collect_contracts.py`).
- **Yazma sırası.** Önce veritabanı satırı, sonra dosya: yazma başarısız olursa işlem geri alınır ve öksüz dosya kalmaz. Dosya, aynı dizinde `.part` uzantılı geçici ada yazılıp `replace` ile taşınır.
- **Kanonik ses.** Kabul edilen kayıt, çözülmüş 16 kHz mono PCM16 biçiminde yeniden kodlanarak saklanır; `sha256` bu saklanan baytların özetidir.
- **Kapı.** `AICLS_ETHICS_APPROVAL_REF` boşsa `accept_consent`/`accept_recording` 503 verir; `withdraw` **kapılı değildir** (KVKK geri çekme her koşulda çalışmalıdır). Referans 80 karakteri aşarsa yapılandırma hatası olarak reddedilir (`consent_versions.ethics_approval_ref` `VARCHAR(80)`).
- **Bölme dosyası.** `export_manifest` varsayılan olarak `out.parent/audio_commands_splits.json` dosyasını okur ve yeniden yazar; içindeki her atama dondurulur, yeni konuşmacılar `(üniversite, cinsiyet, yaş bandı)` tabakaları içinde tohumlanmış SHA-256 sırasıyla kotası en açık bölmeye yerleştirilir.
- **Ek genel API.** Şartnamedeki adlar birebir uygulanmıştır; ek olarak `Demographics`, `hash_ip`, `require_consent`, `consent_texts`, `get_consent_text`, `assign_splits`, `load_splits`/`save_splits`, `SplitFile`, `MANIFEST_FIELDS`, `SPLIT_NAMES`, `contribution_enabled`/`require_contribution_enabled` ve `speaker_audio_dir`/`new_recording_path` dışa verilmiştir. `ip_hash`, `thresholds` ve `data_root` isteğe bağlı hâle getirilmiştir (varsayılanları vardır); adlar ve sıralar değişmemiştir.

---

## aicls.data — sahibi cs (mmt ile) — kapsam ≥ 75 % (ölçülen: 94,2 %)

**Sorumluluk.** Veri manifestleri (SHA-256), katalog, deterministik ve tabakalı bölmeler, InkML→parça dönüşümü, MathWriting/ses/müfredat veri kümesi sınıfları, sentetik üreteçler, collate.

```python
class DataManifest(BaseModel): name; version; license; source_url; retrieved_at; files: list[FileEntry(path, size, sha256)]; fingerprint: str; total_bytes; n_files
    @classmethod build(root: Path, name: str, *, license: str, source_url: str, version=None, retrieved_at=None) -> DataManifest
    def verify(self, root: Path) -> VerifyReport      # ok, checked, missing[], changed[], extra[], fingerprint_matches, summary()
    def save(self, target: Path) -> Path              # <root>/manifest.json
def load_manifest(path: Path) -> DataManifest; MANIFEST_FILENAME = "manifest.json"

# catalog.yaml (yalnızca betimleyici; pakette indirme kodu YOK — ADR-0012)
class CatalogEntry: name; kind: "dataset"|"weights"; version; url|None; license; expected_bytes; sha256|None; gate|None; citation; expected_gib; archive_name; describe()
def load_catalog() -> Catalog; def catalog_entry(name: str) -> CatalogEntry
#   girdiler: mathwriting (CC-BY-NC-SA-4.0, mathwriting-2024.tgz), mathwriting_excerpt (aynı lisans, 1,55 MiB),
#   audio_commands, whisper-small, whisper-large-v3-turbo, wav2vec2-xls-r-300m

class Splits(BaseModel): train; valid; test; extra{symbols, synthetic}; seed; ratios; source; fingerprint; names(); sizes(); split_of(id); index()
def make_splits(ids, *, group_key, stratify=None, ratios=(0.8, 0.1, 0.1), seed=1234, frozen: Splits | Mapping | Path | None = None) -> Splits
def mathwriting_official(root: Path) -> Splits                 # train/valid/test/symbols/synthetic (dizin veya <split>.txt)
def save_splits(splits, path) -> Path; def load_splits(path) -> Splits    # aicls.collect'in {"speakers": {...}} biçimini de okur
def normalize_split_name(name) -> str                          # "dev" -> "valid" (SPLIT_ALIASES)

# InkML kaynakları: açılmış dizin ya da doğrudan .tgz akışı (hiçbir dosya açılmaz)
class InkmlSource(ABC): kind; source_label(split); count_splits(splits, *, cap=20000) -> dict[str, SplitCount]; iter_records(splits, *, limit=None) -> Iterator[tuple[str, InkmlRecord]]
class DirectoryInkmlSource(root: Path): root; split_dir(split)          # <root>/<split> varsa o, yoksa <root>
class TarInkmlSource(archive: Path, *, known_splits=MATHWRITING_SPLITS, default_split=None, max_member_bytes=1 MiB): archive
class InkmlRecord: sample_id; payload: Path | bytes; member
class SplitCount: split; source; n_files; capped; available
def open_inkml_source(path, *, known_splits=MATHWRITING_SPLITS, default_split=None) -> InkmlSource; def looks_like_tar(path) -> bool

def prepare_shards(inkml_root: Path, out: Path, *, split: str, shard_size=4096, render=False, image_size=(96,384), stroke_width=2, is_synthetic=None, limit=None, skip_invalid=False) -> ShardManifest
def prepare_shards_from_source(source: InkmlSource, out: Path, *, splits: Sequence[str], shard_size=4096, render=False, image_size=(96,384), stroke_width=2, is_synthetic=None, limit=None, skip_invalid=False) -> dict[str, ShardManifest]
class ShardManifest(BaseModel): dataset; split; shard_size; n_samples; n_skipped; shards[ShardEntry(name, n_samples, size_bytes, sha256)]; has_images; image_size; source_root; created_at; fingerprint
class ShardReader(path, *, mmap=True): array(name); sample(i) -> ShardSample(sample_id, strokes, label, writer_id, is_synthetic, image); __len__; close()
def load_shard_manifest(root, split) -> ShardManifest; def build_vocab_from_shards(root, out, *, splits=("train",)) -> Path
def normalize_label(latex: str) -> str

class MathWritingDataset(root, split, *, mode="online"|"offline", tokenizer=None, preprocess=None, synthetic_ratio=0.0, max_label_len=None, add_special=True, stroke_width=2)
    # map-style; __len__/__getitem__ -> InkSample(features, label_ids, label, sample_id, writer_id, is_synthetic); .to_torch() tembel torch sarmalayıcı
class SyntheticInkDataset(n, *, seed=1234, tokenizer=None, mode="online", preprocess=None, latexes=None)
class AudioCommandDataset(manifest, split, *, feature_fn, splits=None, max_seconds=30.0)   # -> AudioSample(features, text, speaker, sample_id, sample_rate, duration_s)
class SyntheticAudioDataset(n, *, seed=1234, feature_fn=None, seconds=1.0, texts=None, snr_db=None)
def load_curriculum(path: Path) -> list[Problem]               # problems.jsonl veya *.yaml kart dosyası

def synth_ink(rng, latex: str) -> Ink; def synth_image(rng, latex: str, size, *, noise=0.01) -> np.ndarray   # matplotlib mathtext, TeX kurulumu gerekmez
def synth_audio(rng, seconds: float, *, snr_db: float | None = None) -> Audio; def synth_inkml_dataset(out: Path, n: int, seed: int = 1234) -> Path
def collate_ink(batch, *, to_torch=False) / collate_image(...) / collate_audio(...); def pad_sequences(arrays, *, pad_value=0.0, dtype=None)
```

**Bağımlılıklar.** core, ocr, asr (torch yalnızca `to_torch()` ve `collate_*(to_torch=True)` içinde tembel; `latex2sympy` şu an kullanılmıyor).
**Zorunlu testler.** manifest sahtecilik tespiti, konuşmacı-ayrık ve tabakalı bölme (%5 tolerans), parça gidiş-dönüşü sentetik InkML üzerinde, sentetik üreteçlerin seed ile tekrarlanabilirliği.

### Uygulama notları

- **Kapı disiplini.** `catalog.yaml` betimleyicidir; pakette hiçbir indirme çağrısı yoktur (`test_catalog_module_contains_no_download_helper`, `tools/check_gates.py` temiz). `sha256` indirme sonrası doldurulur.
- **Katalog düzeltmesi (2026-09-08).** MathWriting girdisi üç noktada yanlıştı ve resmi kaynağa karşı doğrulanıp düzeltildi: URL `.zip` değil **`.tgz`** (eski URL HTTP 403 veriyordu), `expected_bytes` 3 113 851 289 değil **3 096 141 721**, ve lisans Apache-2.0 değil **CC-BY-NC-SA-4.0**. Lisans düzeltmesi teknik değil hukuki bir olgudur: NonCommercial ve ShareAlike, bu veriyle eğitilen ağırlıkları da bağlar (`notes_tr`, `test_catalog_records_the_noncommercial_sharealike_licence_of_mathwriting`). Ayrıca resmi 1,55 MiB'lık örneklem `mathwriting_excerpt` adıyla katalogdadır; kapı açıldığında hattı 2,88 GiB harcamadan uçtan uca kanıtlamak için oradadır.
- **Akışlı parçalama.** `prepare_shards_from_source` bir `InkmlSource` alır; `TarInkmlSource` `mathwriting-2024.tgz` üyelerini doğrudan okur ve hiçbir şey açmaz (~20 GB ağaç ve ~630 bin küçük dosya yazımı olmaz — `drvfs` veri kökünde belirleyici). Gzip akışı geri sarılamadığı için istenen **tüm bölmeler tek geçişte** üretilir; `_SplitShardWriter` bölme başına tampon tutar ve iki kaynak da aynı yazıcıyı kullanır. `prepare_shards` bu işlevin dizin kaynaklı ince sarmalayıcısıdır, davranışı değişmedi. Sıra kaynağa bağlıdır (yola göre sıralı ↔ arşiv sırası); `tar` sıralı dizin gezintisi yazdığından iki yol pratikte aynı parçaları ve aynı `fingerprint` değerini üretir (`test_streamed_shards_carry_the_same_samples_as_the_extracted_tree`), ama farklı sırayla paketlenmiş bir arşiv parmak izini değiştirir (`test_shard_order_follows_the_archive_so_the_fingerprint_can_change`) — `source_root` hangisinin kullanıldığını yazar (`tar:<arşiv>!<bölme>`). Konteynerde ölçüldü (2026-09-08, `drvfs`, 20 bin sentetik dosya): açma 166 s + dizinden parçalama 135 s = 301 s; arşivden akıtma 6,5 s, diske hiçbir InkML yazılmadı.
- **Üye boyutu sınırı.** Arşiv üyeleri `MAX_INKML_BYTES` (1 MiB) + 1 bayta kadar okunur; daha büyüğü uyarıyla günlüğe düşer ve `read_inkml` onu `reason="too_large"` ile reddeder — bozuk veya düşmanca bir arşiv sınırsız tampon ayırtamaz. `skip_invalid=True` bozuk bir belgeyi `n_skipped`'a yazıp koşuyu sürdürür (varsayılan kapalı).
- **Parmak izi biçimi.** Manifest, parça manifesti ve bölme dosyası aynı `"sha256:<hex>"` biçimini kullanır (`aicls.core.hashing`), böylece `EvalReport.dataset_fingerprint` üçünü de kabul eder.
- **Bölme adı köprüsü (entegrasyon kapısı, 2026-09-07).** `aicls.collect` orta bölmeye `dev`, eğitim/değerlendirme tarafı `valid` diyor; ayrıca collect'in dışa aktarım satırlarında `id` değil `recording_id` var ve bölme dosyası `{"speakers": {...}}` biçiminde. `aicls.data` üçünü de karşılar (`normalize_split_name`, `recording_id` geri düşüşü, `load_splits`'in konuşmacı haritası desteği); sözleşme `tests/integration/test_data_contracts.py` ile kilitlenmiştir. Kalıcı çözüm için `aicls.collect.SPLIT_NAMES`'in `valid`'e taşınması önerilir (sahibi mmt).
- **Bellek eşlemeli parçalar.** `np.load(..., mmap_mode=...)` `.npz` için çalışmadığından parçalar sıkıştırmasız yazılır ve `ShardReader` zip üyesinin veri konumunu bulup `np.memmap` ile eşler; eşleme mümkün değilse sessizce tam okumaya düşer.
- **Tembel torch.** `MathWritingDataset` `torch.utils.data.Dataset`'ten türemez (map-style protokol yeterlidir); `to_torch()` sarmalayıcıyı torch'u tembel içe aktararak kurar. `import aicls.data` torch'u, transformers'ı ve matplotlib'i yüklemez (`test_importing_aicls_data_does_not_import_torch_or_matplotlib`).
- **Sentetik oran rastgeleliksiz.** `i` konumu sentetiktir ⇔ `floor((i+1)·r) > floor(i·r)`; epoch uzunluğu `round(n_real/(1−r))`, `r=1.0` iken sentetik parçanın boyutu. Her gerçek örnek epoch başına tam bir kez görülür.
- **Türkçe dize yok.** Paket hiçbir Türkçe kullanıcı dizesi tanımlamaz; hatalar `aicls.core.messages_tr` metinleriyle `InputValidationError`/`NotFoundError` olarak çıkar, makine tarafı `details["reason"]` içindedir (`invalid_data`, `missing_path`).
- **Ek genel API** (sözleşmeye eklenenler; yeniden adlandırma yok): `load_manifest`, `VerifyReport`, `FileEntry`, `Catalog`/`CatalogEntry`/`load_catalog`/`catalog_entry`, `save_splits`/`load_splits`/`normalize_split_name`/`CORE_SPLITS`/`MATHWRITING_SPLITS`, `ShardReader`/`ShardSample`/`ShardEntry`/`load_shard_manifest`/`read_shard_manifest`/`build_vocab_from_shards`/`normalize_label`, `prepare_shards_from_source`, `discover_shard_splits`/`shard_fingerprints(root, splits=None, *, verify=False)`/`shards_root_fingerprint` (bir koşumun `run_manifest.json`'a yazdığı veri kimliği; bölme yerleşimini yalnızca bu paket bilir), `InkmlSource`/`DirectoryInkmlSource`/`TarInkmlSource`/`InkmlRecord`/`SplitCount`/`open_inkml_source`, `InkSample`/`AudioSample`/`AudioRow`, `SyntheticInkDataset`/`SyntheticAudioDataset`, `synth_image_dataset`, `pad_sequences`, `Problem`/`ProblemInput`/`ProblemExpected`.

---

## aicls.eval — sahibi cs — kapsam ≥ 95 % (metrics/normalizers ≥ 98 %)

**Sorumluluk.** Dondurulmuş metrik tanımları ve sürümlü normalizasyon, bootstrap güven aralığı, her bileşen için değerlendirme koşumu, `reports/metrics/*.json` yazımı, Markdown tablo. Ayrıntı [../ml/degerlendirme.md](../ml/degerlendirme.md).

```python
def latex_v1(s: str) -> str; def tr_v1(s: str) -> str; NORMALIZER_VERSIONS = {"latex": "v1", "tr": "v1"}
def cer(ref: str, hyp: str, *, norm: Literal["latex_v1","none"] = "latex_v1") -> float
def corpus_cer(pairs: Iterable[tuple[str, str]], *, norm="latex_v1") -> float
def wer(ref: str, hyp: str, *, norm: Literal["tr_v1","none"] = "tr_v1") -> float
def corpus_wer(pairs, *, norm="tr_v1") -> float
def exact_match(ref: str, hyp: str, *, norm="latex_v1") -> bool
def symbolic_equivalence(ref_latex: str, hyp_latex: str) -> Literal["equal","different","unparsable"]
def parse_success_rate(inputs: Iterable[str]) -> RateResult     # ok, total, rate, failures[(id, reason)]
def latency_stats(samples: Sequence[float]) -> LatencyStats     # mean, p50, p95, n
def sus_score(answers: Sequence[int]) -> float                  # 10 madde, 0-100
def bootstrap_ci(values: Sequence[float] | Callable, *, n_resamples: int = 1000, seed: int = 1234) -> tuple[float, float]
class EvalReport(BaseModel): name; kind; git_sha; dirty: bool; dataset_fingerprint; dataset_version; model_version: str | None; model_sha256: str | None; normalizer_versions: dict; n: int; metrics: dict[str, float]; ci95: dict[str, tuple[float,float]]; breakdown: dict; failures_path: str | None; timestamp; citable: bool
def write_report(r: EvalReport, out_dir: Path = Path("reports/metrics")) -> Path
def promote(reports, out_dir=Path("reports/metrics")) -> Path   # latest.json; wiring check ise her ölçüt null + measurement/wiring_check
MODEL_BACKED_KINDS = frozenset({"ocr", "asr"})   # stub arkası = ölçülecek model yok; EvalReport.is_wiring_check bunu kullanır
@dataclass(frozen=True, slots=True)
class WiringCheckReason: code: str; detail: str
@dataclass(frozen=True, slots=True)
class WiringCheckVerdict:
    reasons: tuple[WiringCheckReason, ...]
    is_wiring_check: bool; codes: tuple[str, ...]        # türetilmiş
    def merge(self, other) -> WiringCheckVerdict; def summary(self) -> str
    def to_dict(self) -> dict; @classmethod from_payload(cls, payload: Any)
def assess_wiring_check(*, synthetic_data=False, stub_backend=False, random_init=False,
                        citable: bool | None = None, steps: int | None = None,
                        warmup_steps: int | None = None, n_val: int | None = None,
                        min_steps=MIN_REPORTABLE_STEPS,
                        min_val_samples=MIN_VAL_SAMPLES) -> WiringCheckVerdict
def wiring_check_reason_text(verdict: WiringCheckVerdict) -> str
MIN_REPORTABLE_STEPS = 100; MIN_VAL_SAMPLES = 100; WIRING_CHECK_KEY = "wiring_check"   # breakdown anahtarı
def check_report(report, *, head_sha, ancestors, fingerprints=None, reject_wiring_check=False) -> list[str]
def render_markdown(reports: list[EvalReport]) -> str
def run_parser_eval(path: Path) -> EvalReport
def run_cas_eval(problems: Path, engine: MathEngine) -> EvalReport          # parse_success, correct_rate, per-topic, mean/p95 ms
def run_nlu_eval(path: Path, *, inventory) -> EvalReport                      # intent_acc, slot_f1, command_exact
def blind_sample(sample, *, fields=ANSWER_FIELDS)                            # ground truth alanlarını silen kopya; silemezse InputValidationError
def run_ocr_eval(recognizer: Recognizer, ds, *, max_n: int | None) -> EvalReport   # cer, ter, exprate, symeq, cer CI; her örnek blind_sample'dan geçer
def run_asr_eval(transcriber: Transcriber, ds) -> EvalReport                 # wer_raw, wer_tr_v1, cer, command_intent_accuracy
def run_book_eval(resolver: BookResolver, path: Path) -> EvalReport          # fetch_top1, tagging_coverage
def run_latency_eval(client, commands: Path, *, n: int) -> EvalReport        # stub tabanlı sentetik; gerçek rapor latency_samples tablosundan
```

**Bağımlılıklar.** core, latex2sympy, mathengine, nlu, ocr, asr, book, data. **ml yasak.**
**Zorunlu testler.** elle hesaplanmış referans değerler, özellik testleri (özdeş→0, sınırlar), jiwer çapraz denetim, `tr_v1`'in yalnızca belgelenmiş kurallarla sözcük sayısı değiştirmesi, rapor parmak izi ve SHA alanları, **arkalığın etiketi hiç görmemesi** (`test_the_ocr_runner_never_lets_the_backend_see_the_label`, `test_the_asr_runner_hands_the_backend_a_waveform_and_nothing_else`) ve **stub koşumunun `latest.json`'a sayı yazmaması** (`test_a_stub_backed_run_publishes_no_number_only_a_reason`).

---

## aicls.ml — sahibi cs — kapsam ≥ 75 % (ölçülen: 94,7 % — `aicls.cli` ile birlikte)

**Sorumluluk.** Eğitim altyapısı: tipli yapılandırma, kayıt, koşu manifesti, seed, ortam yakalama, izleyici, OCR için Trainer, Whisper için HF Seq2SeqTrainer sarmalayıcısı, atomik kontrol noktası, profil koşusu, kuyruk. Ayrıntı [../ml/egitim-altyapisi.md](../ml/egitim-altyapisi.md); işletim [../ml/egitim-el-kitabi.md](../ml/egitim-el-kitabi.md).

```python
# --- aicls.ml.config ---------------------------------------------------------------------
class TrainConfig(BaseModel, extra="forbid", frozen=True):
    schema_version: int = 1
    run: RunCfg; paths: PathsCfg; data: DataCfg; model: ModelCfg; optim: OptimCfg
    train: LoopCfg; eval: EvalCfg; tracking: TrackingCfg; checkpoint: CheckpointCfg; gates: GatesCfg
    @property is_gate_free -> bool          # YALNIZCA data.dataset.startswith("synthetic_")
    def to_yaml(self) -> str
# alt düğümler (hepsi extra="forbid", frozen=True):
#   RunCfg(name, seed=1234, deterministic=True, tags, notes)
#   PathsCfg(data_root, models_root, runs_root)
#   DataCfg(dataset, root, manifest, splits_path, split, mode, image_size, preprocess,
#           synthetic_ratio, synthetic_warmup_epochs, augment: AugmentCfg, max_label_len,
#           batch_size, num_workers, feature_extractor, n_samples, sample_seconds, max_seconds)
#   ModelCfg(family, encoder, decoder, d_model, layers, heads, ff_dim, dropout, max_len,
#            vocab_path, base_dir, pretrained_path, weights: None, freeze_encoder,
#            lora: LoraCfg, num_mel_bins)      # weights None dışı bir değer = şema hatası
#   OptimCfg(name, lr, weight_decay, betas, momentum, scheduler, warmup_steps)
#   LoopCfg(epochs, max_steps, grad_accum, amp, grad_clip, eval_every, ckpt_every, log_every,
#           early_stop_patience, allow_dirty)
#   EvalCfg(beam, max_n_val, metrics) / TrackingCfg(backend, mlflow_uri, experiment)
#   CheckpointCfg(keep_last, keep_best, metric, mode) / GatesCfg(synthetic)  # bilgi amaçlı
def load_config(path: Path | str, overrides: Sequence[str] = ()) -> TrainConfig   # _base_ kalıtımı
def build_config(data: Mapping[str, Any], *, source: Path | None = None) -> TrainConfig
def config_hash(cfg: TrainConfig) -> str                 # 64 hex; ÇÖZÜMLENMİŞ config üzerinden
def resolved_dict(cfg) -> dict; def parse_override(text) -> tuple[str, Any]; def expand_variables(v)

# --- aicls.ml.registry -------------------------------------------------------------------
@dataclass(frozen=True) class ModelSpec(model: ModelCfg, image_size, max_len); .from_config(cfg)
@dataclass(frozen=True) class DatasetSpec(data: DataCfg, seed, vocab_path, model_dir, n_mels)
    def tokenizer(self) -> LatexTokenizer; def n_samples(self) -> int; .from_config(cfg)
def register_model(name) / register_dataset(name) -> decorator
def registered_models() / registered_datasets() -> tuple[str, ...]
def get_model_builder(name) / get_dataset_builder(name) -> Callable
def build_model(cfg: ModelCfg | TrainConfig, vocab_size: int) -> torch.nn.Module    # tembel torch
def build_dataset(cfg: DataCfg | TrainConfig, split: str) -> Dataset
# kayıtlı kurucular: build_seq2seq_image / _stroke, build_whisper, build_wav2vec2,
#                    build_mathwriting, build_synthetic_ink, build_audio_commands, build_synthetic_audio

# --- aicls.ml.manifest -------------------------------------------------------------------
class RunManifest(BaseModel, extra="forbid"):
    schema_version; run_id; run_name; seed; started_at; git_sha; branch; dirty; diff_sha256
    git_available; allow_dirty; citable; config_hash; data_fingerprints; pip_freeze_sha256
    python; platform; hostname; torch; cuda; cudnn; cuda_available; device_count
    gpu_name; driver; gpu_memory_total_mb; steps_per_s: float | None
    @classmethod capture(cfg, *, allow_dirty=False, repo_root=None, run_name=None,
                         data_fingerprints=None, data_splits=None) -> RunManifest
                         # kirli ağaç → DirtyWorkingTreeError; data_splits: kaydedilecek bölmeler
    @classmethod load(path) -> RunManifest; def save(target) -> Path
    def to_dict() -> dict; def to_json() -> str; def set_steps_per_s(v, *, run_dir=None) -> None
class DirtyWorkingTreeError(InputValidationError)     # details["reason"] == "dirty_working_tree"
@dataclass(frozen=True) class GitState(sha, branch, dirty, diff_sha256, available)
def seed_everything(seed: int = 1234, *, deterministic: bool = True) -> int    # kullanılan tohum
def capture_env(run_dir: Path) -> dict            # env.json (tam paket listesi)
def capture_git_state(repo_root=None) -> GitState; def config_fingerprint(cfg) -> str
def collect_data_fingerprints(cfg, *, splits=None) -> dict[str, str]
#   hazırlanmış parça kökü: her bölme için <ad>.<bölme> + toplulaştırılmış <ad> anahtarı
#   (aicls.data.shards.shard_fingerprints / shards_root_fingerprint); ayrıca kök manifest.json,
#   data.manifest (<ad>.manifest) ve data.splits_path (<ad>.splits)
def nvidia_smi_devices() -> list | None

# --- aicls.ml.tracking -------------------------------------------------------------------
class Tracker(Protocol):
    def start_run(self, name: str, params: Mapping | None = None) -> str
    def log_metrics(self, metrics: Mapping[str, float], step: int, *, split: str = "train") -> None
    def log_artifact(self, path: Path, *, name: str | None = None) -> dict     # yazdığı kaydı döner
    def end_run(self, status: RunStatus = "finished") -> None                  # + __enter__/__exit__
class JsonlTracker(run_dir, *, run_id=None)   # metrics.jsonl, params.json, artifacts.jsonl
class MlflowTracker(experiment="aicls", *, tracking_uri=None, run_id=None)     # mlflow tembel
class MlflowUnavailableError(InternalError)
def iter_metrics(run_dir) / read_metrics(run_dir) / read_params(run_dir); def flatten_params(...)
# metrics.jsonl satırı: {"schema_version",  "run_id", "step", "split", "wall_time", "ts",
#                        "metrics": {ad: float | null}}      # sonsuz/NaN → null

# --- aicls.ml.checkpoint -----------------------------------------------------------------
def save_checkpoint(ckpt_dir: Path, *, model, step, optimizer=None, scheduler=None, cfg=None,
                    manifest=None, metrics=None, vocab=None, save_rng=True) -> CheckpointMeta
def load_checkpoint(ckpt_dir: Path, *, model=None, optimizer=None, scheduler=None,
                    map_location="cpu", restore_rng=True, verify_sha=True, strict=True) -> Checkpoint
class CheckpointMeta(BaseModel): step; created_at; format; model_file; model_sha256; files;
                                 has_optimizer/scheduler/rng/vocab; metrics; run_id; git_sha; config_hash
@dataclass(frozen=True) class Checkpoint(path, step, meta, config, manifest, metrics, vocab, rng_restored)
class CheckpointSaver(root, *, keep_last=2, keep_best=1, metric="val_loss", mode="min", cfg=None,
                      manifest=None, vocab=None)
    def save(step, *, model, ...) -> CheckpointMeta; def prune() -> list[Path]
    def is_best(metrics) -> bool; property best_dir / last_dir; attrs best_step/best_value/last_step
def rng_state() -> dict; def restore_rng_state(state) -> None
# dizin: model.safetensors (paylaşımlı tensörde model.pt), optimizer.pt, scheduler.pt, rng_state.pt,
#        config.yaml, run_manifest.json, vocab.json, metrics.json, checkpoint.json (+ best/last.json)

# --- aicls.ml.trainer --------------------------------------------------------------------
@dataclass(frozen=True) class DeviceProbe(cuda_available, bf16_supported, device_name, capability)
@dataclass(frozen=True) class DeviceSpec(device, dtype, autocast, amp, fallback_reason, device_name)
    property is_cuda / needs_grad_scaler; def torch_dtype(); def autocast_context()
def probe_device(torch_module=None) -> DeviceProbe
def select_device(amp="off", *, device=None, probe=None) -> DeviceSpec   # test edilebilir karar tablosu
class Trainer:
    def __init__(self, cfg, model, train_ds, val_ds=None, *, metrics=None, tracker=None,
                 run_dir=None, tokenizer=None, repo_root=None, i_have_approval=False,
                 ledger_path=None, device=None, probe=None)
    def fit(self, *, max_steps: int | None = None) -> FitReport   # max_steps MUTLAK bütçedir
    def resume(self, ckpt_dir: Path) -> Checkpoint                # model/optim/sched/adım/RNG
    def evaluate(self, *, max_n: int | None = None) -> dict[str, float]  # toplu çözme; loss,
        # val_loss, eval_seconds, n_val, n_val_total, val_truncated + metrikler
class FitReport(BaseModel, frozen=True): run_id; run_name; steps; epochs; samples_seen; train_loss;
    metrics; best_metric; best_value; best_step; best_checkpoint; last_checkpoint; early_stopped;
    interrupted; device; dtype; amp; wall_time_s; steps_per_s; dataloader_wait_pct;
    train_seconds; eval_seconds; checkpoint_seconds; dataloader_wait_s; other_seconds; evaluations;
    seconds_per_eval; seconds_per_val_sample; eval_share_pct; n_val; n_val_total; val_truncated;
    eval_decoder; run_dir; config_hash; git_sha; citable; wiring_check; notes
class EarlyStopping(patience, *, mode="min", min_delta=0.0); @dataclass class TrainStats
@dataclass class TimeBudget(train_seconds, eval_seconds, checkpoint_seconds, dataloader_wait_s,
    evaluations, eval_samples, wall_time_s)   # + other_seconds / seconds_per_eval /
    # seconds_per_val_sample / eval_share_pct / to_dict()
@dataclass(frozen=True) class ValCoverage(n_val, n_total=None, limit=None)  # truncated,
    # coverage_pct, as_metrics()
@dataclass(frozen=True) class SubsetView(dataset, n)                        # map-style ilk n örnek
def split_size(dataset) -> int | None
def limit_val_dataset(dataset, max_n) -> tuple[Any, ValCoverage]   # eval.max_n_val'i veri kümesi
                                                                   # alan döngüler için (ASR)
def batch_decoders() -> tuple[Callable | None, Callable | None]    # seq2seq.greedy_decode_batch /
                                                                   # beam_search_batch çözümleme
def as_token_ids(result) -> list[int]; def batched_greedy_ids(model, inputs, *, max_len, ...)
def require_training_gate(cfg, *, i_have_approval=False, ledger_path=None) -> bool   # GateError → 3
def make_dataloader(...); def collate_for(cfg); def schedule_factor(...); def seed_worker(id)
def apply_freeze(model, cfg) -> int; def build_optimizer(model, cfg); def score_predictions(pairs, metrics)

# --- aicls.ml.profile --------------------------------------------------------------------
def run_profile(cfg, *, steps=200, warmup_steps=5, eval_samples=8, run_dir=None, out_dir=None,
                model=None, train_ds=None, val_ds=None, tracker=None, device=None, probe=None,
                i_have_approval=False, ledger_path=None, write_report=True) -> ProfileReport
class ProfileReport(BaseModel, frozen=True): name; run_id; steps; warmup_steps; measured_steps;
    steps_per_s; samples_per_s; peak_vram_mb: float | None; dataloader_wait_pct; wall_time_s;
    train_seconds; eval_seconds: float | None; checkpoint_seconds; dataloader_wait_s;
    eval_measured; eval_samples; eval_val_split_size; eval_beam; eval_batch_size; eval_decoder;
    seconds_per_val_sample: float | None; train_split_size; projection: ScheduleProjection;
    batch_size; grad_accum; dataset; model_family; device; device_name; dtype; amp; torch_version;
    config_hash; git_sha; citable; synthetic; wiring_check; run_dir; notes
    def epoch_seconds(self, n_samples: int) -> float
@dataclass(frozen=True) class Schedule(total_steps, eval_count, n_val, n_val_total, train_size,
    steps_per_epoch, epochs, reason)
class ScheduleProjection(BaseModel, frozen=True): projected; reason; total_steps; eval_count;
    n_val_per_eval; n_val_total; epochs; steps_per_epoch; beam; seconds_per_step;
    seconds_per_val_sample; train_seconds; eval_seconds; total_seconds; total_hours; eval_share_pct
def resolve_schedule(cfg, *, train_size=None, val_size=None) -> Schedule
def project_schedule(cfg, *, steps_per_s, seconds_per_val_sample, train_size=None,
                     val_size=None) -> ScheduleProjection   # ölçülmemiş eval → projected=False
def estimate_epoch_seconds(*, steps_per_s, n_samples, batch_size, grad_accum=1) -> float
def throughput(records, *, warmup=5) -> tuple[int, float, float]; def peak_vram_mb(spec) -> float | None

# --- aicls.ml.whisper_ft -----------------------------------------------------------------
class WhisperFineTuner:
    def __init__(self, cfg, *, weights_dir=None, run_dir, tracker=None, model=None, processor=None,
                 allow_dirty=False, repo_root=None, gradient_checkpointing=False)
    property model / processor          # from_pretrained(YEREL dizin, local_files_only=True)
    random_init: bool                   # base_dir yok + kapısız yapılandırma → rastgele model
    def require_training_gate(self, *, i_have_approval=False) -> bool
    def training_args_dict(self, *, with_eval=True) -> dict      # accelerate'siz test edilebilir
    def training_args(self, *, with_eval=True) -> Seq2SeqTrainingArguments
    def compute_metrics(self, eval_prediction) -> dict           # wer (tr_v1), wer_raw, n_samples
    def build_trainer(self, *, train_dataset, eval_dataset=None) -> Seq2SeqTrainer
    def fit(self, *, train_dataset, eval_dataset=None, i_have_approval=False) -> FitReport
    def data_collator() / manifest_callback() / decode() / generation_max_length() / last_checkpoint_dir()
class AiclsManifestCallback(cfg, run_dir, *, tracker=None, allow_dirty=False, repo_root=None, ...)
    # ördek-tipli; on_train_begin → run_manifest.json, on_log/on_evaluate → Tracker, on_save →
    # complete_checkpoint_dir, on_train_end → RunManifest.set_steps_per_s
@dataclass class WhisperDataCollator(processor, label_pad_id=-100, decoder_start_token_id=None,
                                    input_frames=None)   # mel genişliğini kodlayıcıya hizalar
    def align_features(self, features) -> list[dict]
class SmokeTokenizer(alphabet=SMOKE_ALPHABET)   # karakter sözlüğü; ağırlıksız duman koşusu için
@dataclass(frozen=True) class SmokeProcessor(feature_extractor, tokenizer); class SmokeEncoding
def build_smoke_processor(cfg) -> SmokeProcessor        # indirme yok, diskten okuma yok
def encoder_input_frames(config) -> int | None          # 2 * max_source_positions
def silence_hf_printer(trainer) -> bool                 # --json çıktısını HF baskısından korur
@dataclass class WhisperDatasetAdapter(dataset, processor, max_label_len=None, sampling_rate=None)
class DependencyUnavailableError(InternalError)   # accelerate/peft/transformers yoksa tipli hata
def resolve_weights_dir(...) / freeze_encoder(model) / apply_lora(model, lora) /
    configure_generation(model, *, language="tr", task="transcribe") / complete_checkpoint_dir(...)

# --- aicls.ml.queue ----------------------------------------------------------------------
class TrainingQueue(lock_path=None, *, job="", check_gpu=True, gpu_busy_mb=512.0, gpu_probe=None)
    # kilit: training/queue.lock (git-ignore); training/queue.txt yalnızca açıklamadır
    def acquire(self, *, timeout_s=0.0, poll_s=2.0) -> QueueLock; def release(self) -> bool
    def read_lock() -> QueueLock | None; def is_busy() -> bool; property held / lock  # + with
class TrainingQueueBusyError(AiclsError)   # ErrorCode.TRAINING_BUSY; details["reason"]
@dataclass(frozen=True) class QueueLock(pid, host, started_at, job, user)
def process_is_alive(pid) -> bool          # Windows'ta OpenProcess; os.kill ASLA kullanılmaz
def probe_gpu_processes(*, timeout_s=15.0) -> list[GpuProcess] | None      # nvidia-smi
```

**Bağımlılıklar.** core, db, ocr, asr, nlu, mathengine, latex2sympy, book, collect, data, eval (torch/transformers/peft/accelerate tembel, işlev içinde). `aicls.api` ve `aicls.eval` bu paketi **içe aktaramaz**.
**Zorunlu testler.** `training/configs/ocr/smoke_synthetic.yaml` ile 2 adım CPU (kayıp sonlu ve azalıyor), checkpoint gidiş-dönüşü, atomik yazım (çökme simülasyonu), resume adım+RNG eşitliği, manifest git SHA ve kirli bayrağı, `load_config` bilinmeyen anahtar reddi, JSONL izleyici şeması, kapı reddi, `test_registry_builds_every_model_at_tiny_size`, `test_whisper_ft_builds_from_tiny_random_config_without_download`, `test_profile_report_fields_on_synthetic`, `test_training_queue_refuses_second_job`.

### Uygulama notları (2026-09-08)

Tam liste ve gerekçeler [../ml/egitim-altyapisi.md §11](../ml/egitim-altyapisi.md#11-uygulama-notları-aiclsml-aiclscli-training-2026-09-08) içindedir. Bu bölümdeki imzaları etkileyen sapmalar:

- `seed_everything(...)` **`int` döner** (kullanılan tohum), `None` değil.
- Kirli ağaç `RuntimeError` değil **`DirtyWorkingTreeError`** fırlatır (kodlama standardı `AiclsError` dışı istisna yasaklar).
- `Trainer.resume(...)` yüklenen **`Checkpoint`** nesnesini döner, `None` değil.
- `Trainer.fit(max_steps=N)` **mutlak** bütçedir; etkin bütçe `min(N, train.max_steps, epochs×adım)`.
- `save_checkpoint`'ün ilk parametresi `ckpt_dir`'dir (`dir` yerleşik adı gölgelenmez); kontrol noktası dizini **dokuz** dosya taşır (`checkpoint.json` eklendi) ve kök `best.json`/`last.json` işaretçileri alır.
- `build_model`/`build_dataset` tam `TrainConfig` de kabul eder; kayıtlı kurucular çözümlenmiş `ModelSpec`/`DatasetSpec` alır.
- `Tracker.log_metrics` anahtar-kelimeli `split` alır, `log_artifact` yazdığı kaydı döner.
- `TrainConfig.is_gate_free` **yalnızca** `data.dataset` adına bakar; `gates.synthetic` bağlayıcı değildir. Bir yapılandırma kendini kapısız ilan ederek `TRAINING` kapısını aşamaz.
- `Trainer` `whisper`/`wav2vec2` ailesini, aile↔`data.mode` uyuşmazlığını ve `model.lora.enabled` durumunu **tipli hatayla reddeder** (sessizce yok saymaz).
- `Trainer.fit` **kuyruk kilidini almaz**; tek-iş kilidi süreç düzeyindedir ve `training/scripts/train.py` alır.
- `WhisperFineTuner.training_args()` ikiye bölündü: eşleme `training_args_dict()` içindedir (bu venv'de `accelerate` kurulu olmadığı için `Seq2SeqTrainingArguments` kurulamıyor).
- **Bilinen iki kusur:** yapılandırmaların `checkpoint.metric` değeri (`val_cer`/`val_wer`) eğiticinin yazdığı anahtarlarla (`cer`/`wer`) uyuşmuyor, bu yüzden "en iyi" kontrol noktası hiç seçilmiyor; ve `RunManifest` ile `JsonlTracker` **ayrı `run_id`** üretiyor, bu yüzden birleştirme anahtarı koşu dizinidir.

---

## aicls.api — sahibi sg (services için ikinci gözden geçiren cs) — kapsam ≥ 80 %

**Sorumluluk.** FastAPI uygulaması: profil bazlı yönlendiriciler, JWT + roller, RFC 9457 hata eşlemesi, ara katmanlar (request id, Server-Timing, güvenlik başlıkları, hız sınırı, CORS), WebSocket akışları, `PipelineService` orkestrasyonu, model kaydı. Ayrıntı [api.md](api.md).

```python
def create_app(settings: Settings | None = None) -> FastAPI            # aicls.api.app
def get_db(request: Request) -> Iterator[Session]                      # app.state.engine, global yok
def get_current_user(request, token=Depends(bearer_token), session=Depends(get_db)) -> User
def require_role(*roles: Role | str) -> Callable[[User], User]         # admin her zaman geçer
class PipelineService:                                                  # aicls.api.pipeline
    def __init__(self, engine: MathEngine, registry: ModelRegistry, settings: Settings)
    def run_text(self, text, ctx=None, user=None, ws_id=None, *, session=None, timer=None,
                 confirm_intent=None, input_kind=InputKind.TEXT,
                 nlu_source=NluSource.RULES) -> PipelineResult
    def run_latex(self, cmd, user=None, ws_id=None, *, session=None, timer=None) -> PipelineResult
    def run_ink(self, ink, user=None, *, session=None, timer=None, beam=4) -> PipelineResult
    def run_audio(self, audio, user=None, ctx=None, ws_id=None, *, session=None, timer=None,
                  to_command=True) -> PipelineResult
class PipelineResult:   # donmuş dataclass
    kind: str                     # result | clarification | book | ui | recognition | transcript
    input_latex: str | None; command: dict | None; result: Result | None
    clarification: ClarificationPayload | None; book_units: tuple[ContentUnit, ...]
    timings: dict[str, float]; operation_id: str | None; nlu: NluTrace | None
    model_versions: dict[str, str | None]; recognition: Recognition | None
    transcript: Transcript | None; plot_url: str | None; cache_hit: bool
class ModelRegistry:                                                    # model_versions.is_active
    def get_recognizer(self) -> Recognizer; def get_transcriber(self) -> Transcriber
    def reload(self) -> None; def warmup(self) -> None
    def state(self, component) -> "stub"|"loaded"|"absent"|"training"
    def versions(self) -> dict[str, str | None]; def states(self) -> dict[str, str]
class WsEnvelope(BaseModel): type: str; seq: int; payload: dict         # aicls.api.ws.protocol
```

**Not.** `PipelineService` yöntemleri **eşzamanlıdır**: SymPy, SQLAlchemy ve tanıyıcılar
eşzamanlı (senkron) kütüphanelerdir; FastAPI senkron uç noktaları zaten iş parçacığı havuzunda
çalıştırır, WebSocket akışları da `run_in_threadpool` ile çağırır. `run_ink` de artık
`Recognition` değil `PipelineResult` döndürür (tanıma `recognition` alanındadır), böylece dört
girdi yolunun tek bir dönüş tipi olur — `modules/api.json` sözleşmesi budur. `session` ve `timer`
yalnızca anahtar sözcüklü ek parametrelerdir: verilmezse hiçbir şey kalıcılaştırılmaz.

**Bağımlılıklar.** core, db, latex2sympy, mathengine, nlu, ocr, asr, book, collect. **ml, eval, data yasak.**
**Zorunlu testler.** auth akışı ve yenileme döndürme, rol matrisi (her yönlendirici × rol), problem+json şekli, Server-Timing başlığı, WS ink partial→final, WS audio geri basınç, rıza zorunluluğu, sihirli bayt reddi, SQL enjeksiyon dizesinin parametreli olması.

---

## aicls.cli — sahibi cs — kapsam ≥ 70 % (ölçülen: 94,7 % — `aicls.ml` ile birlikte)

**Sorumluluk.** `aicls` konsol komutu (typer). Kapı denetimleri burada çağrılır; `poe_tasks.toml` kapılı komutları içermez. Modüller: `main.py` (kök uygulama), `common.py` (ortak yardımcılar), ve komut başına bir modül — `serve`, `db`, `data`, `train`, `evaluate`, `book`, `models`, `contrib`, `nlu`, `report`.

```
aicls serve --host 127.0.0.1 --port 8000 --profile full|collect --reload [--log-level] [--json] [--dry-run]
aicls db upgrade | downgrade <rev> | revision -m "msg" [--version-path] | seed [--prompts-per-intent 8]
aicls data verify <name> | manifest <root> --name --license --source
              | split <manifest> --group speaker --stratify university,gender [--ratios] [--seed]
              | prepare-mathwriting <root> --out [--split ...] [--shard-size] [--render] [--limit]
aicls train --config training/configs/ocr/offline_resnet34_tfm.yaml [--override k=v ...]
            [--synthetic] [--i-have-approval] [--run-dir] [--max-steps] [--device] [--no-val]
            [--dry-run] [--json]                    # TRAINING kapısı; --synthetic ile kapısız
aicls profile --config ... --steps 200 [--warmup 5] [--synthetic] [--i-have-approval] [--out] [--device]
aicls eval parser|cas|nlu|ocr|asr|book|latency|all [--model <version>] [--split test] [--out]
           [--from-db --since --limit] [--db-url] [--db/--no-db] [--promote] [--json]
aicls book load <dir> | validate <dir> | coverage <dir> | export --jsonld out.json --ttl out.ttl
aicls models import [models/registry.yaml] [--models-root] [--verify/--no-verify]
           | list [--component] | activate --component ocr --version 0.3.0
aicls contrib export-manifest --out ... | withdraw --speaker P00012 | stats
aicls nlu try "x karenin türevini al" [--last-latex] [--active-var] [--modality]
aicls report check | promote <file>... | render
```

```python
# aicls.cli.main
app: typer.Typer                    # pyproject: aicls = "aicls.cli.main:app"
class AiclsGroup(typer.core.TyperGroup):
    def invoke(self, ctx) -> Any    # tüm ağaç için AiclsError → typer.Exit(exit_code)
def main(ctx, version: bool = False) -> None        # kök geri çağırma: --version

# aicls.cli.common
EXIT_OK = 0 ; EXIT_ERROR = 1 ; EXIT_GATE = 3
@dataclass class CliState(json_output: bool = False); def cli_state() -> CliState; def use_json(flag)
class StderrStream                                  # yazma anında sys.stderr'i çözer
def cli_logger(name) -> FilteringBoundLogger        # geç bağlanır (yapılandırmadan SONRA)
def configure_cli_logging(level: str | None = None) -> None       # günlükler → stderr
def emit(payload, *lines) / echo(line) / fail(payload, *lines, code=EXIT_ERROR) -> NoReturn
def render_error(exc: AiclsError) -> int; def dump_json(payload) -> str; def jsonable(v)
def split_csv(v) / parse_ratios(v, default) / read_jsonl(path) / resolve_db_url(url)
@contextmanager def db_session(url=None, *, migrate=True) -> Iterator[Session]

# aicls.cli.train / evaluate / models / report (test edilebilir çekirdekler)
def resolve_config(config: Path, overrides, *, synthetic: bool) -> TrainConfig
def plan_payload(cfg, *, gate_checked: bool, run_dir: Path | None) -> dict
class EvalKindChoice(StrEnum): PARSER CAS NLU OCR ASR BOOK LATENCY ALL
@contextmanager def preserved(paths: Sequence[Path]) -> Iterator[None]   # latest.json/summary.md
class RegistryEntry(BaseModel, extra="forbid") / class RegistryFile      # models/registry.yaml
def load_registry(path: Path) -> RegistryFile
def load_reports(directory: Path) -> list[EvalReport]
```

**Bağımlılıklar.** tüm paketler.
**Zorunlu testler.** her komutun `--json` yükü, kapı reddi (bayrak yok / yalnız bayrak / profil), `--synthetic`'in yapılandırmayı yeniden yazdığı, `aicls eval`'in `latest.json`'ı terfi ettirmediği, `models import`'un sha uyuşmazlığını reddettiği, `report check`'in kirli koşuyu reddettiği.

### Uygulama notları (2026-09-08)

- **`aicls` konsol komutu kurulu DEĞİLDİR.** `pyproject.toml` hedefi gerçektir (`from aicls.cli.main import app` çalışır) ama paket venv'e kurulmamıştır; venv `Scripts/` dizininde `aicls.exe` yoktur. Bugünkü çağrı: `PYTHONPATH=src python -m aicls.cli.main ...` veya bir kereye mahsus `pip install -e .`.
- **Modül adı `evaluate.py`'dir**, `eval.py` değil (yerleşik adı gölgelememek için); `main.py` onu `eval` komutu olarak bağlar.
- **Hata sınırı tek yerdedir:** kök `typer.Typer(cls=AiclsGroup)` `Group.invoke`'u sararak tüm ağaç için `AiclsError` yakalar. `GateError` → 3, diğer `AiclsError` → 1, click kullanım hataları → 2. Yeni bir komutta unutulması yapısal olarak mümkün değildir.
- **Günlükler stderr'e gider** (`configure_cli_logging`), aksi hâlde `aicls train --json | jq` bozulurdu; bu yüzden CLI günlükçüleri modül düzeyinde değil, kullanım yerinde bağlanır (`cli_logger(__name__)`).
- **`--synthetic` kapıyı atlamaz**, `data.dataset`'i ailenin sentetik kardeşine yeniden yazar ve kararı yine `TrainConfig.is_gate_free` verir.
- **`aicls eval` asla terfi ettirmez**; koşumun yazdığı `latest.json`/`summary.md` sonradan geri konur (`preserved`). Terfi `aicls report promote`'un işidir (ADR-0015 §4). `--model` gerçek bir davranış değişikliğidir: sürüm kayıtlı **ve** terfi ettirilmiş olmalıdır, ölçüm API'nin servis ettiği `ModelRegistry` üzerinden koşar, kontrol noktası yoksa rapor `stub: true` yazar.
- **`--split` yalnızca ocr/asr için ve yalnızca `--model` ile** kabul edilir; diğer türlerde `split_not_applicable` hatası verir (sessizce yok sayılan bayrak, hata veren bayraktan kötüdür).
- **`aicls models import` varsayılan olarak SHA-256 doğrular**; `model_versions.path` kontrol noktası **dizinini** saklar; `config_hash`/`git_sha`/`run_id` kontrol noktasının `run_manifest.json`'undan okunur.
- **`aicls db seed`** yayımlanmış rıza sürümlerini ve NLU komut envanterinin örneklerinden üretilen okuma istemlerini yazar (ikisi de idempotent). `datasets/nlu/reading_scenarios.yaml` bir gün eklenirse `prompt_rows` onu okumalıdır.
- **`aicls db revision`** Alembic'in stdout anlatımını stderr'e yönlendirir (`--json` bozulmasın diye) ve `--version-path` kabul eder.
- Yeni bayraklar (şartnamede olmayan): her okuma komutunda `--json`, `train`/`serve`'de `--dry-run`, `eval`'de `--db-url`/`--no-db`/`--promote`, `models import`'ta `--verify/--no-verify`, `db revision`'da `--version-path`, `eval`'de `all` türü.
- **Türkçe dize eklenmedi.** Kullanıcıya görünen hata metni her zaman `AiclsError.message_tr`'dir (`aicls.core.messages_tr`); komut çıktısı, yardım metni ve günlük olayları İngilizcedir.

---

## web/ — sahibi sg — kapsam: vitest ≥ 70 %, Playwright akıllı duman testleri sayfa başına

**Sorumluluk.** React 18 + TypeScript (strict) + Vite tek sayfa uygulaması; özellik klasörleri (workspace, voice, command, results, history, book, contribute, feedback, admin); OpenAPI'den üretilen tipli istemci; tipli WebSocket istemcisi; KaTeX (MathJax yalnızca desteklenmeyen makro yedeği); Türkçe i18n.

```ts
// src/api/client.ts
export const api = createClient<paths>({ baseUrl: API_BASE_URL, credentials: "include", fetch });
export function setAuthTokenProvider(p: () => string | null): void;         // app/auth.ts kaydeder
export function setUnauthorizedHandler(h: (p: ProblemDetail | null) => void): void;
export function isProblemDetail(v: unknown): v is ProblemDetail;
export function problemMessage(error: unknown, fallback: string): string;   // RFC 9457 detail_tr
// src/api/ws.ts
export class TypedSocket<TIn extends ClientEnvelope = ClientEnvelope, TOut extends ServerEnvelope = ServerEnvelope> {
  constructor(path: string, ticket: string, options?: TypedSocketOptions);  // options: test/üretim ayrımı
  send(m: TIn): void; sendBinary(d: ArrayBufferLike | ArrayBufferView): void;
  onMessage(cb: (m: TOut) => void): () => void; close(): void;
  get status(): SocketStatus; get isThrottled(): boolean; get pending(): number;
}
export function resolveWsUrl(path: string, ticket: string, origin?: string): string;
// src/features/workspace/StrokeBuffer.ts
export class StrokeBuffer { push(p: {x: number; y: number; t: number}): void; endStroke(): void; toInkML(): string; toCanvasJson(): InkJson; toPNG(scale?: number): Promise<Blob>; clear(): void; getStrokes(): InkStroke[]; setCanvasSize(s: CanvasSize): void; get strokeCount(): number }
// src/lib/wavEncoder.ts
export async function encodeWav16k(chunks: Float32Array[], inputRate: number): Promise<Blob>   // OfflineAudioContext, yoksa pencereli-sinc yedeği
export function encodeWavBytes(samples: Float32Array, sampleRate: number): ArrayBuffer          // 44 baytlık kanonik RIFF başlığı
export function resampleSinc(input: Float32Array, inputRate: number, outputRate: number, taps?: number): Float32Array
// src/lib/latency.ts
export function markSubmit(opId: string, modality?: Modality): void;
export function markRendered(opId: string): LatencySample | null;          // işaret yoksa null
export async function markRenderedAndReport(opId: string): Promise<LatencySample | null>;
// src/lib/latexSafety.ts  (aicls.core.latex_safety'nin bire bir TS portu)
export function sanitizeLatex(s: string, options?: { maxLen?: number }): string;
export function findDisallowedCommands(s: string): string[];
// src/lib/inkml.ts
export function strokesToInkML(strokes, canvas, options?: { label?: string }): string;
export function strokesToCanvasJson(strokes, canvas): InkJson;             // {version:1, canvas:{w,h}, strokes:[{x[],y[],t[]}]}
// src/features/results/MathView.tsx
export function MathView(props: { latex: string; display?: boolean; onError?: (e: Error) => void }): JSX.Element  // sanitize + katex.renderToString({trust:false, maxExpand:1000})
// src/features/command/CommandPreview.tsx
export function CommandPreview(props: { latex: string; command: CommandDto; confidence: number; onConfirm(): void; onEdit(): void; voice?: boolean }): JSX.Element
// hooks
export function useInkSocket(o?: UseInkSocketOptions): InkSocketState;      // /ws/v1/ink
export function useAudioSocket(o?: UseAudioSocketOptions): AudioSocketState;// /ws/v1/audio
```

**Bağımlılıklar.** api (OpenAPI şeması).
**Zorunlu testler.** `InkCanvas.test.tsx` (2 iz → geçerli InkML), `MathView.test.tsx` (\frac render; bilinmeyen makroda MathJax yedeği), `wavEncoder.test.ts` (16 kHz mono PCM16 başlığı), `ws.test.ts` (yeniden bağlanma, geri basınç), `e2e/browsers.spec.ts` (chromium/firefox/webkit'te komut → KaTeX sonucu).

### Uygulama notları (iskelet adımı, 2026-09-07)

- **Durum.** İskelet tamamlandı: 13 vitest dosyası, 137 test, satır kapsamı **%88,5** (eşik %70). `npx tsc --noEmit`, `npm run lint` (eslint + prettier) ve `npm run build` temiz. `e2e/*.spec.ts` yazıldı ama **çalıştırılmadı** (tarayıcı ikilileri indirme kapısına tabi; `npx playwright install` gerekir).
- **API istemcisi.** `docs/api/openapi.json` henüz üretilmediği için `src/api/paths.ts` elle yazılmış, `openapi-typescript` çıktısıyla **yapısal olarak uyumlu** en küçük `paths` tipini içerir; `src/api/types.ts` de api.md §3'teki gövdelerin elle yazılmış karşılığıdır. Şema üretildiğinde `paths.ts` tek satıra (`export type { paths } from "./generated/schema";`) iner, çağrı yerleri değişmez. Alanlar `snake_case`'tir (tel biçimi).
- **`baseUrl`.** `/api/v1` yolu `location.origin` ile mutlaklaştırılır (`API_BASE_URL`). İstek aynı yere gider; fark, göreli URL kabul etmeyen ortamlarda (jsdom/undici) `Request` kurulabilmesidir. `fetch` çağrı anında `globalThis`'ten çözülür, böylece testler ve olası bir servis-çalışanı katmanı araya girebilir.
- **`markRendered` dönüşü.** Sözleşmedeki imza `LatencySample` idi; uygulama `LatencySample | null` döndürür — gönderim işareti bulunmayan bir işlem için uydurma bir ölçüm üretmek yerine hiçbir şey bildirmemek doğru davranıştır (ADR-0014 ortalamayı bozmamak).
- **`TypedSocket`.** Tip parametreleri `ClientEnvelope`/`ServerEnvelope` ile sınırlandı (zarf `{type, seq, payload}` olmak zorunda); `seq` istemcide otomatik artar. Üçüncü bir `options` parametresi eklendi: soket fabrikası (test), yeniden bağlanma gecikmeleri, `slow_down` varsayılanı, durum geri çağrısı. Biletler tek kullanımlık olduğundan her yeniden bağlanmada `ticketProvider` yeni bilet alır. Geri basınçta gönderimler **kuyruğa** alınır, düşürülmez; sıra korunur.
- **`latexSafety.ts`.** `aicls.core.latex_safety` sunucuda, bu dosya tarayıcıda aynı izin listesini uygular (aynı komut/ortam kümeleri, aynı argüman yutma tablosu, aynı birleşme koruması). İki kopya kasıtlıdır: sunucu güvenlik sınırı, istemci savunma derinliğidir. Python tarafı değişirse **ikisi birlikte** güncellenmelidir.
- **MathJax yedeği.** İzin listesinden geçen ama KaTeX'in tanımadığı makrolar (`\sech`, `\csch`, `\arccot`) `data-mathjax-fallback="true"` işaretli, MathJax sınırlayıcılarıyla sarılmış bir yuvaya düşer; MathJax paketi yüklü olmasa bile kullanıcı boş kutu yerine kaynağı görür. Bilinmeyen makro sanitizer tarafından zaten silindiği için yedeğin gerçek tetikleyicisi budur.
- **Canvas JSON biçimi.** `{version:1, canvas:{w,h}, strokes:[{x:number[], y:number[], t:number[]}]}`; `t` ilk örneğe göre milisaniyedir (duvar saati sızmaz). `aicls.ocr.ink_from_canvas_json` bu biçimi bekleyecek şekilde yazılmalıdır; farklılaşırsa bu not ve `lib/inkml.ts` birlikte güncellenir. (Entegrasyon kapısı 2026-09-07: iç içe `canvas` nesnesi ayrıştırıcıya eklendi, sözleşme `tests/integration/test_web_payload_contracts.py` ile doğrulanıyor.)
- **Ses.** `MediaRecorder` yerine `ScriptProcessorNode` + `encodeWav16k` (ADR-0017). `OfflineAudioContext` yoksa Hann pencereli sinc yeniden örnekleyici devreye girer; DC kazancı 1'dir (test edilir).
- **Durum yönetimi.** Mimari taslağı `zustand` ve `@tanstack/react-query` anıyordu; iskelet, `useSyncExternalStore` üzerine kurulu ~40 satırlık `app/store.ts` ile ilerledi (aynı `getState/setState/subscribe` arayüzü). Bağımlılık azaltma tercihi; gerekirse kütüphaneye geçiş mekaniktir.
- **Dosya adı farkı.** Depo ağacında `features/voice/Recorder.ts` yazıyor; uygulama `Recorder.tsx`'tir (bileşen). `features/<alan>/*Page.tsx` sayfaları ve `lib/commandFormat.ts` ağaçta listelenmeyen eklerdir.
- **Erişilebilirlik/tema.** Tüm metinler `src/i18n/tr.json` (İngilizce anahtar, Türkçe değer); `index.html` `lang="tr"`. Renk şeması `prefers-color-scheme` ile çalışır, açık seçim `<html data-theme>` ve `localStorage` ile saklanır. Yazı tipleri yalnızca aile adıyla istenir (ağ yok, CSP `font-src 'self'`); KaTeX yazı tipleri pakete gömülür.
- **Test ortamı tuzakları.** (1) jsdom'da `getComputedStyle`, KaTeX'in ürettiği MathML alt ağacında patlar → MathML içeren ağaçlarda `getByRole` yerine öznitelik sorgusu kullanılır. (2) jsdom `Blob.arrayBuffer()` sunmaz → `FileReader`. (3) jsdom'da `PointerEvent` yoktur → `src/test/setup.ts` `MouseEvent` tabanlı bir çoklu-dolgu kurar. (4) 2B canvas bağlamı ve `toBlob` sahte nesnelerle karşılanır; piksel davranışı Playwright'a bırakılmıştır.
- **node_modules.** OneDrive senkronizasyonu dışında tutulur: `web/node_modules`, `C:\Users\cagri\.aicls\web\node_modules`'a bağlanan bir dizin bağlantısıdır (junction). Hedef yolun **`node_modules` ile bitmesi zorunludur**; aksi hâlde TypeScript gerçek yolu çözdükten sonra kardeş paketleri bulamaz (`playwright/test`, `@testing-library/dom` çözümlenemez). `npm install` bağlantıyı sildiği için kurulumdan sonra dizin taşınıp bağlantı yeniden kurulmalıdır (bkz. `tools/onedrive_setup.ps1`).
