# LaTeX → SymPy Ayrıştırıcı

Paket: `aicls.latex2sympy`. Karar: [ADR-0004](../adr/0004-latex-ayristirici-lark-dilbilgisi.md). Bu belge İP-4'ün "ayrıştırma algoritmalarını ve eşleştirme kurallarını detaylandıran teknik rapor" ara çıktısının teknik omurgasıdır; rapor sürümü `docs/raporlama/ip-04-cas.md` içine bu belgeden ve `reports/metrics/parser_*.json`'dan derlenir.

## 1. Neden kendi dilbilgimiz

`sympy.parsing.latex.parse_latex` ANTLR çalışma zamanı ister, örtük çarpımı ve `dx` ayrımını sessizce yanlış çözebilir ve sınıf notasyonuna (`\frac{d}{dx}`, `\lim_{x\to a^+}`, `f'(x)`) genişletilemez. Kendi dilbilgimiz: (1) konumlu hata mesajı, (2) NLU'nun da hedeflediği tipli AST, (3) ölçülebilir >%99 ayrıştırma başarısı, (4) saf Python (Lark). `sympy.parsing.latex` yalnızca örtüşen alt kümede farksal test kâhini olarak kullanılır (antlr yoksa test atlanır).

## 2. Araç ve yükleme

- Lark 1.2, `parser="lalr"`, `lexer="contextual"`, `propagate_positions=True`, `maybe_placeholders=False`.
- Dilbilgisi dosyası `src/aicls/latex2sympy/grammar/latex_math.lark`; `commands.yaml` desteklenen komut tablosu (komut → AST düğümü / SymPy karşılığı / KaTeX yazımı) hem dilbilgisi terminallerini üretmek hem de rapordaki eşleme tablosunu otomatik yazmak için kullanılır.
- `LatexParser` modül düzeyinde tek örnek; Lark önbelleği `AICLS_DATA_ROOT/cache/lark/`.

## 3. Boru hattı

```
src ──normalize_latex──▶ metin + repairs[] ──Lark──▶ Tree ──transformer──▶ Node (AST, span'lı)
     ──to_sympy(SymbolTable)──▶ sympy.Basic ──MathEngine──▶ sonuç ──sympy_to_latex──▶ LaTeX (KaTeX)
```

### 3.1 Normalizasyon (`normalize.py`)

Sırayla: NFC; `\left`/`\right` kaldır (dengesizse onarım kaydı); `\,` `\;` `\!` `\quad` `~` `\displaystyle` `\textstyle` sil; `\dfrac`/`\tfrac` → `\frac`; Unicode `×·−²³√∞π≤≥≠→` → komut; boşluk sıkıştır; tek eksik kapanış `}` onarımı (yalnızca 1 adet, sonda). Her onarım `NormalizeResult.repairs` listesine yazılır ve API `ParseOut.repairs` alanında görünür — OCR çıktısındaki düzeltmeler gizlenmez.

### 3.2 Dilbilgisi kapsamı v1 (Temel Analiz müfredatı)

| Sınıf | Desteklenen | SymPy eşlemesi |
|---|---|---|
| Sayılar | tam sayı, ondalık (`.` ve `,` — `,` yalnızca iki rakam arasında), `\frac{a}{b}` sayısal | `Integer`, `Float`(str), `Rational` |
| Semboller | tek Latin harf, alt indis `x_1`, `x_{10}`, Yunan `\alpha..\omega`, `\Gamma..\Omega`, üs `f'`, `f''` | `Symbol` (SymbolTable: `real=True` varsayılan; `n,k,m` toplamda `integer=True`) |
| Sabitler | `e`, `\pi`, `\infty`, `i` (ParserConfig.imaginary_i) | `E`, `pi`, `oo`, `I` |
| Operatörler | `+ - \cdot \times / ^` , örtük çarpım, tek terimli `-`, `\pm` (iki dal), `!` | `Add`, `Mul`, `Pow`, `factorial` |
| Gruplama | `{}`, `()`, `[]`, `\{ \}`, `\left \right` (normalize edilir), `|x|` ve `\lvert \rvert` | `Abs` |
| Fonksiyonlar | `\sin \cos \tan \cot \sec \csc \arcsin \arccos \arctan \sinh \cosh \tanh \ln \log \log_{b} \exp \sqrt \sqrt[n]`, üslü fonksiyon `\sin^{2} x`, `\sin^{-1}`, `f(x)` uygulaması | `sin`, ..., `log(x, b)`, `sqrt`, `root`, `Function("f")(x)` |
| Kalkülüs | `\int`, `\int_a^b`, `\iint` (v1.1), `dx`/`\,dx`/`\mathrm{d}x`, `\frac{d}{dx}`, `\frac{d^n}{dx^n}`, `\frac{\partial}{\partial x}`, `\frac{\partial^n}{\partial x^n}`, kısayol biçimi `\frac{dy}{dx}`, `\frac{\partial f}{\partial x}`, `\frac{d^{n}y}{dx^{n}}`, `\frac{\partial^{n} f}{\partial x^{n}}`, `f'(x)`, `\lim_{x \to a}`, `\lim_{x \to a^{+}}`, `\lim_{x \to \infty}`, `\sum_{k=1}^{n}`, `\prod` | `Integral`, `Derivative`, `Limit(dir)`, `Sum`, `Product` |
| İlişkiler | `= \neq < > \leq \geq \le \ge \approx` (en fazla bir üst düzey ilişki) | `Eq`, `Ne`, `Lt`, ... |
| Kümeler/aralıklar | `[a,b]`, `(a,b)`, `[a,b)`, `\{x \mid ...\}` (v1.1), `\in`, `\mathbb{R}` | `Interval`, `ConditionSet` |
| Matrisler | `\begin{pmatrix}...\end{pmatrix}`, `bmatrix` (v1.1) | `Matrix` |
| Geçiş | `\text{...}` fonksiyon adı değilse yok sayılır, `\mathrm{d}` diferansiyel, `\operatorname{f}` | |

v1'de bilinçli dışlanan ve `UnsupportedConstruct(construct=...)` ile adı verilerek reddedilenler: `\begin{cases}`, `\begin{align}`, `\vec`, `\hat`, `\overline`, `\label`, `\tag`, `\forall`, `\exists`, `\binom`, **karma kısmi türev** (`\frac{\partial^{2} f}{\partial x \partial y}`), tensör indisleri, çok harfli tanımlayıcı (`xy` her zaman `x*y`; `SymbolTable.declare("xy")` ile değiştirilebilir). `datasets/latex/unsupported.jsonl` bu davranışı test eder.

`aicls.ocr.CURRICULUM_LATEX` (32 dize, el yazısı modelinin ürettiği alfabe) içinden **tam
olarak ikisi** bu listeye düşer ve ölçüm bu ikisiyle sınırlıdır: `\forall\varepsilon>0`
(`construct = "\forall"`) ve `\binom{n}{k}` (`construct = "\binom"`). İkisi de v1.1 yol
haritasındadır ve API katmanı bunları "tanındı ama hesaplanamıyor" 422 yükü olarak
döndürmelidir (bkz. KNOWN_ISSUES ayrıştırıcı-2). Bu sınır
`tests/integration/test_latex2sympy_contracts.py` içinde sayıyla sabitlenmiştir.

Kapsam dışı ama **adlandırılmamış** kalan tek bilinen biçim, Leibniz paydasında alt indisli
değişkendir (`\frac{\partial f}{\partial x_{1}}`): hem operatör hem kısayol biçiminde
`ParseError` verir. Temel Analiz v1 tek değişkenli olduğu için v1.1'e bırakıldı.

### 3.3 Belirsizlik çözümü (dilbilgisi içinde, Earley yok)

```
?relation : expr (REL_OP expr)?
?expr     : term (("+"|"-") term)*
?term     : factor (("\\cdot"|"\\times"|"/") factor | juxta_factor)*
?factor   : "-" factor | power
?power    : postfix ("^" exponent)?
?postfix  : atom ("!" | PRIME+)?
```

- **Örtük çarpım:** `juxta_factor` yalnızca ikili operatörle başlamayan atomlar (sayı, sembol, `\frac`, `\sqrt`, parantez, fonksiyon) için izinlidir; `2x`, `x\sin x`, `2\pi r` → `Mul`.
- **Fonksiyon argümanı:** `\sin x y` → `sin(x)*y` (argüman = ilk `power`); parantezli `\sin(x y)` tüm parantez. Kural belgelenir ve `test_parse_sin_x_y_binds_first_power` ile sabitlenir.
- **`/` ile örtük çarpım:** `a/bc` → `(a/b)*c` (soldan sağa, örtük çarpım `/` ile aynı önceliktedir); KaTeX'e geri yazımda `\frac{a}{b} c` gösterilir.
- **Diferansiyel:** `DIFFERENTIAL: /\\,?(\\mathrm\{d\}|d)[a-zA-Z]/` terminali yalnızca `integral` kuralı bağlamında etkindir (bağlamsal lexer); integral dışında `dx` → `d*x`.
- **`\frac{d}{dx}`** tek terminal olarak lexlenir (`LEIBNIZ`), sıradan kesirle çakışmaz; `\frac{d^{2}}{dx^{2}}` mertebe yakalar. Gövdesi payda olan kısayol biçimi (`\frac{\partial f}{\partial x}`, `\frac{d^{2}y}{dx^{2}}`) ayrı bir terminaldir (`LEIBNIZ_OF`); payın gövdesi **tek sembol** olmak zorundadır, böylece `\frac{d + 1}{dx}` ve `\frac{2d}{dx}` sıradan kesir olarak kalır.
- **`\log`:** `ParserConfig.log_base="e"` (varsayılan; sınıf kullanımı `\ln` ile eş) — ADR-0004'te not; `\log_{10}` her zaman açık.
- **`e^{x}`** `E**x`, ancak `SymbolTable.declare("e")` ile sembol yapılabilir; `\exp(x)` eşdeğer.

### 3.4 AST (`ast.py`, dondurulmuş dataclass'lar, `span=(start,end)`)

`Num(value:str)`, `Sym(name:str, sub:str|None)`, `Const(name: "e"|"pi"|"infty"|"i")`, `Neg(x)`, `Add(terms)`, `Mul(factors)`, `Div(num, den)`, `Pow(base, exp)`, `Root(radicand, index)`, `Abs(x)`, `Func(name, arg, power:Node|None)`, `Log(arg, base:Node|None)`, `Call(fname, args)`, `Factorial(x)`, `Integral(body, var, lower, upper)`, `Derivative(body, var, order:int, partial:bool)`, `Prime(fname, arg, order)`, `Limit(body, var, to, direction:"+"|"-"|"+-")`, `Sum(body, var, lower, upper)`, `Product(...)`, `Relation(op, lhs, rhs)`, `Interval(lo, hi, left_open, right_open)`, `SetBuilder(var, cond)`, `Matrix(rows)`, `PlusMinus(a, b)`. Her düğüm `accept(visitor)` sunar; `to_latex` ve `to_sympy` birer `Visitor`.

### 3.5 SymPy'a çevirme (`to_sympy.py`)

`SymbolTable` varsayılanları: `x,y,z,t` `real=True`; `n,k,m` `integer=True, positive=True` yalnızca `Sum/Product` bağlı değişkeni olarak; `e/pi/infty/i` ayrılmış. Öneri A.2.2 eşlemesi birebir: `\frac` → `Rational` (sayısal) / `Mul(a, Pow(b,-1))` (sembolik, `evaluate=False` ile korunur, sonra `MathEngine` sadeleştirir); `\sin` → `sin`; `\log` → `log`; `\sum` → `Sum`; `\int` → `Integral`; `\lim` → `Limit`; `\frac{d}{dx}` → `Derivative`. Kalkülüs düğümleri **değerlendirilmemiş** nesneler üretir; hesaplama `mathengine` sorumluluğudur (`doit()`), böylece `POST /compute/parse` yan etkisizdir.

### 3.6 Yazıcı (`printer.py`)

`sympy_to_latex(expr, style=KATEX)` = `sympy.latex(expr, fold_short_frac=False, mul_symbol="dot", ln_notation=True, inv_trig_style="full", order="lex")` + KaTeX'in desteklemediği makroların (`\operatorname*`, `\substack`) izin listesine göre sadeleştirilmesi. Çıktı her zaman `sanitize_latex` süzgecinden geçer. `\frac{1}{3}` örneği (`test_sympy_one_third_prints_frac_1_3`) öneri A.2.2'yi birebir doğrular.

## 4. Hata politikası

- Lexer/parser hatası → `ParseError(line, col, expected[], snippet)`; Türkçe mesaj `messages_tr` şablonu: "Konum {col}: ifade tamamlanmamış, {expected} bekleniyor". Kısmi sonuç yok.
- Dilbilgisi dışı komut → `UnsupportedConstruct(construct="\\begin{cases}")` → API 422 `UNSUPPORTED_CONSTRUCT`.
- Anlamsal hata (değişkensiz integral, noktasız limit) → `SemanticError`.
- `parse_latex` **yalnızca** `AiclsError` alt sınıfı fırlatır; hypothesis ile rastgele Unicode üzerinde fuzz test edilir (`test_parse_never_raises_non_aicls_error`).
- Girdi uzunluğu 2048 karakterle sınırlıdır (`MAX_INPUT_CHARS`), `to_sympy` 1 s'lik iş parçacığı zaman aşımı taşır (ölçüm için; asıl koruma `mathengine.sandbox`).

## 5. Gidiş-dönüş doğrulama (`roundtrip.py`)

Üç seviye, hepsi CI'da:

1. **AST gidiş-dönüşü (özellik testi):** `parse_latex(to_latex(ast)) == ast` — hypothesis stratejisi `tests/property/strategies.py::st_expr` (Num/Sym/Add/Mul/Div/Pow/Func/Root/Integral/Derivative/Limit üzerinde özyinelemeli, derinlik ≤ 5).
2. **Sembolik gidiş-dönüş:** `A = to_sympy(parse(s))`, `B = to_sympy(parse(sympy_to_latex(A)))`, `simplify(A - B) == 0` veya 5 rastgele reel noktada `|A-B| < 1e-6·(1+|A|)` (tekillikten kaçınarak); ilişkilerde yapısal eşitlik. Bu test **`sympy.latex()` çıktısının dilbilgisi içinde kaldığını** da kanıtlar — A.2.2'de CAS sonuçları `latex()` ile ekrana gider ve kullanıcı bunları yeniden girdi olarak gönderebilir.
3. **Farksal test:** örtüşen alt kümede `sympy.parsing.latex.parse_latex(s)` ile eşdeğerlik (antlr yoksa atlanır).

`roundtrip_check(src)` üretimde `AICLS_ENV=dev` iken `POST /compute/parse` yanıtına `roundtrip_ok` alanı ekler.

## 6. Ölçüm: ayrıştırma başarısı (>%99)

- Korpus `datasets/latex/valid_inputs.jsonl` (≥ 2000 satır): `{id, latex, source: "mathwriting_label"|"curriculum"|"ocr_output"|"sympy_latex", expect_ok: true}`. Kaynak dağılımı: müfredat girdileri (tüm `problems.jsonl` `input_latex` alanları), MathWriting normalize etiketlerinden müfredat sembol kümesiyle sınırlı örnekler (veri seti kapısı açılınca eklenir; o zamana kadar fikstür etiketleri), sentetik OCR-tarzı bozulmalar (`\left` eksik, fazla `{}`, `\dfrac`), `sympy.latex()` çıktıları.
- Başarı tanımı (sıkı): `parse_latex` + `to_sympy` istisnasız **ve** seviye-2 gidiş-dönüş `ok`. Sessiz yanlış ayrıştırma başarısızlık sayılır.
- `aicls eval parser` → `reports/metrics/parser_valid_<gitsha>.json` (`rate`, `n`, `ci95`, kaynak bazında kırılım, başarısızlık kimlikleri). Test: `test_curriculum_and_valid_inputs_parse_success_rate_above_0_99`.

## 7. 1000 problemlik müfredat test seti (İP-4 "test veri seti ve analiz raporu")

### 7.1 Yazım biçimi: YAML kartları

`datasets/curriculum/cards/<topic>.yaml` — M. M. Turhan yazar, Ç. Şahin inceler; 8. haftada 300, 6. ayda 1000 satır. Konu kotaları `datasets/curriculum/README.md` içindeki ağırlık tablosuna göre (limit 120, süreklilik 60, türev 200, türev uygulamaları 120, belirsiz integral 150, belirli integral 120, integral uygulamaları 80, diziler-seriler 80, denklem/eşitsizlik 50, grafik 20 = 1000).

```yaml
- id: turev-0042
  topic: turev
  subtopic: zincir_kurali
  chapter_ref: aicls:bolum-3
  difficulty: 2
  input:
    kind: latex            # latex | text_tr  (text_tr satırları NLU'yu da sınar)
    value: "\\frac{d}{dx} \\sin(x^{2})"
  command: {op: diff, expr: "\\sin(x^{2})", var: x, order: 1}
  expected:
    latex: "2 x \\cos{\\left(x^{2} \\right)}"
    srepr: "Mul(Integer(2), Symbol('x'), cos(Pow(Symbol('x'), Integer(2))))"
  checker: symbolic        # symbolic | numeric | set | bool | exact_latex | plot_ok
  checker_args: {tolerance: 1.0e-9, domain: [-3, 3]}
  source: "kitap:3.4"
  author: mmturhan
  reviewed_by: csahin
  status: reviewed         # draft | reviewed | frozen
  notes_tr: ""
```

### 7.2 Derleme ve dondurma

`scripts/build_curriculum_testset.py` kartları `datasets/curriculum/problems.jsonl`'a derler; her `expected.srepr`'ın `sympy` ile yeniden kurulup `expected.latex` ile eşdeğer olduğunu, `input.value`'nun ayrıştığını, kimliklerin tekil olduğunu, konu kotalarının sağlandığını doğrular. `status: frozen` satırların SHA-256'sı `datasets/curriculum/FROZEN.sha256`'ya yazılır; `test_frozen_curriculum_rows_unchanged` bu satırların değişmediğini garanti eder. Rapor etiketi (`report-2027-05`) anında tüm satırlar `frozen` olur.

### 7.3 Değerlendirme

`aicls eval cas` (`aicls.eval.cas_eval.run_cas_eval`): her satır için ayrıştır → komutu kur → `MathEngine.execute` (üretimdekiyle aynı sandbox) → `checker` ile karşılaştır (symbolic: `symbolic_equivalent`; numeric: alan içinde 20 nokta; set: `FiniteSet` eşitliği; exact_latex: `latex_v1` sonrası dize). Çıktı `reports/metrics/cas_curriculum_<gitsha>.json`: `parse_success`, `correct_rate` (hedef > 0.99), `ci95`, konu/alt konu kırılımı, ortalama ve p95 SymPy süresi, başarısız kimlikler ve nedenleri, `unsupported` sayısı ayrı sütun. Bu tablo İP-4 analiz raporunun gövdesidir.

## 8. Testler (seçki; tam liste [test-stratejisi](../gelistirme/test-stratejisi.md))

`test_parse_frac_nested_returns_div_node`, `test_parse_implicit_multiplication_2x_sin_x`, `test_parse_derivative_leibniz_frac_d_dx`, `test_parse_definite_integral_bounds_to_sympy_integral`, `test_parse_indefinite_integral_dx_detected`, `test_parse_limit_with_direction_plus`, `test_parse_greek_alpha_maps_to_symbol_alpha`, `test_parse_sin_x_y_binds_first_power`, `test_parse_unsupported_cases_raises_with_construct_name`, `test_parse_error_reports_line_and_column`, `test_normalize_latex_records_single_brace_repair`, `test_property_roundtrip_ast_to_latex_to_ast`, `test_property_latex_to_sympy_symbolically_equivalent_after_reprint`, `test_sympy_latex_output_is_within_grammar_on_curriculum`, `test_differential_against_sympy_parse_latex_subset`, `test_sympy_one_third_prints_frac_1_3`, `test_curriculum_and_valid_inputs_parse_success_rate_above_0_99`, `test_frozen_curriculum_rows_unchanged`.

## 9. Uygulama notları

Bu bölüm, `aicls.latex2sympy` uygulanırken (2026-09-07) yukarıdaki tasarımdan **saptığımız** ve
tasarıma **eklediğimiz** noktaları kaydeder. Bölüm 1–8 tasarım otoritesidir; burada anlatılan
her madde kodda ve testlerde sabitlenmiştir.

### 9.1 `dx` bağlamsal terminal yerine AST üzerinde ayrılıyor (sapma)

§3.3 diferansiyeli "yalnızca `integral` kuralı bağlamında etkin" bir `DIFFERENTIAL` terminali
olarak tanımlıyordu. LALR'de bu **mümkün değil**: bir atomdan sonraki durum (`postfix: atom •`)
tüm bağlamlarda aynı LR durumudur ve LALR ileri-bakış kümelerini birleştirdiği için
`DIFFERENTIAL` integral dışında da kabul edilebilir hâle gelir; sonuç olarak `2dx` ifadesi
integral dışında da diferansiyel olarak lexlenip ayrıştırma hatası verirdi. Bunu önlemenin tek
dilbilgisel yolu integral gövdesi için tüm ifade omurgasını (expr/term/mulchain/factor/power/
atom) kopyalamaktı; bakım maliyeti kabul edilmedi.

Uygulanan çözüm: `dx` her yerde `d · x` çarpımıdır; integral gövdesi kurulurken
`transformer._split_differential` **düz `Mul` çarpanları arasında** `Sym("d")` + değişken
ikilisini arar ve ifadeyi orada böler. Bu, sözleşme davranışını (`test_parse_indefinite_integral_dx_detected`)
aynen korur ve iki ek doğru okuma sağlar:

| Girdi | Sonuç |
|---|---|
| `\int x^{2} + 1 \, dx` | `Integral(x^2 + 1, x)` (gövde diferansiyele kadar açgözlü) |
| `\int x \, dx + 1` | `Integral(x, x) + 1` (diferansiyel ilk terimde) |
| `\int x \, dx \cdot 2` | `Integral(x, x) · 2` |
| `\int x^{2}` | `SemanticError(reason="integral_without_differential")` |

`\int \frac{dx}{x}` biçimi (diferansiyel kesir payında) v1 kapsamı dışındadır ve
`SemanticError` verir. Bağlamsal lexer yine de kullanılıyor (`lexer="contextual"`); tek yönlü
limit için `DIRECTION` terminali ondan yararlanıyor.

### 9.2 Bilinçli çözülen iki kaydır/indirge çakışması

Dilbilgisi LALR(1)'dir; Lark iki çakışma ailesini varsayılan biçimde **kaydırma** lehine
çözer. Her ikisi de istenen okumadır ve testle sabitlenmiştir:

1. `<power : postfix>` ile `^ _ ! '` — sonek işleci en içteki `postfix`'e bağlanır:
   `\sin x^{2}` = `sin(x²)`, `\sin x_1` = `sin(x_1)`.
2. `<bigop : _INT expr>` ile `+ - \pm` — integral gövdesi diferansiyeline kadar açgözlüdür
   (§9.1). Sum/Product/Limit/`\frac{d}{dx}` gövdesi ise `term` düzeyindedir, yani
   `\sum_{k=1}^{n} k + 1` = `(Σk) + 1`.

Büyük işleçler çarpanla çarpılamaz; yalnızca **önlerine** katsayı gelebilir
(`2\int x\,dx` çalışır, `\int x\,dx \cdot 2` diferansiyelden sonraki çarpan olarak okunur).
Diğer durumlar için parantez gerekir: `\left(\sum_{k=1}^{n} k\right)^{2}`.

### 9.3 `to_sympy` aritmetiği değerlendirir; `evaluate=False` seçenektir (sapma)

§3.5 `\frac`'ın sembolik durumda `evaluate=False` ile korunmasını öngörüyordu. Varsayılan
olarak **değerlendirilmiş** SymPy nesnesi üretiyoruz: `simplify`, `srepr`, `Eq` ve
`MathEngine`'in beklediği kanonik biçimler ancak böyle doğru çalışıyor. Yapısal biçim gerektiğinde
`to_sympy(node, evaluate=False)` tüm kurulumu `sympy.evaluate(False)` içinde yapar
(`\frac{2}{4}` → `2 · 4^{-1}`). Kalkülüs düğümleri (`Integral`, `Derivative`, `Limit`, `Sum`,
`Product`) her iki durumda da **değerlendirilmemiş** kalır — `doit()` `mathengine`'in işidir.

`x \in \mathbb{R}` de bilerek değerlendirilmez (`Contains(..., evaluate=False)`): aksi hâlde
reel tanımlı bir `x` için ifade `True`'ya çöker ve ekrana geri yazılamazdı.

### 9.4 Hata kodları ve Türkçe metin

`ErrorCode` içinde anlamsal hata için ayrı bir kod yok; `SemanticError` `PARSE_ERROR` (422)
kullanır ve makine okunur nedeni `details["reason"]` alanında taşır
(`integral_without_differential`, `nested_plus_minus`, `matrix_ragged_rows`, …).
`ParseError` Türkçe metni `messages_tr.MSG_PARSE_ERROR_AT` + `MSG_PARSE_ERROR` şablonundan
üretilir; §4'teki "…{expected} bekleniyor" şablonu `aicls.core.messages_tr` içinde **yok**.
Beklenen terminal listesi kaybolmaz, `details["expected"]` alanında döner. Çekirdeğe
`MSG_PARSE_ERROR_EXPECTED` eklenirse mesaj tek satır değişiklikle zenginleşir (rapor:
*config_changes_needed*).

### 9.5 Dilbilgisine eklenenler

- **`\frac{dy}{dx}`** (`LEIBNIZ_OF` terminali) → `Derivative(y, x)`; kapalı türev ve bağlı hız
  problemlerinde yaygın. Terminal §9.9'da genişletildi: her iki yazım (`d` / `\partial`) ve
  mertebe (`\frac{d^{2}y}{dx^{2}}`) artık kısayol biçiminde de tanınır.
- **Fonksiyon uygulaması** yalnızca `f g h F G H` harfleri için tanınır (`f(x)`, `f''(x)`,
  `f_1` değil). `x(x+1)` örtük çarpımdır. Harf kümesi `to_latex.FUNCTION_LETTERS`.
- **`\operatorname{ad}(x)`** → `Call("ad", …)` → `Function("ad")(…)`.
- **`\mathbb{R|N|Z|Q|C}`** için `MathSet` düğümü eklendi (§3.4 listesinde yoktu); `\in` ile
  birlikte `Contains` üretir.
- **`e` ve `i`** ayrılmış *sabit* düğümü olarak değil, `Sym` olarak ayrıştırılır; sabite
  dönüşüm `to_sympy` içinde `SymbolTable.reserved` ve `ParserConfig.imaginary_i` ile yapılır.
  Böylece `SymbolTable.declare("e")` gerçekten sembol üretir.
- **`\pm`** → `FiniteSet(a+b, a-b)`; yalnızca ifadenin kökünde veya bir ilişkinin bir yanında
  serbesttir, iç içe kullanımda `SemanticError(reason="nested_plus_minus")`.
- **`pmatrix` v1'de açık**: `matrix/pmatrix/bmatrix/vmatrix/Vmatrix/Bmatrix` ayrıştırılır
  (`sympy.latex()` matrisleri `\begin{matrix}` ile yazdığı için gidiş-dönüş bunu gerektirdi).
- **`\{x \mid …\}` (SetBuilder) v1'de yok**: `\mid`, `\cup`, `\cap`, `\subset` … adlarıyla
  `UnsupportedConstruct` verir.
- **Büyük operatör `\cdot` / `\times` sağında** da katsayı alabilir (`2 \cdot \int x \, dx`);
  gerekçe ve tarihçe §9.9 madde 2.

### 9.6 Normalizasyona eklenen onarımlar

§3.1'deki listeye ek olarak (hepsi `NormalizeResult.repairs` içinde görünür):

| Onarım | Ne yapar | Neden |
|---|---|---|
| `function_power_moved` | `\ln{x}^{2}` → `\ln^{2}{x}` | `sympy.latex` üslü fonksiyonu böyle yazar; düzeltilmezse üs **sessizce** argümana girerdi |
| `abs_bars` | dengeli `\|…\|` → `\lvert … \rvert` | çubuklar LALR'de ayrıştırılamaz |
| `decimal_comma` | `3,5` → `3.5`, `[0,1]` korunur | Türkçe ondalık ayraç ile aralık virgülü ayrımı: yalnızca *iki sayıdan ibaret* parantez içi virgül ayraç sayılır |
| `text_dropped`, `mathrm_unwrapped` | `\text{...}` atılır, `\mathrm{d}x` → `dx` | OCR/kitap kopyası |
| `empty_group`, `math_delimiters` | `{}` ve `$…$`/`\[…\]` temizlenir | |

Ayrıca `f{\left(x \right)}` → `f(x)` sadeleştirmesi yapılır (yine `function_power_moved`
etiketiyle): `sympy.latex` tanımsız fonksiyon uygulamasını süslü parantezle yazar, düzeltilmezse
örtük çarpıma dönüşürdü. `f{x}` (parantezsiz) **değiştirilmez**, çünkü TeX'te çarpımdır.
`normalize_latex` sabit noktaya kadar (en çok 6 geçiş) yinelenir; böylece idempotanttır
(`test_property_normalize_latex_idempotent`).

### 9.7 Ölçüm sonucu ve dürüstlük notu

`datasets/latex/valid_inputs.jsonl` **3709 satır** (müfredat 2315, `sympy_latex` 272,
`ocr_output` 972, `mathwriting_label` 150) — sıkı tanımla (ayrıştırma + `to_sympy` +
gidiş-dönüş) başarı oranı **%100.0** (2026-09-07). Hedef > %99 sağlanıyor, ancak korpus dilbilgisiyle
birlikte geliştirildi: bugünkü işlevi **regresyon kapısı** olmaktır, bağımsız bir genelleme
kanıtı değil. Bağımsız ölçüm, veri kapısı açıldığında eklenecek gerçek MathWriting etiketleri
ve OCR çıktılarıyla yapılacaktır (`datasets/latex/README.md` §1).

`datasets/latex/unsupported.jsonl` 106 satır: 70 adlandırılmış kapsam-dışı yapı, 10 bilinmeyen
makro, 20 bozuk girdi, 6 anlamsal hata. Test her satırda **tam olarak** beklenen hata sınıfını
ve `construct` adını doğrular; son ölçümde 106/106 satır beyan edildiği gibi reddedildi ve
hiçbiri sessizce ayrışmadı (2026-09-07).

### 9.8 Küçük uygulama kararları

- Lark dilbilgisi disk önbelleğine yazılmıyor (§2'deki `AICLS_DATA_ROOT/cache/lark`): LALR
  tablosu süreç başına ~0.2 s'de kuruluyor ve `lru_cache` ile bir kez kuruluyor; disk önbelleği
  OneDrive'da gereksiz yazma üretirdi. Gerekirse tek satırla açılır.
- Özellik testi stratejisi (`st_expr`) `tests/property/strategies.py` yerine
  `tests/property/test_latex2sympy_roundtrip.py` içindedir (paylaşılan dosya başka modüllerin
  de sahibi olduğu için).
- `sympy.parsing.latex` farksal testi `antlr4` kurulu olmadığından **atlanıyor** (skip) —
  kapı politikası gereği kurulum yapılmadı.
- Karmaşık sonuçlar (`zoo`, `nan`, `i`) v1 dilbilgisi dışıdır: `sympy.latex` bunları
  `\tilde{\infty}` / `\text{NaN}` / `i` olarak yazar; ilk ikisi `UnsupportedConstruct`,
  `i` ise (varsayılan `imaginary_i=False` ile) sembol olarak geri okunur. Karmaşık analiz
  gerektiğinde `ParserConfig(imaginary_i=True)` verilmelidir.
- `\sin 2x` = `sin(2)·x`'tir (§3.3'teki "argüman = ilk power" kuralının doğrudan sonucu).
  Müfredat verisi bu yazımı içermez; parantezli `\sin(2x)` önerilir ve kural
  `test_parse_sin_x_y_binds_first_power` ile sabitlenmiştir.

### 9.9 Uygulama notları — onarım dalgası (2026-09-07)

Bu dalga yeni bir yetenek eklemedi; KNOWN_ISSUES "ayrıştırıcı-1" maddesini kapattı ve onu
ararken bulunan iki **sessiz yanlış ayrıştırmayı** giderdi. Genel kamu imzaları (`parse_latex`,
`to_sympy`, `to_latex`, AST düğümleri) değişmedi.

**1. `LEIBNIZ_OF` genişletildi (asıl hata).** Terminal yalnızca `d` yazımını kabul ediyor, hem
`\partial`'ı hem de mertebeyi dışarıda bırakıyordu; `\frac{\partial}{\partial x} f` çalışırken
`\frac{\partial f}{\partial x}` sütun 7'de `ParseError` veriyordu. Yeni terminal `LEIBNIZ` ile
aynı yapıdadır: `\frac{(\partial|d)[^n] <gövde>}{(\partial|d) <değişken>[^n]}`. `transformer`
tarafında `_LEIBNIZ_OF_RE` artık `num` ve mertebe gruplarını da taşır; `leibniz_of` sabit
`partial=False` yerine `_parse_leibniz_of()` sonucunu kullanır. **Mertebe ve kısmi olma bilgisi
paydadan değil paydan okunur** — `LEIBNIZ` ile birebir aynı kural, bu yüzden `\frac{d^{2}y}{dx}`
gibi tutarsız yazımlar da ikinci mertebe kabul edilir (sessizce reddetmek yerine okumayı tercih
ettik; OCR çıktısı paydadaki üssü sık düşürür).

*Yan etki (asıl kazanç):* `\frac{d^{2}y}{dx^{2}}` daha önce **sessizce** `d·y/x²` olarak
ayrışıyordu ve `datasets/latex/valid_inputs.jsonl` içinde bu yanlış okumayı "başarı" sayan
21 satır vardı — gidiş-dönüş kapısı bunu yakalayamaz, çünkü yanlış ama *tutarlı* bir okuma da
gidiş-dönüşü geçer. Şimdi `Derivative(y, x, order=2)` üretiyor.

**2. Büyük operatör açık çarpma işaretinin sağında (yeni dilbilgisi alternatifi).** `?term`
kuralına `mulchain _CDOT bigop` ve `mulchain _TIMES bigop` eklendi (ikisi de `scaled_bigop`).
Öncesinde büyük operatöre yalnızca yan yana yazımla (`2 \int x \, dx`) ulaşılabiliyordu; oysa
`sympy.latex` ölçeklenmiş kalkülüs nesnelerini `mul_symbol="dot"` ile `2 \cdot \int …` diye
yazar. Sonuç: `\int`/`\sum`/`\prod`/`\lim` için `ParseError`, `\frac{d}{dx}` için ise **sessiz
yanlış ayrıştırma** (`2 \cdot \frac{d}{dx} f` → `2f/x`, çünkü bağlamsal lexer `LEIBNIZ` kabul
etmeyen bir durumda `_FRAC`'e düşüyordu). Bu, §5'teki seviye-2 gidiş-dönüş sözleşmesinin
doğrudan ihlaliydi: kullanıcıya gönderdiğimiz CAS çıktısı geri gönderildiğinde başka bir ifade
oluyordu. LALR çakışma sayısı değişmedi (10 kaydır/indirge, §9.2'deki iki aile).

**3. Karma kısmi türev artık adıyla reddediliyor.** `\frac{\partial^{2} f}{\partial x \partial y}`
şekil olarak tanınır ama v1 `Derivative` düğümü tek değişken taşır. Anonim `ParseError` yerine
`parser.MIXED_PARTIAL_CONSTRUCT` adıyla `UnsupportedConstruct` verilir (hem `strict=True` hem
`strict=False`), yol haritası notu `grammar/commands.yaml` içindedir. Yakalama, iki ardışık
`\partial <değişken>` grubunu arayan doğrusal bir düzenli ifadedir; `\frac{\partial f}{\partial x}
+ \frac{\partial f}{\partial y}` ve iç içe operatör yazımı `\frac{\partial}{\partial x}
\frac{\partial}{\partial y} f` (önerilen geçici çözüm) etkilenmez.

**Ölçüm (sağlama, 2026-09-07).** `aicls.ocr.CURRICULUM_LATEX`: **29/32 → 30/32 (%90,6 → %93,8)**;
kalan iki dize §3.2'de sayılan `\forall` ve `\binom`'dur ve ikisi de adlandırılmış
`UnsupportedConstruct` verir. `valid_inputs.jsonl`: 3599 → 3709 satır, sıkı tanımla %100,0
(yeni `kismi_turev` satırları eklenmeden önceki dilbilgisiyle ölçüldüğünde 3695/3709 = %99,6
olurdu; farkı kapatan madde 2'dir). `unsupported.jsonl`: 103 → 106 satır, 106/106 beyan edilen
hata sınıfıyla reddedildi.
