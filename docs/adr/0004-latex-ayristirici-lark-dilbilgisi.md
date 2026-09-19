# ADR-0004: LaTeX → SymPy için kendi Lark LALR dilbilgisi ve tipli AST

- **Durum:** kabul edildi
- **Tarih:** 2026-09-07
- **Karar vericiler:** M. M. Turhan, Ç. Şahin
- **İlgili İP:** İP-4

## Bağlam ve problem

İP-4, OCR ve ASR'dan gelen LaTeX'i "%99'undan fazlasını hatasız olarak SymPy formatına dönüştürebilmek" ölçütü ve "ayrıştırma algoritmalarını ve eşleştirme kurallarını detaylandıran teknik rapor" çıktısı ile tanımlar. `sympy.parsing.latex.parse_latex` ANTLR çalışma zamanı ister, örtük çarpım/`dx` ayrımında sessiz hata yapabilir ve sınıf notasyonuna (`\frac{d}{dx}`, tek yönlü limit, `f'(x)`) genişletilemez. Ekip kararı (memory: bekleyen kararlar #3) kendi dilbilgisidir.

## Değerlendirilen seçenekler

1. **`sympy.parsing.latex`** — hazır; ANTLR bağımlılığı, hoşgörülü/sessiz, genişletilemez, konumlu hata yok.
2. **`latex2sympy2`** — benzer sınırlar, bakım belirsiz.
3. **Kendi Lark dilbilgisi (LALR, bağlamsal lexer) → tipli AST → SymPy** — saf Python, konumlu hata, NLU ile ortak AST, ölçülebilir kapsam; bakım bize ait.
4. **Regex yeniden yazım** — ölçeklenmez.

## Karar

Seçenek 3. `src/aicls/latex2sympy/grammar/latex_math.lark`; AST `ast.py` (span'lı dondurulmuş dataclass'lar); `to_sympy` `SymbolTable` ile (x,y,z,t reel; toplam bağlı değişkenleri tam sayı); `\log` tabanı `ParserConfig.log_base="e"` (sınıf kullanımında `\log` ≡ `\ln`; `\log_{10}` açık yazılır); belirsizlikler dilbilgisi içinde çözülür (örtük çarpım yalnızca operatörle başlamayan atomlar; `\sin x y` → `sin(x)·y`; `dx` yalnızca integral bağlamında); dilbilgisi dışı komutlar adıyla `UnsupportedConstruct`; kısmi sonuç yok; `normalize_latex` yalnızca tek eksik parantezi onarır ve onarımı izler. `sympy.parsing.latex` yalnızca farksal test kâhini. Üç seviyeli gidiş-dönüş doğrulama (AST, sembolik, farksal) ve `sympy.latex()` çıktısının dilbilgisi içinde kalması testi.

## Sonuçlar

- Olumlu: >%99 hedefi ölçülebilir ve sıkı (sessiz yanlış ayrıştırma başarısızlık sayılır); İP-4 teknik raporu dilbilgisi + `commands.yaml` tablosundan üretilir.
- Olumsuz: dilbilgisi bakımı bizde; OCR çıktısı çeşitliliği kapsam dışına düşerse başarı düşer → `valid_inputs.jsonl` OCR çıktılarıyla büyür, haftalık başarısızlık triyajı.

## Öneri metniyle uyum

İP-4 görevleri: "LaTeX'teki semboller, operatörler ve fonksiyonlar (\int, \frac, \sin vb.), SymPy'da karşılık gelen fonksiyonlara (integrate(), Rational(), sin() vb.) eşleştirilecektir." A.2.2 eşleme örnekleri (`\frac{d}{dx}x^2` → `diff(x**2, x)`, `x^2-4=0` → `solve(Eq(...))`, `latex(...)` → `\frac{1}{3}`) test adlarıyla birebir doğrulanır. Öneri metni belirli bir ayrıştırıcı kütüphane adlandırmaz.

## Doğrulama

`test_property_roundtrip_ast_to_latex_to_ast`, `test_sympy_latex_output_is_within_grammar_on_curriculum`, `test_differential_against_sympy_parse_latex_subset`, `test_curriculum_and_valid_inputs_parse_success_rate_above_0_99`, `test_sympy_one_third_prints_frac_1_3`.

## İlgili

[mimari/latex-ayristirici.md](../mimari/latex-ayristirici.md), ADR-0005, ADR-0015.
