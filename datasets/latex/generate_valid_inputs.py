"""Regenerate ``valid_inputs.jsonl`` and ``unsupported.jsonl`` deterministically.

Run from the repository root::

    python datasets/latex/generate_valid_inputs.py

The corpus is a **versioned evaluation asset**: it is committed, and the generator exists so
the rows can be audited and extended, not so they are rebuilt on every run. Row schemas are
documented in ``datasets/latex/README.md``.

Sources (``source`` field):

``curriculum``
    Hand-written Temel Analiz templates expanded over small integer parameters.
``sympy_latex``
    ``sympy_to_latex`` output of curriculum-scoped SymPy expressions - these are the strings
    the platform itself sends to the browser, and a user may send them straight back.
``ocr_output``
    Curriculum rows with realistic OCR/hand-writing spellings (``\\dfrac``, ``\\left(``, a
    missing closing brace, unicode operators, a Turkish decimal comma, ``\\mathrm{d}x``).
``mathwriting_label``
    Hand-written strings in the *style* of MathWriting labels (token-spaced braces). They
    carry ``"synthetic": true``: no MathWriting data is downloaded (approval gate 1), the real
    labels are added when the dataset gate opens.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sympy

from aicls.latex2sympy import sympy_to_latex

OUT_DIR = Path(__file__).resolve().parent
VALID_PATH = OUT_DIR / "valid_inputs.jsonl"
UNSUPPORTED_PATH = OUT_DIR / "unsupported.jsonl"

VARIABLES = ("x", "t", "u")
SMALL = (1, 2, 3, 4, 5)
FUNCTIONS = ("sin", "cos", "tan", "ln")

# --------------------------------------------------------------------------------------------
# curriculum templates: (topic, template, parameter grid)
# --------------------------------------------------------------------------------------------
POLYNOMIALS = [
    "{a}{v}^{{2}} + {b}{v} - {c}",
    "{a}{v}^{{3}} - {b}{v}^{{2}} + {c}",
    "({v} + {a})({v} - {b})",
    "({v} + {a})^{{2}}",
    "{a}{v}^{{4}} - {b}",
]
RATIONALS = [
    "\\frac{{{v} + {a}}}{{{v} - {b}}}",
    "\\frac{{{v}^{{2}} - {a}}}{{{v} + {b}}}",
    "\\frac{{{a}}}{{{v}^{{2}} + {b}}}",
    "\\frac{{{a}{v} + {b}}}{{{c}{v} - {a}}}",
]
ROOTS = [
    "\\sqrt{{{v} + {a}}}",
    "\\sqrt{{{v}^{{2}} + {a}}}",
    "\\sqrt[3]{{{v}^{{2}} - {a}}}",
    "{a}\\sqrt{{{v}}}",
]
TRIGONOMETRY = [
    "\\{f}({a}{v})",
    "\\{f}^{{2}}({v})",
    "\\{f}({v}) + \\{f}({a}{v})",
    "{a}\\{f}({v}^{{2}})",
]
EXPONENTIALS = [
    "e^{{{a}{v}}}",
    "\\ln({v} + {a})",
    "\\log_{{10}}({v}^{{2}} + {a})",
    "{a}e^{{-{v}}}",
    "\\exp({a}{v})",
]
LIMITS = [
    "\\lim_{{{v} \\to {a}}} \\frac{{{v}^{{2}} - {a}^{{2}}}}{{{v} - {a}}}",
    "\\lim_{{{v} \\to 0}} \\frac{{\\sin({a}{v})}}{{{v}}}",
    "\\lim_{{{v} \\to \\infty}} \\frac{{{a}{v} + {b}}}{{{v} - {c}}}",
    "\\lim_{{{v} \\to 0^{{+}}}} \\frac{{{a}}}{{{v}}}",
    "\\lim_{{{v} \\to 0^{{-}}}} \\frac{{{a}}}{{{v}^{{2}}}}",
]
DERIVATIVES = [
    "\\frac{{d}}{{d{v}}} ({v}^{{3}} + {a}{v})",
    "\\frac{{d}}{{d{v}}} \\sin({a}{v})",
    "\\frac{{d^{{2}}}}{{d{v}^{{2}}}} ({v}^{{{a}}} - {b})",
    "\\frac{{\\partial}}{{\\partial {v}}} ({v} y + {a})",
    "\\frac{{d}}{{d{v}}} \\frac{{{v}}}{{{v} + {a}}}",
]
PARTIAL_DERIVATIVES = [
    # The Leibniz *shorthand* (body inside the numerator) in both spellings and both orders.
    # ``\frac{\partial f}{\partial x}`` is what aicls.ocr.CURRICULUM_LATEX emits; before the
    # KNOWN_ISSUES parser-1 fix it raised ParseError and ``\frac{d^{2}y}{dx^{2}}`` misparsed
    # silently as ``d y / x^{2}``, so these rows are the regression gate for both.
    "\\frac{{\\partial f}}{{\\partial {v}}}",
    "\\frac{{\\partial f}}{{\\partial {v}}} = {a}{v}",
    "\\frac{{\\partial^{{2}} f}}{{\\partial {v}^{{2}}}}",
    "\\frac{{\\partial y}}{{\\partial {v}}} + {a}y",
    "\\frac{{\\partial \\theta}}{{\\partial {v}}}",
    "\\frac{{d y}}{{d {v}}}",
    "\\frac{{d^{{2}}y}}{{d{v}^{{2}}}} = {a}y",
    "\\frac{{d^{{{a}}}y}}{{d{v}^{{{a}}}}}",
    "\\frac{{d f}}{{d {v}}} \\cdot {a}",
]
INDEFINITE_INTEGRALS = [
    "\\int {v}^{{{a}}} \\, d{v}",
    "\\int ({v}^{{2}} + {a}) \\, d{v}",
    "\\int \\frac{{{a}}}{{{v}}} \\, d{v}",
    "\\int \\sin({a}{v}) \\, d{v}",
    "\\int e^{{{a}{v}}} \\, d{v}",
]
DEFINITE_INTEGRALS = [
    "\\int_{{0}}^{{{a}}} {v}^{{2}} \\, d{v}",
    "\\int_{{{a}}}^{{{b}}} ({v} + {c}) \\, d{v}",
    "\\int_{{0}}^{{\\pi}} \\sin({a}{v}) \\, d{v}",
    "\\int_{{1}}^{{{a}}} \\frac{{1}}{{{v}}} \\, d{v}",
]
SERIES = [
    "\\sum_{{k=1}}^{{n}} k^{{{a}}}",
    "\\sum_{{k=1}}^{{{a}}} \\frac{{1}}{{k}}",
    "\\sum_{{n=1}}^{{\\infty}} \\frac{{1}}{{n^{{{a}}}}}",
    "\\prod_{{k=1}}^{{n}} ({a}k)",
]
EQUATIONS = [
    "{v}^{{2}} - {a} = 0",
    "{a}{v} + {b} = {c}",
    "{v}^{{2}} + {a}{v} + {b} = 0",
    "{a}{v} - {b} \\le {c}",
    "\\lvert {v} - {a} \\rvert < {b}",
    "{v} \\ge {a}",
    "{v} \\in \\mathbb{{R}}",
]
APPLICATIONS = [
    "\\pi r^{{2}} + {a}",
    "\\frac{{1}}{{2}} b h + {a}",
    "{a}\\pi r^{{{b}}}",
    "\\lvert {v}^{{2}} - {a} \\rvert",
    "{v}! + {a}",
    "\\left[-{a}, {b}\\right]",
    "\\left(-{a}, {b}\\right)",
]

CLASSIC = [
    # the derivative definition, the classic limits and the shapes a lecturer writes verbatim
    "\\lim_{{h \\to 0}} \\frac{{f({v}+h) - f({v})}}{{h}}",
    "\\lim_{{{v} \\to {a}}} \\frac{{f({v}) - f({a})}}{{{v} - {a}}}",
    "f({v}) = {a}{v}^{{2}} + {b}{v} - {c}",
    "f'({v}) = {a}{v} + {b}",
    "f''({v}) = {a}",
    "y = {a}{v} + {b}",
    "\\frac{{d}}{{d{v}}}\\left({v}^{{2}} + {a}\\right)",
    "\\frac{{{v}^{{2}}}}{{{a}}} + \\frac{{y^{{2}}}}{{{b}}} = 1",
    "\\sqrt{{{v}^{{2}} + y^{{2}}}} = {a}",
    "\\int_{{0}}^{{1}} \\frac{{1}}{{1 + {v}^{{2}}}} \\, d{v}",
    "\\int \\frac{{{v}}}{{{v}^{{2}} + {a}}} \\, d{v}",
    "\\sum_{{k=1}}^{{n}} \\frac{{1}}{{k(k + {a})}}",
    "a_{{n}} = \\frac{{n}}{{n + {a}}}",
    "\\lvert f({v}) - L \\rvert < \\varepsilon",
    "\\tan^{{-1}}({a}{v})",
    "\\sin^{{2}}({v}) + \\cos^{{2}}({v}) = 1",
    "{a}\\pi r h + {b}",
    "\\frac{{\\sin({v})}}{{\\cos({v})}} = \\tan({v})",
    "\\begin{{pmatrix}} {a} & {b} \\\\ {c} & {a} \\end{{pmatrix}}",
    "\\lim_{{{v} \\to \\infty}} \\left(1 + \\frac{{{a}}}{{{v}}}\\right)^{{{v}}}",
    "e^{{{v}}} = \\sum_{{n=1}}^{{\\infty}} \\frac{{{v}^{{n}}}}{{n!}}",
    "\\frac{{d^{{2}}y}}{{d{v}^{{2}}}} + {a}y = 0",
    "{v}^{{2}} + y^{{2}} \\le {a}",
    "g(f({v})) = {a}",
    "\\ln\\left(\\frac{{{v}}}{{{a}}}\\right)",
    "\\sqrt[{a}]{{{v}^{{{b}}}}}",
    "\\frac{{{a}}}{{{v} - {b}}} + \\frac{{{b}}}{{{v} + {a}}}",
    "\\left(\\frac{{{v}}}{{{a}}}\\right)^{{{b}}}",
    "-\\frac{{{a}}}{{{v}^{{2}}}}",
    "{a}{v} \\cdot {b}{v}",
]

TEMPLATE_GROUPS: list[tuple[str, list[str]]] = [
    ("klasik", CLASSIC),
    ("polinom", POLYNOMIALS),
    ("rasyonel", RATIONALS),
    ("kok", ROOTS),
    ("trigonometri", TRIGONOMETRY),
    ("ustel_logaritma", EXPONENTIALS),
    ("limit", LIMITS),
    ("turev", DERIVATIVES),
    ("kismi_turev", PARTIAL_DERIVATIVES),
    ("belirsiz_integral", INDEFINITE_INTEGRALS),
    ("belirli_integral", DEFINITE_INTEGRALS),
    ("diziler_seriler", SERIES),
    ("denklem_esitsizlik", EQUATIONS),
    ("uygulama", APPLICATIONS),
]


def _curriculum_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for topic, templates in TEMPLATE_GROUPS:
        for template in templates:
            for variable in VARIABLES:
                for a in SMALL:
                    for b in SMALL:
                        if b == a:
                            continue
                        c = (a + b) % 7 + 1
                        latex = template.format(v=variable, a=a, b=b, c=c, f=FUNCTIONS[a % 4])
                        if latex in seen:
                            continue
                        seen.add(latex)
                        rows.append({"latex": latex, "source": "curriculum", "topic": topic})
    return rows


def _sympy_rows() -> list[dict[str, Any]]:
    x, y = sympy.symbols("x y", real=True)
    n = sympy.Symbol("n", integer=True)
    expressions: list[sympy.Basic] = []
    for a in range(1, 10):
        expressions.extend(
            [
                a * x**2 + x - a,
                (x + a) * (x - a),
                sympy.Rational(1, a + 1),
                sympy.Rational(a, a + 2) * x,
                sympy.sqrt(x + a),
                sympy.root(x**2 + a, 3),
                sympy.sin(a * x),
                sympy.cos(x) ** 2 + a,
                sympy.tan(x) * a,
                sympy.asin(x / a),
                sympy.exp(a * x),
                sympy.log(x + a),
                sympy.log(x, 10) + a,
                sympy.Abs(x - a),
                x / (y + a),
                sympy.pi * x**2 / a,
                sympy.Integral(x**a, x),
                sympy.Integral(x**2, (x, 0, a)),
                sympy.Derivative(x**a, x),
                sympy.Derivative(sympy.sin(x), (x, 2)),
                sympy.Limit(sympy.sin(a * x) / x, x, 0),
                sympy.Limit(1 / x, x, 0, "+"),
                sympy.Sum(n**2, (n, 1, a)),
                sympy.Product(n, (n, 1, a)),
                sympy.Eq(x**2 - a, 0),
                sympy.Interval(0, a),
                sympy.Interval.open(-a, a),
                sympy.factorial(n) + a,
                sympy.cosh(a * x),
                x ** sympy.Rational(1, 2) + a,
                -sympy.Rational(a, 4) * x**2,
                sympy.sqrt(a) * x,
            ]
        )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for expression in expressions:
        latex = sympy_to_latex(expression)
        if latex in seen:
            continue
        seen.add(latex)
        rows.append({"latex": latex, "source": "sympy_latex", "topic": "cas_ciktisi"})
    return rows


def _corrupt(latex: str, variant: int) -> str | None:
    """Apply one realistic OCR/hand-writing spelling; None when it does not apply."""
    if variant == 0 and "\\frac" in latex:
        return latex.replace("\\frac", "\\dfrac", 1)
    if variant == 1 and "(" in latex and ")" in latex:
        return latex.replace("(", "\\left(", 1).replace(")", "\\right)", 1)
    if variant == 2 and latex.endswith("}"):
        return latex[:-1]
    if variant == 3 and "\\cdot" not in latex:
        return "\\displaystyle " + latex
    if variant == 4 and " \\, d" in latex:
        return latex.replace(" \\, d", " \\mathrm{d}", 1)
    if variant == 5 and "\\lvert" in latex:
        return latex.replace("\\lvert", "|").replace("\\rvert", "|")
    if variant == 6 and "+ 1" in latex:
        return latex.replace("+ 1", "+ 1,5", 1)
    if variant == 7 and "\\times" not in latex and "2" in latex and "d^{" not in latex:
        # ``d^{`` is excluded on purpose: in ``\frac{d^{2}y}{dx^{2}}`` the corruption would break
        # the Leibniz token, the row would silently become an ordinary fraction and the corpus
        # would then assert a *misparse* as a success.
        return latex.replace("2", "2 \\times 1", 1)
    if variant == 8:
        return latex.replace(" ", "  ") + " "
    return None


def _ocr_rows(curriculum: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(curriculum):
        variant = index % 9
        corrupted = _corrupt(row["latex"], variant)
        if corrupted is None or corrupted in seen:
            continue
        seen.add(corrupted)
        rows.append(
            {
                "latex": corrupted,
                "source": "ocr_output",
                "topic": row["topic"],
                "variant": variant,
            }
        )
    return rows


MATHWRITING_STYLE = [
    "\\frac { 1 } { 2 }",
    "\\frac { x + 1 } { x - 1 }",
    "x ^ { 2 } + y ^ { 2 }",
    "\\sqrt { x + 1 }",
    "\\alpha + \\beta",
    "\\sum _ { k = 1 } ^ { n } k",
    "\\int _ { 0 } ^ { 1 } x ^ { 2 } d x",
    "\\lim _ { x \\to 0 } \\frac { \\sin x } { x }",
    "\\frac { d } { d x } x ^ { 2 }",
    "\\theta + \\varphi",
    "a _ { n + 1 } = a _ { n } + 1",
    "\\lvert x \\rvert \\le 1",
    "2 \\pi r",
    "e ^ { i } + 1",
    "\\cos ^ { 2 } x + \\sin ^ { 2 } x = 1",
]


def _mathwriting_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for base in MATHWRITING_STYLE:
        for a in range(1, 11):
            latex = base.replace("1", str(a)) if "1" in base else f"{base} + {a}"
            if latex in seen:
                continue
            seen.add(latex)
            rows.append(
                {
                    "latex": latex,
                    "source": "mathwriting_label",
                    "topic": "el_yazisi_etiketi",
                    "synthetic": True,
                }
            )
    return rows


# --------------------------------------------------------------------------------------------
# out-of-scope corpus
# --------------------------------------------------------------------------------------------
UNSUPPORTED_CONSTRUCT_SAMPLES: list[tuple[str, str]] = [
    ("\\begin{cases} x & x > 0 \\\\ -x & x \\le 0 \\end{cases}", "\\begin{cases}"),
    ("\\begin{align} x &= 1 \\end{align}", "\\begin{align}"),
    ("\\begin{aligned} y &= 2x \\end{aligned}", "\\begin{aligned}"),
    ("\\begin{array}{cc} 1 & 2 \\end{array}", "\\begin{array}"),
    ("\\vec{v} + \\vec{w}", "\\vec"),
    ("\\hat{x} \\cdot \\hat{y}", "\\hat"),
    ("\\widehat{ABC}", "\\widehat"),
    ("\\bar{z} + 1", "\\bar"),
    ("\\overline{z} \\cdot z", "\\overline"),
    ("\\underline{x}", "\\underline"),
    ("\\tilde{f}(x)", "\\tilde"),
    ("\\widetilde{g}(x)", "\\widetilde"),
    ("\\dot{x} + \\dot{y}", "\\dot"),
    ("\\ddot{x}", "\\ddot"),
    ("\\overrightarrow{AB}", "\\overrightarrow"),
    ("\\mathbf{v}", "\\mathbf"),
    ("\\boldsymbol{\\alpha}", "\\boldsymbol"),
    ("\\mathcal{L}(f)", "\\mathcal"),
    ("\\mathfrak{g}", "\\mathfrak"),
    ("\\mathsf{A}", "\\mathsf"),
    ("\\mathtt{B}", "\\mathtt"),
    ("\\label{eq:1} x = 1", "\\label"),
    ("x = 1 \\tag{1}", "\\tag"),
    ("\\ref{eq:1}", "\\ref"),
    ("\\iint f(x, y) \\, dx", "\\iint"),
    ("\\iiint f \\, dx", "\\iiint"),
    ("\\oint_{C} f \\, dx", "\\oint"),
    ("\\binom{n}{k}", "\\binom"),
    ("\\dbinom{n}{2}", "\\dbinom"),
    ("A \\cup B", "\\cup"),
    ("A \\cap B", "\\cap"),
    ("A \\setminus B", "\\setminus"),
    ("A \\subset B", "\\subset"),
    ("A \\subseteq B", "\\subseteq"),
    ("A \\supset B", "\\supset"),
    ("A \\supseteq B", "\\supseteq"),
    ("x \\notin A", "\\notin"),
    ("A = \\emptyset", "\\emptyset"),
    ("A = \\varnothing", "\\varnothing"),
    ("\\{x \\mid x > 0\\}", "\\mid"),
    ("\\forall x > 0", "\\forall"),
    ("\\exists x > 0", "\\exists"),
    ("\\nabla f", "\\nabla"),
    ("a \\equiv b", "\\equiv"),
    ("y \\propto x", "\\propto"),
    ("f \\sim g", "\\sim"),
    ("A \\cong B", "\\cong"),
    ("a \\perp b", "\\perp"),
    ("a \\parallel b", "\\parallel"),
    ("\\angle ABC", "\\angle"),
    ("90 \\degree", "\\degree"),
    ("f \\circ g", "\\circ"),
    ("1, 2, \\ldots, n", "\\ldots"),
    ("1 + 2 + \\cdots + n", "\\cdots"),
    ("1, 2, \\dots", "\\dots"),
    ("\\limsup_{n} a_n", "\\limsup"),
    ("\\liminf_{n} a_n", "\\liminf"),
    ("\\arg(z)", "\\arg"),
    ("\\deg(p)", "\\deg"),
    ("\\det(A)", "\\det"),
    ("\\dim(V)", "\\dim"),
    ("\\gcd(a, b)", "\\gcd"),
    ("\\max(a, b)", "\\max"),
    ("\\min(a, b)", "\\min"),
    ("\\sup(A)", "\\sup"),
    ("\\inf(A)", "\\inf"),
    ("\\substack{i < j}", "\\substack"),
    # Mixed partials: recognised as a shape, refused by name (parser.MIXED_PARTIAL_CONSTRUCT).
    (
        "\\frac{\\partial^{2} f}{\\partial x \\partial y}",
        "\\frac{\\partial^{2}}{\\partial x \\partial y}",
    ),
    (
        "\\frac{\\partial^{2} u}{\\partial x \\partial t}",
        "\\frac{\\partial^{2}}{\\partial x \\partial y}",
    ),
    (
        "\\frac{\\partial^{3} f}{\\partial x^{2} \\partial y}",
        "\\frac{\\partial^{2}}{\\partial x \\partial y}",
    ),
]

UNKNOWN_MACROS = [
    "\\foo{x}",
    "\\bazqux",
    "\\Coloneqq",
    "\\varoiint",
    "\\bigstar",
    "\\clubsuit",
    "\\heartsuit",
    "\\lightning",
    "\\smiley",
    "\\notacommand{1}",
]

MALFORMED = [
    "\\frac{1}{2}}}",
    "{{{x",
    "}",
    "x @ y",
    "x^",
    "\\frac{}{}",
    "\\sqrt[]{}",
    "\\lim_{}",
    "\\sum_{}^{}",
    "\\int_",
    "f(",
    "2 +",
    "* 3",
    "\\frac",
    "\\left(",
    "x_",
    "\\log_",
    "\\begin{pmatrix} 1 & 2 \\end{pmatrix} \\end{pmatrix}",
    "===",
    "((()))",
]

SEMANTIC = [
    "\\int x^{2}",
    "\\int_{0}^{1} x^{2}",
    "2_{3}",
    "1_{x}",
    "\\int (x + 1)",
    "\\frac{1 \\pm 2}{3}",
]


def _unsupported_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for latex, construct in UNSUPPORTED_CONSTRUCT_SAMPLES:
        rows.append({"latex": latex, "expect_error": "unsupported", "construct": construct})
    for latex in UNKNOWN_MACROS:
        rows.append(
            {"latex": latex, "expect_error": "unsupported", "construct": latex.split("{")[0]}
        )
    for latex in MALFORMED:
        rows.append({"latex": latex, "expect_error": "parse_error", "construct": ""})
    for latex in SEMANTIC:
        rows.append({"latex": latex, "expect_error": "semantic", "construct": ""})
    return rows


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build() -> tuple[int, int]:
    """Write both corpora and return their row counts."""
    curriculum = _curriculum_rows()
    rows = curriculum + _sympy_rows() + _ocr_rows(curriculum) + _mathwriting_rows()
    numbered: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    for row in rows:
        source = str(row["source"])
        counters[source] = counters.get(source, 0) + 1
        numbered.append({"id": f"{source}-{counters[source]:04d}", "expect_ok": True, **row})
    _write(VALID_PATH, numbered)

    unsupported = _unsupported_rows()
    numbered_unsupported = [
        {"id": f"unsupported-{index + 1:04d}", **row} for index, row in enumerate(unsupported)
    ]
    _write(UNSUPPORTED_PATH, numbered_unsupported)
    return len(numbered), len(numbered_unsupported)


if __name__ == "__main__":
    valid_count, unsupported_count = build()
    print(f"valid_inputs.jsonl: {valid_count} rows")
    print(f"unsupported.jsonl: {unsupported_count} rows")
