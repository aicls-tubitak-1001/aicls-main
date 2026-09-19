"""Author the Temel Analiz curriculum cards under ``datasets/curriculum/cards/``.

Run from the repository root with the project venv::

    python datasets/curriculum/generate_cards.py

**What this file is.** The İP-4 evaluation corpus is written here, not measured here. Every
family below is a *curriculum template*: a syllabus rule (power rule, chain rule, partial
fractions, the ratio test, the Maclaurin series of ``e^{ax}``) together with the closed-form
answer that rule produces, written out by the template author. The generator expands each
template over a small integer parameter grid and emits one YAML card per problem.

**Why the answer is authored and not computed.** ``scripts/build_curriculum_testset.py`` runs
every card through :class:`aicls.mathengine.MathEngine` and compares the engine's answer with
the answer written here. That comparison is only worth something if the two sides are
independent: if the card's ``expected`` were filled in from the engine, "CAS accuracy" would be
100 % by construction and would measure nothing. So this file never calls the engine's
operations (``sympy.diff``, ``sympy.integrate``, ``sympy.limit``, ``sympy.solveset``,
``sympy.series``); it only *builds* the expression the textbook rule predicts, out of SymPy
constructors. ``python -m pytest tests/unit/datasets`` re-checks that rule.

**Provenance, stated plainly.** These cards were authored inside the project (machine-expanded
from hand-written templates), not sampled from an independent problem corpus and not copied
from a textbook. See ``datasets/curriculum/README.md``; the 2027-05-01 progress report must
repeat that sentence next to any number measured on this set.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

import sympy
import yaml  # type: ignore[import-untyped]  # PyYAML ships no stubs and types-PyYAML is absent

from aicls.latex2sympy import sympy_to_latex

CURRICULUM_DIR = Path(__file__).resolve().parent
CARDS_DIR = CURRICULUM_DIR / "cards"

#: Who wrote the cards. Deliberately not a person: the rows are machine-expanded templates and
#: no human has reviewed them yet (``reviewed_by`` stays empty until they do).
AUTHOR = "generator:datasets/curriculum/generate_cards.py"

#: Nobody has reviewed the rows yet; the field exists so that the gap is visible in the data.
REVIEWED_BY = ""

#: ``frozen`` here means "byte-frozen" (``FROZEN.sha256``), not "editorially reviewed".
STATUS = "frozen"

#: Chapter each topic belongs to, in the numbering of ``datasets/book/sample``.
CHAPTER_BY_TOPIC: dict[str, str] = {
    "limit": "aicls:bolum-1",
    "sureklilik": "aicls:bolum-1",
    "turev": "aicls:bolum-2",
    "turev_uygulamalari": "aicls:bolum-3",
    "belirsiz_integral": "aicls:bolum-4",
    "belirli_integral": "aicls:bolum-4",
    "integral_uygulamalari": "aicls:bolum-5",
    "diziler_seriler": "aicls:bolum-6",
    "denklem_esitsizlik": "aicls:bolum-0",
    "grafik": "aicls:bolum-0",
}

#: Every 17th card is re-phrased as a Turkish sentence (``input.kind == "text_tr"``), so the
#: set also carries NLU material. The gold NLU set itself is ``datasets/nlu/``.
TEXT_TR_EVERY = 17

#: Turkish sentence templates, by command ``op`` (only used for ``text_tr`` rows).
SENTENCE_TR: dict[str, str] = {
    "diff": "{tex} ifadesinin {var} değişkenine göre türevini al",
    "integrate": "{tex} integralini hesapla",
    "limit": "{tex} limitini hesapla",
    "solve": "{tex} denklemini çöz",
    "simplify": "{tex} ifadesini sadeleştir",
    "expand": "{tex} ifadesinin açılımını yaz",
    "factor": "{tex} ifadesini çarpanlarına ayır",
    "series": "{tex} ifadesinin Maclaurin serisini yaz",
    "evaluate": "{tex} ifadesinin değerini hesapla",
    "plot": "{tex} fonksiyonunun grafiğini çiz",
}

X = sympy.Symbol("x", real=True)
Y = sympy.Symbol("y", real=True)
T = sympy.Symbol("t", real=True)
N = sympy.Symbol("n", real=True)
H = sympy.Symbol("h", real=True)

#: LaTeX fragments kept as constants: an f-string expression may not contain a backslash.
ABS_X = "\\lvert x \\rvert"
SIN_SQ = "\\sin^{2}(x)"
COS_SQ = "\\cos^{2}(x)"


@dataclass(frozen=True, slots=True)
class Draft:
    """One authored problem before it is given an id and rendered to YAML.

    Attributes:
        topic: Curriculum topic (quota key).
        subtopic: Template family inside the topic.
        difficulty: 1 (drill), 2 (standard), 3 (multi-step).
        input_value: What the student writes, in LaTeX.
        command: The typed CAS command, as a plain mapping.
        answer: The answer the curriculum rule predicts, as a SymPy object.
        checker: Comparison strategy the compiler must use.
        checker_args: Extra arguments for the checker.
        template: Template identifier, recorded in ``source``.
        notes_tr: Turkish note naming the rule the answer comes from.
    """

    topic: str
    subtopic: str
    difficulty: int
    input_value: str
    command: dict[str, Any]
    answer: Any
    checker: str = "symbolic"
    checker_args: dict[str, Any] = field(default_factory=dict)
    template: str = ""
    notes_tr: str = ""


# --------------------------------------------------------------------------------------------
# small LaTeX helpers (inputs are written the way a lecturer writes them, not printed by SymPy)
# --------------------------------------------------------------------------------------------
def co(value: int, symbol: str = "") -> str:
    """Render an integer coefficient in front of ``symbol`` (``1x`` -> ``x``, ``-1x`` -> ``-x``)."""
    if not symbol:
        return str(value)
    if value == 1:
        return symbol
    if value == -1:
        return f"-{symbol}"
    return f"{value}{symbol}"


def signed(value: int, symbol: str = "") -> str:
    """Render ``+ 3x`` / ``- 3x`` for a term that follows another term."""
    if value >= 0:
        return f"+ {co(value, symbol)}"
    return f"- {co(-value, symbol)}"


def pw(var: str, power: int) -> str:
    """Render ``x^{3}``; ``x^{1}`` collapses to ``x`` and ``x^{0}`` to ``1``."""
    if power == 0:
        return "1"
    if power == 1:
        return var
    return f"{var}^{{{power}}}"


def indefinite(integrand: str, var: str = "x") -> str:
    """Render ``\\int <integrand> \\, dx``."""
    return f"\\int {integrand} \\, d{var}"


def definite(integrand: str, lower: str, upper: str, var: str = "x") -> str:
    """Render ``\\int_{a}^{b} <integrand> \\, dx``."""
    return f"\\int_{{{lower}}}^{{{upper}}} {integrand} \\, d{var}"


def leibniz(expr: str, var: str = "x", order: int = 1) -> str:
    """Render ``\\frac{d}{dx} (expr)`` or its higher-order form."""
    head = (
        f"\\frac{{d}}{{d{var}}}" if order == 1 else f"\\frac{{d^{{{order}}}}}{{d{var}^{{{order}}}}}"
    )
    return f"{head} \\left({expr}\\right)"


def partial(expr: str, var: str) -> str:
    """Render ``\\frac{\\partial}{\\partial x} (expr)``."""
    return f"\\frac{{\\partial}}{{\\partial {var}}} \\left({expr}\\right)"


def lim(expr: str, var: str, to: str, direction: str = "") -> str:
    """Render ``\\lim_{x \\to a} expr`` with an optional one-sided arrow."""
    target = to if not direction else f"{to}^{{{direction}}}"
    return f"\\lim_{{{var} \\to {target}}} {expr}"


def frac(numerator: str, denominator: str) -> str:
    """Render ``\\frac{a}{b}``."""
    return f"\\frac{{{numerator}}}{{{denominator}}}"


# --------------------------------------------------------------------------------------------
# 1. limit (quota 120)
# --------------------------------------------------------------------------------------------
def limit_drafts() -> list[Draft]:
    """Nine limit families: substitution, indeterminate forms, one-sided and infinite limits."""
    out: list[Draft] = []

    # L1 -- direct substitution into a polynomial.
    for a in range(1, 5):
        for b, c, d in ((1, 2, -3), (2, -1, 4), (3, 0, -2), (1, -4, 5)):
            expr = f"{co(b, 'x^{2}')} {signed(c, 'x')} {signed(d)}"
            out.append(
                Draft(
                    "limit",
                    "polinom_limiti",
                    1,
                    lim(f"\\left({expr}\\right)", "x", str(a)),
                    {"op": "limit", "expr": expr, "var": "x", "to": str(a)},
                    sympy.Integer(b * a * a + c * a + d),
                    template="limit/polinom",
                    notes_tr=(
                        "Polinomlar her noktada süreklidir; limit doğrudan yerine koymayla bulunur."
                    ),
                )
            )

    # L2 -- (x^n - a^n)/(x - a): the difference-of-powers factorisation.
    for n in range(2, 6):
        for a in range(1, 5):
            expr = frac(f"{pw('x', n)} - {a**n}", f"x - {a}")
            out.append(
                Draft(
                    "limit",
                    "belirsizlik_0_0",
                    2,
                    lim(expr, "x", str(a)),
                    {"op": "limit", "expr": expr, "var": "x", "to": str(a)},
                    sympy.Integer(n * a ** (n - 1)),
                    template="limit/fark_carpanlari",
                    notes_tr=(
                        "0/0 belirsizliği: x^n - a^n = (x - a)(x^{n-1} + ... + a^{n-1}) "
                        "çarpanlarına ayrılır."
                    ),
                )
            )

    # L3 -- rational functions at infinity: the ratio of leading coefficients.
    for a, c in ((1, 2), (3, 4), (5, 2), (2, 7), (9, 3), (4, 5), (7, 2), (6, 4)):
        expr = frac(f"{co(a, 'x')} + 1", f"{co(c, 'x')} - 3")
        out.append(
            Draft(
                "limit",
                "sonsuzda_limit",
                2,
                lim(expr, "x", "\\infty"),
                {"op": "limit", "expr": expr, "var": "x", "to": "\\infty"},
                sympy.Rational(a, c),
                template="limit/sonsuzda_birinci_derece",
                notes_tr=(
                    "Sonsuzda limit: pay ve payda aynı dereceden ise limit baş katsayıların "
                    "oranıdır."
                ),
            )
        )
    for a, c in ((1, 2), (3, 4), (5, 2), (2, 7), (9, 3), (4, 5), (7, 2), (6, 4)):
        expr = frac(f"{co(a, 'x^{2}')} - 1", f"{co(c, 'x^{2}')} + 5")
        out.append(
            Draft(
                "limit",
                "sonsuzda_limit",
                2,
                lim(expr, "x", "\\infty"),
                {"op": "limit", "expr": expr, "var": "x", "to": "\\infty"},
                sympy.Rational(a, c),
                template="limit/sonsuzda_ikinci_derece",
                notes_tr=(
                    "Sonsuzda limit: pay ve payda aynı dereceden ise limit baş katsayıların "
                    "oranıdır."
                ),
            )
        )

    # L4 -- the trigonometric standard limits.
    for a, b in ((1, 1), (2, 1), (3, 2), (5, 3), (4, 7), (7, 5)):
        expr = frac(f"\\sin({co(a, 'x')})", co(b, "x"))
        out.append(
            Draft(
                "limit",
                "trigonometrik_limit",
                2,
                lim(expr, "x", "0"),
                {"op": "limit", "expr": expr, "var": "x", "to": "0"},
                sympy.Rational(a, b),
                template="limit/sin_ax_bx",
                notes_tr="Temel trigonometrik limit: lim_{x->0} sin(ax)/(bx) = a/b.",
            )
        )
    for a, b in ((1, 1), (2, 3), (5, 2), (3, 4), (6, 5)):
        expr = frac(f"\\tan({co(a, 'x')})", co(b, "x"))
        out.append(
            Draft(
                "limit",
                "trigonometrik_limit",
                2,
                lim(expr, "x", "0"),
                {"op": "limit", "expr": expr, "var": "x", "to": "0"},
                sympy.Rational(a, b),
                template="limit/tan_ax_bx",
                notes_tr=(
                    "tan(ax)/(bx) = [sin(ax)/(bx)] * [1/cos(ax)] ayrışmasından limit a/b bulunur."
                ),
            )
        )
    for a in (1, 2, 3, 4, 5):
        expr = frac(f"1 - \\cos({co(a, 'x')})", "x^{2}")
        out.append(
            Draft(
                "limit",
                "trigonometrik_limit",
                3,
                lim(expr, "x", "0"),
                {"op": "limit", "expr": expr, "var": "x", "to": "0"},
                sympy.Rational(a * a, 2),
                template="limit/bir_eksi_cos",
                notes_tr="1 - cos(ax) = 2 sin^2(ax/2) yazılırsa limit a^2/2 çıkar.",
            )
        )

    # L5 -- one-sided limits (four infinite, four finite).
    for a in (1, 2, 3, 4):
        expr = frac(str(a), "x")
        out.append(
            Draft(
                "limit",
                "tek_yonlu_limit",
                2,
                lim(expr, "x", "0", "+"),
                {"op": "limit", "expr": expr, "var": "x", "to": "0", "direction": "+"},
                sympy.oo,
                checker="exact_latex",
                template="limit/sagdan_sonsuz",
                notes_tr="Sağdan limit sonsuzdur; sonsuz değerler dizgesel olarak karşılaştırılır.",
            )
        )
    for a in (1, 2, 3, 4):
        expr = frac(str(a), "x")
        out.append(
            Draft(
                "limit",
                "tek_yonlu_limit",
                2,
                lim(expr, "x", "0", "-"),
                {"op": "limit", "expr": expr, "var": "x", "to": "0", "direction": "-"},
                -sympy.oo,
                checker="exact_latex",
                template="limit/soldan_sonsuz",
                notes_tr="Soldan limit eksi sonsuzdur; iki taraflı limit yoktur.",
            )
        )
    for a in (1, 2, 3, 4):
        expr = frac(co(a, ABS_X), "x")
        out.append(
            Draft(
                "limit",
                "tek_yonlu_limit",
                2,
                lim(expr, "x", "0", "+"),
                {"op": "limit", "expr": expr, "var": "x", "to": "0", "direction": "+"},
                sympy.Integer(a),
                template="limit/mutlak_deger_sagdan",
                notes_tr="x > 0 için |x| = x olduğundan sağdan limit a olur.",
            )
        )
    for a in (1, 2, 3, 4):
        expr = frac(co(a, ABS_X), "x")
        out.append(
            Draft(
                "limit",
                "tek_yonlu_limit",
                2,
                lim(expr, "x", "0", "-"),
                {"op": "limit", "expr": expr, "var": "x", "to": "0", "direction": "-"},
                sympy.Integer(-a),
                template="limit/mutlak_deger_soldan",
                notes_tr="x < 0 için |x| = -x olduğundan soldan limit -a olur.",
            )
        )

    # L6 -- the number e.
    for a in range(1, 9):
        expr = f"\\left(1 + {frac(str(a), 'x')}\\right)^{{x}}"
        out.append(
            Draft(
                "limit",
                "e_limiti",
                3,
                lim(expr, "x", "\\infty"),
                {"op": "limit", "expr": expr, "var": "x", "to": "\\infty"},
                sympy.exp(a),
                template="limit/e_ustel",
                notes_tr="lim_{x->oo} (1 + a/x)^x = e^a temel limitidir.",
            )
        )
    for a in range(2, 10):
        expr = f"\\left(1 + {frac('1', co(a, 'x'))}\\right)^{{x}}"
        out.append(
            Draft(
                "limit",
                "e_limiti",
                3,
                lim(expr, "x", "\\infty"),
                {"op": "limit", "expr": expr, "var": "x", "to": "\\infty"},
                sympy.exp(sympy.Rational(1, a)),
                template="limit/e_ustel_kesirli",
                notes_tr="(1 + 1/(ax))^x = [(1 + 1/(ax))^{ax}]^{1/a} -> e^{1/a}.",
            )
        )

    # L7 -- exponential and logarithmic standard limits.
    for a in range(1, 7):
        expr = frac(f"e^{{{co(a, 'x')}}} - 1", "x")
        out.append(
            Draft(
                "limit",
                "ustel_logaritmik_limit",
                2,
                lim(expr, "x", "0"),
                {"op": "limit", "expr": expr, "var": "x", "to": "0"},
                sympy.Integer(a),
                template="limit/e_ax_eksi_bir",
                notes_tr="lim_{x->0} (e^{ax} - 1)/x = a temel limitidir.",
            )
        )
    for a in range(1, 6):
        expr = frac(f"\\ln(1 + {co(a, 'x')})", "x")
        out.append(
            Draft(
                "limit",
                "ustel_logaritmik_limit",
                2,
                lim(expr, "x", "0"),
                {"op": "limit", "expr": expr, "var": "x", "to": "0"},
                sympy.Integer(a),
                template="limit/ln_bir_arti_ax",
                notes_tr="lim_{x->0} ln(1 + ax)/x = a temel limitidir.",
            )
        )
    for a in range(2, 7):
        expr = frac(f"{a}^{{x}} - 1", "x")
        out.append(
            Draft(
                "limit",
                "ustel_logaritmik_limit",
                3,
                lim(expr, "x", "0"),
                {"op": "limit", "expr": expr, "var": "x", "to": "0"},
                sympy.log(a),
                template="limit/a_ux_eksi_bir",
                notes_tr="a^x = e^{x ln a} yazılırsa limit ln a olur.",
            )
        )

    # L8 -- limits that need conjugate multiplication.
    for a in range(1, 9):
        expr = frac(f"\\sqrt{{x + {a * a}}} - {a}", "x")
        out.append(
            Draft(
                "limit",
                "kok_belirsizligi",
                3,
                lim(expr, "x", "0"),
                {"op": "limit", "expr": expr, "var": "x", "to": "0"},
                sympy.Rational(1, 2 * a),
                template="limit/eslenik_carpani",
                notes_tr="Eşlenikle çarpılırsa (sqrt(x+a^2) - a)/x -> 1/(2a).",
            )
        )
    for a in range(1, 9):
        expr = f"\\sqrt{{x^{{2}} + {co(a, 'x')}}} - x"
        out.append(
            Draft(
                "limit",
                "kok_belirsizligi",
                3,
                lim(f"\\left({expr}\\right)", "x", "\\infty"),
                {"op": "limit", "expr": expr, "var": "x", "to": "\\infty"},
                sympy.Rational(a, 2),
                template="limit/sonsuz_eksi_sonsuz",
                notes_tr="oo - oo belirsizliği eşlenikle çarpılarak a/2'ye indirgenir.",
            )
        )

    # L9 -- removable 0/0 from a factorised quadratic.
    for a in range(1, 5):
        for b in range(1, 5):
            expr = frac(f"x^{{2}} {signed(-(a + b), 'x')} {signed(a * b)}", f"x - {a}")
            out.append(
                Draft(
                    "limit",
                    "belirsizlik_0_0",
                    2,
                    lim(expr, "x", str(a)),
                    {"op": "limit", "expr": expr, "var": "x", "to": str(a)},
                    sympy.Integer(a - b),
                    template="limit/carpanlara_ayirma",
                    notes_tr="Pay (x - a)(x - b) olarak çarpanlarına ayrılıp sadeleştirilir.",
                )
            )
    return out


# --------------------------------------------------------------------------------------------
# 2. sureklilik (quota 60)
# --------------------------------------------------------------------------------------------
def continuity_drafts() -> list[Draft]:
    """Five continuity families: removable gaps, continuous extension, one-sided values, IVT."""
    out: list[Draft] = []

    # S1 -- removable discontinuity: the limit that patches the hole.
    for a in range(1, 6):
        expr = frac(f"x^{{2}} - {a * a}", f"x - {a}")
        out.append(
            Draft(
                "sureklilik",
                "kaldirilabilir_sureksizlik",
                2,
                lim(expr, "x", str(a)),
                {"op": "limit", "expr": expr, "var": "x", "to": str(a)},
                sympy.Integer(2 * a),
                template="sureklilik/kare_farki",
                notes_tr=(
                    "x = a noktasındaki boşluk kaldırılabilirdir; f(a) := 2a tanımlanırsa "
                    "fonksiyon süreklidir."
                ),
            )
        )
    for a in range(1, 6):
        expr = frac(f"x^{{3}} - {a**3}", f"x - {a}")
        out.append(
            Draft(
                "sureklilik",
                "kaldirilabilir_sureksizlik",
                2,
                lim(expr, "x", str(a)),
                {"op": "limit", "expr": expr, "var": "x", "to": str(a)},
                sympy.Integer(3 * a * a),
                template="sureklilik/kup_farki",
                notes_tr="x^3 - a^3 = (x - a)(x^2 + ax + a^2) sadeleştirmesiyle limit 3a^2 olur.",
            )
        )
    for a in range(1, 6):
        expr = frac(f"x^{{2}} {signed(-(a + 1), 'x')} + {a}", "x - 1")
        out.append(
            Draft(
                "sureklilik",
                "kaldirilabilir_sureksizlik",
                2,
                lim(expr, "x", "1"),
                {"op": "limit", "expr": expr, "var": "x", "to": "1"},
                sympy.Integer(1 - a),
                template="sureklilik/carpanlara_ayirma",
                notes_tr="Pay (x - 1)(x - a) olarak ayrılır; x = 1 bosluğu kaldırılabilirdir.",
            )
        )

    # S2 -- continuous extension at 0.
    for a in range(1, 6):
        expr = frac(f"\\sin({co(a, 'x')})", "x")
        out.append(
            Draft(
                "sureklilik",
                "surekli_genisletme",
                2,
                lim(expr, "x", "0"),
                {"op": "limit", "expr": expr, "var": "x", "to": "0"},
                sympy.Integer(a),
                template="sureklilik/sin_genisletme",
                notes_tr="f(0) := a tanımlanırsa sin(ax)/x fonksiyonu 0 noktasında sürekli olur.",
            )
        )
    for a in range(1, 6):
        expr = frac(f"e^{{{co(a, 'x')}}} - 1", "x")
        out.append(
            Draft(
                "sureklilik",
                "surekli_genisletme",
                2,
                lim(expr, "x", "0"),
                {"op": "limit", "expr": expr, "var": "x", "to": "0"},
                sympy.Integer(a),
                template="sureklilik/ustel_genisletme",
                notes_tr=(
                    "f(0) := a tanımlanırsa (e^{ax} - 1)/x fonksiyonu 0 noktasında sürekli olur."
                ),
            )
        )
    for a in range(1, 6):
        expr = frac(f"1 - \\cos({co(a, 'x')})", "x^{2}")
        out.append(
            Draft(
                "sureklilik",
                "surekli_genisletme",
                3,
                lim(expr, "x", "0"),
                {"op": "limit", "expr": expr, "var": "x", "to": "0"},
                sympy.Rational(a * a, 2),
                template="sureklilik/kosinus_genisletme",
                notes_tr=(
                    "f(0) := a^2/2 tanımlanırsa (1 - cos(ax))/x^2 fonksiyonu 0 noktasında sürekli "
                    "olur."
                ),
            )
        )

    # S3 -- one-sided values of a jump discontinuity.
    for a in range(1, 8):
        expr = frac(co(a, ABS_X), "x")
        out.append(
            Draft(
                "sureklilik",
                "tek_yonlu_sureklilik",
                2,
                lim(expr, "x", "0", "+"),
                {"op": "limit", "expr": expr, "var": "x", "to": "0", "direction": "+"},
                sympy.Integer(a),
                template="sureklilik/sicrama_sag",
                notes_tr=(
                    "Sağ limit a, sol limit -a olduğundan 0 noktasında sıçrama süreksizliği vardır."
                ),
            )
        )
    for a in range(1, 8):
        expr = frac(co(a, ABS_X), "x")
        out.append(
            Draft(
                "sureklilik",
                "tek_yonlu_sureklilik",
                2,
                lim(expr, "x", "0", "-"),
                {"op": "limit", "expr": expr, "var": "x", "to": "0", "direction": "-"},
                sympy.Integer(-a),
                template="sureklilik/sicrama_sol",
                notes_tr=(
                    "Sağ limit a, sol limit -a olduğundan 0 noktasında sıçrama süreksizliği vardır."
                ),
            )
        )

    # S4 -- the value a continuous function actually takes.
    grid = [(a, b, c, d) for a in (1, 2) for b in (-3, 2) for c in (1, -4) for d in (0, 1, 2)]
    for a, b, c, d in grid[:14]:
        expr = f"{co(a, 'x^{2}')} {signed(b, 'x')} {signed(c)}"
        out.append(
            Draft(
                "sureklilik",
                "deger_kontrolu",
                1,
                expr,
                {"op": "evaluate", "expr": expr, "subs": {"x": str(d)}},
                sympy.Integer(a * d * d + b * d + c),
                template="sureklilik/deger",
                notes_tr=(
                    "Polinom her yerde süreklidir; f(d) değeri doğrudan yerine koymayla bulunur."
                ),
            )
        )

    # S5 -- intermediate value theorem: the sign change at the two endpoints.
    for a in range(1, 8):
        expr = f"x^{{3}} - {a}"
        out.append(
            Draft(
                "sureklilik",
                "ara_deger_teoremi",
                2,
                expr,
                {"op": "evaluate", "expr": expr, "subs": {"x": "0"}},
                sympy.Integer(-a),
                template="sureklilik/ara_deger_alt",
                notes_tr=(
                    "f(0) < 0 ve f(2) > 0 ise Ara Değer Teoremi gereği (0, 2) aralığında bir kök "
                    "vardır."
                ),
            )
        )
    for a in range(1, 8):
        expr = f"x^{{3}} - {a}"
        out.append(
            Draft(
                "sureklilik",
                "ara_deger_teoremi",
                2,
                expr,
                {"op": "evaluate", "expr": expr, "subs": {"x": "2"}},
                sympy.Integer(8 - a),
                template="sureklilik/ara_deger_ust",
                notes_tr=(
                    "f(0) < 0 ve f(2) > 0 ise Ara Değer Teoremi gereği (0, 2) aralığında bir kök "
                    "vardır."
                ),
            )
        )
    return out


# --------------------------------------------------------------------------------------------
# 3. turev (quota 200)
# --------------------------------------------------------------------------------------------
def derivative_drafts() -> list[Draft]:
    """Twelve derivative families, including the definition and implicit differentiation."""
    out: list[Draft] = []

    # T1 -- power rule.
    for a in range(1, 5):
        for n in range(2, 7):
            expr = co(a, pw("x", n))
            out.append(
                Draft(
                    "turev",
                    "kuvvet_kurali",
                    1,
                    leibniz(expr),
                    {"op": "diff", "expr": expr, "var": "x", "order": 1},
                    sympy.Integer(a * n) * X ** (n - 1),
                    template="turev/kuvvet",
                    notes_tr="Kuvvet kuralı: (a x^n)' = a n x^{n-1}.",
                )
            )

    # T2 -- sum rule.
    grid = [(a, m, b, c) for a in (1, 2, 3) for m in (2, 3, 4) for b in (-2, 5) for c in (0, 7)]
    for a, m, b, c in grid[:20]:
        expr = f"{co(a, pw('x', m))} {signed(b, 'x')} {signed(c)}"
        out.append(
            Draft(
                "turev",
                "toplam_kurali",
                1,
                leibniz(expr),
                {"op": "diff", "expr": expr, "var": "x", "order": 1},
                sympy.Integer(a * m) * X ** (m - 1) + sympy.Integer(b),
                template="turev/toplam",
                notes_tr="Toplamın türevi türevlerin toplamıdır; sabitin türevi sıfırdır.",
            )
        )

    # T3 -- product rule.
    for m in range(1, 5):
        for a in range(1, 6):
            expr = f"{pw('x', m)} \\sin({co(a, 'x')})"
            out.append(
                Draft(
                    "turev",
                    "carpim_kurali",
                    2,
                    leibniz(expr),
                    {"op": "diff", "expr": expr, "var": "x", "order": 1},
                    sympy.Integer(m) * X ** (m - 1) * sympy.sin(a * X)
                    + sympy.Integer(a) * X**m * sympy.cos(a * X),
                    template="turev/carpim",
                    notes_tr="Çarpım kuralı: (uv)' = u'v + uv'.",
                )
            )

    # T4 -- quotient rule.
    quotients = [
        (a, b, c, d)
        for a in (1, 2, 3)
        for b in (1, -2)
        for c in (1, 3)
        for d in (2, 4)
        if a * d - b * c != 0
    ]
    for a, b, c, d in quotients[:20]:
        expr = frac(f"{co(a, 'x')} {signed(b)}", f"{co(c, 'x')} {signed(d)}")
        out.append(
            Draft(
                "turev",
                "bolum_kurali",
                2,
                leibniz(expr),
                {"op": "diff", "expr": expr, "var": "x", "order": 1},
                sympy.Integer(a * d - b * c) / (sympy.Integer(c) * X + sympy.Integer(d)) ** 2,
                template="turev/bolum",
                notes_tr="Bölüm kuralı: ((ax+b)/(cx+d))' = (ad - bc)/(cx + d)^2.",
            )
        )

    # T5 -- chain rule.
    for a in (1, 2, 3, 4):
        for n in (2, 3):
            expr = f"\\sin({co(a, pw('x', n))})"
            out.append(
                Draft(
                    "turev",
                    "zincir_kurali",
                    2,
                    leibniz(expr),
                    {"op": "diff", "expr": expr, "var": "x", "order": 1},
                    sympy.Integer(a * n) * X ** (n - 1) * sympy.cos(a * X**n),
                    template="turev/zincir_sinus",
                    notes_tr="Zincir kuralı: (sin(u))' = u' cos(u), burada u = a x^n.",
                )
            )
    for a, b, n in ((1, 1, 3), (2, 1, 2), (3, -1, 2), (1, 2, 4), (2, 3, 3), (5, 1, 2)):
        expr = f"\\left({co(a, 'x')} {signed(b)}\\right)^{{{n}}}"
        out.append(
            Draft(
                "turev",
                "zincir_kurali",
                2,
                leibniz(expr),
                {"op": "diff", "expr": expr, "var": "x", "order": 1},
                sympy.Integer(n * a) * (sympy.Integer(a) * X + sympy.Integer(b)) ** (n - 1),
                template="turev/zincir_kuvvet",
                notes_tr="Zincir kuralı: ((ax+b)^n)' = n a (ax+b)^{n-1}.",
            )
        )
    for a in (1, 2, 3, 4, 5, 6):
        expr = f"e^{{{co(a, 'x^{2}')}}}"
        out.append(
            Draft(
                "turev",
                "zincir_kurali",
                2,
                leibniz(expr),
                {"op": "diff", "expr": expr, "var": "x", "order": 1},
                sympy.Integer(2 * a) * X * sympy.exp(a * X**2),
                template="turev/zincir_ustel",
                notes_tr="Zincir kuralı: (e^{u})' = u' e^{u}, burada u = a x^2.",
            )
        )

    # T6 -- trigonometric derivatives.
    for a in range(1, 6):
        expr = f"\\sin({co(a, 'x')})"
        out.append(
            Draft(
                "turev",
                "trigonometrik_turev",
                1,
                leibniz(expr),
                {"op": "diff", "expr": expr, "var": "x", "order": 1},
                sympy.Integer(a) * sympy.cos(a * X),
                template="turev/sinus",
                notes_tr="(sin(ax))' = a cos(ax).",
            )
        )
    for a in range(1, 6):
        expr = f"\\cos({co(a, 'x')})"
        out.append(
            Draft(
                "turev",
                "trigonometrik_turev",
                1,
                leibniz(expr),
                {"op": "diff", "expr": expr, "var": "x", "order": 1},
                sympy.Integer(-a) * sympy.sin(a * X),
                template="turev/kosinus",
                notes_tr="(cos(ax))' = -a sin(ax).",
            )
        )
    for a in range(1, 6):
        expr = f"\\tan({co(a, 'x')})"
        out.append(
            Draft(
                "turev",
                "trigonometrik_turev",
                2,
                leibniz(expr),
                {"op": "diff", "expr": expr, "var": "x", "order": 1},
                sympy.Integer(a) * sympy.sec(a * X) ** 2,
                template="turev/tanjant",
                notes_tr="(tan(ax))' = a sec^2(ax) = a (1 + tan^2(ax)).",
            )
        )
    for a in range(1, 6):
        expr = f"\\sec({co(a, 'x')})"
        out.append(
            Draft(
                "turev",
                "trigonometrik_turev",
                2,
                leibniz(expr),
                {"op": "diff", "expr": expr, "var": "x", "order": 1},
                sympy.Integer(a) * sympy.sec(a * X) * sympy.tan(a * X),
                template="turev/sekant",
                notes_tr="(sec(ax))' = a sec(ax) tan(ax).",
            )
        )

    # T7 -- exponential and logarithmic derivatives.
    for a in range(1, 6):
        expr = f"e^{{{co(a, 'x')}}}"
        out.append(
            Draft(
                "turev",
                "ustel_logaritmik_turev",
                1,
                leibniz(expr),
                {"op": "diff", "expr": expr, "var": "x", "order": 1},
                sympy.Integer(a) * sympy.exp(a * X),
                template="turev/ustel",
                notes_tr="(e^{ax})' = a e^{ax}.",
            )
        )
    for a, b in ((1, 1), (2, 1), (3, 2), (4, 5), (5, 3)):
        expr = f"\\ln({co(a, 'x')} {signed(b)})"
        out.append(
            Draft(
                "turev",
                "ustel_logaritmik_turev",
                2,
                leibniz(expr),
                {"op": "diff", "expr": expr, "var": "x", "order": 1},
                sympy.Integer(a) / (sympy.Integer(a) * X + sympy.Integer(b)),
                template="turev/logaritma",
                notes_tr="(ln(ax + b))' = a/(ax + b).",
            )
        )
    for a in range(2, 7):
        expr = f"{a}^{{x}}"
        out.append(
            Draft(
                "turev",
                "ustel_logaritmik_turev",
                2,
                leibniz(expr),
                {"op": "diff", "expr": expr, "var": "x", "order": 1},
                sympy.Integer(a) ** X * sympy.log(a),
                template="turev/genel_ustel",
                notes_tr="(a^x)' = a^x ln a.",
            )
        )
    for a in range(1, 6):
        expr = f"{co(a, 'x')} \\ln(x)"
        out.append(
            Draft(
                "turev",
                "ustel_logaritmik_turev",
                2,
                leibniz(expr),
                {"op": "diff", "expr": expr, "var": "x", "order": 1},
                sympy.Integer(a) * sympy.log(X) + sympy.Integer(a),
                template="turev/carpim_logaritma",
                notes_tr="(a x ln x)' = a ln x + a (çarpım kuralı).",
            )
        )

    # T8 -- implicit differentiation, in three steps: F_x, F_y and dy/dx = -F_x/F_y.
    implicit = [(1, 1, 25), (1, 4, 16), (4, 9, 36), (1, 1, 9), (2, 3, 12), (1, 2, 8), (3, 1, 27)]
    for p, q, r in implicit:
        expr = f"{co(p, 'x^{2}')} + {co(q, 'y^{2}')} - {r}"
        out.append(
            Draft(
                "turev",
                "kapali_turev",
                3,
                partial(expr, "x"),
                {"op": "diff", "expr": expr, "var": "x", "order": 1},
                sympy.Integer(2 * p) * X,
                template="turev/kapali_fx",
                notes_tr=(
                    "Kapalı türevin ilk adımı: F(x,y) = 0 eşitliğinin x'e göre kısmi türevi F_x."
                ),
            )
        )
    for p, q, r in implicit:
        expr = f"{co(p, 'x^{2}')} + {co(q, 'y^{2}')} - {r}"
        out.append(
            Draft(
                "turev",
                "kapali_turev",
                3,
                partial(expr, "y"),
                {"op": "diff", "expr": expr, "var": "y", "order": 1},
                sympy.Integer(2 * q) * Y,
                template="turev/kapali_fy",
                notes_tr="Kapalı türevin ikinci adımı: aynı eşitliğin y'ye göre kısmi türevi F_y.",
            )
        )
    for p, q, _r in implicit[:6]:
        expr = f"-{frac(co(2 * p, 'x'), co(2 * q, 'y'))}"
        out.append(
            Draft(
                "turev",
                "kapali_turev",
                3,
                expr,
                {"op": "simplify", "expr": expr},
                -sympy.Rational(p, q) * X / Y,
                template="turev/kapali_dydx",
                notes_tr="Kapalı türevin son adımı: dy/dx = -F_x/F_y sadeleştirilir.",
            )
        )

    # T9 -- higher-order derivatives.
    for n in range(2, 12):
        expr = pw("x", n)
        out.append(
            Draft(
                "turev",
                "yuksek_mertebeden_turev",
                2,
                leibniz(expr, order=2),
                {"op": "diff", "expr": expr, "var": "x", "order": 2},
                sympy.Integer(n * (n - 1)) * X ** (n - 2),
                template="turev/ikinci_mertebe",
                notes_tr="(x^n)'' = n(n-1) x^{n-2}.",
            )
        )
    for n in range(3, 8):
        expr = pw("x", n)
        out.append(
            Draft(
                "turev",
                "yuksek_mertebeden_turev",
                2,
                leibniz(expr, order=3),
                {"op": "diff", "expr": expr, "var": "x", "order": 3},
                sympy.Integer(n * (n - 1) * (n - 2)) * X ** (n - 3),
                template="turev/ucuncu_mertebe",
                notes_tr="(x^n)''' = n(n-1)(n-2) x^{n-3}.",
            )
        )
    for a in range(1, 6):
        expr = f"\\sin({co(a, 'x')})"
        out.append(
            Draft(
                "turev",
                "yuksek_mertebeden_turev",
                2,
                leibniz(expr, order=2),
                {"op": "diff", "expr": expr, "var": "x", "order": 2},
                sympy.Integer(-a * a) * sympy.sin(a * X),
                template="turev/ikinci_mertebe_sinus",
                notes_tr="(sin(ax))'' = -a^2 sin(ax).",
            )
        )

    # T10 -- inverse trigonometric derivatives.
    for a in range(1, 8):
        expr = f"\\arctan({co(a, 'x')})"
        out.append(
            Draft(
                "turev",
                "ters_trigonometrik_turev",
                2,
                leibniz(expr),
                {"op": "diff", "expr": expr, "var": "x", "order": 1},
                sympy.Integer(a) / (1 + sympy.Integer(a * a) * X**2),
                template="turev/arctan",
                notes_tr="(arctan(ax))' = a/(1 + a^2 x^2).",
            )
        )
    for a in range(1, 8):
        expr = f"\\arcsin({co(a, 'x')})"
        out.append(
            Draft(
                "turev",
                "ters_trigonometrik_turev",
                2,
                leibniz(expr),
                {"op": "diff", "expr": expr, "var": "x", "order": 1},
                sympy.Integer(a) / sympy.sqrt(1 - sympy.Integer(a * a) * X**2),
                template="turev/arcsin",
                notes_tr="(arcsin(ax))' = a/sqrt(1 - a^2 x^2).",
            )
        )
    for a in range(1, 7):
        expr = f"\\arccos({co(a, 'x')})"
        out.append(
            Draft(
                "turev",
                "ters_trigonometrik_turev",
                2,
                leibniz(expr),
                {"op": "diff", "expr": expr, "var": "x", "order": 1},
                -sympy.Integer(a) / sympy.sqrt(1 - sympy.Integer(a * a) * X**2),
                template="turev/arccos",
                notes_tr="(arccos(ax))' = -a/sqrt(1 - a^2 x^2).",
            )
        )

    # T11 -- hyperbolic derivatives.
    for a in range(1, 8):
        expr = f"\\sinh({co(a, 'x')})"
        out.append(
            Draft(
                "turev",
                "hiperbolik_turev",
                2,
                leibniz(expr),
                {"op": "diff", "expr": expr, "var": "x", "order": 1},
                sympy.Integer(a) * sympy.cosh(a * X),
                template="turev/sinh",
                notes_tr="(sinh(ax))' = a cosh(ax).",
            )
        )
    for a in range(1, 8):
        expr = f"\\cosh({co(a, 'x')})"
        out.append(
            Draft(
                "turev",
                "hiperbolik_turev",
                2,
                leibniz(expr),
                {"op": "diff", "expr": expr, "var": "x", "order": 1},
                sympy.Integer(a) * sympy.sinh(a * X),
                template="turev/cosh",
                notes_tr="(cosh(ax))' = a sinh(ax).",
            )
        )
    for a in range(1, 7):
        expr = f"\\tanh({co(a, 'x')})"
        out.append(
            Draft(
                "turev",
                "hiperbolik_turev",
                2,
                leibniz(expr),
                {"op": "diff", "expr": expr, "var": "x", "order": 1},
                sympy.Integer(a) * (1 - sympy.tanh(a * X) ** 2),
                template="turev/tanh",
                notes_tr="(tanh(ax))' = a (1 - tanh^2(ax)).",
            )
        )

    # T12 -- the derivative from its definition (a limit in h).
    for a in range(1, 6):
        for n in range(2, 6):
            expr = frac(f"{co(a, '(x + h)^{' + str(n) + '}')} - {co(a, pw('x', n))}", "h")
            out.append(
                Draft(
                    "turev",
                    "turev_tanimi",
                    3,
                    lim(expr, "h", "0"),
                    {"op": "limit", "expr": expr, "var": "h", "to": "0"},
                    sympy.Integer(a * n) * X ** (n - 1),
                    template="turev/tanim",
                    notes_tr="Türevin tanımı: f'(x) = lim_{h->0} (f(x+h) - f(x))/h.",
                )
            )
    return out


# --------------------------------------------------------------------------------------------
# 4. turev_uygulamalari (quota 120)
# --------------------------------------------------------------------------------------------
def derivative_application_drafts() -> list[Draft]:
    """Eight families: critical points, tangents, monotonicity, concavity, optimisation, MVT."""
    out: list[Draft] = []

    # A1 -- critical points of x^3 - 3a^2 x and of x^2 - 2ax.
    for a in range(1, 10):
        equation = f"3x^{{2}} - {3 * a * a} = 0"
        out.append(
            Draft(
                "turev_uygulamalari",
                "kritik_noktalar",
                2,
                equation,
                {"op": "solve", "equation": equation, "var": "x", "domain": "reals"},
                sympy.FiniteSet(-a, a),
                checker="set",
                template="uygulama/kritik_kubik",
                notes_tr=(
                    "f(x) = x^3 - 3a^2 x için f'(x) = 3x^2 - 3a^2; kritik noktalar x = -a ve x = a."
                ),
            )
        )
    for a in range(1, 10):
        equation = f"2x - {2 * a} = 0"
        out.append(
            Draft(
                "turev_uygulamalari",
                "kritik_noktalar",
                1,
                equation,
                {"op": "solve", "equation": equation, "var": "x", "domain": "reals"},
                sympy.FiniteSet(a),
                checker="set",
                template="uygulama/kritik_parabol",
                notes_tr="f(x) = x^2 - 2ax için f'(x) = 2x - 2a; tek kritik nokta x = a.",
            )
        )

    # A2 -- the slope of the tangent at a point.
    tangent = [(a, b, x0) for a in (1, 2, 3) for b in (-1, 4) for x0 in (0, 1, 2)]
    for a, b, x0 in tangent[:18]:
        expr = f"{co(2 * a, 'x')} {signed(b)}"
        out.append(
            Draft(
                "turev_uygulamalari",
                "teget_egimi",
                2,
                expr,
                {"op": "evaluate", "expr": expr, "subs": {"x": str(x0)}},
                sympy.Integer(2 * a * x0 + b),
                template="uygulama/teget_egimi",
                notes_tr="f(x) = a x^2 + b x için f'(x) = 2ax + b; teğet eğimi f'(x0) değeridir.",
            )
        )

    # A3 -- where the function increases or decreases.
    for k in range(1, 10):
        equation = f"x^{{2}} - {k * k} = 0"
        out.append(
            Draft(
                "turev_uygulamalari",
                "artan_azalan",
                2,
                equation,
                {"op": "solve", "equation": equation, "var": "x", "domain": "reals"},
                sympy.FiniteSet(-k, k),
                checker="set",
                template="uygulama/artan_azalan_kubik",
                notes_tr=(
                    "f(x) = x^3/3 - k^2 x için f'(x) = x^2 - k^2; işaret değişim noktaları -k ve k."
                ),
            )
        )
    for a in range(1, 10):
        equation = f"2x - {a} = 0"
        out.append(
            Draft(
                "turev_uygulamalari",
                "artan_azalan",
                1,
                equation,
                {"op": "solve", "equation": equation, "var": "x", "domain": "reals"},
                sympy.FiniteSet(sympy.Rational(a, 2)),
                checker="set",
                template="uygulama/artan_azalan_parabol",
                notes_tr="f'(x) = 2x - a sıfırdan önce azalan, sonra artandır.",
            )
        )

    # A4 -- concavity and inflection points.
    for a in range(1, 10):
        equation = f"6x + {2 * a} = 0"
        out.append(
            Draft(
                "turev_uygulamalari",
                "ikinci_turev_bukeylik",
                2,
                equation,
                {"op": "solve", "equation": equation, "var": "x", "domain": "reals"},
                sympy.FiniteSet(sympy.Rational(-a, 3)),
                checker="set",
                template="uygulama/donum_noktasi",
                notes_tr="f(x) = x^3 + a x^2 için f''(x) = 6x + 2a; dönüm noktası x = -a/3.",
            )
        )
    for a in range(1, 10):
        expr = f"x^{{3}} + {co(a, 'x^{2}')}"
        out.append(
            Draft(
                "turev_uygulamalari",
                "ikinci_turev_bukeylik",
                2,
                leibniz(expr, order=2),
                {"op": "diff", "expr": expr, "var": "x", "order": 2},
                6 * X + sympy.Integer(2 * a),
                template="uygulama/ikinci_turev",
                notes_tr="Bükeylik ikinci türevin işaretiyle belirlenir: f''(x) = 6x + 2a.",
            )
        )

    # A5 -- optimisation.
    for a in range(2, 11):
        equation = f"{a} - 2x = 0"
        out.append(
            Draft(
                "turev_uygulamalari",
                "optimizasyon",
                2,
                equation,
                {"op": "solve", "equation": equation, "var": "x", "domain": "reals"},
                sympy.FiniteSet(sympy.Rational(a, 2)),
                checker="set",
                template="uygulama/dikdortgen_alan",
                notes_tr=(
                    "Çevresi 2a olan dikdörtgenin alanı A(x) = x(a - x); A'(x) = a - 2x = 0 için x "
                    "= a/2."
                ),
            )
        )
    for k in range(1, 10):
        equation = f"x^{{2}} - {k * k} = 0"
        out.append(
            Draft(
                "turev_uygulamalari",
                "optimizasyon",
                3,
                equation,
                {"op": "solve", "equation": equation, "var": "x", "domain": "reals"},
                sympy.FiniteSet(-k, k),
                checker="set",
                template="uygulama/toplam_minimum",
                notes_tr="f(x) = x + k^2/x için f'(x) = 1 - k^2/x^2 = 0, yani x^2 = k^2.",
            )
        )

    # A6 -- velocity and acceleration.
    for a, b in ((1, 3), (2, -1), (3, 4), (4, 0), (5, 2), (2, 7), (6, -3), (7, 1), (3, 9)):
        expr = f"{co(a, 't^{2}')} {signed(b, 't')}"
        out.append(
            Draft(
                "turev_uygulamalari",
                "hiz_ivme",
                2,
                leibniz(expr, "t"),
                {"op": "diff", "expr": expr, "var": "t", "order": 1},
                sympy.Integer(2 * a) * T + sympy.Integer(b),
                template="uygulama/hiz",
                notes_tr="Konum s(t) = a t^2 + b t için hız v(t) = s'(t) = 2at + b.",
            )
        )
    for a, b in ((1, 3), (2, -1), (3, 4), (4, 0), (5, 2), (2, 7), (6, -3), (7, 1), (3, 9)):
        expr = f"{co(a, 't^{2}')} {signed(b, 't')}"
        out.append(
            Draft(
                "turev_uygulamalari",
                "hiz_ivme",
                2,
                leibniz(expr, "t", order=2),
                {"op": "diff", "expr": expr, "var": "t", "order": 2},
                sympy.Integer(2 * a),
                template="uygulama/ivme",
                notes_tr="İvme konumun ikinci türevidir: a(t) = s''(t) = 2a.",
            )
        )

    # A7 -- L'Hopital's rule.
    for a, b in ((1, 1), (2, 3), (3, 2), (4, 5), (5, 2), (6, 7)):
        expr = frac(f"e^{{{co(a, 'x')}}} - 1", co(b, "x"))
        out.append(
            Draft(
                "turev_uygulamalari",
                "l_hospital",
                3,
                lim(expr, "x", "0"),
                {"op": "limit", "expr": expr, "var": "x", "to": "0"},
                sympy.Rational(a, b),
                template="uygulama/lhospital_ustel",
                notes_tr="0/0 belirsizliği L'Hospital kuralıyla a e^{ax}/b -> a/b olur.",
            )
        )
    for a, b in ((1, 1), (2, 3), (3, 2), (4, 5), (5, 2), (6, 7)):
        expr = frac(f"1 - \\cos({co(a, 'x')})", co(b, "x^{2}"))
        out.append(
            Draft(
                "turev_uygulamalari",
                "l_hospital",
                3,
                lim(expr, "x", "0"),
                {"op": "limit", "expr": expr, "var": "x", "to": "0"},
                sympy.Rational(a * a, 2 * b),
                template="uygulama/lhospital_trig",
                notes_tr="L'Hospital iki kez uygulanırsa (1 - cos(ax))/(b x^2) -> a^2/(2b).",
            )
        )
    for a in range(1, 7):
        expr = frac("\\ln(x)", pw("x", a))
        out.append(
            Draft(
                "turev_uygulamalari",
                "l_hospital",
                3,
                lim(expr, "x", "\\infty"),
                {"op": "limit", "expr": expr, "var": "x", "to": "\\infty"},
                sympy.Integer(0),
                template="uygulama/lhospital_log",
                notes_tr="oo/oo belirsizliği: L'Hospital ile 1/(a x^a) -> 0.",
            )
        )

    # A8 -- the mean value theorem.
    for a in range(1, 10):
        equation = f"2x - {2 * a} = 0"
        out.append(
            Draft(
                "turev_uygulamalari",
                "ortalama_deger_teoremi",
                3,
                equation,
                {"op": "solve", "equation": equation, "var": "x", "domain": "reals"},
                sympy.FiniteSet(a),
                checker="set",
                template="uygulama/odt_kok",
                notes_tr=(
                    "f(x) = x^2, [0, 2a] aralığında ortalama eğim 2a; f'(c) = 2c = 2a için c = a."
                ),
            )
        )
    for a in range(1, 10):
        b = a + 2
        expr = frac(f"{b * b} - {a * a}", f"{b} - {a}")
        out.append(
            Draft(
                "turev_uygulamalari",
                "ortalama_deger_teoremi",
                2,
                expr,
                {"op": "simplify", "expr": expr},
                sympy.Integer(a + b),
                template="uygulama/ortalama_degisim",
                notes_tr="f(x) = x^2 için [a, b] aralığındaki ortalama değişim oranı a + b'dir.",
            )
        )
    return out


# --------------------------------------------------------------------------------------------
# 5. belirsiz_integral (quota 150)
# --------------------------------------------------------------------------------------------
#: Indefinite integrals are only determined up to an additive constant, so their checker is
#: ``symbolic`` with ``up_to_constant`` (see ``datasets/curriculum/README.md``).
UP_TO_C: dict[str, Any] = {"up_to_constant": True, "var": "x"}


def indefinite_integral_drafts() -> list[Draft]:
    """Nine antiderivative families, one per integration technique on the syllabus."""
    out: list[Draft] = []

    # I1 -- power rule.
    for a in range(1, 5):
        for n in range(0, 5):
            # ``co(a, pw("x", 0))`` would spell ``a1``; a constant integrand is just ``a``.
            expr = co(a, pw("x", n)) if n else str(a)
            out.append(
                Draft(
                    "belirsiz_integral",
                    "kuvvet_kurali",
                    1,
                    indefinite(expr),
                    {"op": "integrate", "expr": expr, "var": "x"},
                    sympy.Rational(a, n + 1) * X ** (n + 1),
                    checker_args=dict(UP_TO_C),
                    template="integral/kuvvet",
                    notes_tr="Kuvvet kuralı: int a x^n dx = a x^{n+1}/(n+1) + C.",
                )
            )

    # I2 -- exponentials.
    for a in range(1, 11):
        expr = f"e^{{{co(a, 'x')}}}"
        out.append(
            Draft(
                "belirsiz_integral",
                "ustel_integral",
                1,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                sympy.exp(a * X) / sympy.Integer(a),
                checker_args=dict(UP_TO_C),
                template="integral/ustel",
                notes_tr="int e^{ax} dx = e^{ax}/a + C.",
            )
        )
    for a in range(2, 12):
        expr = f"{a}^{{x}}"
        out.append(
            Draft(
                "belirsiz_integral",
                "ustel_integral",
                2,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                sympy.Integer(a) ** X / sympy.log(a),
                checker_args=dict(UP_TO_C),
                template="integral/genel_ustel",
                notes_tr="int a^x dx = a^x/ln a + C.",
            )
        )

    # I3 -- trigonometric antiderivatives.
    for a in range(1, 8):
        expr = f"\\sin({co(a, 'x')})"
        out.append(
            Draft(
                "belirsiz_integral",
                "trigonometrik_integral",
                1,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                -sympy.cos(a * X) / sympy.Integer(a),
                checker_args=dict(UP_TO_C),
                template="integral/sinus",
                notes_tr="int sin(ax) dx = -cos(ax)/a + C.",
            )
        )
    for a in range(1, 8):
        expr = f"\\cos({co(a, 'x')})"
        out.append(
            Draft(
                "belirsiz_integral",
                "trigonometrik_integral",
                1,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                sympy.sin(a * X) / sympy.Integer(a),
                checker_args=dict(UP_TO_C),
                template="integral/kosinus",
                notes_tr="int cos(ax) dx = sin(ax)/a + C.",
            )
        )
    for a in range(1, 7):
        expr = f"\\sec^{{2}}({co(a, 'x')})"
        out.append(
            Draft(
                "belirsiz_integral",
                "trigonometrik_integral",
                2,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                sympy.tan(a * X) / sympy.Integer(a),
                checker_args=dict(UP_TO_C),
                template="integral/sekant_kare",
                notes_tr="int sec^2(ax) dx = tan(ax)/a + C.",
            )
        )

    # I4 -- basic rational integrands.
    for a in range(1, 8):
        expr = frac(str(a), "x")
        out.append(
            Draft(
                "belirsiz_integral",
                "rasyonel_integral",
                1,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                sympy.Integer(a) * sympy.log(X),
                checker_args=dict(UP_TO_C),
                template="integral/bir_bolu_x",
                notes_tr="int a/x dx = a ln|x| + C; SymPy gerçel bölgede ln(x) yazar.",
            )
        )
    for a, b in ((1, 1), (2, 3), (3, 5), (4, 2), (5, 7), (6, 4), (7, 9)):
        expr = frac(str(a), f"x + {b}")
        out.append(
            Draft(
                "belirsiz_integral",
                "rasyonel_integral",
                2,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                sympy.Integer(a) * sympy.log(X + b),
                checker_args=dict(UP_TO_C),
                template="integral/bir_bolu_x_arti_b",
                notes_tr="int a/(x + b) dx = a ln|x + b| + C.",
            )
        )
    for a, b in ((2, 1), (3, 2), (4, 1), (5, 3), (6, 5), (7, 2)):
        expr = frac("1", f"{co(a, 'x')} + {b}")
        out.append(
            Draft(
                "belirsiz_integral",
                "rasyonel_integral",
                2,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                sympy.log(sympy.Integer(a) * X + b) / sympy.Integer(a),
                checker_args=dict(UP_TO_C),
                template="integral/bir_bolu_ax_arti_b",
                notes_tr="u = ax + b değişken değiştirmesi ile int dx/(ax+b) = ln|ax+b|/a + C.",
            )
        )

    # I5 -- substitution.
    for a in range(1, 8):
        expr = frac("2x", f"x^{{2}} + {a}")
        out.append(
            Draft(
                "belirsiz_integral",
                "degisken_degistirme",
                2,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                sympy.log(X**2 + a),
                checker_args=dict(UP_TO_C),
                template="integral/u_turevi_bolu_u",
                notes_tr="u = x^2 + a alınırsa int du/u = ln|u| + C.",
            )
        )
    for a in range(1, 8):
        expr = f"x e^{{{co(a, 'x^{2}')}}}"
        out.append(
            Draft(
                "belirsiz_integral",
                "degisken_degistirme",
                2,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                sympy.exp(a * X**2) / sympy.Integer(2 * a),
                checker_args=dict(UP_TO_C),
                template="integral/x_e_ax2",
                notes_tr="u = a x^2, du = 2ax dx değişken değiştirmesi kullanılır.",
            )
        )
    for n in range(1, 7):
        expr = f"\\sin^{{{n}}}(x) \\cos(x)"
        out.append(
            Draft(
                "belirsiz_integral",
                "degisken_degistirme",
                2,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                sympy.sin(X) ** (n + 1) / sympy.Integer(n + 1),
                checker_args=dict(UP_TO_C),
                template="integral/sin_us_cos",
                notes_tr="u = sin x, du = cos x dx değişken değiştirmesi kullanılır.",
            )
        )

    # I6 -- integration by parts.
    for a in range(1, 6):
        expr = f"x e^{{{co(a, 'x')}}}"
        out.append(
            Draft(
                "belirsiz_integral",
                "kismi_integrasyon",
                3,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                sympy.exp(a * X) * (sympy.Integer(a) * X - 1) / sympy.Integer(a * a),
                checker_args=dict(UP_TO_C),
                template="integral/kismi_ustel",
                notes_tr="u = x, dv = e^{ax} dx kısmi integrasyonu: e^{ax}(ax - 1)/a^2 + C.",
            )
        )
    for a in range(1, 6):
        expr = f"x \\sin({co(a, 'x')})"
        out.append(
            Draft(
                "belirsiz_integral",
                "kismi_integrasyon",
                3,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                (sympy.sin(a * X) - sympy.Integer(a) * X * sympy.cos(a * X)) / sympy.Integer(a * a),
                checker_args=dict(UP_TO_C),
                template="integral/kismi_sinus",
                notes_tr="u = x, dv = sin(ax) dx kısmi integrasyonu uygulanır.",
            )
        )
    for a in range(1, 6):
        expr = f"x \\cos({co(a, 'x')})"
        out.append(
            Draft(
                "belirsiz_integral",
                "kismi_integrasyon",
                3,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                (sympy.cos(a * X) + sympy.Integer(a) * X * sympy.sin(a * X)) / sympy.Integer(a * a),
                checker_args=dict(UP_TO_C),
                template="integral/kismi_kosinus",
                notes_tr="u = x, dv = cos(ax) dx kısmi integrasyonu uygulanır.",
            )
        )
    for n in range(0, 5):
        expr = f"{pw('x', n)} \\ln(x)" if n else "\\ln(x)"
        out.append(
            Draft(
                "belirsiz_integral",
                "kismi_integrasyon",
                3,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                X ** (n + 1)
                * (sympy.Integer(n + 1) * sympy.log(X) - 1)
                / sympy.Integer((n + 1) ** 2),
                checker_args=dict(UP_TO_C),
                template="integral/kismi_logaritma",
                notes_tr=(
                    "u = ln x, dv = x^n dx kısmi integrasyonu: x^{n+1}((n+1)ln x - 1)/(n+1)^2 + C."
                ),
            )
        )

    # I7 -- partial fractions.
    pairs = [(1, 2), (1, 3), (2, 5), (1, 4), (3, 7), (2, 6), (1, 5), (4, 9), (2, 3), (3, 8)]
    for a, b in pairs:
        expr = frac("1", f"(x - {a})(x - {b})")
        out.append(
            Draft(
                "belirsiz_integral",
                "basit_kesirlere_ayirma",
                3,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                (sympy.log(X - a) - sympy.log(X - b)) / sympy.Integer(a - b),
                checker_args=dict(UP_TO_C),
                template="integral/basit_kesir_iki_kok",
                notes_tr="1/((x-a)(x-b)) = [1/(a-b)][1/(x-a) - 1/(x-b)] ayrışımı kullanılır.",
            )
        )
    for a in range(1, 11):
        expr = frac("1", f"x^{{2}} - {a * a}")
        out.append(
            Draft(
                "belirsiz_integral",
                "basit_kesirlere_ayirma",
                3,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                (sympy.log(X - a) - sympy.log(X + a)) / sympy.Integer(2 * a),
                checker_args=dict(UP_TO_C),
                template="integral/basit_kesir_kare_farki",
                notes_tr="1/(x^2 - a^2) = [1/(2a)][1/(x-a) - 1/(x+a)] ayrışımı kullanılır.",
            )
        )

    # I8 -- trigonometric techniques.
    for a in range(1, 8):
        expr = f"\\sin^{{2}}({co(a, 'x')})"
        out.append(
            Draft(
                "belirsiz_integral",
                "trigonometrik_teknikler",
                3,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                X / 2 - sympy.sin(2 * a * X) / sympy.Integer(4 * a),
                checker_args=dict(UP_TO_C),
                template="integral/sinus_kare",
                notes_tr="sin^2(ax) = (1 - cos(2ax))/2 yarım açı formülü kullanılır.",
            )
        )
    for a in range(1, 8):
        expr = f"\\cos^{{2}}({co(a, 'x')})"
        out.append(
            Draft(
                "belirsiz_integral",
                "trigonometrik_teknikler",
                3,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                X / 2 + sympy.sin(2 * a * X) / sympy.Integer(4 * a),
                checker_args=dict(UP_TO_C),
                template="integral/kosinus_kare",
                notes_tr="cos^2(ax) = (1 + cos(2ax))/2 yarım açı formülü kullanılır.",
            )
        )
    for a in range(1, 7):
        expr = f"\\tan({co(a, 'x')})"
        out.append(
            Draft(
                "belirsiz_integral",
                "trigonometrik_teknikler",
                3,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                -sympy.log(sympy.cos(a * X)) / sympy.Integer(a),
                checker_args=dict(UP_TO_C),
                template="integral/tanjant",
                notes_tr="int tan(ax) dx = -ln|cos(ax)|/a + C.",
            )
        )

    # I9 -- integrals that produce inverse trigonometric functions.
    for a in range(1, 8):
        expr = frac("1", f"x^{{2}} + {a * a}")
        out.append(
            Draft(
                "belirsiz_integral",
                "ters_trigonometrik_sonuc",
                3,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                sympy.atan(X / sympy.Integer(a)) / sympy.Integer(a),
                checker_args=dict(UP_TO_C),
                template="integral/arctan_sonucu",
                notes_tr="int dx/(x^2 + a^2) = arctan(x/a)/a + C.",
            )
        )
    for a in range(1, 8):
        expr = frac("1", f"\\sqrt{{{a * a} - x^{{2}}}}")
        out.append(
            Draft(
                "belirsiz_integral",
                "ters_trigonometrik_sonuc",
                3,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                sympy.asin(X / sympy.Integer(a)),
                checker_args=dict(UP_TO_C),
                template="integral/arcsin_sonucu",
                notes_tr="int dx/sqrt(a^2 - x^2) = arcsin(x/a) + C.",
            )
        )
    for a in range(1, 7):
        expr = frac(str(a), "1 + x^{2}")
        out.append(
            Draft(
                "belirsiz_integral",
                "ters_trigonometrik_sonuc",
                2,
                indefinite(expr),
                {"op": "integrate", "expr": expr, "var": "x"},
                sympy.Integer(a) * sympy.atan(X),
                checker_args=dict(UP_TO_C),
                template="integral/arctan_katsayili",
                notes_tr="int a dx/(1 + x^2) = a arctan(x) + C.",
            )
        )
    return out


# --------------------------------------------------------------------------------------------
# 6. belirli_integral (quota 120)
# --------------------------------------------------------------------------------------------
def definite_integral_drafts() -> list[Draft]:
    """Eight definite-integral families, including improper integrals."""
    out: list[Draft] = []

    # D1 -- the fundamental theorem on a power.
    grid = [(a, n, b) for a in (1, 2, 3) for n in (1, 2, 3) for b in (1, 2)]
    for a, n, b in grid[:18]:
        expr = co(a, pw("x", n))
        out.append(
            Draft(
                "belirli_integral",
                "kuvvet_belirli",
                1,
                definite(expr, "0", str(b)),
                {"op": "integrate", "expr": expr, "var": "x", "lower": "0", "upper": str(b)},
                sympy.Rational(a * b ** (n + 1), n + 1),
                template="belirli/kuvvet",
                notes_tr="Analizin temel teoremi: int_0^b a x^n dx = a b^{n+1}/(n+1).",
            )
        )

    # D2 -- trigonometric integrals over [0, pi].
    for a in range(1, 10):
        expr = f"\\sin({co(a, 'x')})"
        out.append(
            Draft(
                "belirli_integral",
                "trigonometrik_belirli",
                2,
                definite(expr, "0", "\\pi"),
                {"op": "integrate", "expr": expr, "var": "x", "lower": "0", "upper": "\\pi"},
                (1 - sympy.cos(sympy.Integer(a) * sympy.pi)) / sympy.Integer(a),
                template="belirli/sinus",
                notes_tr="int_0^pi sin(ax) dx = (1 - cos(a pi))/a.",
            )
        )
    for a in range(1, 10):
        expr = f"\\cos({co(a, 'x')})"
        out.append(
            Draft(
                "belirli_integral",
                "trigonometrik_belirli",
                2,
                definite(expr, "0", "\\pi"),
                {"op": "integrate", "expr": expr, "var": "x", "lower": "0", "upper": "\\pi"},
                sympy.sin(sympy.Integer(a) * sympy.pi) / sympy.Integer(a),
                template="belirli/kosinus",
                notes_tr="int_0^pi cos(ax) dx = sin(a pi)/a = 0 (a tam sayı için).",
            )
        )

    # D3 -- exponentials.
    grid_e = [(a, b) for a in (1, 2, 3) for b in (1, 2, 3)]
    for a, b in grid_e[:9]:
        expr = f"e^{{{co(a, 'x')}}}"
        out.append(
            Draft(
                "belirli_integral",
                "ustel_belirli",
                2,
                definite(expr, "0", str(b)),
                {"op": "integrate", "expr": expr, "var": "x", "lower": "0", "upper": str(b)},
                (sympy.exp(sympy.Integer(a * b)) - 1) / sympy.Integer(a),
                template="belirli/ustel",
                notes_tr="int_0^b e^{ax} dx = (e^{ab} - 1)/a.",
            )
        )
    for a in range(1, 10):
        expr = f"e^{{-{co(a, 'x')}}}"
        out.append(
            Draft(
                "belirli_integral",
                "ustel_belirli",
                2,
                definite(expr, "0", "1"),
                {"op": "integrate", "expr": expr, "var": "x", "lower": "0", "upper": "1"},
                (1 - sympy.exp(-sympy.Integer(a))) / sympy.Integer(a),
                template="belirli/ustel_negatif",
                notes_tr="int_0^1 e^{-ax} dx = (1 - e^{-a})/a.",
            )
        )

    # D4 -- logarithmic values.
    for b in range(2, 11):
        expr = frac("1", "x")
        out.append(
            Draft(
                "belirli_integral",
                "rasyonel_belirli",
                2,
                definite(expr, "1", str(b)),
                {"op": "integrate", "expr": expr, "var": "x", "lower": "1", "upper": str(b)},
                sympy.log(b),
                template="belirli/logaritma",
                notes_tr="int_1^b dx/x = ln b.",
            )
        )
    for a in range(1, 10):
        expr = frac(str(a), "x")
        out.append(
            Draft(
                "belirli_integral",
                "rasyonel_belirli",
                2,
                definite(expr, "1", "2"),
                {"op": "integrate", "expr": expr, "var": "x", "lower": "1", "upper": "2"},
                sympy.Integer(a) * sympy.log(2),
                template="belirli/logaritma_katsayili",
                notes_tr="int_1^2 a dx/x = a ln 2.",
            )
        )

    # D5 -- inverse trigonometric values.
    for a in range(1, 10):
        expr = frac("1", f"x^{{2}} + {a * a}")
        out.append(
            Draft(
                "belirli_integral",
                "ters_trigonometrik_belirli",
                3,
                definite(expr, "0", str(a)),
                {"op": "integrate", "expr": expr, "var": "x", "lower": "0", "upper": str(a)},
                sympy.pi / sympy.Integer(4 * a),
                template="belirli/arctan",
                notes_tr="int_0^a dx/(x^2 + a^2) = arctan(1)/a = pi/(4a).",
            )
        )
    for a in range(1, 10):
        expr = frac("1", f"x^{{2}} + {a * a}")
        out.append(
            Draft(
                "belirli_integral",
                "ters_trigonometrik_belirli",
                3,
                definite(expr, f"-{a}", str(a)),
                {"op": "integrate", "expr": expr, "var": "x", "lower": f"-{a}", "upper": str(a)},
                sympy.pi / sympy.Integer(2 * a),
                template="belirli/arctan_simetrik",
                notes_tr="Çift fonksiyonun simetrik aralıkta integrali iki katıdır: pi/(2a).",
            )
        )

    # D6 -- substitution in a definite integral.
    for a in range(1, 10):
        expr = f"x e^{{{co(a, 'x^{2}')}}}"
        out.append(
            Draft(
                "belirli_integral",
                "degisken_degistirme_belirli",
                3,
                definite(expr, "0", "1"),
                {"op": "integrate", "expr": expr, "var": "x", "lower": "0", "upper": "1"},
                (sympy.exp(sympy.Integer(a)) - 1) / sympy.Integer(2 * a),
                template="belirli/degisken_degistirme_ustel",
                notes_tr="u = a x^2 değişken değiştirmesiyle sınırlar 0 ve a olur.",
            )
        )
    for a in range(1, 10):
        expr = frac("2x", f"x^{{2}} + {a}")
        out.append(
            Draft(
                "belirli_integral",
                "degisken_degistirme_belirli",
                3,
                definite(expr, "0", "1"),
                {"op": "integrate", "expr": expr, "var": "x", "lower": "0", "upper": "1"},
                sympy.log(sympy.Integer(a + 1)) - sympy.log(sympy.Integer(a)),
                template="belirli/degisken_degistirme_log",
                notes_tr="u = x^2 + a değişken değiştirmesiyle ln((1+a)/a) elde edilir.",
            )
        )

    # D7 -- integration by parts on a definite integral.
    for a in range(1, 10):
        expr = f"{co(a, 'x')} e^{{x}}"
        out.append(
            Draft(
                "belirli_integral",
                "kismi_integrasyon_belirli",
                3,
                definite(expr, "0", "1"),
                {"op": "integrate", "expr": expr, "var": "x", "lower": "0", "upper": "1"},
                sympy.Integer(a),
                template="belirli/kismi_ustel",
                notes_tr="int_0^1 x e^x dx = 1 olduğundan katsayılı integral a olur.",
            )
        )
    for a in range(1, 10):
        expr = f"{co(a, 'x')} \\ln(x)"
        out.append(
            Draft(
                "belirli_integral",
                "kismi_integrasyon_belirli",
                3,
                definite(expr, "1", "e"),
                {"op": "integrate", "expr": expr, "var": "x", "lower": "1", "upper": "e"},
                sympy.Rational(a, 4) * (sympy.exp(2) + 1),
                template="belirli/kismi_logaritma",
                notes_tr="int_1^e x ln x dx = (e^2 + 1)/4; katsayılı hali a(e^2 + 1)/4.",
            )
        )

    # D8 -- improper integrals.
    for p in range(2, 11):
        expr = frac("1", pw("x", p))
        out.append(
            Draft(
                "belirli_integral",
                "has_olmayan_integral",
                3,
                definite(expr, "1", "\\infty"),
                {"op": "integrate", "expr": expr, "var": "x", "lower": "1", "upper": "\\infty"},
                sympy.Rational(1, p - 1),
                template="belirli/has_olmayan_p",
                notes_tr="int_1^oo dx/x^p = 1/(p-1) (p > 1 için yakınsar).",
            )
        )
    for a in range(1, 10):
        expr = f"e^{{-{co(a, 'x')}}}"
        out.append(
            Draft(
                "belirli_integral",
                "has_olmayan_integral",
                3,
                definite(expr, "0", "\\infty"),
                {"op": "integrate", "expr": expr, "var": "x", "lower": "0", "upper": "\\infty"},
                sympy.Rational(1, a),
                template="belirli/has_olmayan_ustel",
                notes_tr="int_0^oo e^{-ax} dx = 1/a (a > 0 için yakınsar).",
            )
        )
    return out


# --------------------------------------------------------------------------------------------
# 7. integral_uygulamalari (quota 80)
# --------------------------------------------------------------------------------------------
def integral_application_drafts() -> list[Draft]:
    """Six application families: area, area between curves, volume, arc length, mean, work."""
    out: list[Draft] = []

    # P1 -- area under a curve.
    grid = [(a, n, c, b) for a in (1, 2) for n in (1, 2) for c in (0, 1) for b in (1, 2)]
    for a, n, c, b in grid[:16]:
        expr = f"{co(a, pw('x', n))} {signed(c)}"
        out.append(
            Draft(
                "integral_uygulamalari",
                "egri_alti_alan",
                2,
                definite(expr, "0", str(b)),
                {"op": "integrate", "expr": expr, "var": "x", "lower": "0", "upper": str(b)},
                sympy.Rational(a * b ** (n + 1), n + 1) + sympy.Integer(c * b),
                template="uygulama_integral/alan",
                notes_tr="[0, b] aralığında eğri altında kalan alan belirli integralle bulunur.",
            )
        )

    # P2 -- area between two curves.
    for a in range(1, 9):
        expr = f"{co(a, 'x')} - x^{{2}}"
        out.append(
            Draft(
                "integral_uygulamalari",
                "iki_egri_arasi_alan",
                3,
                definite(expr, "0", str(a)),
                {"op": "integrate", "expr": expr, "var": "x", "lower": "0", "upper": str(a)},
                sympy.Rational(a**3, 6),
                template="uygulama_integral/iki_egri_parabol",
                notes_tr="y = ax ile y = x^2 eğrileri arasındaki alan a^3/6'dır.",
            )
        )
    for m, n in ((1, 2), (1, 3), (2, 3), (1, 4), (2, 5), (3, 4), (1, 5), (2, 7)):
        expr = f"{pw('x', m)} - {pw('x', n)}"
        out.append(
            Draft(
                "integral_uygulamalari",
                "iki_egri_arasi_alan",
                3,
                definite(expr, "0", "1"),
                {"op": "integrate", "expr": expr, "var": "x", "lower": "0", "upper": "1"},
                sympy.Rational(1, m + 1) - sympy.Rational(1, n + 1),
                template="uygulama_integral/iki_egri_kuvvet",
                notes_tr="[0, 1] aralığında y = x^m ve y = x^n eğrileri arasındaki alan.",
            )
        )

    # P3 -- volume of a solid of revolution (disc method).
    grid_v = [(n, b) for n in (1, 2, 3, 4) for b in (1, 2, 3, 4)]
    for n, b in grid_v[:16]:
        expr = f"\\pi {pw('x', 2 * n)}"
        out.append(
            Draft(
                "integral_uygulamalari",
                "donel_cisim_hacmi",
                3,
                definite(expr, "0", str(b)),
                {"op": "integrate", "expr": expr, "var": "x", "lower": "0", "upper": str(b)},
                sympy.pi * sympy.Rational(b ** (2 * n + 1), 2 * n + 1),
                template="uygulama_integral/hacim_disk",
                notes_tr=(
                    "y = x^n eğrisinin x ekseni etrafında dönmesiyle V = pi int_0^b x^{2n} dx."
                ),
            )
        )

    # P4 -- arc length.
    for b in range(1, 9):
        expr = "\\sqrt{1 + x}"
        out.append(
            Draft(
                "integral_uygulamalari",
                "yay_uzunlugu",
                3,
                definite(expr, "0", str(b)),
                {"op": "integrate", "expr": expr, "var": "x", "lower": "0", "upper": str(b)},
                sympy.Rational(2, 3) * ((1 + sympy.Integer(b)) ** sympy.Rational(3, 2) - 1),
                template="uygulama_integral/yay_kok",
                notes_tr="y = (2/3) x^{3/2} eğrisinin yay uzunluğu int_0^b sqrt(1 + x) dx'tir.",
            )
        )
    for a in range(1, 9):
        expr = f"\\sqrt{{1 + {a * a}}}"
        out.append(
            Draft(
                "integral_uygulamalari",
                "yay_uzunlugu",
                2,
                definite(expr, "0", "1"),
                {"op": "integrate", "expr": expr, "var": "x", "lower": "0", "upper": "1"},
                sympy.sqrt(1 + a * a),
                template="uygulama_integral/yay_dogru",
                notes_tr="y = ax doğrusunun [0, 1] aralığındaki yay uzunluğu sqrt(1 + a^2)'dir.",
            )
        )

    # P5 -- mean value of a function.
    grid_m = [(n, b) for n in (1, 2, 3, 4) for b in (1, 2, 3, 4)]
    for n, b in grid_m[:16]:
        expr = frac(pw("x", n), str(b))
        out.append(
            Draft(
                "integral_uygulamalari",
                "ortalama_deger",
                3,
                definite(expr, "0", str(b)),
                {"op": "integrate", "expr": expr, "var": "x", "lower": "0", "upper": str(b)},
                sympy.Rational(b**n, n + 1),
                template="uygulama_integral/ortalama",
                notes_tr="f'nin [0, b] ortalama değeri (1/b) int_0^b x^n dx = b^n/(n+1).",
            )
        )

    # P6 -- work done by a spring-like force.
    grid_w = [(k, b) for k in (1, 2, 3, 4) for b in (1, 2, 3, 4)]
    for k, b in grid_w[:16]:
        expr = co(k, "x")
        out.append(
            Draft(
                "integral_uygulamalari",
                "is_ve_kuvvet",
                2,
                definite(expr, "0", str(b)),
                {"op": "integrate", "expr": expr, "var": "x", "lower": "0", "upper": str(b)},
                sympy.Rational(k * b * b, 2),
                template="uygulama_integral/is",
                notes_tr="F(x) = kx kuvvetinin [0, b] aralığında yaptığı is W = k b^2/2'dir.",
            )
        )
    return out


# --------------------------------------------------------------------------------------------
# 8. diziler_seriler (quota 80)
# --------------------------------------------------------------------------------------------
def series_drafts() -> list[Draft]:
    """Six families: sequence limits, geometric and telescoping sums, Maclaurin series, tests."""
    out: list[Draft] = []

    # Q1 -- sequence limits.
    for a, c in ((1, 2), (3, 4), (5, 2), (2, 7), (9, 3), (4, 5), (7, 2), (6, 4)):
        expr = frac(f"{co(a, 'n')} + 1", f"{co(c, 'n')} - 3")
        out.append(
            Draft(
                "diziler_seriler",
                "dizi_limiti",
                2,
                lim(expr, "n", "\\infty"),
                {"op": "limit", "expr": expr, "var": "n", "to": "\\infty"},
                sympy.Rational(a, c),
                template="seri/dizi_birinci_derece",
                notes_tr="a_n = (an + 1)/(cn - 3) dizisinin limiti a/c'dir.",
            )
        )
    for a, c in ((1, 2), (3, 4), (5, 2), (2, 7), (9, 3), (4, 5), (7, 2), (6, 4)):
        expr = frac(f"{co(a, 'n^{2}')} + n", f"{co(c, 'n^{2}')} + 1")
        out.append(
            Draft(
                "diziler_seriler",
                "dizi_limiti",
                2,
                lim(expr, "n", "\\infty"),
                {"op": "limit", "expr": expr, "var": "n", "to": "\\infty"},
                sympy.Rational(a, c),
                template="seri/dizi_ikinci_derece",
                notes_tr="Pay ve payda aynı dereceden olduğundan limit baş katsayıların oranıdır.",
            )
        )

    # Q2 -- geometric series as the limit of its partial sums.
    # The partial sum a(1 - r^n)/(1 - r) with r = 1/k is written out as
    # S_n = p/q - p/(q k^n), because the compound constant denominator (1 - 1/k) is folded by
    # SymPy on parse and would break the level-2 round trip of the input string.
    geometric = [(a, k) for a in (1, 2, 3) for k in (2, 3, 4, 5)]
    for a, k in geometric[:12]:
        total = sympy.Rational(a * k, k - 1)
        p, q = total.p, total.q
        head = str(p) if q == 1 else frac(str(p), str(q))
        tail = frac(str(p), f"{k}^{{n}}") if q == 1 else frac(str(p), f"{q} \\cdot {k}^{{n}}")
        expr = f"{head} - {tail}"
        out.append(
            Draft(
                "diziler_seriler",
                "geometrik_seri",
                3,
                lim(f"\\left({expr}\\right)", "n", "\\infty"),
                {"op": "limit", "expr": expr, "var": "n", "to": "\\infty"},
                total,
                template="seri/geometrik",
                notes_tr=(
                    "Geometrik serinin toplamı kısmi toplamların limitidir: a/(1 - r) = ak/(k-1)."
                ),
            )
        )

    # Q3 -- telescoping series.
    for a in range(2, 14):
        expr = f"\\frac{{1}}{{{a}}} - \\frac{{1}}{{n + {a}}}"
        out.append(
            Draft(
                "diziler_seriler",
                "teleskopik_seri",
                3,
                lim(f"\\left({expr}\\right)", "n", "\\infty"),
                {"op": "limit", "expr": expr, "var": "n", "to": "\\infty"},
                sympy.Rational(1, a),
                template="seri/teleskopik",
                notes_tr="Teleskopik serinin kısmi toplamı S_n = 1/a - 1/(n+a); limiti 1/a'dır.",
            )
        )

    # Q4 -- closed forms of power sums.
    for m in range(2, 8):
        expr = frac("n(n + 1)", "2")
        out.append(
            Draft(
                "diziler_seriler",
                "kuvvet_toplamlari",
                2,
                expr,
                {"op": "evaluate", "expr": expr, "subs": {"n": str(m)}},
                sympy.Integer(m * (m + 1) // 2),
                template="seri/gauss_toplami",
                notes_tr="1 + 2 + ... + n = n(n+1)/2 (Gauss formülü).",
            )
        )
    for m in range(2, 8):
        expr = frac("n(n + 1)(2n + 1)", "6")
        out.append(
            Draft(
                "diziler_seriler",
                "kuvvet_toplamlari",
                2,
                expr,
                {"op": "evaluate", "expr": expr, "subs": {"n": str(m)}},
                sympy.Integer(m * (m + 1) * (2 * m + 1) // 6),
                template="seri/kare_toplami",
                notes_tr="1^2 + 2^2 + ... + n^2 = n(n+1)(2n+1)/6.",
            )
        )

    # Q5 -- Maclaurin series (the Taylor family of the syllabus).
    def order_term(m: int) -> Any:
        return sympy.Order(X**m, (X, 0))

    for a in range(1, 5):
        for m in (4, 5):
            answer = sum(
                (sympy.Integer(a) ** k * X**k / sympy.factorial(k) for k in range(m)),
                sympy.Integer(0),
            )
            expr = f"e^{{{co(a, 'x')}}}"
            out.append(
                Draft(
                    "diziler_seriler",
                    "taylor_maclaurin",
                    3,
                    expr,
                    {"op": "series", "expr": expr, "var": "x", "point": "0", "order": m},
                    answer + order_term(m),
                    template="seri/maclaurin_ustel",
                    notes_tr="e^{ax} = sum_{k>=0} (ax)^k/k! Maclaurin serisi.",
                )
            )
    for a in range(1, 4):
        for m in (5, 6):
            answer = sum(
                (
                    sympy.Integer(-1) ** j
                    * sympy.Integer(a) ** (2 * j + 1)
                    * X ** (2 * j + 1)
                    / sympy.factorial(2 * j + 1)
                    for j in range(0, (m + 1) // 2)
                    if 2 * j + 1 < m
                ),
                sympy.Integer(0),
            )
            expr = f"\\sin({co(a, 'x')})"
            out.append(
                Draft(
                    "diziler_seriler",
                    "taylor_maclaurin",
                    3,
                    expr,
                    {"op": "series", "expr": expr, "var": "x", "point": "0", "order": m},
                    answer + order_term(m),
                    template="seri/maclaurin_sinus",
                    notes_tr="sin(ax) = sum_{j>=0} (-1)^j (ax)^{2j+1}/(2j+1)! Maclaurin serisi.",
                )
            )
    for a in range(1, 4):
        for m in (5, 6):
            answer = sum(
                (
                    sympy.Integer(-1) ** j
                    * sympy.Integer(a) ** (2 * j)
                    * X ** (2 * j)
                    / sympy.factorial(2 * j)
                    for j in range(0, m // 2 + 1)
                    if 2 * j < m
                ),
                sympy.Integer(0),
            )
            expr = f"\\cos({co(a, 'x')})"
            out.append(
                Draft(
                    "diziler_seriler",
                    "taylor_maclaurin",
                    3,
                    expr,
                    {"op": "series", "expr": expr, "var": "x", "point": "0", "order": m},
                    answer + order_term(m),
                    template="seri/maclaurin_kosinus",
                    notes_tr="cos(ax) = sum_{j>=0} (-1)^j (ax)^{2j}/(2j)! Maclaurin serisi.",
                )
            )
    for a in range(1, 5):
        for m in (4, 5):
            answer = sum(
                (
                    sympy.Integer(-1) ** (k + 1) * sympy.Integer(a) ** k * X**k / sympy.Integer(k)
                    for k in range(1, m)
                ),
                sympy.Integer(0),
            )
            expr = f"\\ln(1 + {co(a, 'x')})"
            out.append(
                Draft(
                    "diziler_seriler",
                    "taylor_maclaurin",
                    3,
                    expr,
                    {"op": "series", "expr": expr, "var": "x", "point": "0", "order": m},
                    answer + order_term(m),
                    template="seri/maclaurin_logaritma",
                    notes_tr="ln(1 + ax) = sum_{k>=1} (-1)^{k+1} (ax)^k/k Maclaurin serisi.",
                )
            )
    for a in range(1, 5):
        for m in (4, 5):
            answer = sum(
                (sympy.Integer(a) ** k * X**k for k in range(m)),
                sympy.Integer(0),
            )
            expr = frac("1", f"1 - {co(a, 'x')}")
            out.append(
                Draft(
                    "diziler_seriler",
                    "taylor_maclaurin",
                    3,
                    expr,
                    {"op": "series", "expr": expr, "var": "x", "point": "0", "order": m},
                    answer + order_term(m),
                    template="seri/maclaurin_geometrik",
                    notes_tr="1/(1 - ax) = sum_{k>=0} (ax)^k geometrik seri açılımı.",
                )
            )

    # Q6 -- convergence tests (the n-th term and ratio tests).
    for p in range(1, 7):
        expr = frac("1", pw("n", p))
        out.append(
            Draft(
                "diziler_seriler",
                "yakinsaklik_testleri",
                2,
                lim(expr, "n", "\\infty"),
                {"op": "limit", "expr": expr, "var": "n", "to": "\\infty"},
                sympy.Integer(0),
                template="seri/terim_testi",
                notes_tr=(
                    "p-serisinin genel teriminin limiti sıfırdır; yakınsaklık için p > 1 gerekir."
                ),
            )
        )
    for a in range(1, 7):
        expr = frac(str(a), "n + 1")
        out.append(
            Draft(
                "diziler_seriler",
                "yakinsaklik_testleri",
                3,
                lim(expr, "n", "\\infty"),
                {"op": "limit", "expr": expr, "var": "n", "to": "\\infty"},
                sympy.Integer(0),
                template="seri/oran_testi",
                notes_tr=(
                    "sum a^n/n! serisi için oran testi limiti a/(n+1) -> 0 < 1; seri yakınsar."
                ),
            )
        )
    return out


# --------------------------------------------------------------------------------------------
# 9. denklem_esitsizlik (quota 50)
# --------------------------------------------------------------------------------------------
def equation_drafts() -> list[Draft]:
    """Six algebra families the CAS is asked for as a preparatory step."""
    out: list[Draft] = []

    for a in range(1, 11):
        equation = f"x^{{2}} - {a * a} = 0"
        out.append(
            Draft(
                "denklem_esitsizlik",
                "ikinci_derece_denklem",
                1,
                equation,
                {"op": "solve", "equation": equation, "var": "x", "domain": "reals"},
                sympy.FiniteSet(-a, a),
                checker="set",
                template="denklem/kare_farki",
                notes_tr="x^2 - a^2 = 0 denkleminin gerçel kökleri -a ve a'dır.",
            )
        )
    for a, b in (
        (2, 1),
        (3, -6),
        (4, 2),
        (5, 10),
        (6, 3),
        (7, -7),
        (8, 4),
        (9, 6),
        (3, 12),
        (5, -5),
    ):
        equation = f"{co(a, 'x')} {signed(b)} = 0"
        out.append(
            Draft(
                "denklem_esitsizlik",
                "birinci_derece_denklem",
                1,
                equation,
                {"op": "solve", "equation": equation, "var": "x", "domain": "reals"},
                sympy.FiniteSet(sympy.Rational(-b, a)),
                checker="set",
                template="denklem/birinci_derece",
                notes_tr="ax + b = 0 denkleminin kokü x = -b/a'dır.",
            )
        )
    for a in range(1, 6):
        expr = f"x^{{2}} - {a * a}"
        out.append(
            Draft(
                "denklem_esitsizlik",
                "carpanlara_ayirma",
                1,
                expr,
                {"op": "factor", "expr": expr},
                (X - a) * (X + a),
                template="denklem/carpan_kare_farki",
                notes_tr="Kare farkı özdeşliği: x^2 - a^2 = (x - a)(x + a).",
            )
        )
    for a in range(1, 6):
        expr = f"x^{{2}} + {co(2 * a, 'x')} + {a * a}"
        out.append(
            Draft(
                "denklem_esitsizlik",
                "carpanlara_ayirma",
                1,
                expr,
                {"op": "factor", "expr": expr},
                (X + a) ** 2,
                template="denklem/carpan_tam_kare",
                notes_tr="Tam kare özdeşliği: x^2 + 2ax + a^2 = (x + a)^2.",
            )
        )
    for a in range(1, 6):
        expr = f"\\left(x + {a}\\right)^{{2}}"
        out.append(
            Draft(
                "denklem_esitsizlik",
                "acilim",
                1,
                expr,
                {"op": "expand", "expr": expr},
                X**2 + sympy.Integer(2 * a) * X + sympy.Integer(a * a),
                template="denklem/acilim_tam_kare",
                notes_tr="(x + a)^2 = x^2 + 2ax + a^2.",
            )
        )
    for a, b in ((1, 2), (2, 3), (3, 1), (4, 5), (5, 2)):
        expr = f"\\left(x + {a}\\right)\\left(x - {b}\\right)"
        out.append(
            Draft(
                "denklem_esitsizlik",
                "acilim",
                1,
                expr,
                {"op": "expand", "expr": expr},
                X**2 + sympy.Integer(a - b) * X - sympy.Integer(a * b),
                template="denklem/acilim_iki_terim",
                notes_tr="(x + a)(x - b) = x^2 + (a - b)x - ab.",
            )
        )
    for a in range(1, 6):
        expr = frac(f"x^{{2}} - {a * a}", f"x - {a}")
        out.append(
            Draft(
                "denklem_esitsizlik",
                "sadelestirme",
                2,
                expr,
                {"op": "simplify", "expr": expr},
                X + sympy.Integer(a),
                template="denklem/sadelestirme_kare_farki",
                notes_tr="Ortak çarpan sadeleştirilirse (x^2 - a^2)/(x - a) = x + a olur.",
            )
        )
    for a in range(1, 6):
        expr = co(a, SIN_SQ) + " + " + co(a, COS_SQ)
        out.append(
            Draft(
                "denklem_esitsizlik",
                "sadelestirme",
                2,
                expr,
                {"op": "simplify", "expr": expr},
                sympy.Integer(a),
                template="denklem/pisagor_ozdesligi",
                notes_tr="Pisagor özdeşliği: a sin^2 x + a cos^2 x = a.",
            )
        )
    for a in range(2, 7):
        equation = f"e^{{x}} = {a}"
        out.append(
            Draft(
                "denklem_esitsizlik",
                "ustel_logaritmik_denklem",
                2,
                equation,
                {"op": "solve", "equation": equation, "var": "x", "domain": "reals"},
                sympy.FiniteSet(sympy.log(a)),
                checker="set",
                template="denklem/ustel",
                notes_tr="e^x = a denkleminin çözümü x = ln a'dır.",
            )
        )
    for a in range(1, 6):
        equation = f"\\ln(x) = {a}"
        out.append(
            Draft(
                "denklem_esitsizlik",
                "ustel_logaritmik_denklem",
                2,
                equation,
                {"op": "solve", "equation": equation, "var": "x", "domain": "reals"},
                sympy.FiniteSet(sympy.exp(a)),
                checker="set",
                template="denklem/logaritmik",
                notes_tr="ln x = a denkleminin çözümü x = e^a'dır.",
            )
        )
    return out


# --------------------------------------------------------------------------------------------
# 10. grafik (quota 20)
# --------------------------------------------------------------------------------------------
def plot_drafts() -> list[Draft]:
    """Four plotting families; the checker only asserts that a PNG was produced."""
    out: list[Draft] = []
    for a, b in ((1, 4), (2, 1), (3, 9), (1, 1), (2, 8), (4, 4)):
        expr = f"{co(a, 'x^{2}')} - {b}"
        out.append(
            Draft(
                "grafik",
                "polinom_grafigi",
                1,
                expr,
                {"op": "plot", "expr": expr, "var": "x", "xmin": -3.0, "xmax": 3.0},
                sympy.Integer(a) * X**2 - sympy.Integer(b),
                checker="plot_ok",
                template="grafik/parabol",
                notes_tr="Parabolün [-3, 3] aralığındaki grafiği çizilir.",
            )
        )
    for a in range(1, 7):
        expr = f"\\sin({co(a, 'x')})"
        out.append(
            Draft(
                "grafik",
                "trigonometrik_grafik",
                1,
                expr,
                {"op": "plot", "expr": expr, "var": "x", "xmin": -6.0, "xmax": 6.0},
                sympy.sin(sympy.Integer(a) * X),
                checker="plot_ok",
                template="grafik/sinus",
                notes_tr="sin(ax) fonksiyonunun periyodik grafiği çizilir.",
            )
        )
    for a in range(1, 7):
        expr = f"e^{{{frac('x', str(a))}}}"
        out.append(
            Draft(
                "grafik",
                "ustel_grafik",
                1,
                expr,
                {"op": "plot", "expr": expr, "var": "x", "xmin": -2.0, "xmax": 2.0},
                sympy.exp(X / sympy.Integer(a)),
                checker="plot_ok",
                template="grafik/ustel",
                notes_tr="Üstel fonksiyonun [-2, 2] aralığındaki grafiği çizilir.",
            )
        )
    for a, b in ((1, 1), (2, 1), (3, 4), (1, 9), (4, 2), (5, 3)):
        expr = frac(str(a), f"x^{{2}} + {b}")
        out.append(
            Draft(
                "grafik",
                "rasyonel_grafik",
                2,
                expr,
                {"op": "plot", "expr": expr, "var": "x", "xmin": -4.0, "xmax": 4.0},
                sympy.Integer(a) / (X**2 + sympy.Integer(b)),
                checker="plot_ok",
                template="grafik/rasyonel",
                notes_tr="Paydası sıfır olmayan rasyonel fonksiyonun grafiği çizilir.",
            )
        )
    return out


#: Topic -> the function that authors that topic's cards, in the order of the quota table.
FAMILIES: tuple[tuple[str, Any], ...] = (
    ("limit", limit_drafts),
    ("sureklilik", continuity_drafts),
    ("turev", derivative_drafts),
    ("turev_uygulamalari", derivative_application_drafts),
    ("belirsiz_integral", indefinite_integral_drafts),
    ("belirli_integral", definite_integral_drafts),
    ("integral_uygulamalari", integral_application_drafts),
    ("diziler_seriler", series_drafts),
    ("denklem_esitsizlik", equation_drafts),
    ("grafik", plot_drafts),
)


def sentence_tr(draft: Draft) -> str:
    """Turkish phrasing of one command, used for ``input.kind == "text_tr"`` rows."""
    template = SENTENCE_TR[draft.command["op"]]
    return template.format(tex=f"${draft.input_value}$", var=draft.command.get("var", "x"))


def to_card(draft: Draft, index: int, *, as_text: bool) -> dict[str, Any]:
    """Render one authored draft into the card mapping written to YAML.

    Args:
        draft: The authored problem.
        index: 1-based index inside its topic; becomes the numeric part of the id.
        as_text: When True the input is a Turkish sentence instead of LaTeX.

    Returns:
        A mapping with the exact key order of the card schema.
    """
    kind = "text_tr" if as_text else "latex"
    value = sentence_tr(draft) if as_text else draft.input_value
    return {
        "id": f"{draft.topic}-{index:04d}",
        "topic": draft.topic,
        "subtopic": draft.subtopic,
        "chapter_ref": CHAPTER_BY_TOPIC[draft.topic],
        "difficulty": draft.difficulty,
        "input": {"kind": kind, "value": value},
        "command": dict(draft.command),
        "expected": {
            "latex": sympy_to_latex(draft.answer),
            "srepr": sympy.srepr(draft.answer),
        },
        "checker": draft.checker,
        "checker_args": dict(draft.checker_args),
        "source": f"sablon:{draft.template}",
        "author": AUTHOR,
        "reviewed_by": REVIEWED_BY,
        "status": STATUS,
        "notes_tr": draft.notes_tr,
    }


def build_cards() -> dict[str, list[dict[str, Any]]]:
    """Author every card, grouped by topic and ordered deterministically."""
    cards: dict[str, list[dict[str, Any]]] = {}
    position = 0
    for topic, factory in FAMILIES:
        drafts: list[Draft] = factory()
        rows: list[dict[str, Any]] = []
        for index, draft in enumerate(drafts, start=1):
            position += 1
            as_text = position % TEXT_TR_EVERY == 0 and draft.command["op"] in SENTENCE_TR
            rows.append(to_card(draft, index, as_text=as_text))
        cards[topic] = rows
    return cards


def header_comment(topic: str, rows: list[dict[str, Any]]) -> str:
    """The Turkish comment block written at the top of every card file."""
    subtopics = sorted({row["subtopic"] for row in rows})
    return (
        f"# datasets/curriculum/cards/{topic}.yaml\n"
        f"# {len(rows)} kart. Alt konular: {', '.join(subtopics)}.\n"
        "# KÖKEN: Bu kartlar proje içinde yazılmıştır (şablonlardan makine ile üretilmiştir);\n"
        "#        bağımsız bir derlemden ÖRNEKLENMEMİŞTİR ve bir ders kitabından alıntı değildir.\n"
        "#        Beklenen yanıtlar müfredat kuralından türetilmiştir, CAS'a sorulmamıştır.\n"
        "# Üretici: datasets/curriculum/generate_cards.py -- elle düzenlemeyin, yeniden üretin.\n"
        "# Derleyici: scripts/build_curriculum_testset.py\n"
    )


def write_cards(cards: dict[str, list[dict[str, Any]]], cards_dir: Path) -> list[Path]:
    """Write one YAML file per topic and return the paths in a stable order."""
    cards_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for topic, rows in cards.items():
        body = yaml.safe_dump(
            rows, allow_unicode=True, sort_keys=False, default_flow_style=False, width=100
        )
        reloaded = yaml.safe_load(body)
        if reloaded != rows:  # pragma: no cover - a YAML round-trip failure is a hard stop
            raise RuntimeError(f"YAML round trip changed the cards of topic {topic!r}")
        path = cards_dir / f"{topic}.yaml"
        path.write_text(header_comment(topic, rows) + body, encoding="utf-8", newline="\n")
        written.append(path)
    return written


def iter_rows(cards: dict[str, list[dict[str, Any]]]) -> Iterator[dict[str, Any]]:
    """Yield every card in topic order."""
    for rows in cards.values():
        yield from rows


def main(argv: list[str] | None = None) -> int:
    """Author the cards and print a per-topic summary."""
    _ = argv
    cards = build_cards()
    written = write_cards(cards, CARDS_DIR)
    total = sum(len(rows) for rows in cards.values())
    for path, (topic, rows) in zip(written, cards.items(), strict=True):
        text_rows = sum(1 for row in rows if row["input"]["kind"] == "text_tr")
        print(f"{topic:24s} {len(rows):5d} kart ({text_rows} text_tr)  -> {path.name}")
    print(f"{'TOPLAM':24s} {total:5d} kart")
    ids = [row["id"] for row in iter_rows(cards)]
    if len(set(ids)) != len(ids):
        print("HATA: yinelenen kimlik var", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
