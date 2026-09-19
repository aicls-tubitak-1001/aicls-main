"""Cross-module contracts of :mod:`aicls.latex2sympy` (wave-1 lesson: test the boundary).

One boundary is exercised here with the *real* peer module rather than a local fixture:

``aicls.ocr`` -> ``aicls.latex2sympy`` -- :data:`aicls.ocr.CURRICULUM_LATEX` is the label
alphabet the handwriting recogniser is trained to emit, so every one of those strings reaches
``parse_latex`` verbatim in production.  The contract has two halves:

1. a **measured** parse rate over that table (KNOWN_ISSUES parser item 1 was found exactly
   here: ``\\frac{\\partial f}{\\partial x}`` raised ``ParseError``), and
2. an **enumerated** out-of-scope set: the strings that do not parse must raise
   :class:`~aicls.latex2sympy.errors.UnsupportedConstruct` *with the construct name*, never a
   bare ``ParseError`` and never a non-``AiclsError`` exception, because the API turns the
   named error into a 422 "recognised but not computable" payload rather than a 500.

Both halves are pinned as numbers so a grammar regression shows up as a failing count, not as a
quietly degraded metric in the next progress report.
"""

from __future__ import annotations

import pytest

from aicls.core.errors import AiclsError
from aicls.latex2sympy import (
    UnsupportedConstruct,
    parse_latex,
    roundtrip_check,
    to_latex,
    to_sympy,
)
from aicls.ocr import CURRICULUM_LATEX

pytestmark = pytest.mark.integration

#: Curriculum strings that are deliberately outside the v1 grammar scope
#: (docs/mimari/latex-ayristirici.md 3.2), mapped to the construct name the user must see.
OUT_OF_SCOPE: dict[str, str] = {
    r"\forall\varepsilon>0": "\\forall",
    r"\binom{n}{k}": "\\binom",
}

#: Measured floor for the ocr -> latex2sympy hand-off: 32 strings minus the two rows of
#: :data:`OUT_OF_SCOPE` = 30/32 = 93.75 %.  Before the KNOWN_ISSUES fix it was 29/32 = 90.6 %.
MIN_CURRICULUM_PARSED = len(CURRICULUM_LATEX) - len(OUT_OF_SCOPE)


def _parsable() -> list[str]:
    return [latex for latex in CURRICULUM_LATEX if latex not in OUT_OF_SCOPE]


def test_curriculum_out_of_scope_set_is_exactly_the_documented_two() -> None:
    """A new unparsable curriculum string must be a deliberate, documented decision."""
    failing = {}
    for latex in CURRICULUM_LATEX:
        try:
            parse_latex(latex)
        except AiclsError as exc:
            failing[latex] = type(exc).__name__
    assert sorted(failing) == sorted(OUT_OF_SCOPE), failing


@pytest.mark.parametrize("latex", sorted(OUT_OF_SCOPE))
def test_out_of_scope_curriculum_strings_are_named_not_anonymous(latex: str) -> None:
    """The OCR -> CAS hand-off needs the construct name to build its 422 payload."""
    with pytest.raises(UnsupportedConstruct) as excinfo:
        parse_latex(latex)
    assert excinfo.value.construct == OUT_OF_SCOPE[latex]
    assert excinfo.value.details["construct"] == OUT_OF_SCOPE[latex]
    assert OUT_OF_SCOPE[latex] in str(excinfo.value)


def test_curriculum_parse_rate_is_30_of_32() -> None:
    """Measured 2026-09-07: 30/32 = 93.75 % (was 29/32 = 90.6 % before the partial fix)."""
    parsed = 0
    for latex in CURRICULUM_LATEX:
        try:
            parse_latex(latex)
            parsed += 1
        except AiclsError:
            continue
    assert parsed >= MIN_CURRICULUM_PARSED, f"{parsed}/{len(CURRICULUM_LATEX)}"
    assert len(CURRICULUM_LATEX) == 32


@pytest.mark.parametrize("latex", _parsable())
def test_every_parsable_curriculum_string_reaches_sympy_and_round_trips(latex: str) -> None:
    """Parsing is not enough: the CAS layer needs a SymPy object and a stable canonical form."""
    node = parse_latex(latex)
    to_sympy(node)
    assert parse_latex(to_latex(node)) == node
    assert roundtrip_check(latex, numeric_samples=3).ok


def test_partial_derivative_from_the_curriculum_lowers_to_a_sympy_derivative() -> None:
    """KNOWN_ISSUES parser item 1: this exact string is emitted by the recogniser."""
    import sympy

    node = parse_latex(r"\frac{\partial f}{\partial x}")
    expression = to_sympy(node)
    assert isinstance(expression, sympy.Derivative)
    assert expression.variables == (sympy.Symbol("x", real=True),)


def test_lenient_mode_still_names_structural_constructs() -> None:
    """``strict=False`` (what OCR output gets) drops decoration but not real constructs."""
    assert parse_latex(r"\overline{x} + 1", strict=False) == parse_latex("x + 1")
    with pytest.raises(UnsupportedConstruct) as excinfo:
        parse_latex(r"\binom{n}{k}", strict=False)
    assert excinfo.value.construct == "\\binom"
