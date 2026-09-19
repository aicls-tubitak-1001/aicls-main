"""SymPy -> KaTeX-safe LaTeX (docs/mimari/latex-ayristirici.md 3.6).

Every result the CAS produces leaves the platform through this function, and the user may
send it straight back as input - so the output has to stay *inside our own grammar*.  That is
enforced by ``test_sympy_latex_output_is_within_grammar_on_curriculum`` and by the round-trip
levels in :mod:`aicls.latex2sympy.roundtrip`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import sympy

from aicls.core.errors import InputValidationError
from aicls.core.latex_safety import sanitize_latex


class PrintStyle(StrEnum):
    """Output flavour for :func:`sympy_to_latex`."""

    #: KaTeX-compatible, allow-list filtered (default).
    KATEX = "katex"
    #: Plain ``sympy.latex`` output, for debugging and differential tests.
    RAW = "raw"


#: ``sympy.latex`` options that keep the output inside the v1 grammar.
KATEX_PRINT_OPTIONS: dict[str, Any] = {
    "fold_short_frac": False,
    "mul_symbol": "dot",
    "ln_notation": True,
    "inv_trig_style": "full",
    "order": "lex",
}

#: Literal replacements applied before the allow-list filter.
_FIXUPS: tuple[tuple[str, str], ...] = (
    # \operatorname* (lim-like operators) is outside the KaTeX allow-list.
    (r"\operatorname*", r"\operatorname"),
    # sympy prints norms with double bars; the grammar knows single bars only.
    (r"\left\|", r"\lvert "),
    (r"\right\|", r" \rvert"),
)


def _katex_fixups(text: str) -> str:
    result = text
    for needle, replacement in _FIXUPS:
        result = result.replace(needle, replacement)
    return result


def sympy_to_latex(expr: sympy.Basic, *, style: PrintStyle = PrintStyle.KATEX) -> str:
    """Render a SymPy object as LaTeX.

    Args:
        expr: Any ``sympy.Basic`` (expression, relation, set, matrix, unevaluated integral).
        style: :attr:`PrintStyle.KATEX` (default) applies the curriculum print options, the
            KaTeX fixups and :func:`aicls.core.latex_safety.sanitize_latex`;
            :attr:`PrintStyle.RAW` returns ``sympy.latex(expr)`` untouched.

    Returns:
        The LaTeX string. ``sympy.Rational(1, 3)`` renders as ``\\frac{1}{3}`` (proposal A.2.2).

    Raises:
        aicls.core.errors.InputValidationError: If ``expr`` is not a SymPy object.
    """
    # A mutable sympy.Matrix is a MatrixBase but not a Basic, hence the union.
    if not isinstance(expr, sympy.Basic | sympy.MatrixBase):
        raise InputValidationError(details={"type": type(expr).__name__})
    if style is PrintStyle.RAW:
        raw: str = sympy.latex(expr)
        return raw
    rendered: str = sympy.latex(expr, **KATEX_PRINT_OPTIONS)
    return sanitize_latex(_katex_fixups(rendered))


__all__ = ["KATEX_PRINT_OPTIONS", "PrintStyle", "sympy_to_latex"]
