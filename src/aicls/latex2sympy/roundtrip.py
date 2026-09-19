"""Three-level round-trip verification (docs/mimari/latex-ayristirici.md 5).

``roundtrip_check(src)`` runs the loop the platform actually performs in production -
parse, convert to SymPy, print the result back with :func:`sympy_to_latex`, re-parse it - and
reports the strongest level at which the two agree:

#. :data:`LEVEL_CANONICAL` - the canonical LaTeX of both ASTs is identical (strongest);
#. :data:`LEVEL_SREPR` - the two SymPy objects have the same ``srepr``;
#. :data:`LEVEL_NUMERIC` - ``simplify(a - b) == 0``, or the two agree at random real sample
   points (weakest, used for expressions SymPy re-associates while printing).

A silent misparse therefore counts as a failure, which is what makes the ">99 % parse success"
metric of IP-4 strict: ``ok=False`` is reported, never raised, so a corpus run can collect the
failing ids instead of stopping at the first one.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import sympy
from sympy.core.relational import Relational

from aicls.core.errors import AiclsError
from aicls.latex2sympy.parser import parse_latex
from aicls.latex2sympy.printer import sympy_to_latex
from aicls.latex2sympy.to_latex import to_latex
from aicls.latex2sympy.to_sympy import to_sympy

#: Round-trip levels, strongest first.
LEVEL_CANONICAL = "canonical"
LEVEL_SREPR = "srepr"
LEVEL_NUMERIC = "numeric"

#: Failure reasons (prefixes; the concrete error is appended).
REASON_PARSE_FAILED = "parse_failed"
REASON_REPARSE_FAILED = "reparse_failed"
REASON_MISMATCH = "mismatch"

#: Sampling window for the numeric level; avoids 0 and negative values so that logarithms,
#: square roots and 1/x stay finite.
SAMPLE_LOW = 0.5
SAMPLE_HIGH = 2.5
SAMPLE_SEED = 1234
NUMERIC_TOLERANCE = 1e-6
MAX_NUMERIC_SYMBOLS = 3

_UNEVALUATED = (sympy.Integral, sympy.Sum, sympy.Product, sympy.Limit, sympy.Derivative)


@dataclass(frozen=True, slots=True)
class RoundTripResult:
    """Outcome of :func:`roundtrip_check`."""

    ok: bool
    reason: str
    canonical: str


def numerically_equal(left: sympy.Basic, right: sympy.Basic, *, samples: int = 5) -> bool:
    """Compare two expressions at ``samples`` deterministic random points.

    Args:
        left: First expression.
        right: Second expression.
        samples: Number of sample points; ``0`` disables the comparison (returns False).

    Returns:
        True when both expressions evaluated at the same points agree within
        :data:`NUMERIC_TOLERANCE` at least once and never disagree.
    """
    if samples <= 0:
        return False
    symbols = sorted(left.free_symbols | right.free_symbols, key=str)
    if len(symbols) > MAX_NUMERIC_SYMBOLS:
        return False
    rng = random.Random(SAMPLE_SEED)
    checked = 0
    for _ in range(samples * 3):
        if checked >= samples:
            break
        point = {symbol: rng.uniform(SAMPLE_LOW, SAMPLE_HIGH) for symbol in symbols}
        try:
            left_value = complex(left.evalf(subs=point))
            right_value = complex(right.evalf(subs=point))
        except (TypeError, ValueError, ZeroDivisionError, AttributeError):
            continue
        if not (_is_finite(left_value) and _is_finite(right_value)):
            continue
        checked += 1
        if abs(left_value - right_value) > NUMERIC_TOLERANCE * (1.0 + abs(left_value)):
            return False
    return checked > 0


def _is_finite(value: complex) -> bool:
    return math.isfinite(value.real) and math.isfinite(value.imag) and abs(value) < 1e12


def expressions_equivalent(
    left: sympy.Basic, right: sympy.Basic, *, numeric_samples: int = 5
) -> bool:
    """Whether two SymPy objects mean the same thing (level 3 of the round trip).

    Structural equality first, then ``simplify(left - right) == 0``, then numeric sampling.
    Relations, sets, matrices and unevaluated calculus objects are compared structurally only.

    Args:
        left: First object.
        right: Second object.
        numeric_samples: Points for :func:`numerically_equal`.

    Returns:
        True when the two are equivalent.
    """
    if left == right:
        return True
    if isinstance(left, Relational | sympy.Set | sympy.MatrixBase):
        return False
    if isinstance(right, Relational | sympy.Set | sympy.MatrixBase):
        return False
    if left.has(*_UNEVALUATED) or right.has(*_UNEVALUATED):
        return False
    try:
        if sympy.simplify(left - right) == 0:
            return True
    except (TypeError, ValueError, AttributeError, NotImplementedError):
        return False
    return numerically_equal(left, right, samples=numeric_samples)


def roundtrip_check(src: str, *, numeric_samples: int = 5) -> RoundTripResult:
    """Verify that ``src`` survives the parse -> SymPy -> print -> parse loop.

    Args:
        src: LaTeX source.
        numeric_samples: How many random real points the numeric level may use; ``0``
            disables that level.

    Returns:
        A :class:`RoundTripResult`; ``reason`` names the level that succeeded, or the failure.
        Errors are reported, not raised.
    """
    try:
        node = parse_latex(src)
        expression = to_sympy(node)
    except AiclsError as exc:
        return RoundTripResult(False, f"{REASON_PARSE_FAILED}: {exc.code.value}", "")
    canonical = to_latex(node)
    try:
        printed = sympy_to_latex(expression)
        reparsed_node = parse_latex(printed)
        reparsed = to_sympy(reparsed_node)
    except AiclsError as exc:
        return RoundTripResult(False, f"{REASON_REPARSE_FAILED}: {exc.code.value}", canonical)
    if to_latex(reparsed_node) == canonical:
        return RoundTripResult(True, LEVEL_CANONICAL, canonical)
    if sympy.srepr(expression) == sympy.srepr(reparsed):
        return RoundTripResult(True, LEVEL_SREPR, canonical)
    if expressions_equivalent(expression, reparsed, numeric_samples=numeric_samples):
        return RoundTripResult(True, LEVEL_NUMERIC, canonical)
    return RoundTripResult(False, REASON_MISMATCH, canonical)


__all__ = [
    "LEVEL_CANONICAL",
    "LEVEL_NUMERIC",
    "LEVEL_SREPR",
    "MAX_NUMERIC_SYMBOLS",
    "NUMERIC_TOLERANCE",
    "REASON_MISMATCH",
    "REASON_PARSE_FAILED",
    "REASON_REPARSE_FAILED",
    "RoundTripResult",
    "expressions_equivalent",
    "numerically_equal",
    "roundtrip_check",
]
