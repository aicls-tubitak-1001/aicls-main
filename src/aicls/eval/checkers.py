"""How a curriculum answer is judged (docs/ml/degerlendirme.md §2, CAS accuracy).

``datasets/curriculum/problems.jsonl`` carries a ``checker`` per row, because "is this answer
correct?" is not one question: an antiderivative is only defined up to a constant, a solution
set has no order, a limit may be infinite and a plot is judged by the expression it drew. The
strategies live here so that the compiler (``scripts/build_curriculum_testset.py``, which builds
the set) and the evaluation harness (which reports on it) can never drift apart in what they
call "correct" — and so that the *reported* definition is the one under test.

Every comparison ends in :func:`aicls.mathengine.symbolic_equivalent`, whose third state
(``unparsable``) is preserved: a row the CAS cannot judge is reported as its own column, never
silently scored as wrong.
"""

from __future__ import annotations

import cmath
from collections.abc import Mapping
import re
from typing import Any

import sympy

from aicls.core import messages_tr
from aicls.core.errors import AiclsError, InputValidationError
from aicls.latex2sympy import latex_to_sympy
from aicls.mathengine import symbolic_equivalent

#: Sampling grid of the ``numeric`` checker (deterministic, no ``random``).
NUMERIC_SAMPLES = 20

#: Default absolute tolerance of the ``numeric`` checker.
NUMERIC_TOLERANCE = 1e-9

#: Default sampling domain of the ``numeric`` checker.
NUMERIC_DOMAIN = (-3.0, 3.0)

#: Characters an ``srepr`` may contain. ``srepr`` is machine-generated (by SymPy) and comes
#: either from a committed dataset row or from our own engine, but it is still *text that gets
#: evaluated*, so it is validated before :func:`sympy.sympify` sees it (CODING_STANDARDS:
#: "never eval/exec on user text").
_SREPR_ALLOWED_RE = re.compile(r"^[A-Za-z0-9_ ,.:()\[\]{}'\"/*+\-=]*$")

#: Names that must never appear in an ``srepr``; their presence means it is not an ``srepr``.
_SREPR_FORBIDDEN = ("__", "import", "lambda", "eval", "exec", "open", "os.", "sys.")


def rebuild_srepr(text: str) -> Any:
    """Rebuild a SymPy object from an ``srepr`` string, after validating the text.

    Args:
        text: The ``srepr`` (``"Add(Symbol('x'), Integer(1))"``).

    Returns:
        The SymPy object.

    Raises:
        InputValidationError: If the text is empty, contains characters an ``srepr`` never has,
            names a forbidden construct, or SymPy cannot rebuild it.
    """
    if not text or not text.strip():
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR, details={"reason": "empty_srepr"}
        )
    lowered = text.lower()
    if not _SREPR_ALLOWED_RE.match(text) or any(bad in lowered for bad in _SREPR_FORBIDDEN):
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "srepr_rejected", "srepr": text[:120]},
        )
    try:
        return sympy.sympify(text)
    except (SyntaxError, TypeError, ValueError, AttributeError) as exc:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={
                "reason": "srepr_not_rebuildable",
                "srepr": text[:120],
                "error": str(exc)[:200],
            },
        ) from exc


def _symbol(args: Mapping[str, Any]) -> sympy.Symbol:
    """The variable a checker differentiates or samples over (``checker_args.var``)."""
    return sympy.Symbol(str(args.get("var", "x")), real=True)


def compare_latex(actual: str, expected: str) -> tuple[bool, str]:
    """``exact_latex``: the rendered answer must be the declared string, character for character.

    Used where symbolic comparison abstains — infinite limits, ``\\infty`` results — and where
    the *rendering* is the thing being checked.
    """
    if actual == expected:
        return True, ""
    return False, f"latex {actual!r} != {expected!r}"


def compare_numeric(actual: Any, expected: Any, args: Mapping[str, Any]) -> tuple[bool, str]:
    """``numeric``: sample both answers on a deterministic grid inside ``domain``."""
    tolerance = float(args.get("tolerance", NUMERIC_TOLERANCE))
    lower, upper = (float(bound) for bound in args.get("domain", NUMERIC_DOMAIN))
    variable = _symbol(args)
    free = (
        (actual.free_symbols | expected.free_symbols) if hasattr(actual, "free_symbols") else set()
    )
    if not free:
        try:
            difference = complex(sympy.N(actual - expected, 25))
        except (TypeError, ValueError, ZeroDivisionError, AttributeError) as exc:
            return False, f"numeric evaluation failed: {type(exc).__name__}"
        if cmath.isnan(difference) or cmath.isinf(difference):
            # zoo / nan / oo: the metric abstains instead of claiming a difference
            return False, "numeric checker could not evaluate the difference"
        gap = abs(difference)
        return (gap <= tolerance, "" if gap <= tolerance else f"numeric gap {gap:.3e}")
    evaluated = 0
    for index in range(NUMERIC_SAMPLES):
        point = lower + (upper - lower) * index / (NUMERIC_SAMPLES - 1)
        substitution = {variable: sympy.Rational(point).limit_denominator(1000)}
        try:
            value = complex(sympy.N((actual - expected).subs(substitution), 25))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if cmath.isnan(value) or cmath.isinf(value):  # a pole: this probe abstains
            continue
        evaluated += 1
        if abs(value) > tolerance:
            return False, f"numeric gap {abs(value):.3e} at {point:.3f}"
    if evaluated == 0:
        return False, "numeric checker could not evaluate any probe"
    return True, ""


def compare(checker: str, actual: Any, expected: Any, args: Mapping[str, Any]) -> tuple[bool, str]:
    """Compare an engine answer with the authored answer under one strategy.

    Args:
        checker: ``symbolic``, ``numeric``, ``set``, ``bool`` or ``plot_ok``; ``exact_latex``
            works on strings and lives in :func:`compare_latex`.
        actual: What the engine produced (a SymPy object).
        expected: What the dataset row says the answer is (a SymPy object).
        args: The row's ``checker_args``.

    Returns:
        ``(ok, reason)``; ``reason`` is empty when ``ok`` is True and carries the abstention
        verdict (``"symbolic: unparsable"``) when the comparison could not decide.
    """
    if actual == expected:
        # Identity first: symbolic_equivalent(oo, oo) abstains, and an exact match never needs
        # simplification anyway.
        return True, ""
    if checker == "bool":
        return False, f"bool {actual} != {expected}"
    if checker == "numeric":
        return compare_numeric(actual, expected, args)
    if checker == "symbolic" and args.get("up_to_constant"):
        variable = _symbol(args)
        verdict = symbolic_equivalent(sympy.diff(actual, variable), sympy.diff(expected, variable))
        if verdict == "equal":
            return True, ""
        return False, f"antiderivatives differ by more than a constant ({verdict})"
    verdict = symbolic_equivalent(actual, expected)
    if verdict == "equal":
        return True, ""
    if verdict == "different":
        # SymPy's Symbol('x') and Symbol('x', real=True) are *different symbols*, so an engine
        # answer built with assumptions never simplifies against a hand-written one without
        # them: 2*x - 2*x stays non-zero and the probes disagree. Retrying once with the
        # assumptions dropped can only turn a nominal difference into equality (a genuinely
        # different pair stays different), so it removes a class of false "wrong answers" from
        # the headline CAS metric without weakening it.
        stripped = symbolic_equivalent(*_strip_assumptions(actual, expected))
        if stripped == "equal":
            return True, ""
    return False, f"{checker}: {verdict}"


def _strip_assumptions(*expressions: Any) -> tuple[Any, ...]:
    """Rewrite every free symbol as a plain ``Symbol(name)`` in each expression."""
    out: list[Any] = []
    for expression in expressions:
        symbols: set[Any] = getattr(expression, "free_symbols", set())
        mapping = {symbol: sympy.Symbol(symbol.name) for symbol in symbols}
        out.append(expression.subs(mapping) if mapping else expression)
    return tuple(out)


def check_antiderivative(actual: Any, integrand_latex: str, args: Mapping[str, Any]) -> str:
    """For an indefinite integral, verify the answer really differentiates to the integrand.

    Args:
        actual: The engine's antiderivative.
        integrand_latex: The ``expr`` field of the row's command.
        args: The row's ``checker_args`` (``var``).

    Returns:
        An empty string when it does, otherwise the problem description.
    """
    variable = _symbol(args)
    try:
        integrand = latex_to_sympy(integrand_latex)
    except AiclsError as exc:
        return f"integrand did not parse: {type(exc).__name__}: {exc}"[:160]
    verdict = symbolic_equivalent(sympy.diff(actual, variable), integrand)
    if verdict == "equal":
        return ""
    return f"d/d{variable} of the answer is not the integrand ({verdict})"


__all__ = [
    "NUMERIC_DOMAIN",
    "NUMERIC_SAMPLES",
    "NUMERIC_TOLERANCE",
    "check_antiderivative",
    "compare",
    "compare_latex",
    "compare_numeric",
    "rebuild_srepr",
]
