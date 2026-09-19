"""Cheap solution-step narration.

The proposal promises *steps*, not a proof assistant. Producing a full derivation for an
arbitrary expression is a research problem of its own, so the policy here is deliberately
narrow and stated in the module contract: narrate the two rules a first-year Calculus student
is actually asked to name -- the **product rule** and the **chain rule** -- and return an empty
list for everything else. A missing narration is a normal, documented outcome, never an error.

Narration is *structural*: it reads the SymPy tree that the user's own input produced, so the
``u`` and ``v`` in the story are the factors the user wrote. It never calls ``simplify``, which
is what keeps it inside the latency budget.
"""

from __future__ import annotations

import sympy

from aicls.latex2sympy import sympy_to_latex
from aicls.mathengine.messages import tr
from aicls.mathengine.models import Step

#: Largest number of factors a product-rule narration is produced for.
MAX_PRODUCT_FACTORS = 2


def _latex(expr: sympy.Basic) -> str:
    return sympy_to_latex(expr)


def _derivative_notation(expr: sympy.Basic, var: sympy.Symbol) -> str:
    return rf"\frac{{d}}{{d{var.name}}}\left({_latex(expr)}\right)"


def _split_product(expr: sympy.Mul, var: sympy.Symbol) -> tuple[sympy.Expr, sympy.Expr] | None:
    """Split ``expr`` into exactly two ``var``-dependent factors, folding constants into the first.

    Returns None when the product does not have exactly two variable factors -- ``2x`` is a
    constant multiple (the constant-multiple rule, not the product rule) and ``x y z`` needs a
    nested narration the policy does not promise.
    """
    variable_factors = [factor for factor in expr.args if factor.has(var)]
    constant_factors = [factor for factor in expr.args if not factor.has(var)]
    if len(variable_factors) != MAX_PRODUCT_FACTORS:
        return None
    first, second = variable_factors
    if constant_factors:
        first = sympy.Mul(*constant_factors, first)
    return first, second


def _product_rule(expr: sympy.Basic, var: sympy.Symbol) -> list[Step]:
    if not isinstance(expr, sympy.Mul):
        return []
    split = _split_product(expr, var)
    if split is None:
        return []
    u, v = split
    du = sympy.diff(u, var)
    dv = sympy.diff(v, var)
    rule = (
        rf"{_derivative_notation(expr, var)} = "
        rf"{_derivative_notation(u, var)} \cdot {_latex(v)} + "
        rf"{_latex(u)} \cdot {_derivative_notation(v, var)}"
    )
    applied = rf"= {_latex(du)} \cdot {_latex(v)} + {_latex(u)} \cdot {_latex(dv)}"
    return [
        Step(title_tr=tr("STEP_PRODUCT_RULE"), latex=rule, rule="product_rule"),
        Step(title_tr=tr("STEP_DERIVATIVE"), latex=applied, rule="product_rule_applied"),
    ]


def _chain_parts(expr: sympy.Basic, var: sympy.Symbol) -> tuple[sympy.Expr, sympy.Expr] | None:
    """Return ``(outer(u), inner)`` when ``expr`` is a genuine composition in ``var``."""
    placeholder = sympy.Symbol("u")
    if isinstance(expr, sympy.Pow):
        base, exponent = expr.args
        if exponent.has(var) or not base.has(var) or base == var:
            return None
        return sympy.Pow(placeholder, exponent), base
    if isinstance(expr, sympy.Function) and len(expr.args) == 1:
        inner = expr.args[0]
        if inner == var or not inner.has(var):
            return None
        return expr.func(placeholder), inner
    return None


def _chain_rule(expr: sympy.Basic, var: sympy.Symbol) -> list[Step]:
    parts = _chain_parts(expr, var)
    if parts is None:
        return []
    outer, inner = parts
    placeholder = sympy.Symbol("u")
    outer_derivative = sympy.diff(outer, placeholder).subs(placeholder, inner)
    inner_derivative = sympy.diff(inner, var)
    rule = (
        rf"{_derivative_notation(expr, var)} = "
        rf"\frac{{d}}{{du}}\left({_latex(outer)}\right) \cdot "
        rf"{_derivative_notation(inner, var)}, \quad u = {_latex(inner)}"
    )
    applied = rf"= {_latex(outer_derivative)} \cdot {_latex(inner_derivative)}"
    return [
        Step(title_tr=tr("STEP_CHAIN_RULE"), latex=rule, rule="chain_rule"),
        Step(title_tr=tr("STEP_DERIVATIVE"), latex=applied, rule="chain_rule_applied"),
    ]


def diff_steps(expr: sympy.Basic, var: sympy.Symbol, value: sympy.Basic) -> list[Step]:
    """Narrate a first-order derivative when the product or chain rule applies.

    Args:
        expr: The expression that was differentiated (as the user wrote it).
        var: The differentiation variable.
        value: The derivative SymPy computed; it closes the narration.

    Returns:
        Two or three :class:`~aicls.mathengine.models.Step` objects, or an empty list when
        neither rule applies -- which is the documented, non-exceptional default.
    """
    steps = _product_rule(expr, var) or _chain_rule(expr, var)
    if not steps:
        return []
    steps.append(Step(title_tr=tr("STEP_RESULT"), latex=_latex(value), rule="result"))
    return steps


__all__ = ["MAX_PRODUCT_FACTORS", "diff_steps"]
