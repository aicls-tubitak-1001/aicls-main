"""Property-based round-trip tests for aicls.latex2sympy (ADR-0004, ADR-0010).

The strategy ``st_expr`` builds *parser-shaped* ASTs: nested ``Add``/``Mul`` are flattened
the way the grammar flattens them, so ``parse_latex(to_latex(node)) == node`` is an exact
identity rather than an approximation.
"""

from __future__ import annotations

import contextlib

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
import pytest
import sympy

from aicls.core.errors import AiclsError
from aicls.latex2sympy import (
    Abs,
    Add,
    Div,
    Func,
    Mul,
    Neg,
    Node,
    Num,
    Pow,
    Root,
    Sym,
    normalize_latex,
    parse_latex,
    roundtrip_check,
    to_latex,
    to_sympy,
)

pytestmark = pytest.mark.property

FUNCTION_NAMES = ("sin", "cos", "tan", "ln", "exp")
VARIABLES = ("x", "y", "z", "t", "a", "b")

st_num = st.integers(min_value=0, max_value=20).map(lambda i: Num(str(i)))
st_nonzero_num = st.integers(min_value=1, max_value=9).map(lambda i: Num(str(i)))
st_sym = st.sampled_from(VARIABLES).map(Sym)
st_leaf: st.SearchStrategy[Node] = st.one_of(st_num, st_sym)


def _flat_terms(children: list[Node]) -> tuple[Node, ...]:
    """Reject nested Add so the printed form re-parses into the very same node."""
    assume(not any(isinstance(child, Add) for child in children))
    return tuple(children)


def _flat_factors(children: list[Node]) -> tuple[Node, ...]:
    assume(not any(isinstance(child, Mul) for child in children))
    # only the leading factor may be a negation (``-x y`` parses as Mul(Neg(x), y))
    assume(not any(isinstance(child, Neg) for child in children[1:]))
    return tuple(children)


def st_expr(max_leaves: int = 12) -> st.SearchStrategy[Node]:
    """Curriculum-scoped random expressions (depth <= 5, no singular denominators)."""

    def extend(children: st.SearchStrategy[Node]) -> st.SearchStrategy[Node]:
        return st.one_of(
            st.lists(children, min_size=2, max_size=3).map(_flat_terms).map(Add),
            st.lists(children, min_size=2, max_size=3).map(_flat_factors).map(Mul),
            children.map(Neg),
            st.tuples(children, st.one_of(st_nonzero_num, st_sym)).map(
                lambda pair: Div(pair[0], pair[1])
            ),
            st.tuples(children, st.integers(min_value=1, max_value=3)).map(
                lambda pair: Pow(pair[0], Num(str(pair[1])))
            ),
            st.tuples(st.sampled_from(FUNCTION_NAMES), children).map(
                lambda pair: Func(pair[0], pair[1])
            ),
            children.map(Abs),
            st.tuples(children, st.integers(min_value=2, max_value=3)).map(
                lambda pair: Root(pair[0], Num(str(pair[1])))
            ),
        )

    return st.recursive(st_leaf, extend, max_leaves=max_leaves)


PROFILE = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@PROFILE
@given(st_expr())
def test_property_roundtrip_ast_to_latex_to_ast(node: Node) -> None:
    printed = to_latex(node)
    assert parse_latex(printed) == node
    # printing is stable: a second pass produces the identical string
    assert to_latex(parse_latex(printed)) == printed


@PROFILE
@given(st_expr(max_leaves=8))
def test_property_latex_to_sympy_symbolically_equivalent_after_reprint(node: Node) -> None:
    src = to_latex(node)
    # Complex infinity, NaN and the imaginary unit are outside the v1 grammar: sympy prints
    # them as \tilde{\infty}/\text{NaN}/i, which parse_latex rejects (or reads as a symbol,
    # because ParserConfig.imaginary_i defaults to False). See "Uygulama notlari".
    value = to_sympy(parse_latex(src))
    assume(not value.has(sympy.zoo, sympy.nan, sympy.I))
    result = roundtrip_check(src, numeric_samples=4)
    assert result.ok is True, (src, result.reason)


LATEX_ALPHABET = "abxyzn0123456789\\{}[]()^_+-*/,.|=<>&~$! \t"


@PROFILE
@given(st.text(alphabet=LATEX_ALPHABET, max_size=40))
def test_property_parse_never_raises_non_aicls_error(src: str) -> None:
    with contextlib.suppress(AiclsError):
        parse_latex(src)


@PROFILE
@given(st.text(alphabet=LATEX_ALPHABET + "×·−²³√∞π≤≥≠→çöğü", max_size=40))
def test_property_normalize_latex_idempotent(src: str) -> None:
    once = normalize_latex(src).text
    assert normalize_latex(once).text == once


@PROFILE
@given(
    st.text(
        alphabet=st.characters(codec="utf-8", exclude_categories=("Cs",)),
        max_size=30,
    )
)
def test_property_parse_never_raises_on_arbitrary_unicode(src: str) -> None:
    with contextlib.suppress(AiclsError):
        parse_latex(src)
