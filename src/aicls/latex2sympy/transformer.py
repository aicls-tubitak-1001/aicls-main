"""Lark parse tree -> typed AST (docs/mimari/latex-ayristirici.md 3.3-3.4).

The transformer is where the two "reading rules" of the grammar become explicit:

* ``Add``/``Mul`` chains are **flattened**, so ``2x\\sin x`` is one ``Mul`` with three factors
  and the canonical printer can reproduce the very same tree;
* the **differential of an integral** is recognised here rather than in the lexer.  An LALR
  state that accepts a ``dx`` terminal is shared with every other "after an atom" state, so a
  contextual differential terminal would also swallow ``d x`` outside integrals; instead the
  greedy integral body is split at the ``d`` factor (:func:`_split_differential`), which also
  gives ``\\int x\\,dx + 1`` and ``\\int x^{2}+1\\,dx`` their conventional readings.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from lark import Token, Transformer, v_args
from lark.tree import Meta

from aicls.latex2sympy.ast import (
    NO_SPAN,
    Abs,
    Add,
    Call,
    Const,
    Derivative,
    Div,
    Factorial,
    Func,
    Integral,
    Interval,
    Limit,
    Log,
    MathSet,
    Matrix,
    Mul,
    Neg,
    Node,
    Num,
    PlusMinus,
    Pow,
    Prime,
    Product,
    Relation,
    Root,
    Span,
    Sum,
    Sym,
)
from aicls.latex2sympy.errors import SemanticError

_LEIBNIZ_RE = re.compile(
    r"\\frac\s*\{\s*(?P<num>\\partial|d)\s*"
    r"(?:\^\s*(?:\{\s*(?P<order_braced>\d+)\s*\}|(?P<order_plain>\d+)))?\s*\}\s*"
    r"\{\s*(?:\\partial|d)\s*(?P<var>\\[a-zA-Z]+|[A-Za-z])"
)
_LEIBNIZ_OF_RE = re.compile(
    r"\\frac\s*\{\s*(?P<num>\\partial|d)\s*"
    r"(?:\^\s*(?:\{\s*(?P<order_braced>\d+)\s*\}|(?P<order_plain>\d+)))?\s*"
    r"(?P<body>\\[a-zA-Z]+|[A-Za-z])\s*\}\s*"
    r"\{\s*(?:\\partial|d)\s*(?P<var>\\[a-zA-Z]+|[A-Za-z])"
)
_ENV_RE = re.compile(r"\{\s*([A-Za-z]+)\s*\}")

#: Relation token type -> canonical operator string stored on :class:`Relation`.
RELATION_OPS: dict[str, str] = {
    "EQ": "=",
    "LT": "<",
    "GT": ">",
    "LE": "<=",
    "GE": ">=",
    "NEQ": "!=",
    "APPROX": "~=",
    "IN": "in",
}

#: The differential variable letter (``dx`` -> ``d`` + ``x``).
DIFFERENTIAL_LETTER = "d"


@dataclass(frozen=True, slots=True)
class _Split:
    """Result of splitting an integral body at its differential."""

    integrand: Node
    var: Sym
    extra_terms: tuple[Node, ...] = ()
    extra_factors: tuple[Node, ...] = ()


def _flatten_add(left: Node, right: Node, span: Span) -> Add:
    terms = left.terms if isinstance(left, Add) else (left,)
    return Add((*terms, right), span)


def _flatten_mul(left: Node, right: Node, span: Span) -> Mul:
    factors = left.factors if isinstance(left, Mul) else (left,)
    return Mul((*factors, right), span)


def _find_differential(factors: tuple[Node, ...]) -> tuple[int, Sym] | None:
    """Index of the ``d`` factor and the differential variable, if the product carries one."""
    for index in range(len(factors) - 1):
        marker = factors[index]
        variable = factors[index + 1]
        if (
            isinstance(marker, Sym)
            and marker.sub is None
            and marker.name == DIFFERENTIAL_LETTER
            and isinstance(variable, Sym)
            and variable.sub is None
        ):
            return index, variable
    return None


def _split_mul(node: Node) -> _Split | None:
    """Split a single (possibly implicit) product at ``d<var>``."""
    factors = node.factors if isinstance(node, Mul) else (node,)
    found = _find_differential(factors)
    if found is None:
        return None
    index, variable = found
    before = factors[:index]
    after = factors[index + 2 :]
    if not before:
        integrand: Node = Num("1")
    elif len(before) == 1:
        integrand = before[0]
    else:
        integrand = Mul(before)
    return _Split(integrand=integrand, var=variable, extra_factors=after)


def _split_differential(body: Node) -> _Split | None:
    """Find the differential in an integral body and split the expression around it."""
    if isinstance(body, Add):
        for index, term in enumerate(body.terms):
            inner = _split_mul(term.x if isinstance(term, Neg) else term)
            if inner is None:
                continue
            integrand_part = Neg(inner.integrand) if isinstance(term, Neg) else inner.integrand
            head = (*body.terms[:index], integrand_part)
            integrand: Node = head[0] if len(head) == 1 else Add(head)
            return _Split(
                integrand=integrand,
                var=inner.var,
                extra_terms=body.terms[index + 1 :],
                extra_factors=inner.extra_factors,
            )
        return None
    if isinstance(body, Neg):
        inner = _split_mul(body.x)
        if inner is None:
            return None
        return _Split(
            integrand=Neg(inner.integrand),
            var=inner.var,
            extra_factors=inner.extra_factors,
        )
    return _split_mul(body)


def _span(meta: Meta) -> Span:
    if getattr(meta, "empty", True):
        return NO_SPAN
    return Span(meta.start_pos, meta.end_pos)


@v_args(meta=True)
class LatexTransformer(Transformer[Token, Any]):
    """Turn the Lark tree into the typed AST.

    Args:
        src: The *normalised* source, used to recover the literal text of a subscript.
    """

    def __init__(self, src: str) -> None:
        super().__init__()
        self._src = src

    # -- helpers ------------------------------------------------------------------------
    def _text(self, node: Node) -> str:
        span = node.span
        if span.end > span.start:
            return self._src[span.start : span.end].strip()
        return ""

    # -- entry --------------------------------------------------------------------------
    def start(self, meta: Meta, children: list[Any]) -> Node:
        node: Node = children[0]
        return node

    # -- leaves -------------------------------------------------------------------------
    def number(self, meta: Meta, children: list[Any]) -> Node:
        return Num(str(children[0]), _span(meta))

    def symbol(self, meta: Meta, children: list[Any]) -> Node:
        return Sym(str(children[0]), None, _span(meta))

    def greek(self, meta: Meta, children: list[Any]) -> Node:
        name = str(children[0])[1:]
        if name == "pi":
            return Const("pi", _span(meta))
        return Sym(name, None, _span(meta))

    def infty(self, meta: Meta, children: list[Any]) -> Node:
        return Const("infty", _span(meta))

    def mathbb(self, meta: Meta, children: list[Any]) -> Node:
        match = _ENV_RE.search(str(children[0]))
        name = match.group(1) if match else "R"
        return MathSet(name, _span(meta))

    # -- grouping -----------------------------------------------------------------------
    def group(self, meta: Meta, children: list[Any]) -> Node:
        node: Node = children[0]
        return node

    def paren(self, meta: Meta, children: list[Any]) -> Node:
        node: Node = children[0]
        return node

    def interval_oo(self, meta: Meta, children: list[Any]) -> Node:
        return Interval(children[0], children[1], True, True, _span(meta))

    def interval_cc(self, meta: Meta, children: list[Any]) -> Node:
        return Interval(children[0], children[1], False, False, _span(meta))

    def interval_co(self, meta: Meta, children: list[Any]) -> Node:
        return Interval(children[0], children[1], False, True, _span(meta))

    def interval_oc(self, meta: Meta, children: list[Any]) -> Node:
        return Interval(children[0], children[1], True, False, _span(meta))

    # -- arithmetic ---------------------------------------------------------------------
    def add(self, meta: Meta, children: list[Any]) -> Node:
        return _flatten_add(children[0], children[1], _span(meta))

    def sub(self, meta: Meta, children: list[Any]) -> Node:
        return _flatten_add(children[0], Neg(children[1], children[1].span), _span(meta))

    def plus_minus(self, meta: Meta, children: list[Any]) -> Node:
        return PlusMinus(children[0], children[1], _span(meta))

    def mul(self, meta: Meta, children: list[Any]) -> Node:
        return _flatten_mul(children[0], children[1], _span(meta))

    def div(self, meta: Meta, children: list[Any]) -> Node:
        return Div(children[0], children[1], _span(meta))

    def neg(self, meta: Meta, children: list[Any]) -> Node:
        return Neg(children[0], _span(meta))

    def pos(self, meta: Meta, children: list[Any]) -> Node:
        node: Node = children[0]
        return node

    def pow(self, meta: Meta, children: list[Any]) -> Node:
        return Pow(children[0], children[1], _span(meta))

    def subscript(self, meta: Meta, children: list[Any]) -> Node:
        base, index = children[0], children[1]
        if not isinstance(base, Sym) or base.sub is not None:
            raise SemanticError("subscript_on_non_symbol", details={"base": type(base).__name__})
        text = self._text(index)
        if not text:
            raise SemanticError("empty_subscript")
        return Sym(base.name, text, _span(meta))

    def factorial(self, meta: Meta, children: list[Any]) -> Node:
        return Factorial(children[0], _span(meta))

    def prime(self, meta: Meta, children: list[Any]) -> Node:
        base, token = children[0], children[1]
        if not isinstance(base, Sym) or base.sub is not None:
            raise SemanticError("prime_on_non_symbol", details={"base": type(base).__name__})
        return Prime(base.name, None, len(str(token)), _span(meta))

    # -- functions ----------------------------------------------------------------------
    def frac(self, meta: Meta, children: list[Any]) -> Node:
        return Div(children[0], children[1], _span(meta))

    def sqrt(self, meta: Meta, children: list[Any]) -> Node:
        return Root(children[0], None, _span(meta))

    def root(self, meta: Meta, children: list[Any]) -> Node:
        return Root(children[1], children[0], _span(meta))

    def func(self, meta: Meta, children: list[Any]) -> Node:
        return Func(str(children[0])[1:], children[1], None, _span(meta))

    def func_pow(self, meta: Meta, children: list[Any]) -> Node:
        return Func(str(children[0])[1:], children[2], children[1], _span(meta))

    def logf(self, meta: Meta, children: list[Any]) -> Node:
        return Log(children[0], None, _span(meta))

    def log_base(self, meta: Meta, children: list[Any]) -> Node:
        return Log(children[1], children[0], _span(meta))

    def arglist(self, meta: Meta, children: list[Any]) -> tuple[Node, ...]:
        return tuple(children)

    def call(self, meta: Meta, children: list[Any]) -> Node:
        raw = str(children[0])
        args: tuple[Node, ...] = children[1]
        name = raw.rstrip("'")
        primes = len(raw) - len(name)
        if primes:
            if len(args) != 1:
                raise SemanticError("prime_arity", details={"args": len(args)})
            return Prime(name, args[0], primes, _span(meta))
        return Call(name, args, _span(meta))

    def opname_call(self, meta: Meta, children: list[Any]) -> Node:
        match = _ENV_RE.search(str(children[0]))
        name = match.group(1) if match else "f"
        return Call(name, children[1], _span(meta))

    def abs_expr(self, meta: Meta, children: list[Any]) -> Node:
        return Abs(children[0], _span(meta))

    # -- matrices -----------------------------------------------------------------------
    def mrow(self, meta: Meta, children: list[Any]) -> tuple[Node, ...]:
        return tuple(children)

    def matrix(self, meta: Meta, children: list[Any]) -> Node:
        begin, end = str(children[0]), str(children[-1])
        begin_match, end_match = _ENV_RE.search(begin), _ENV_RE.search(end)
        kind = begin_match.group(1) if begin_match else "matrix"
        if end_match is None or end_match.group(1) != kind:
            raise SemanticError("matrix_environment_mismatch", details={"begin": kind})
        rows: tuple[tuple[Node, ...], ...] = tuple(children[1:-1])
        if not rows or len({len(row) for row in rows}) != 1:
            raise SemanticError("matrix_ragged_rows", details={"rows": len(rows)})
        return Matrix(rows, kind, _span(meta))

    # -- relations ----------------------------------------------------------------------
    def rel_op(self, meta: Meta, children: list[Any]) -> str:
        token: Token = children[0]
        return RELATION_OPS[token.type]

    def relation(self, meta: Meta, children: list[Any]) -> Node:
        return Relation(children[1], children[0], children[2], _span(meta))

    # -- big operators ------------------------------------------------------------------
    def _integral(self, body: Node, lower: Node | None, upper: Node | None, span: Span) -> Node:
        split = _split_differential(body)
        if split is None:
            raise SemanticError(
                "integral_without_differential",
                details={"hint": "dx"},
            )
        node: Node = Integral(split.integrand, split.var, lower, upper, span)
        if split.extra_factors:
            node = Mul((node, *split.extra_factors), span)
        if split.extra_terms:
            node = Add((node, *split.extra_terms), span)
        return node

    def integral(self, meta: Meta, children: list[Any]) -> Node:
        return self._integral(children[0], None, None, _span(meta))

    def integral_bounds(self, meta: Meta, children: list[Any]) -> Node:
        return self._integral(children[2], children[0], children[1], _span(meta))

    def summation(self, meta: Meta, children: list[Any]) -> Node:
        variable = Sym(str(children[0]))
        return Sum(children[4], variable, children[2], children[3], _span(meta))

    def production(self, meta: Meta, children: list[Any]) -> Node:
        variable = Sym(str(children[0]))
        return Product(children[4], variable, children[2], children[3], _span(meta))

    def limit(self, meta: Meta, children: list[Any]) -> Node:
        return self._limit(children[0], children[1], "+-", children[2], _span(meta))

    def limit_dir(self, meta: Meta, children: list[Any]) -> Node:
        direction = "-" if "-" in str(children[2]) else "+"
        return self._limit(children[0], children[1], direction, children[3], _span(meta))

    def _limit(self, var: Node, to: Node, direction: str, body: Node, span: Span) -> Node:
        if not isinstance(var, Sym):
            raise SemanticError("limit_variable", details={"var": type(var).__name__})
        return Limit(body, var, to, direction, span)

    def leibniz(self, meta: Meta, children: list[Any]) -> Node:
        order, var, partial = _parse_leibniz(str(children[0]))
        return Derivative(children[1], var, order, partial, _span(meta))

    def leibniz_of(self, meta: Meta, children: list[Any]) -> Node:
        order, var, partial, body = _parse_leibniz_of(str(children[0]))
        return Derivative(body, var, order, partial, _span(meta))

    def scaled_bigop(self, meta: Meta, children: list[Any]) -> Node:
        return _flatten_mul(children[0], children[1], _span(meta))


def _leibniz_order(match: re.Match[str]) -> int:
    """Differentiation order of a Leibniz token (``d^{2}``, ``d^2`` or none -> 1)."""
    return int(match.group("order_braced") or match.group("order_plain") or "1")


def _leibniz_symbol(raw: str) -> Sym:
    """``\\theta`` or ``x`` (as written inside a Leibniz token) -> :class:`Sym`."""
    return Sym(raw.lstrip("\\"))


def _parse_leibniz(text: str) -> tuple[int, Sym, bool]:
    """Read order, variable and partial-ness out of a ``\\frac{d}{dx}`` token."""
    match = _LEIBNIZ_RE.match(text)
    if match is None:  # pragma: no cover - the terminal guarantees a match
        raise SemanticError("leibniz_operator", details={"text": text})
    return _leibniz_order(match), _leibniz_symbol(match.group("var")), _is_partial(match)


def _parse_leibniz_of(text: str) -> tuple[int, Sym, bool, Sym]:
    """Same as :func:`_parse_leibniz` plus the body, for ``\\frac{\\partial f}{\\partial x}``.

    The numerator decides both the order and the partial-ness, exactly as in the operator
    form, so ``\\frac{d^{2}y}{dx^{2}}`` is a second-order total derivative of ``y`` and
    ``\\frac{\\partial f}{\\partial x}`` a first-order partial derivative of ``f``.
    """
    match = _LEIBNIZ_OF_RE.match(text)
    if match is None:  # pragma: no cover - the terminal guarantees a match
        raise SemanticError("leibniz_shorthand", details={"text": text})
    return (
        _leibniz_order(match),
        _leibniz_symbol(match.group("var")),
        _is_partial(match),
        _leibniz_symbol(match.group("body")),
    )


def _is_partial(match: re.Match[str]) -> bool:
    """``\\partial`` in the numerator marks a partial derivative."""
    return match.group("num") == "\\partial"


__all__ = ["RELATION_OPS", "LatexTransformer"]
