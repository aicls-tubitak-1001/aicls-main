"""Lark tree -> :mod:`aicls.latex2sympy.ast` nodes.

Building the *typed AST* rather than a LaTeX string is what makes the spoken-math path safe: the
canonical printer (:func:`aicls.latex2sympy.to_latex`) then emits KaTeX-safe LaTeX that
``parse_latex`` accepts by construction, so the NLU layer can never hand the CAS a string it
made up. ``\\infty`` for "sonsuz", ``e`` for the constant and ``\\alpha`` for "alfa" all come out
of the same table the LaTeX front end uses.
"""

from __future__ import annotations

from typing import Any

from lark import Token, Transformer

from aicls.latex2sympy.ast import (
    Abs,
    Add,
    Const,
    Div,
    Factorial,
    Func,
    Log,
    Mul,
    Neg,
    Node,
    Num,
    Pow,
    Relation,
    Root,
    Sym,
)
from aicls.nlu.errors import SpokenMathError

#: Spoken constant word -> :class:`aicls.latex2sympy.ast.Const` name.
CONST_NAMES: dict[str, str] = {"pi": "pi", "sonsuz": "infty"}

#: Single letters that are read as a constant rather than a free symbol.
LETTER_CONSTANTS: dict[str, str] = {"e": "e"}

#: Function words the grammar accepts that map to :class:`~aicls.latex2sympy.ast.Log`.
LOG_FUNCTIONS = frozenset({"log"})


def _num(value: str) -> Num:
    return Num(value)


class SpokenMathTransformer(Transformer[Token, Node]):
    """Turn the spoken-math parse tree into AST nodes."""

    # -- leaves ------------------------------------------------------------------------
    def number(self, children: list[Token]) -> Node:
        """``123`` / ``2.5`` -> :class:`Num`."""
        return _num(str(children[0]))

    def symbol(self, children: list[Token]) -> Node:
        """A single latin letter -> :class:`Sym`, except ``e`` which is the constant."""
        name = str(children[0])
        constant = LETTER_CONSTANTS.get(name)
        if constant is not None:
            return Const(constant)
        return Sym(name)

    def greek(self, children: list[Token]) -> Node:
        """``alpha`` / ``theta`` -> :class:`Sym` with the LaTeX greek name."""
        return Sym(str(children[0]))

    def const(self, children: list[Token]) -> Node:
        """``pi`` / ``sonsuz`` -> :class:`Const`."""
        name = str(children[0])
        return Const(CONST_NAMES[name])

    # -- arithmetic --------------------------------------------------------------------
    def add(self, children: list[Node]) -> Node:
        """``a artı b`` -> flat :class:`Add`."""
        return self._flat_add(children[0], children[1])

    def sub(self, children: list[Node]) -> Node:
        """``a eksi b`` -> :class:`Add` with a :class:`Neg` term."""
        return self._flat_add(children[0], Neg(children[1]))

    def raised(self, children: list[Node]) -> Node:
        """``a üssü b`` / ``a üzeri b`` -> :class:`Pow`."""
        return Pow(children[0], children[1])

    @staticmethod
    def _flat_add(left: Node, right: Node) -> Node:
        if isinstance(left, Add):
            return Add((*left.terms, right))
        return Add((left, right))

    def mul(self, children: list[Node]) -> Node:
        """``a çarpı b`` -> flat :class:`Mul`."""
        return self._flat_mul(children[0], children[1])

    def imul(self, children: list[Node]) -> Node:
        """Implicit product (``iki x``) -> flat :class:`Mul`.

        A *numeric* right factor is refused. Nobody says "iki üç" for ``2·3``; two bare numerals
        next to each other mean the numeral folder shortened a run it could not read, or the
        speaker said something the layer must ask about. Reading them as a product is how
        ``"x kare eksi dört sıfır denklemini çöz"`` became ``x² − 4·0`` -- a confidently computed
        answer to a question nobody asked (KNOWN_ISSUES PRIORITY 1).

        Raises:
            aicls.nlu.errors.SpokenMathError: If the right factor is a bare number.
        """
        right = children[1]
        if isinstance(right, Num):
            raise SpokenMathError(str(right.value), reason="adjacent_numerals")
        return self._flat_mul(children[0], right)

    @staticmethod
    def _flat_mul(left: Node, right: Node) -> Node:
        if isinstance(left, Mul):
            return Mul((*left.factors, right))
        return Mul((left, right))

    def div(self, children: list[Node]) -> Node:
        """``a bölü b`` -> :class:`Div` (binds the two neighbouring powers only)."""
        return Div(children[0], children[1])

    def neg(self, children: list[Node]) -> Node:
        """``eksi a`` -> :class:`Neg`."""
        return Neg(children[0])

    def square(self, children: list[Any]) -> Node:
        """``a kare`` -> ``a^{2}``."""
        return Pow(children[0], _num("2"))

    def cube(self, children: list[Any]) -> Node:
        """``a küp`` -> ``a^{3}``."""
        return Pow(children[0], _num("3"))

    def factorial(self, children: list[Any]) -> Node:
        """``a faktöriyel`` -> :class:`Factorial`."""
        return Factorial(children[0])

    # -- functions and grouping --------------------------------------------------------
    def func(self, children: list[Any]) -> Node:
        """``sin x`` -> :class:`Func`; ``log x`` -> :class:`Log`."""
        name = str(children[0])
        argument = children[1]
        if name in LOG_FUNCTIONS:
            return Log(argument)
        return Func(name, argument)

    def sqrt(self, children: list[Any]) -> Node:
        """``karekök x`` -> ``\\sqrt{x}``."""
        return Root(children[0])

    def cbrt(self, children: list[Any]) -> Node:
        """``küpkök x`` -> ``\\sqrt[3]{x}``."""
        return Root(children[0], _num("3"))

    def absolute(self, children: list[Any]) -> Node:
        """``mutlak değer x`` -> ``\\lvert x \\rvert``."""
        return Abs(children[0])

    def paren(self, children: list[Any]) -> Node:
        """``parantez aç ... parantez kapa`` -> the inner expression (printer re-brackets)."""
        node = children[0]
        assert isinstance(node, Node)
        return node

    def equation(self, children: list[Any]) -> Node:
        """``a eşittir b`` -> :class:`Relation`."""
        return Relation("=", children[0], children[1])

    def start(self, children: list[Node]) -> Node:
        """Unwrap the single child of ``start``."""
        return children[0]


__all__ = ["CONST_NAMES", "LETTER_CONSTANTS", "LOG_FUNCTIONS", "SpokenMathTransformer"]
