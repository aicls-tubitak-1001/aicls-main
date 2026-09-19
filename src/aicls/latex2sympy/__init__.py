"""LaTeX -> typed AST -> SymPy via a Lark grammar, canonical printer and round-trip.

Public entry points (docs/mimari/moduller.md, ADR-0004)::

    parse_latex(src, *, strict=True) -> Node
    to_sympy(node, symbols=None) -> sympy.Basic
    latex_to_sympy(src, *, symbols=None) -> sympy.Basic
    to_latex(node) -> str
    sympy_to_latex(expr, *, style=PrintStyle.KATEX) -> str
    normalize_latex(src) -> NormalizeResult
    roundtrip_check(src, *, numeric_samples=5) -> RoundTripResult

Everything raises only :class:`aicls.core.errors.AiclsError` subclasses:
:class:`ParseError`, :class:`UnsupportedConstruct`, :class:`SemanticError` and
``InputValidationError`` for the :data:`MAX_INPUT_CHARS` limit.
"""

from __future__ import annotations

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
    Visitor,
)
from aicls.latex2sympy.errors import ParseError, SemanticError, UnsupportedConstruct
from aicls.latex2sympy.normalize import NormalizeResult, normalize_latex
from aicls.latex2sympy.parser import MAX_INPUT_CHARS, LatexParser, parse_latex
from aicls.latex2sympy.printer import PrintStyle, sympy_to_latex
from aicls.latex2sympy.roundtrip import RoundTripResult, roundtrip_check
from aicls.latex2sympy.symbols import DEFAULT_CONFIG, ParserConfig, SymbolTable
from aicls.latex2sympy.to_latex import to_latex
from aicls.latex2sympy.to_sympy import latex_to_sympy, to_sympy

__all__ = [
    "DEFAULT_CONFIG",
    "MAX_INPUT_CHARS",
    "NO_SPAN",
    "Abs",
    "Add",
    "Call",
    "Const",
    "Derivative",
    "Div",
    "Factorial",
    "Func",
    "Integral",
    "Interval",
    "LatexParser",
    "Limit",
    "Log",
    "MathSet",
    "Matrix",
    "Mul",
    "Neg",
    "Node",
    "NormalizeResult",
    "Num",
    "ParseError",
    "ParserConfig",
    "PlusMinus",
    "Pow",
    "Prime",
    "PrintStyle",
    "Product",
    "Relation",
    "Root",
    "RoundTripResult",
    "SemanticError",
    "Span",
    "Sum",
    "Sym",
    "SymbolTable",
    "UnsupportedConstruct",
    "Visitor",
    "latex_to_sympy",
    "normalize_latex",
    "parse_latex",
    "roundtrip_check",
    "sympy_to_latex",
    "to_latex",
    "to_sympy",
]
