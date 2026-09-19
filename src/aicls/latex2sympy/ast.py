"""Typed AST with source spans (docs/mimari/latex-ayristirici.md 3.4).

Every node is a frozen, slotted dataclass carrying a :class:`Span` into the *normalised*
source.  ``span`` is excluded from equality (``compare=False``) so two structurally identical
expressions parsed from differently spaced strings compare equal - that is what makes
``parse_latex(to_latex(node)) == node`` an exact identity.

Traversal uses the visitor pattern: :meth:`Node.accept` dispatches to the matching
``visit_*`` method of a :class:`Visitor`; :mod:`aicls.latex2sympy.to_latex` and
:mod:`aicls.latex2sympy.to_sympy` are the two production visitors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Span:
    """Half-open character range ``[start, end)`` in the normalised input."""

    start: int
    end: int


#: Span used for hand-built nodes (printers, tests, NLU-generated expressions).
NO_SPAN = Span(0, 0)


class Node(ABC):
    """Base class of every AST node."""

    __slots__ = ()

    span: Span

    @abstractmethod
    def accept(self, visitor: Visitor[T]) -> T:
        """Dispatch to the visitor method for this node type."""


# ---------------------------------------------------------------------------------------------
# leaves
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Num(Node):
    """Numeric literal kept as written (``"2"``, ``"3.5"``)."""

    value: str
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_num(self)


@dataclass(frozen=True, slots=True)
class Sym(Node):
    """Symbol with an optional subscript (``x``, ``x_1``, ``a_{n+1}``, ``alpha``)."""

    name: str
    sub: str | None = None
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_sym(self)


@dataclass(frozen=True, slots=True)
class Const(Node):
    """Mathematical constant: ``e``, ``pi``, ``infty`` or ``i``."""

    name: str
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_const(self)


@dataclass(frozen=True, slots=True)
class MathSet(Node):
    """A blackboard-bold number set (``\\mathbb{R}`` -> ``name="R"``)."""

    name: str
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_mathset(self)


# ---------------------------------------------------------------------------------------------
# arithmetic
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Neg(Node):
    """Unary minus."""

    x: Node
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_neg(self)


@dataclass(frozen=True, slots=True)
class Add(Node):
    """Flat sum; subtraction is stored as a :class:`Neg` term."""

    terms: tuple[Node, ...]
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_add(self)


@dataclass(frozen=True, slots=True)
class Mul(Node):
    """Flat product; explicit (``\\cdot``, ``\\times``) and implicit products are the same node."""

    factors: tuple[Node, ...]
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_mul(self)


@dataclass(frozen=True, slots=True)
class Div(Node):
    """Quotient from ``\\frac{a}{b}`` or ``a/b``."""

    num: Node
    den: Node
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_div(self)


@dataclass(frozen=True, slots=True)
class Pow(Node):
    """Exponentiation."""

    base: Node
    exp: Node
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_pow(self)


@dataclass(frozen=True, slots=True)
class Root(Node):
    """``\\sqrt{x}`` (``index=None``) or ``\\sqrt[n]{x}``."""

    radicand: Node
    index: Node | None = None
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_root(self)


@dataclass(frozen=True, slots=True)
class Abs(Node):
    """Absolute value (``\\lvert x \\rvert``, ``|x|`` after normalisation)."""

    x: Node
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_abs(self)


@dataclass(frozen=True, slots=True)
class Factorial(Node):
    """``n!``."""

    x: Node
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_factorial(self)


@dataclass(frozen=True, slots=True)
class PlusMinus(Node):
    """``a \\pm b`` - a two-branch value, mapped to a ``FiniteSet`` by ``to_sympy``."""

    a: Node
    b: Node
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_plus_minus(self)


# ---------------------------------------------------------------------------------------------
# functions
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Func(Node):
    """A named function such as ``\\sin``; ``power`` holds ``\\sin^{2}`` / ``\\sin^{-1}``."""

    name: str
    arg: Node
    power: Node | None = None
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_func(self)


@dataclass(frozen=True, slots=True)
class Log(Node):
    """``\\log`` with an optional explicit base (``\\log_{10} x``)."""

    arg: Node
    base: Node | None = None
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_log(self)


@dataclass(frozen=True, slots=True)
class Call(Node):
    """Application of an (undefined) function: ``f(x)``, ``\\operatorname{sgn}(x)``."""

    fname: str
    args: tuple[Node, ...]
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_call(self)


@dataclass(frozen=True, slots=True)
class Prime(Node):
    """Lagrange notation: ``f'`` (``arg=None``) or ``f''(x)``."""

    fname: str
    arg: Node | None = None
    order: int = 1
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_prime(self)


# ---------------------------------------------------------------------------------------------
# calculus
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Integral(Node):
    """``\\int body \\, dvar`` with optional bounds."""

    body: Node
    var: Sym
    lower: Node | None = None
    upper: Node | None = None
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_integral(self)


@dataclass(frozen=True, slots=True)
class Derivative(Node):
    """``\\frac{d}{dx} body``, ``\\frac{d^{n}}{dx^{n}} body`` or the partial variant."""

    body: Node
    var: Sym
    order: int = 1
    partial: bool = False
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_derivative(self)


@dataclass(frozen=True, slots=True)
class Limit(Node):
    """``\\lim_{var \\to to} body``; ``direction`` is ``"+"``, ``"-"`` or ``"+-"``."""

    body: Node
    var: Sym
    to: Node
    direction: str = "+-"
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_limit(self)


@dataclass(frozen=True, slots=True)
class Sum(Node):
    """``\\sum_{var=lower}^{upper} body``."""

    body: Node
    var: Sym
    lower: Node
    upper: Node
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_sum(self)


@dataclass(frozen=True, slots=True)
class Product(Node):
    """``\\prod_{var=lower}^{upper} body``."""

    body: Node
    var: Sym
    lower: Node
    upper: Node
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_product(self)


# ---------------------------------------------------------------------------------------------
# relations, sets, matrices
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Relation(Node):
    """A single top-level relation; ``op`` is ``=``, ``!=``, ``<``, ``<=``, ``>``, ``>=``,
    ``~=`` (``\\approx``) or ``in``."""

    op: str
    lhs: Node
    rhs: Node
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_relation(self)


@dataclass(frozen=True, slots=True)
class Interval(Node):
    """A real interval; ``[0, 1)`` is ``left_open=False, right_open=True``."""

    lo: Node
    hi: Node
    left_open: bool = False
    right_open: bool = False
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_interval(self)


@dataclass(frozen=True, slots=True)
class Matrix(Node):
    """``\\begin{pmatrix} ... \\end{pmatrix}``; ``kind`` keeps the environment name."""

    rows: tuple[tuple[Node, ...], ...]
    kind: str = "pmatrix"
    span: Span = field(default=NO_SPAN, compare=False, repr=False)

    def accept(self, visitor: Visitor[T]) -> T:
        return visitor.visit_matrix(self)


# ---------------------------------------------------------------------------------------------
# visitor
# ---------------------------------------------------------------------------------------------
class Visitor(ABC, Generic[T]):
    """Exhaustive visitor over the AST; implementations must handle every node type."""

    @abstractmethod
    def visit_num(self, node: Num) -> T: ...

    @abstractmethod
    def visit_sym(self, node: Sym) -> T: ...

    @abstractmethod
    def visit_const(self, node: Const) -> T: ...

    @abstractmethod
    def visit_mathset(self, node: MathSet) -> T: ...

    @abstractmethod
    def visit_neg(self, node: Neg) -> T: ...

    @abstractmethod
    def visit_add(self, node: Add) -> T: ...

    @abstractmethod
    def visit_mul(self, node: Mul) -> T: ...

    @abstractmethod
    def visit_div(self, node: Div) -> T: ...

    @abstractmethod
    def visit_pow(self, node: Pow) -> T: ...

    @abstractmethod
    def visit_root(self, node: Root) -> T: ...

    @abstractmethod
    def visit_abs(self, node: Abs) -> T: ...

    @abstractmethod
    def visit_factorial(self, node: Factorial) -> T: ...

    @abstractmethod
    def visit_plus_minus(self, node: PlusMinus) -> T: ...

    @abstractmethod
    def visit_func(self, node: Func) -> T: ...

    @abstractmethod
    def visit_log(self, node: Log) -> T: ...

    @abstractmethod
    def visit_call(self, node: Call) -> T: ...

    @abstractmethod
    def visit_prime(self, node: Prime) -> T: ...

    @abstractmethod
    def visit_integral(self, node: Integral) -> T: ...

    @abstractmethod
    def visit_derivative(self, node: Derivative) -> T: ...

    @abstractmethod
    def visit_limit(self, node: Limit) -> T: ...

    @abstractmethod
    def visit_sum(self, node: Sum) -> T: ...

    @abstractmethod
    def visit_product(self, node: Product) -> T: ...

    @abstractmethod
    def visit_relation(self, node: Relation) -> T: ...

    @abstractmethod
    def visit_interval(self, node: Interval) -> T: ...

    @abstractmethod
    def visit_matrix(self, node: Matrix) -> T: ...


__all__ = [
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
    "Limit",
    "Log",
    "MathSet",
    "Matrix",
    "Mul",
    "Neg",
    "Node",
    "Num",
    "PlusMinus",
    "Pow",
    "Prime",
    "Product",
    "Relation",
    "Root",
    "Span",
    "Sum",
    "Sym",
    "Visitor",
]
