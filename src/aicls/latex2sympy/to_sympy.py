"""AST -> ``sympy.Basic`` (docs/mimari/latex-ayristirici.md 3.5).

The mapping follows the proposal's A.2.2 table one to one: ``\\frac`` -> ``Rational`` /
``Mul(a, Pow(b, -1))``, ``\\sin`` -> ``sin``, ``\\log`` -> ``log``, ``\\sum`` -> ``Sum``,
``\\int`` -> ``Integral``, ``\\lim`` -> ``Limit``, ``\\frac{d}{dx}`` -> ``Derivative``.

Calculus nodes are built **unevaluated** (``Integral``, ``Derivative``, ``Limit``, ``Sum``,
``Product``); calling ``doit()`` is :mod:`aicls.mathengine`'s job, which keeps
``POST /compute/parse`` free of side effects.  Ordinary arithmetic *is* evaluated, so the
result is a canonical SymPy object that ``simplify`` and ``srepr`` behave normally on; pass
``evaluate=False`` to keep the literal structure instead.
"""

from __future__ import annotations

from typing import Any

import sympy

from aicls.core.errors import AiclsError
from aicls.latex2sympy.ast import (
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
    Sum,
    Sym,
    Visitor,
)
from aicls.latex2sympy.errors import SemanticError
from aicls.latex2sympy.symbols import DEFAULT_CONFIG, ParserConfig, SymbolTable

#: LaTeX function name -> sympy function.
FUNCTIONS: dict[str, Any] = {
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "cot": sympy.cot,
    "sec": sympy.sec,
    "csc": sympy.csc,
    "arcsin": sympy.asin,
    "arccos": sympy.acos,
    "arctan": sympy.atan,
    "arccot": sympy.acot,
    "sinh": sympy.sinh,
    "cosh": sympy.cosh,
    "tanh": sympy.tanh,
    "coth": sympy.coth,
    "sech": sympy.sech,
    "csch": sympy.csch,
    "ln": sympy.log,
    "exp": sympy.exp,
}

#: Functions whose ``^{-1}`` power means the inverse function, not a reciprocal.
INVERSE_FUNCTIONS: dict[str, Any] = {
    "sin": sympy.asin,
    "cos": sympy.acos,
    "tan": sympy.atan,
    "cot": sympy.acot,
    "sec": sympy.asec,
    "csc": sympy.acsc,
    "sinh": sympy.asinh,
    "cosh": sympy.acosh,
    "tanh": sympy.atanh,
    "coth": sympy.acoth,
}

#: Relation operator -> sympy constructor.
RELATIONS: dict[str, Any] = {
    "=": sympy.Eq,
    "!=": sympy.Ne,
    "<": sympy.Lt,
    "<=": sympy.Le,
    ">": sympy.Gt,
    ">=": sympy.Ge,
    "~=": sympy.Eq,
}

#: ``\mathbb{X}`` -> sympy set.
MATH_SETS: dict[str, Any] = {
    "R": sympy.S.Reals,
    "N": sympy.S.Naturals,
    "Z": sympy.S.Integers,
    "Q": sympy.S.Rationals,
    "C": sympy.S.Complexes,
}

#: Assumptions given to a bound summation/product index.
BOUND_INDEX_ASSUMPTIONS: dict[str, bool] = {"integer": True}

_MINUS_ONE = Neg(Num("1"))


class SympyBuilder(Visitor[Any]):
    """Visitor that materialises the AST as SymPy objects."""

    def __init__(self, symbols: SymbolTable, config: ParserConfig) -> None:
        self.symbols = symbols
        self.config = config
        self._allow_plus_minus = False

    # -- traversal ----------------------------------------------------------------------
    def build(self, node: Node) -> Any:
        """Entry point: ``\\pm`` is only meaningful at the very top of an expression."""
        return self._visit(node, plus_minus_allowed=True)

    def _visit(self, node: Node, *, plus_minus_allowed: bool = False) -> Any:
        previous = self._allow_plus_minus
        self._allow_plus_minus = plus_minus_allowed
        try:
            return node.accept(self)
        finally:
            self._allow_plus_minus = previous

    def _symbol_name(self, node: Sym) -> str:
        return node.name if node.sub is None else f"{node.name}_{{{node.sub}}}"

    def _variable(self, node: Sym) -> Any:
        return self.symbols.symbol(self._symbol_name(node))

    # -- leaves -------------------------------------------------------------------------
    def visit_num(self, node: Num) -> Any:
        if "." in node.value:
            return sympy.Float(node.value)
        return sympy.Integer(int(node.value))

    def visit_sym(self, node: Sym) -> Any:
        name = self._symbol_name(node)
        if node.sub is None and name in self.symbols.reserved:
            constant = self._constant(name)
            if constant is not None:
                return constant
        return self.symbols.symbol(name)

    def visit_const(self, node: Const) -> Any:
        constant = self._constant(node.name)
        if constant is None:
            return self.symbols.symbol(node.name)
        return constant

    def _constant(self, name: str) -> Any:
        if name not in self.symbols.reserved:
            return None
        if name == "e":
            return sympy.E
        if name == "pi":
            return sympy.pi
        if name in {"infty", "oo"}:
            return sympy.oo
        if name == "i" and self.config.imaginary_i:
            return sympy.I
        return None

    def visit_mathset(self, node: MathSet) -> Any:
        try:
            return MATH_SETS[node.name]
        except KeyError as exc:
            raise SemanticError("unknown_number_set", details={"name": node.name}) from exc

    # -- arithmetic ---------------------------------------------------------------------
    def visit_neg(self, node: Neg) -> Any:
        return -self._visit(node.x)

    def visit_add(self, node: Add) -> Any:
        return sympy.Add(*(self._visit(term) for term in node.terms))

    def visit_mul(self, node: Mul) -> Any:
        return sympy.Mul(*(self._visit(factor) for factor in node.factors))

    def visit_div(self, node: Div) -> Any:
        return self._visit(node.num) / self._visit(node.den)

    def visit_pow(self, node: Pow) -> Any:
        return self._visit(node.base) ** self._visit(node.exp)

    def visit_root(self, node: Root) -> Any:
        radicand = self._visit(node.radicand)
        if node.index is None:
            return sympy.sqrt(radicand)
        return sympy.root(radicand, self._visit(node.index))

    def visit_abs(self, node: Abs) -> Any:
        return sympy.Abs(self._visit(node.x))

    def visit_factorial(self, node: Factorial) -> Any:
        return sympy.factorial(self._visit(node.x))

    def visit_plus_minus(self, node: PlusMinus) -> Any:
        if not self._allow_plus_minus:
            raise SemanticError("nested_plus_minus")
        left = self._visit(node.a)
        right = self._visit(node.b)
        return sympy.FiniteSet(left + right, left - right)

    # -- functions ----------------------------------------------------------------------
    def visit_func(self, node: Func) -> Any:
        argument = self._visit(node.arg)
        name = node.name
        if node.power is not None and node.power == _MINUS_ONE and name in INVERSE_FUNCTIONS:
            return INVERSE_FUNCTIONS[name](argument)
        if name == "lg":
            value = sympy.log(argument, 10)
        else:
            try:
                value = FUNCTIONS[name](argument)
            except KeyError as exc:  # pragma: no cover - the grammar limits the names
                raise SemanticError("unknown_function", details={"name": name}) from exc
        if node.power is None:
            return value
        return value ** self._visit(node.power)

    def visit_log(self, node: Log) -> Any:
        argument = self._visit(node.arg)
        if node.base is not None:
            return sympy.log(argument, self._visit(node.base))
        if self.config.log_base == "10":
            return sympy.log(argument, 10)
        return sympy.log(argument)

    def visit_call(self, node: Call) -> Any:
        return sympy.Function(node.fname)(*(self._visit(arg) for arg in node.args))

    def visit_prime(self, node: Prime) -> Any:
        function = sympy.Function(node.fname)
        if node.arg is None:
            variable = self.symbols.symbol("x")
            return sympy.Derivative(function(variable), (variable, node.order))
        if not isinstance(node.arg, Sym):
            raise SemanticError("prime_of_expression", details={"arg": type(node.arg).__name__})
        variable = self._variable(node.arg)
        return sympy.Derivative(function(variable), (variable, node.order))

    # -- calculus -----------------------------------------------------------------------
    def visit_integral(self, node: Integral) -> Any:
        variable = self._variable(node.var)
        body = self._visit(node.body)
        if node.lower is None or node.upper is None:
            return sympy.Integral(body, variable)
        return sympy.Integral(body, (variable, self._visit(node.lower), self._visit(node.upper)))

    def visit_derivative(self, node: Derivative) -> Any:
        # ``node.partial`` deliberately does not change the lowering: SymPy has no separate
        # partial-derivative class, and ``Derivative(expr, (x, n))`` *is* the partial derivative
        # with respect to x when expr depends on more variables.  The flag is kept on the AST so
        # ``to_latex`` can reproduce the spelling the user wrote (\partial vs d).
        variable = self._variable(node.var)
        return sympy.Derivative(self._visit(node.body), (variable, node.order))

    def visit_limit(self, node: Limit) -> Any:
        variable = self._variable(node.var)
        return sympy.Limit(self._visit(node.body), variable, self._visit(node.to), node.direction)

    def visit_sum(self, node: Sum) -> Any:
        return self._accumulate(sympy.Sum, node.var, node.lower, node.upper, node.body)

    def visit_product(self, node: Product) -> Any:
        return self._accumulate(sympy.Product, node.var, node.lower, node.upper, node.body)

    def _accumulate(self, factory: Any, var: Sym, lower: Node, upper: Node, body: Node) -> Any:
        name = self._symbol_name(var)
        with self.symbols.bound(name, **BOUND_INDEX_ASSUMPTIONS) as index:
            return factory(self._visit(body), (index, self._visit(lower), self._visit(upper)))

    # -- relations, sets, matrices ------------------------------------------------------
    def visit_relation(self, node: Relation) -> Any:
        left = self._visit(node.lhs, plus_minus_allowed=True)
        right = self._visit(node.rhs, plus_minus_allowed=True)
        if node.op == "in":
            # Unevaluated: ``x \in \mathbb{R}`` is a statement to display and re-parse, not a
            # truth value to collapse (a real-declared x would make it literally ``True``).
            return sympy.Contains(left, right, evaluate=False)
        try:
            return RELATIONS[node.op](left, right)
        except KeyError as exc:  # pragma: no cover - the grammar limits the operators
            raise SemanticError("unknown_relation", details={"op": node.op}) from exc

    def visit_interval(self, node: Interval) -> Any:
        return sympy.Interval(
            self._visit(node.lo), self._visit(node.hi), node.left_open, node.right_open
        )

    def visit_matrix(self, node: Matrix) -> Any:
        return sympy.Matrix([[self._visit(cell) for cell in row] for row in node.rows])


def to_sympy(
    node: Node,
    symbols: SymbolTable | None = None,
    *,
    config: ParserConfig | None = None,
    evaluate: bool = True,
) -> sympy.Basic:
    """Convert a typed AST to a SymPy expression.

    Args:
        node: AST root, typically from :func:`~aicls.latex2sympy.parser.parse_latex`.
        symbols: Symbol table to use (and to extend); a fresh default table is created when
            omitted.
        config: Parser configuration; defaults to the table's configuration.
        evaluate: When False the expression is built inside ``sympy.evaluate(False)``, which
            keeps the literal structure (``\\frac{2}{4}`` stays ``2 \\cdot 4^{-1}``).

    Returns:
        The SymPy object; calculus nodes stay unevaluated.

    Raises:
        SemanticError: If the AST cannot be given a meaning (unknown number set, ``\\pm``
            nested inside another operator, ``f'`` of a non-symbol).
    """
    table = symbols if symbols is not None else SymbolTable(config=config or DEFAULT_CONFIG)
    effective = config if config is not None else table.config
    builder = SympyBuilder(table, effective)
    try:
        if evaluate:
            return builder.build(node)
        with sympy.evaluate(False):
            return builder.build(node)
    except AiclsError:
        raise
    except Exception as exc:
        # SymPy raises plain exceptions for type-invalid combinations (``\ln`` of an interval,
        # a matrix exponent). The module contract is AiclsError only.
        raise SemanticError(
            "sympy_construction_failed",
            details={"error": type(exc).__name__, "message": str(exc)[:200]},
        ) from exc


def latex_to_sympy(
    src: str,
    *,
    symbols: SymbolTable | None = None,
    config: ParserConfig | None = None,
    strict: bool = True,
    evaluate: bool = True,
) -> sympy.Basic:
    """Parse ``src`` and convert it to SymPy in one step.

    Args:
        src: LaTeX source.
        symbols: Optional symbol table.
        config: Optional parser configuration.
        strict: Passed to :func:`~aicls.latex2sympy.parser.parse_latex`.
        evaluate: Passed to :func:`to_sympy`.

    Returns:
        The SymPy expression.
    """
    from aicls.latex2sympy.parser import parse_latex

    node = parse_latex(src, strict=strict)
    return to_sympy(node, symbols, config=config, evaluate=evaluate)


__all__ = [
    "BOUND_INDEX_ASSUMPTIONS",
    "FUNCTIONS",
    "INVERSE_FUNCTIONS",
    "MATH_SETS",
    "RELATIONS",
    "SympyBuilder",
    "latex_to_sympy",
    "to_sympy",
]
