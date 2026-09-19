"""AST -> canonical LaTeX (docs/mimari/latex-ayristirici.md 3.6).

The printer is *canonical*: for every node the grammar can produce, ``parse_latex(to_latex(n))``
returns exactly ``n`` again (property-tested in
``tests/property/test_latex2sympy_roundtrip.py``).  That is why parentheses are inserted by
precedence rather than by taste, why an implicit product between two numbers becomes
``\\cdot`` (``2 3`` would lex as one number) and why the differential of an integral is always
printed as ``\\, dx``.

Output is KaTeX-safe: only commands on the ``aicls.core.latex_safety`` allow-list are emitted.
"""

from __future__ import annotations

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
from aicls.latex2sympy.commands import command_table

# Precedence levels used for parenthesisation.
P_RELATION = 0
P_ADD = 1
P_MUL = 2
P_UNIT = 3
P_ATOM = 4

#: Relation operator -> LaTeX spelling.
RELATION_LATEX: dict[str, str] = {
    "=": "=",
    "!=": "\\neq",
    "<": "<",
    "<=": "\\le",
    ">": ">",
    ">=": "\\ge",
    "~=": "\\approx",
    "in": "\\in",
}

#: Constant name -> LaTeX spelling.
CONST_LATEX: dict[str, str] = {"e": "e", "i": "i", "pi": "\\pi", "infty": "\\infty"}

#: Single letters that the grammar reads as an applied function (``f(x)``).
FUNCTION_LETTERS = frozenset("fghFGH")

_GREEK_NAMES = frozenset(
    entry.latex.lstrip("\\") for entry in command_table().entries if entry.kind == "greek"
)

_PRECEDENCE: dict[type[Node], int] = {
    Relation: P_RELATION,
    PlusMinus: P_RELATION,
    Add: P_ADD,
    Neg: P_ADD,
    Integral: P_ADD,
    Sum: P_ADD,
    Product: P_ADD,
    Limit: P_ADD,
    Derivative: P_ADD,
    Mul: P_MUL,
    Div: P_UNIT,
    Pow: P_UNIT,
    Func: P_UNIT,
    Log: P_UNIT,
}


def _precedence(node: Node) -> int:
    return _PRECEDENCE.get(type(node), P_ATOM)


def symbol_latex(node: Sym) -> str:
    """Render a symbol: ``x``, ``x_{1}``, ``\\alpha``."""
    name = f"\\{node.name}" if node.name in _GREEK_NAMES else node.name
    return name if node.sub is None else f"{name}_{{{node.sub}}}"


class LatexPrinter(Visitor[str]):
    """Visitor producing the canonical LaTeX string."""

    # -- helpers ------------------------------------------------------------------------
    def wrap(self, node: Node, min_precedence: int) -> str:
        """Render ``node``, parenthesised when its precedence is below ``min_precedence``."""
        text = node.accept(self)
        if _precedence(node) < min_precedence:
            return f"\\left({text}\\right)"
        return text

    def _bigop_body(self, node: Node) -> str:
        return self.wrap(node, P_MUL)

    def _bounds(self, lower: Node | None, upper: Node | None) -> str:
        if lower is None or upper is None:
            return ""
        return f"_{{{lower.accept(self)}}}^{{{upper.accept(self)}}}"

    # -- leaves -------------------------------------------------------------------------
    def visit_num(self, node: Num) -> str:
        return node.value

    def visit_sym(self, node: Sym) -> str:
        return symbol_latex(node)

    def visit_const(self, node: Const) -> str:
        return CONST_LATEX.get(node.name, node.name)

    def visit_mathset(self, node: MathSet) -> str:
        return f"\\mathbb{{{node.name}}}"

    # -- arithmetic ---------------------------------------------------------------------
    def visit_neg(self, node: Neg) -> str:
        return f"-{self.wrap(node.x, P_UNIT)}"

    def visit_add(self, node: Add) -> str:
        parts: list[str] = []
        for index, term in enumerate(node.terms):
            if index == 0:
                parts.append(self.wrap(term, P_ADD))
            elif isinstance(term, Neg):
                parts.append(f" - {self.wrap(term.x, P_MUL)}")
            else:
                parts.append(f" + {self.wrap(term, P_ADD)}")
        return "".join(parts)

    def visit_mul(self, node: Mul) -> str:
        parts: list[str] = []
        for index, factor in enumerate(node.factors):
            if index == 0:
                text = factor.accept(self) if isinstance(factor, Neg) else self.wrap(factor, P_MUL)
                parts.append(text)
                continue
            text = self.wrap(factor, P_MUL)
            separator = " \\cdot " if text[:1].isdigit() or text.startswith(".") else " "
            parts.append(separator + text)
        return "".join(parts)

    def visit_div(self, node: Div) -> str:
        return f"\\frac{{{node.num.accept(self)}}}{{{node.den.accept(self)}}}"

    def visit_pow(self, node: Pow) -> str:
        return f"{self.wrap(node.base, P_ATOM)}^{{{node.exp.accept(self)}}}"

    def visit_root(self, node: Root) -> str:
        radicand = node.radicand.accept(self)
        if node.index is None:
            return f"\\sqrt{{{radicand}}}"
        return f"\\sqrt[{node.index.accept(self)}]{{{radicand}}}"

    def visit_abs(self, node: Abs) -> str:
        return f"\\lvert {node.x.accept(self)} \\rvert"

    def visit_factorial(self, node: Factorial) -> str:
        return f"{self.wrap(node.x, P_ATOM)}!"

    def visit_plus_minus(self, node: PlusMinus) -> str:
        return f"{self.wrap(node.a, P_ADD)} \\pm {self.wrap(node.b, P_ADD)}"

    # -- functions ----------------------------------------------------------------------
    def visit_func(self, node: Func) -> str:
        power = "" if node.power is None else f"^{{{node.power.accept(self)}}}"
        return f"\\{node.name}{power}\\left({node.arg.accept(self)}\\right)"

    def visit_log(self, node: Log) -> str:
        base = "" if node.base is None else f"_{{{node.base.accept(self)}}}"
        return f"\\log{base}\\left({node.arg.accept(self)}\\right)"

    def visit_call(self, node: Call) -> str:
        name = node.fname if node.fname in FUNCTION_LETTERS else f"\\operatorname{{{node.fname}}}"
        arguments = ", ".join(argument.accept(self) for argument in node.args)
        return f"{name}\\left({arguments}\\right)"

    def visit_prime(self, node: Prime) -> str:
        primes = "'" * node.order
        if node.arg is None:
            return f"{node.fname}{primes}"
        return f"{node.fname}{primes}\\left({node.arg.accept(self)}\\right)"

    # -- calculus -----------------------------------------------------------------------
    def visit_integral(self, node: Integral) -> str:
        bounds = self._bounds(node.lower, node.upper)
        body = self.wrap(node.body, P_ADD)
        return f"\\int{bounds} {body} \\, d{symbol_latex(node.var)}"

    def visit_derivative(self, node: Derivative) -> str:
        letter = "\\partial " if node.partial else "d"
        head = letter.rstrip()
        variable = symbol_latex(node.var)
        if node.order == 1:
            operator = f"\\frac{{{head}}}{{{letter}{variable}}}"
        else:
            operator = f"\\frac{{{head}^{{{node.order}}}}}{{{letter}{variable}^{{{node.order}}}}}"
        return f"{operator} {self._bigop_body(node.body)}"

    def visit_limit(self, node: Limit) -> str:
        direction = "" if node.direction == "+-" else f"^{{{node.direction}}}"
        target = node.to.accept(self)
        variable = symbol_latex(node.var)
        return f"\\lim_{{{variable} \\to {target}{direction}}} {self._bigop_body(node.body)}"

    def visit_sum(self, node: Sum) -> str:
        return self._accumulate("\\sum", node.var, node.lower, node.upper, node.body)

    def visit_product(self, node: Product) -> str:
        return self._accumulate("\\prod", node.var, node.lower, node.upper, node.body)

    def _accumulate(self, command: str, var: Sym, lower: Node, upper: Node, body: Node) -> str:
        head = f"{command}_{{{symbol_latex(var)}={lower.accept(self)}}}^{{{upper.accept(self)}}}"
        return f"{head} {self._bigop_body(body)}"

    # -- relations, sets, matrices ------------------------------------------------------
    def visit_relation(self, node: Relation) -> str:
        operator = RELATION_LATEX[node.op]
        left = self.wrap(node.lhs, P_ADD)
        right = self.wrap(node.rhs, P_ADD)
        return f"{left} {operator} {right}"

    def visit_interval(self, node: Interval) -> str:
        left = "\\left(" if node.left_open else "\\left["
        right = "\\right)" if node.right_open else "\\right]"
        return f"{left}{node.lo.accept(self)}, {node.hi.accept(self)}{right}"

    def visit_matrix(self, node: Matrix) -> str:
        rows = " \\\\ ".join(" & ".join(cell.accept(self) for cell in row) for row in node.rows)
        return f"\\begin{{{node.kind}}} {rows} \\end{{{node.kind}}}"


def to_latex(node: Node) -> str:
    """Render an AST as canonical, KaTeX-safe LaTeX.

    Args:
        node: Any AST node.

    Returns:
        The canonical LaTeX string; re-parsing it yields an equal AST.
    """
    return node.accept(LatexPrinter())


__all__ = [
    "CONST_LATEX",
    "FUNCTION_LETTERS",
    "RELATION_LATEX",
    "LatexPrinter",
    "symbol_latex",
    "to_latex",
]
