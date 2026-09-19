"""Grammar-level parse tests (docs/mimari/latex-ayristirici.md 3.2-3.3).

The test names in this module are contractual: they come from the module spec and from
docs/gelistirme/test-stratejisi.md 6.
"""

from __future__ import annotations

import pytest
import sympy

from aicls.latex2sympy import (
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
    Matrix,
    Mul,
    Neg,
    Num,
    Pow,
    Prime,
    Product,
    Relation,
    Root,
    Sum,
    Sym,
    latex_to_sympy,
    parse_latex,
    to_latex,
    to_sympy,
)


# ---------------------------------------------------------------------------------------------
# contract test names
# ---------------------------------------------------------------------------------------------
def test_parse_frac_nested_returns_div_node() -> None:
    node = parse_latex(r"\frac{\frac{1}{2}}{3}")
    assert isinstance(node, Div)
    assert isinstance(node.num, Div)
    assert node.num.num == Num("1")
    assert node.num.den == Num("2")
    assert node.den == Num("3")
    assert to_sympy(node) == sympy.Rational(1, 6)


def test_parse_implicit_multiplication_2x_sin_x() -> None:
    node = parse_latex(r"2x\sin x")
    assert isinstance(node, Mul)
    assert node.factors == (Num("2"), Sym("x"), Func("sin", Sym("x")))
    x = sympy.Symbol("x", real=True)
    assert to_sympy(node) == 2 * x * sympy.sin(x)


def test_parse_derivative_leibniz_frac_d_dx() -> None:
    node = parse_latex(r"\frac{d}{dx} x^{2}")
    assert isinstance(node, Derivative)
    assert node.var == Sym("x")
    assert node.order == 1
    assert node.partial is False
    assert node.body == Pow(Sym("x"), Num("2"))
    x = sympy.Symbol("x", real=True)
    assert to_sympy(node) == sympy.Derivative(x**2, x)
    assert to_sympy(node).doit() == 2 * x


def test_parse_definite_integral_bounds_to_sympy_integral() -> None:
    node = parse_latex(r"\int_{0}^{1} x^{2}\,dx")
    assert isinstance(node, Integral)
    assert node.var == Sym("x")
    assert node.lower == Num("0")
    assert node.upper == Num("1")
    x = sympy.Symbol("x", real=True)
    expr = to_sympy(node)
    assert expr == sympy.Integral(x**2, (x, 0, 1))
    assert expr.doit() == sympy.Rational(1, 3)


def test_parse_indefinite_integral_dx_detected() -> None:
    node = parse_latex(r"\int x^{2}\,dx")
    assert isinstance(node, Integral)
    assert node.var == Sym("x")
    assert node.lower is None and node.upper is None
    assert node.body == Pow(Sym("x"), Num("2"))
    x = sympy.Symbol("x", real=True)
    assert to_sympy(node) == sympy.Integral(x**2, x)
    # Outside an integral ``dx`` stays an ordinary product (ADR-0004).
    assert parse_latex("2dx") == Mul((Num("2"), Sym("d"), Sym("x")))


def test_parse_limit_with_direction_plus() -> None:
    node = parse_latex(r"\lim_{x \to 0^{+}} \frac{1}{x}")
    assert isinstance(node, Limit)
    assert node.var == Sym("x")
    assert node.to == Num("0")
    assert node.direction == "+"
    x = sympy.Symbol("x", real=True)
    assert to_sympy(node) == sympy.Limit(1 / x, x, 0, "+")
    assert to_sympy(node).doit() is sympy.oo


def test_parse_greek_alpha_maps_to_symbol_alpha() -> None:
    node = parse_latex(r"\alpha + \beta")
    assert node == Add((Sym("alpha"), Sym("beta")))
    alpha = sympy.Symbol("alpha", real=True)
    beta = sympy.Symbol("beta", real=True)
    assert to_sympy(node) == alpha + beta
    assert str(to_sympy(parse_latex(r"\alpha"))) == "alpha"


def test_parse_sin_x_y_binds_first_power() -> None:
    node = parse_latex(r"\sin x y")
    assert node == Mul((Func("sin", Sym("x")), Sym("y")))
    x = sympy.Symbol("x", real=True)
    y = sympy.Symbol("y", real=True)
    assert to_sympy(node) == sympy.sin(x) * y
    # parentheses take the whole group
    assert parse_latex(r"\sin(x y)") == Func("sin", Mul((Sym("x"), Sym("y"))))
    assert latex_to_sympy(r"\sin(x y)") == sympy.sin(x * y)


def test_parse_sum_with_limits() -> None:
    node = parse_latex(r"\sum_{k=1}^{n} k^{2}")
    assert isinstance(node, Sum)
    assert node.var == Sym("k")
    assert node.lower == Num("1")
    assert node.upper == Sym("n")
    k = sympy.Symbol("k", integer=True)
    n = sympy.Symbol("n", real=True)
    assert to_sympy(node) == sympy.Sum(k**2, (k, 1, n))


def test_parse_error_reports_line_and_column() -> None:
    from aicls.latex2sympy import ParseError

    with pytest.raises(ParseError) as excinfo:
        parse_latex("2 + ")
    err = excinfo.value
    assert err.line == 1
    assert err.col >= 1
    assert err.expected
    assert err.snippet
    assert err.details["position"] == err.col
    assert err.http_status == 422


def test_unbalanced_braces_is_parse_error_not_crash() -> None:
    from aicls.core.errors import AiclsError

    for src in (r"\frac{1}{2}}}", r"{{{x", r"\frac{{1}{2}", "}", r"x^{{}"):
        with pytest.raises(AiclsError):
            parse_latex(src)


# ---------------------------------------------------------------------------------------------
# further grammar coverage
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ("1", Num("1")),
        ("3.5", Num("3.5")),
        ("x", Sym("x")),
        ("x_1", Sym("x", "1")),
        ("x_{10}", Sym("x", "10")),
        ("a_{n+1}", Sym("a", "n+1")),
        (r"\pi", Const("pi")),
        (r"\infty", Const("infty")),
        ("-x", Neg(Sym("x"))),
        ("+x", Sym("x")),
        ("x + y", Add((Sym("x"), Sym("y")))),
        ("x - y", Add((Sym("x"), Neg(Sym("y"))))),
        ("2 \\cdot 3", Mul((Num("2"), Num("3")))),
        ("2 \\times 3", Mul((Num("2"), Num("3")))),
        ("x/y", Div(Sym("x"), Sym("y"))),
        ("x^{2}", Pow(Sym("x"), Num("2"))),
        ("x^2", Pow(Sym("x"), Num("2"))),
        (r"\sqrt{x}", Root(Sym("x"))),
        (r"\sqrt[3]{x}", Root(Sym("x"), Num("3"))),
        (r"\lvert x \rvert", Abs(Sym("x"))),
        ("n!", Factorial(Sym("n"))),
        (r"\ln x", Func("ln", Sym("x"))),
        (r"\log x", Log(Sym("x"))),
        (r"\log_{10} x", Log(Sym("x"), Num("10"))),
        (r"\sin^{2} x", Func("sin", Sym("x"), Num("2"))),
        ("f(x)", Call("f", (Sym("x"),))),
        ("f(x, y)", Call("f", (Sym("x"), Sym("y")))),
        (r"\operatorname{sgn}(x)", Call("sgn", (Sym("x"),))),
        ("f'", Prime("f", None, 1)),
        ("f''(x)", Prime("f", Sym("x"), 2)),
        ("x = 1", Relation("=", Sym("x"), Num("1"))),
        ("x < 1", Relation("<", Sym("x"), Num("1"))),
        (r"x \le 1", Relation("<=", Sym("x"), Num("1"))),
        (r"x \geq 1", Relation(">=", Sym("x"), Num("1"))),
        (r"x \neq 1", Relation("!=", Sym("x"), Num("1"))),
        ("[0, 1]", Interval(Num("0"), Num("1"), False, False)),
        ("(0, 1)", Interval(Num("0"), Num("1"), True, True)),
        ("[0, 1)", Interval(Num("0"), Num("1"), False, True)),
        ("(0, 1]", Interval(Num("0"), Num("1"), True, False)),
    ],
)
def test_parse_atoms_and_operators(src: str, expected: object) -> None:
    assert parse_latex(src) == expected


def test_parse_grouping_and_precedence() -> None:
    assert parse_latex("(x + 1)(x - 1)") == Mul(
        (Add((Sym("x"), Num("1"))), Add((Sym("x"), Neg(Num("1")))))
    )
    assert parse_latex("2 + 3 \\cdot 4") == Add((Num("2"), Mul((Num("3"), Num("4")))))
    # a/bc is (a/b)*c - left to right, implicit multiplication has "/" precedence
    assert parse_latex("a/bc") == Mul((Div(Sym("a"), Sym("b")), Sym("c")))
    assert parse_latex("{x + 1}") == Add((Sym("x"), Num("1")))


def test_parse_matrix_pmatrix() -> None:
    node = parse_latex(r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}")
    assert isinstance(node, Matrix)
    assert node.kind == "pmatrix"
    assert node.rows == ((Num("1"), Num("2")), (Num("3"), Num("4")))
    assert to_sympy(node) == sympy.Matrix([[1, 2], [3, 4]])


def test_parse_product_and_partial_derivative() -> None:
    node = parse_latex(r"\prod_{k=1}^{n} k")
    assert isinstance(node, Product)
    k = sympy.Symbol("k", integer=True)
    n = sympy.Symbol("n", real=True)
    assert to_sympy(node) == sympy.Product(k, (k, 1, n))

    partial = parse_latex(r"\frac{\partial}{\partial x} x y")
    assert isinstance(partial, Derivative)
    assert partial.partial is True
    assert partial.order == 1


def test_parse_second_order_leibniz() -> None:
    node = parse_latex(r"\frac{d^{2}}{dx^{2}} \sin x")
    assert isinstance(node, Derivative)
    assert node.order == 2
    x = sympy.Symbol("x", real=True)
    assert to_sympy(node) == sympy.Derivative(sympy.sin(x), (x, 2))


def test_parse_dy_dx_shorthand() -> None:
    node = parse_latex(r"\frac{dy}{dx}")
    assert isinstance(node, Derivative)
    assert node.var == Sym("x")
    assert node.body == Sym("y")


def test_integral_body_is_greedy_up_to_the_differential() -> None:
    node = parse_latex(r"\int x^{2} + 1 \, dx")
    assert isinstance(node, Integral)
    assert node.body == Add((Pow(Sym("x"), Num("2")), Num("1")))

    split = parse_latex(r"\int x \, dx + 1")
    assert isinstance(split, Add)
    assert isinstance(split.terms[0], Integral)
    assert split.terms[1] == Num("1")


def test_integral_without_differential_is_semantic_error() -> None:
    from aicls.latex2sympy import SemanticError

    with pytest.raises(SemanticError) as excinfo:
        parse_latex(r"\int_{0}^{1} x^{2}")
    assert "differential" in str(excinfo.value.details["reason"])


def test_scaled_big_operator_keeps_coefficient() -> None:
    node = parse_latex(r"2\int_{0}^{1} x \, dx")
    assert isinstance(node, Mul)
    assert node.factors[0] == Num("2")
    assert isinstance(node.factors[1], Integral)
    assert latex_to_sympy(r"2\int_{0}^{1} x \, dx").doit() == 1


def test_function_power_minus_one_is_inverse() -> None:
    x = sympy.Symbol("x", real=True)
    assert latex_to_sympy(r"\sin^{-1}(x)") == sympy.asin(x)
    assert latex_to_sympy(r"\sin^{2}(x)") == sympy.sin(x) ** 2
    assert latex_to_sympy(r"\cos^{2}x + \sin^{2}x") == sympy.cos(x) ** 2 + sympy.sin(x) ** 2


def test_constants_e_pi_and_exponential() -> None:
    x = sympy.Symbol("x", real=True)
    assert latex_to_sympy("e^{x}") == sympy.exp(x)
    assert latex_to_sympy(r"\exp(x)") == sympy.exp(x)
    assert latex_to_sympy(r"\pi r^{2}") == sympy.pi * sympy.Symbol("r", real=True) ** 2
    assert latex_to_sympy(r"\infty") is sympy.oo


def test_set_membership_and_intervals() -> None:
    node = parse_latex(r"x \in \mathbb{R}")
    assert isinstance(node, Relation)
    assert node.op == "in"
    membership = sympy.Contains(sympy.Symbol("x", real=True), sympy.S.Reals, evaluate=False)
    assert to_sympy(node) == membership
    assert to_sympy(parse_latex("[0, 1]")) == sympy.Interval(0, 1)
    assert to_sympy(parse_latex("(0, 1)")) == sympy.Interval.open(0, 1)


def test_plus_minus_becomes_finite_set() -> None:
    node = parse_latex(r"1 \pm 2")
    assert to_sympy(node) == sympy.FiniteSet(-1, 3)


def test_left_right_and_spacing_macros_are_ignored() -> None:
    assert parse_latex(r"\left( x + 1 \right)^{2}") == Pow(Add((Sym("x"), Num("1"))), Num("2"))
    assert parse_latex(r"x \, + \; y \quad") == Add((Sym("x"), Sym("y")))
    assert parse_latex(r"\displaystyle \frac{1}{2}") == Div(Num("1"), Num("2"))


def test_sympy_printed_function_call_form_parses() -> None:
    # sympy.latex() prints ``\sin{\left(x \right)}``
    x = sympy.Symbol("x", real=True)
    assert latex_to_sympy(r"\sin{\left(x \right)}") == sympy.sin(x)
    assert latex_to_sympy(r"\log{\left(x \right)}") == sympy.log(x)


# ---------------------------------------------------------------------------------------------
# Leibniz shorthand ``\frac{<d|\partial>[^n] f}{<d|\partial> x[^n]}``  (KNOWN_ISSUES parser 1)
# ---------------------------------------------------------------------------------------------
def test_parse_partial_derivative_shorthand_frac_partial_f_partial_x() -> None:
    r"""``\frac{\partial f}{\partial x}`` is what ``aicls.ocr.CURRICULUM_LATEX`` emits."""
    node = parse_latex(r"\frac{\partial f}{\partial x}")
    assert isinstance(node, Derivative)
    assert node.body == Sym("f")
    assert node.var == Sym("x")
    assert node.order == 1
    assert node.partial is True
    f, x = sympy.Symbol("f", real=True), sympy.Symbol("x", real=True)
    assert to_sympy(node) == sympy.Derivative(f, x)


def test_parse_leibniz_shorthand_accepts_both_spellings() -> None:
    r"""``d`` and ``\partial`` produce the same node except for the ``partial`` flag."""
    total = parse_latex(r"\frac{d u}{d t}")
    partial = parse_latex(r"\frac{\partial u}{\partial t}")
    assert isinstance(total, Derivative)
    assert isinstance(partial, Derivative)
    assert (total.body, total.var, total.order) == (Sym("u"), Sym("t"), 1)
    assert (partial.body, partial.var, partial.order) == (Sym("u"), Sym("t"), 1)
    assert total.partial is False
    assert partial.partial is True


def test_parse_higher_order_leibniz_shorthand_both_spellings() -> None:
    r"""``\frac{d^{2}y}{dx^{2}}`` used to misparse silently as ``d y / x^{2}``."""
    total = parse_latex(r"\frac{d^{2}y}{dx^{2}}")
    assert total == Derivative(Sym("y"), Sym("x"), 2, False)
    partial = parse_latex(r"\frac{\partial^{2} f}{\partial x^{2}}")
    assert partial == Derivative(Sym("f"), Sym("x"), 2, True)
    unbraced = parse_latex(r"\frac{d^2 y}{d x^2}")
    assert unbraced == Derivative(Sym("y"), Sym("x"), 2, False)
    y, x = sympy.Symbol("y", real=True), sympy.Symbol("x", real=True)
    assert to_sympy(total) == sympy.Derivative(y, (x, 2))


def test_parse_leibniz_shorthand_with_greek_names() -> None:
    node = parse_latex(r"\frac{\partial \phi}{\partial \theta}")
    assert node == Derivative(Sym("phi"), Sym("theta"), 1, True)


def test_leibniz_shorthand_round_trips_to_the_operator_form() -> None:
    """The printer has one canonical spelling; re-parsing it yields the very same AST."""
    for src in (
        r"\frac{\partial f}{\partial x}",
        r"\frac{d^{2}y}{dx^{2}}",
        r"\frac{\partial^{2} f}{\partial x^{2}}",
    ):
        node = parse_latex(src)
        assert parse_latex(to_latex(node)) == node


def test_second_order_leibniz_equation_is_not_a_plain_fraction() -> None:
    r"""``\frac{d^{2}y}{dt^{2}} + 4y = 0`` is an ODE, not ``d y / t^{2} + 4 y = 0``."""
    node = parse_latex(r"\frac{d^{2}y}{dt^{2}} + 4y = 0")
    assert isinstance(node, Relation)
    assert isinstance(node.lhs, Add)
    assert node.lhs.terms[0] == Derivative(Sym("y"), Sym("t"), 2, False)


def test_ordinary_fraction_starting_with_d_is_still_a_fraction() -> None:
    """Widening the shorthand must not swallow fractions that merely start with ``d``."""
    assert isinstance(parse_latex(r"\frac{d + 1}{dx}"), Div)
    assert isinstance(parse_latex(r"\frac{2d}{dx}"), Div)
    assert isinstance(parse_latex(r"\frac{dx}{2}"), Div)


# ---------------------------------------------------------------------------------------------
# a big operator on the right of an explicit multiplication sign
# ---------------------------------------------------------------------------------------------
def test_big_operator_after_an_explicit_multiplication_sign() -> None:
    r"""``sympy.latex`` writes a scaled calculus object as ``2 \cdot \int ...``.

    Only juxtaposition (``2 \int x \, dx``) used to reach the ``bigop`` rule, so this spelling
    -- the platform's own CAS output, which a user may send straight back -- raised a
    ``ParseError`` for ``\int``/``\sum``/``\prod``/``\lim`` and, worse, *silently misparsed*
    ``2 \cdot \frac{d}{dx} f`` as ``2 f / x`` because the contextual lexer fell back to _FRAC.
    """
    x, f = sympy.Symbol("x", real=True), sympy.Symbol("f", real=True)
    k = sympy.Symbol("k", integer=True)
    assert to_sympy(parse_latex(r"2 \cdot \frac{d}{dx} f")) == 2 * sympy.Derivative(f, x)
    assert to_sympy(parse_latex(r"2 \cdot \int x^{2} \, dx")) == 2 * sympy.Integral(x**2, x)
    assert to_sympy(parse_latex(r"2 \times \int x^{2} \, dx")) == 2 * sympy.Integral(x**2, x)
    assert to_sympy(parse_latex(r"3 \cdot \sum_{k=1}^{5} k")) == 3 * sympy.Sum(k, (k, 1, 5))
    assert to_sympy(parse_latex(r"3 \cdot \prod_{k=1}^{5} k")) == 3 * sympy.Product(k, (k, 1, 5))
    limit = to_sympy(parse_latex(r"3 \cdot \lim_{x \to 0} \frac{\sin x}{x}"))
    assert limit == 3 * sympy.Limit(sympy.sin(x) / x, x, 0, "+-")


def test_juxtaposed_big_operator_still_binds_the_same_way() -> None:
    """The new alternative must not change the reading of the juxtaposed form."""
    x = sympy.Symbol("x", real=True)
    assert to_sympy(parse_latex(r"2 \int x \, dx")) == 2 * sympy.Integral(x, x)
    assert to_sympy(parse_latex(r"2 \cdot 3 \int x \, dx")) == 6 * sympy.Integral(x, x)
