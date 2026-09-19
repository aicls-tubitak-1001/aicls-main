"""Canonical AST -> LaTeX printer (docs/mimari/latex-ayristirici.md 3.6)."""

from __future__ import annotations

import pytest

from aicls.latex2sympy import (
    Abs,
    Add,
    Const,
    Div,
    Factorial,
    Func,
    Mul,
    Neg,
    Num,
    Pow,
    Sym,
    parse_latex,
    to_latex,
)

CANONICAL: list[tuple[str, str]] = [
    ("1", "1"),
    ("x", "x"),
    ("x_1", "x_{1}"),
    (r"\pi", r"\pi"),
    (r"\infty", r"\infty"),
    ("x + y", "x + y"),
    ("x - y", "x - y"),
    ("-x", "-x"),
    ("2x", "2 x"),
    ("2 \\cdot 3", r"2 \cdot 3"),
    (r"\frac{1}{2}", r"\frac{1}{2}"),
    ("x^{2}", "x^{2}"),
    ("(x + 1)^{2}", r"\left(x + 1\right)^{2}"),
    (r"\sqrt{x}", r"\sqrt{x}"),
    (r"\sqrt[3]{x}", r"\sqrt[3]{x}"),
    (r"\lvert x \rvert", r"\lvert x \rvert"),
    ("n!", "n!"),
    (r"\sin x", r"\sin\left(x\right)"),
    (r"\sin^{2} x", r"\sin^{2}\left(x\right)"),
    (r"\log_{10} x", r"\log_{10}\left(x\right)"),
    ("f(x)", r"f\left(x\right)"),
    ("f''(x)", r"f''\left(x\right)"),
    ("x = 1", "x = 1"),
    (r"x \le 1", r"x \le 1"),
    ("[0, 1]", r"\left[0, 1\right]"),
    ("(0, 1]", r"\left(0, 1\right]"),
    (r"\int x^{2} \, dx", r"\int x^{2} \, dx"),
    (r"\int_{0}^{1} x^{2} \, dx", r"\int_{0}^{1} x^{2} \, dx"),
    (r"\sum_{k=1}^{n} k^{2}", r"\sum_{k=1}^{n} k^{2}"),
    (r"\prod_{k=1}^{n} k", r"\prod_{k=1}^{n} k"),
    (r"\lim_{x \to 0} \frac{1}{x}", r"\lim_{x \to 0} \frac{1}{x}"),
    (r"\lim_{x \to 0^{+}} \frac{1}{x}", r"\lim_{x \to 0^{+}} \frac{1}{x}"),
    (r"\frac{d}{dx} x^{2}", r"\frac{d}{dx} x^{2}"),
    (r"\frac{d^{2}}{dx^{2}} x^{3}", r"\frac{d^{2}}{dx^{2}} x^{3}"),
    (r"\frac{\partial}{\partial x} x y", r"\frac{\partial}{\partial x} x y"),
    (r"\frac{\partial f}{\partial x}", r"\frac{\partial}{\partial x} f"),
    (r"\frac{dy}{dx}", r"\frac{d}{dx} y"),
    (r"\frac{d^{2}y}{dx^{2}}", r"\frac{d^{2}}{dx^{2}} y"),
    (r"\frac{\partial^{2} f}{\partial x^{2}}", r"\frac{\partial^{2}}{\partial x^{2}} f"),
    (r"1 \pm 2", r"1 \pm 2"),
    (r"x \in \mathbb{R}", r"x \in \mathbb{R}"),
    (
        r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}",
        r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}",
    ),
]


@pytest.mark.parametrize(("src", "expected"), CANONICAL)
def test_to_latex_is_canonical(src: str, expected: str) -> None:
    assert to_latex(parse_latex(src)) == expected


@pytest.mark.parametrize(("src", "_expected"), CANONICAL)
def test_to_latex_output_reparses_to_the_same_ast(src: str, _expected: str) -> None:
    node = parse_latex(src)
    assert parse_latex(to_latex(node)) == node


def test_to_latex_parenthesises_where_precedence_requires_it() -> None:
    assert to_latex(Mul((Add((Sym("x"), Num("1"))), Sym("y")))) == r"\left(x + 1\right) y"
    assert to_latex(Pow(Div(Num("1"), Sym("x")), Num("2"))) == r"\left(\frac{1}{x}\right)^{2}"
    assert to_latex(Mul((Sym("x"), Neg(Sym("y"))))) == r"x \left(-y\right)"
    assert to_latex(Factorial(Add((Sym("x"), Num("1"))))) == r"\left(x + 1\right)!"
    assert to_latex(Mul((Num("2"), Num("3")))) == r"2 \cdot 3"
    assert to_latex(Pow(Func("sin", Sym("x")), Num("2"))) == r"\left(\sin\left(x\right)\right)^{2}"


def test_to_latex_of_handmade_nodes() -> None:
    assert to_latex(Abs(Sym("x"))) == r"\lvert x \rvert"
    assert to_latex(Const("e")) == "e"
    assert to_latex(Const("i")) == "i"
    assert to_latex(Sym("x", "n+1")) == "x_{n+1}"


def test_to_latex_output_is_katex_safe() -> None:
    from aicls.core.latex_safety import find_disallowed_commands

    for src, _ in CANONICAL:
        rendered = to_latex(parse_latex(src))
        assert find_disallowed_commands(rendered) == []
