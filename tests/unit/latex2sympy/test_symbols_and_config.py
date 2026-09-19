"""SymbolTable assumptions, reserved names and ParserConfig switches."""

from __future__ import annotations

import pytest
import sympy

from aicls.latex2sympy import (
    ParserConfig,
    SemanticError,
    SymbolTable,
    latex_to_sympy,
    parse_latex,
    to_sympy,
)


def test_default_assumptions_are_real() -> None:
    table = SymbolTable()
    for name in ("x", "y", "z", "t", "a", "alpha"):
        assert table.symbol(name).is_real is True


def test_declare_overrides_assumptions() -> None:
    table = SymbolTable()
    table.declare("n", integer=True, positive=True)
    n = table.symbol("n")
    assert n.is_integer is True
    assert n.is_positive is True
    # the same object comes back on the next lookup
    assert table.symbol("n") is n


def test_reserved_names_and_override() -> None:
    table = SymbolTable()
    assert "e" in table.reserved
    assert "pi" in table.reserved
    assert "infty" in table.reserved
    assert latex_to_sympy("e") is sympy.E
    declared = SymbolTable()
    declared.declare("e", positive=True)
    assert "e" not in declared.reserved
    assert latex_to_sympy("e", symbols=declared) == sympy.Symbol("e", positive=True)


def test_bound_scope_is_restored() -> None:
    table = SymbolTable()
    outer = table.symbol("k")
    with table.bound("k", integer=True):
        assert table.symbol("k").is_integer is True
    assert table.symbol("k") == outer


def test_sum_bound_variable_is_integer_everywhere_in_the_body() -> None:
    expr = latex_to_sympy(r"\sum_{k=1}^{n} k^{2}")
    assert isinstance(expr, sympy.Sum)
    index = expr.limits[0][0]
    assert index.is_integer is True
    # the body uses exactly the same symbol, otherwise Sum would not bind it
    assert expr.function.free_symbols == {index}


def test_parser_config_imaginary_i() -> None:
    assert latex_to_sympy("i") == sympy.Symbol("i", real=True)
    assert latex_to_sympy("i", config=ParserConfig(imaginary_i=True)) is sympy.I


def test_parser_config_log_base() -> None:
    x = sympy.Symbol("x", real=True)
    assert latex_to_sympy(r"\log x") == sympy.log(x)
    assert latex_to_sympy(r"\log x", config=ParserConfig(log_base="10")) == sympy.log(x, 10)
    assert latex_to_sympy(r"\log_{10} x") == sympy.log(x, 10)


def test_parser_config_real_symbols_off() -> None:
    plain = latex_to_sympy("x", config=ParserConfig(real_symbols=False))
    assert plain == sympy.Symbol("x")
    assert plain.is_real is None


def test_symbol_with_subscript_keeps_latex_name() -> None:
    expr = latex_to_sympy("x_{1} + x_{2}")
    names = sorted(str(s) for s in expr.free_symbols)
    assert names == ["x_{1}", "x_{2}"]


def test_declaring_a_multi_letter_identifier() -> None:
    # ``xy`` is x*y by default (ADR-0004)
    x = sympy.Symbol("x", real=True)
    y = sympy.Symbol("y", real=True)
    assert latex_to_sympy("xy") == x * y


def test_to_sympy_unevaluated_mode_keeps_structure() -> None:
    node = parse_latex(r"\frac{2}{4}")
    assert to_sympy(node) == sympy.Rational(1, 2)
    raw = to_sympy(node, evaluate=False)
    assert raw != sympy.Rational(1, 2)
    assert raw.doit() == sympy.Rational(1, 2)


def test_subscript_on_non_symbol_is_semantic_error() -> None:
    with pytest.raises(SemanticError):
        parse_latex("2_{3}")
