"""Edge cases of the individual stages: error branches, helpers, lenient mode."""

from __future__ import annotations

import pytest
import sympy

from aicls.core.errors import AiclsError
from aicls.latex2sympy import (
    Add,
    Const,
    Integral,
    Mul,
    Neg,
    Num,
    ParseError,
    Prime,
    SemanticError,
    Sym,
    SymbolTable,
    UnsupportedConstruct,
    latex_to_sympy,
    normalize_latex,
    parse_latex,
    to_latex,
    to_sympy,
)
from aicls.latex2sympy.normalize import _find_group_end, _wrap_sqrt
from aicls.latex2sympy.roundtrip import (
    LEVEL_NUMERIC,
    expressions_equivalent,
    numerically_equal,
)

x = sympy.Symbol("x", real=True)
y = sympy.Symbol("y", real=True)


# ---------------------------------------------------------------------------------------------
# roundtrip helpers
# ---------------------------------------------------------------------------------------------
def test_numerically_equal_detects_equal_and_different() -> None:
    assert numerically_equal((x + 1) ** 2, x**2 + 2 * x + 1) is True
    assert numerically_equal(x**2, x**3) is False
    assert numerically_equal(x, x, samples=0) is False


def test_numerically_equal_skips_too_many_symbols() -> None:
    a, b, c, d = sympy.symbols("a b c d", real=True)
    assert numerically_equal(a + b + c + d, a + b + c + d) is False


def test_numerically_equal_ignores_singular_points() -> None:
    # 1/(x - 1) is finite on the sampling window, so the comparison still concludes
    assert numerically_equal(1 / (x - 10), 1 / (x - 10)) is True


def test_expressions_equivalent_levels() -> None:
    assert expressions_equivalent(x + x, 2 * x) is True
    assert expressions_equivalent(sympy.sin(x) ** 2 + sympy.cos(x) ** 2, sympy.Integer(1)) is True
    assert expressions_equivalent(x, y) is False
    # relations, sets and matrices are compared structurally only
    assert expressions_equivalent(sympy.Eq(x, 1), sympy.Eq(x, 1)) is True
    assert expressions_equivalent(sympy.Eq(x, 1), sympy.Eq(x, 2)) is False
    assert expressions_equivalent(sympy.Interval(0, 1), sympy.Interval(0, 2)) is False
    assert expressions_equivalent(sympy.Matrix([[1]]), sympy.Matrix([[2]])) is False
    # unevaluated calculus objects are never sampled
    assert expressions_equivalent(sympy.Integral(x, x), sympy.Integral(x + 0, x)) is True
    assert expressions_equivalent(sympy.Integral(x, x), sympy.Integral(x**2, x)) is False


def test_expressions_equivalent_uses_the_numeric_level() -> None:
    left = sympy.sqrt(x**2 + 2 * x + 1)
    right = sympy.Abs(x + 1)
    assert expressions_equivalent(left, right, numeric_samples=4) is True
    assert LEVEL_NUMERIC == "numeric"


# ---------------------------------------------------------------------------------------------
# normalisation helpers
# ---------------------------------------------------------------------------------------------
def test_wrap_sqrt_and_group_end_helpers() -> None:
    assert _wrap_sqrt("{x+1}") == r"\sqrt{x+1}"
    assert _wrap_sqrt("2") == r"\sqrt{2}"
    assert _find_group_end("{a}", 0) == 3
    assert _find_group_end("{a", 0) == -1
    assert normalize_latex("√{x+1}").text == r"\sqrt{x+1}"


def test_unwrap_applied_function_only_for_parenthesised_groups() -> None:
    assert normalize_latex(r"f{(x + 1)}").text == "f(x + 1)"
    assert normalize_latex(r"g{\left(f{\left(x \right)} \right)} = 1").text == "g(f(x ) ) = 1"
    # a plain brace group after a function letter is a product, not an application
    assert normalize_latex("f{x}").text == "f{x}"
    # an unterminated group is left alone
    assert normalize_latex("f{(x").text == "f{(x}"


def test_move_function_power_handles_missing_and_plain_exponents() -> None:
    assert normalize_latex(r"\sin{x}^2").text == r"\sin^{2}{x}"
    assert normalize_latex(r"\sin{x} + 1").text == r"\sin{x} + 1"
    assert latex_to_sympy(r"\ln{(x)}^{2}") == sympy.log(x) ** 2


def test_move_function_power_accepts_a_bare_command_exponent() -> None:
    r"""``\sin{x}^\alpha`` and ``\sin{x}^{\alpha}`` must mean the same thing.

    They did not: the braced spelling was repaired to ``\sin^{\alpha}{x}`` and parsed as
    ``sin(x)**alpha`` while the bare one was left alone and parsed as ``sin(x**alpha)`` -- the
    silent misparse ``_move_function_power`` exists to prevent, reached through the one spelling
    it did not recognise. Found on 2026-09-08 by ``test_latex_v1_is_idempotent``, because
    ``latex_v1``'s own spaced output is the spelling that used to match.
    """
    assert normalize_latex(r"\sin{x}^\alpha").text == r"\sin^{\alpha}{x}"
    assert normalize_latex(r"\sin{xx}^\alpha").text == r"\sin^{\alpha}{xx}"
    assert parse_latex(r"\sin{x}^\alpha") == parse_latex(r"\sin{x}^{\alpha}")
    # A command that takes its own argument is NOT the exponent atom: moving only its name would
    # tear ``\frac`` away from ``{1}{2}``.
    assert normalize_latex(r"\sin{x}^\frac{1}{2}").text == r"\sin{x}^\frac{1}{2}"


def test_decimal_comma_inside_a_function_argument() -> None:
    assert normalize_latex(r"\ln(x + 1,5)").text == r"\ln(x + 1.5)"
    assert normalize_latex("[0,1]").text == "[0,1]"


# ---------------------------------------------------------------------------------------------
# parser / lenient mode
# ---------------------------------------------------------------------------------------------
def test_lenient_mode_drops_unknown_macros() -> None:
    assert parse_latex(r"\foo{x} + 1", strict=False) == Add((Sym("x"), Num("1")))
    with pytest.raises(UnsupportedConstruct) as excinfo:
        parse_latex(r"\foo{x} + 1")
    assert excinfo.value.construct == r"\foo"


def test_lenient_mode_can_empty_the_input() -> None:
    with pytest.raises(ParseError):
        parse_latex(r"\foo", strict=False)


def test_unexpected_character_reports_the_offending_char() -> None:
    with pytest.raises(ParseError) as excinfo:
        parse_latex("x @ y")
    assert excinfo.value.details["char"] == "@"


def test_parse_with_repairs_returns_the_repair_list() -> None:
    from aicls.latex2sympy import LatexParser

    node, repairs = LatexParser().parse_with_repairs(r"\dfrac{1}{2")
    assert node is not None
    assert "frac_variant" in repairs
    assert "missing_closing_brace" in repairs


# ---------------------------------------------------------------------------------------------
# transformer error branches
# ---------------------------------------------------------------------------------------------
def test_matrix_environment_must_match_and_be_rectangular() -> None:
    with pytest.raises(SemanticError) as mismatch:
        parse_latex(r"\begin{pmatrix} 1 & 2 \end{bmatrix}")
    assert mismatch.value.details["reason"] == "matrix_environment_mismatch"
    with pytest.raises(SemanticError) as ragged:
        parse_latex(r"\begin{pmatrix} 1 & 2 \\ 3 \end{pmatrix}")
    assert ragged.value.details["reason"] == "matrix_ragged_rows"


def test_prime_error_branches() -> None:
    with pytest.raises(SemanticError) as arity:
        parse_latex("f'(x, y)")
    assert arity.value.details["reason"] == "prime_arity"
    with pytest.raises(SemanticError) as base:
        parse_latex("(x + 1)'")
    assert base.value.details["reason"] == "prime_on_non_symbol"


def test_limit_variable_must_be_a_symbol() -> None:
    with pytest.raises(SemanticError) as excinfo:
        parse_latex(r"\lim_{\pi \to 0} x")
    assert excinfo.value.details["reason"] == "limit_variable"


def test_empty_subscript_is_rejected() -> None:
    with pytest.raises(AiclsError):
        parse_latex("x_{}")


def test_negative_integral_body_keeps_the_sign() -> None:
    node = parse_latex(r"\int -x \, dx")
    assert isinstance(node, Integral)
    assert node.body == Neg(Sym("x"))
    assert to_sympy(node) == sympy.Integral(-x, x)


def test_integral_with_factors_after_the_differential() -> None:
    node = parse_latex(r"\int x \, dx \cdot 2")
    assert isinstance(node, Mul)
    assert isinstance(node.factors[0], Integral)
    assert node.factors[1] == Num("2")


def test_integral_of_a_bare_differential() -> None:
    node = parse_latex(r"\int_{0}^{1} dx")
    assert isinstance(node, Integral)
    assert node.body == Num("1")
    assert to_sympy(node).doit() == 1


# ---------------------------------------------------------------------------------------------
# to_sympy branches
# ---------------------------------------------------------------------------------------------
def test_unknown_number_set_is_a_semantic_error() -> None:
    with pytest.raises(SemanticError) as excinfo:
        latex_to_sympy(r"x \in \mathbb{X}")
    assert excinfo.value.details["reason"] == "unknown_number_set"


def test_sympy_construction_failure_becomes_semantic_error() -> None:
    with pytest.raises(SemanticError) as excinfo:
        latex_to_sympy(r"\ln((0, 1))")
    assert excinfo.value.details["reason"] == "sympy_construction_failed"


def test_lg_is_base_ten() -> None:
    assert latex_to_sympy(r"\lg(x)") == sympy.log(x, 10)


def test_prime_without_argument_and_of_an_expression() -> None:
    node = parse_latex("f'")
    assert node == Prime("f", None, 1)
    assert to_sympy(node) == sympy.Derivative(sympy.Function("f")(x), (x, 1))
    with pytest.raises(SemanticError) as excinfo:
        to_sympy(Prime("f", Add((Sym("x"), Num("1"))), 1))
    assert excinfo.value.details["reason"] == "prime_of_expression"


def test_constants_via_hand_built_nodes() -> None:
    assert to_sympy(Const("e")) is sympy.E
    assert to_sympy(Const("infty")) is sympy.oo
    assert to_sympy(Const("i")) == sympy.Symbol("i", real=True)
    declared = SymbolTable()
    declared.declare("pi", positive=True)
    assert to_sympy(Const("pi"), declared) == sympy.Symbol("pi", positive=True)


def test_symbol_table_declaration_helpers() -> None:
    table = SymbolTable()
    assert table.is_declared("q") is False
    table.declare("q", integer=True)
    assert table.is_declared("q") is True
    with table.bound("q", positive=True):
        assert table.symbol("q").is_positive is True
    assert table.symbol("q").is_integer is True


# ---------------------------------------------------------------------------------------------
# printer branches
# ---------------------------------------------------------------------------------------------
def test_to_latex_of_a_prime_without_argument() -> None:
    assert to_latex(Prime("f", None, 3)) == "f'''"


def test_error_construction_options() -> None:
    custom = ParseError(line=2, col=7, expected=["NUMBER"], snippet="x", message_tr="hata")
    assert custom.message_tr == "hata"
    assert custom.details["line"] == 2
    plain = UnsupportedConstruct(r"\foo")
    assert "note" not in plain.details
    assert plain.construct == r"\foo"
    detailed = UnsupportedConstruct(r"\foo", note="v1.1", details={"where": "test"})
    assert detailed.details == {"construct": r"\foo", "note": "v1.1", "where": "test"}
    assert SemanticError("reason", details={"where": "test"}).details["where"] == "test"


# ---------------------------------------------------------------------------------------------
# round-trip failure levels
# ---------------------------------------------------------------------------------------------
def test_roundtrip_reports_a_reparse_failure() -> None:
    from aicls.latex2sympy.roundtrip import REASON_REPARSE_FAILED, roundtrip_check

    # ln(0) is complex infinity; sympy prints \tilde{\infty}, which is out of scope
    result = roundtrip_check(r"\ln(0)")
    assert result.ok is False
    assert result.reason.startswith(REASON_REPARSE_FAILED)
    assert result.canonical


def test_roundtrip_reports_a_mismatch() -> None:
    from aicls.latex2sympy.roundtrip import REASON_MISMATCH, roundtrip_check

    # sqrt(-1) becomes the imaginary unit, which prints as ``i`` and re-parses as a symbol
    # (ParserConfig.imaginary_i defaults to False)
    result = roundtrip_check(r"\sqrt{-1}")
    assert result.ok is False
    assert result.reason == REASON_MISMATCH


def test_bound_restores_a_reserved_name() -> None:
    table = SymbolTable()
    assert "e" in table.reserved
    with table.bound("e", integer=True):
        assert "e" not in table.reserved
    assert "e" in table.reserved


def test_negated_group_integral_body() -> None:
    node = parse_latex(r"\int -(x \, dx)")
    assert isinstance(node, Integral)
    assert node.body == Neg(Sym("x"))
