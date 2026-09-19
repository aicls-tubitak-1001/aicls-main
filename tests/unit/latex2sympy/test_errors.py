"""Error policy: ParseError / UnsupportedConstruct / SemanticError only."""

from __future__ import annotations

import pytest

from aicls.core.errors import AiclsError, ErrorCode, InputValidationError
from aicls.latex2sympy import (
    MAX_INPUT_CHARS,
    ParseError,
    SemanticError,
    UnsupportedConstruct,
    latex_to_sympy,
    parse_latex,
)

UNSUPPORTED_CASES: list[tuple[str, str]] = [
    (r"\begin{cases} x & x > 0 \\ -x & x \le 0 \end{cases}", r"\begin{cases}"),
    (r"\begin{align} x &= 1 \end{align}", r"\begin{align}"),
    (r"\vec{v}", r"\vec"),
    (r"\hat{x}", r"\hat"),
    (r"\overline{z}", r"\overline"),
    (r"\label{eq:1}", r"\label"),
    (r"\tag{1}", r"\tag"),
    (r"\iint f dA", r"\iint"),
    (r"\binom{n}{k}", r"\binom"),
    (r"x \cup y", r"\cup"),
]


def test_parse_unsupported_cases_raises_with_construct_name() -> None:
    for src, construct in UNSUPPORTED_CASES:
        with pytest.raises(UnsupportedConstruct) as excinfo:
            parse_latex(src)
        err = excinfo.value
        assert err.construct == construct
        assert err.code is ErrorCode.UNSUPPORTED_CONSTRUCT
        assert err.http_status == 422
        assert construct in err.message_tr
        assert err.details["construct"] == construct


def test_unsupported_construct_is_reported_before_a_parse_error() -> None:
    # the input is also syntactically broken; the named construct wins
    with pytest.raises(UnsupportedConstruct):
        parse_latex(r"\vec{v} + ")


def test_non_strict_mode_drops_decorative_commands() -> None:
    node = parse_latex(r"\overline{x} + 1", strict=False)
    assert node is not None
    # structural constructs stay unsupported even in lenient mode
    with pytest.raises(UnsupportedConstruct):
        parse_latex(r"\begin{cases} 1 \end{cases}", strict=False)


def test_parse_error_carries_position_and_snippet() -> None:
    with pytest.raises(ParseError) as excinfo:
        parse_latex(r"\frac{1}{2} +")
    err = excinfo.value
    assert err.code is ErrorCode.PARSE_ERROR
    assert err.line == 1
    assert isinstance(err.expected, list)
    assert err.snippet
    problem = err.to_problem()
    assert problem["code"] == "PARSE_ERROR"
    assert problem["position"] == err.col


def test_lexer_error_is_a_parse_error() -> None:
    with pytest.raises(ParseError):
        parse_latex("x @ y")


def test_empty_input_is_rejected() -> None:
    for src in ("", "   ", r"\,"):
        with pytest.raises(AiclsError):
            parse_latex(src)


def test_input_over_max_chars_is_rejected() -> None:
    assert MAX_INPUT_CHARS == 2048
    with pytest.raises(InputValidationError) as excinfo:
        parse_latex("x + " * 1000)
    assert excinfo.value.details["max_len"] == MAX_INPUT_CHARS


def test_semantic_errors() -> None:
    with pytest.raises(SemanticError):
        parse_latex(r"\int x^{2}")  # no differential
    with pytest.raises(SemanticError):
        parse_latex("2_{3}")  # subscript on a number


def test_parse_never_raises_a_non_aicls_error_on_hostile_input() -> None:
    hostile = [
        "\\",
        "\\\\",
        "{",
        "}",
        "^",
        "_",
        "()",
        "[]",
        "$$",
        "\x00x",
        "\\frac",
        "\\frac{}{}",
        "\\sqrt[]{}",
        "\\lim_{}",
        "\\sum_{}^{}",
        "\\int_",
        "f(",
        "'''",
        "....",
        "1 2 3 \\cdot",
        "çöğü",  # Turkish letters are not math symbols
        "\U0001f600",
    ]
    for src in hostile:
        try:
            parse_latex(src)
        except AiclsError:
            pass
        else:
            pass  # parsing successfully is fine too, crashing is not


def test_to_sympy_of_unsupported_node_combination() -> None:
    # \pm nested inside another operator has no single sympy expression
    with pytest.raises(SemanticError):
        latex_to_sympy(r"\frac{1 \pm 2}{3}")


def test_mixed_partial_derivative_is_refused_by_name_not_as_a_syntax_error() -> None:
    """A shape the v1 AST cannot hold must still reach the user with a construct name."""
    from aicls.latex2sympy.parser import MIXED_PARTIAL_CONSTRUCT

    for src in (
        r"\frac{\partial^{2} f}{\partial x \partial y}",
        r"\frac{\partial^{2} u}{\partial x \partial t}",
        r"\frac{\partial^{3} f}{\partial x^{2} \partial y}",
    ):
        for strict in (True, False):
            with pytest.raises(UnsupportedConstruct) as excinfo:
                parse_latex(src, strict=strict)
            assert excinfo.value.construct == MIXED_PARTIAL_CONSTRUCT
            assert excinfo.value.details["note"].startswith("v1.1")


def test_single_variable_partials_are_not_mistaken_for_mixed_partials() -> None:
    """The guard must not fire on a sum of partials or on nested partial operators."""
    for src in (
        r"\frac{\partial f}{\partial x} + \frac{\partial f}{\partial y}",
        r"\frac{\partial}{\partial x}\frac{\partial}{\partial y} f",
        r"\frac{\partial^{2} f}{\partial x^{2}}",
    ):
        assert parse_latex(src) is not None
