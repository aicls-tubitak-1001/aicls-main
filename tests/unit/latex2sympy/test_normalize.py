"""Normalisation and tracked repairs (docs/mimari/latex-ayristirici.md 3.1)."""

from __future__ import annotations

import pytest

from aicls.latex2sympy import normalize_latex, parse_latex
from aicls.latex2sympy.normalize import (
    REPAIR_ABS_BARS,
    REPAIR_DECIMAL_COMMA,
    REPAIR_EMPTY_GROUP,
    REPAIR_FRAC_VARIANT,
    REPAIR_LEFT_RIGHT,
    REPAIR_MISSING_BRACE,
    REPAIR_SPACING,
    REPAIR_TEXT_DROPPED,
    REPAIR_UNICODE,
)


def test_normalize_latex_records_single_brace_repair() -> None:
    result = normalize_latex(r"\frac{1}{2")
    assert result.text == r"\frac{1}{2}"
    assert REPAIR_MISSING_BRACE in result.repairs
    # and the repaired text parses
    assert parse_latex(r"\frac{1}{2") is not None


def test_normalize_repairs_only_one_missing_brace() -> None:
    result = normalize_latex(r"\frac{1}{2")
    assert result.text.count("}") == 2
    two_missing = normalize_latex(r"\frac{{1}{2")
    assert REPAIR_MISSING_BRACE not in two_missing.repairs
    assert two_missing.text == r"\frac{{1}{2"


def test_normalize_removes_spacing_macros() -> None:
    result = normalize_latex(r"x \, + \; y \quad \! \displaystyle")
    assert "\\," not in result.text
    assert "\\quad" not in result.text
    assert "\\displaystyle" not in result.text
    assert REPAIR_SPACING in result.repairs


def test_normalize_left_right_pairs() -> None:
    result = normalize_latex(r"\left( x + 1 \right)")
    assert result.text == "( x + 1 )"
    assert REPAIR_LEFT_RIGHT in result.repairs
    dotted = normalize_latex(r"\left. x \right|")
    assert "\\left" not in dotted.text and "\\right" not in dotted.text


def test_normalize_dfrac_and_tfrac() -> None:
    assert normalize_latex(r"\dfrac{1}{2}").text == r"\frac{1}{2}"
    assert normalize_latex(r"\tfrac{1}{2}").text == r"\frac{1}{2}"
    assert REPAIR_FRAC_VARIANT in normalize_latex(r"\dfrac{1}{2}").repairs


def test_normalize_decimal_comma_between_digits_only() -> None:
    result = normalize_latex("3,5 + 1")
    assert result.text == "3.5 + 1"
    assert REPAIR_DECIMAL_COMMA in result.repairs
    # a comma that separates interval bounds must survive
    kept = normalize_latex("[0, 1]")
    assert kept.text == "[0, 1]"
    assert REPAIR_DECIMAL_COMMA not in kept.repairs
    assert normalize_latex("f(x, y)").text == "f(x, y)"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2×3", r"2\times 3"),
        ("2·3", r"2\cdot 3"),
        ("x−1", "x-1"),
        ("x²", "x^{2}"),
        ("x³", "x^{3}"),
        ("√2", r"\sqrt{2}"),
        ("∞", r"\infty"),
        ("π", r"\pi"),
        ("x≤1", r"x\le 1"),
        ("x≥1", r"x\ge 1"),
        ("x≠1", r"x\neq 1"),
        ("x→0", r"x\to 0"),
    ],
)
def test_normalize_unicode_operators(raw: str, expected: str) -> None:
    result = normalize_latex(raw)
    assert result.text == expected
    assert REPAIR_UNICODE in result.repairs


def test_normalize_absolute_value_bars() -> None:
    result = normalize_latex("|x| + |y|")
    assert result.text == r"\lvert x \rvert + \lvert y \rvert"
    assert REPAIR_ABS_BARS in result.repairs
    assert normalize_latex(r"\left| x \right|").text == r"\lvert x \rvert"


def test_normalize_drops_text_and_mathrm_d() -> None:
    result = normalize_latex(r"\text{alan} x")
    assert result.text == "x"
    assert REPAIR_TEXT_DROPPED in result.repairs
    assert normalize_latex(r"\int x \mathrm{d}x").text == r"\int x dx"


def test_normalize_drops_empty_groups() -> None:
    result = normalize_latex("x^{2}{}")
    assert result.text == "x^{2}"
    assert REPAIR_EMPTY_GROUP in result.repairs


def test_normalize_is_idempotent_on_examples() -> None:
    samples = [
        r"\frac{1}{2",
        r"\left( x \right)",
        "3,5",
        "|x|",
        r"\dfrac{a}{b}",
        "2×3",
        r"x \, y",
        "",
        "   ",
    ]
    for src in samples:
        once = normalize_latex(src).text
        twice = normalize_latex(once).text
        assert once == twice, src


def test_normalize_clean_input_reports_no_repairs() -> None:
    result = normalize_latex(r"\frac{x+1}{x-1}")
    assert result.text == r"\frac{x+1}{x-1}"
    assert result.repairs == []


def test_normalize_strips_control_characters_and_math_delimiters() -> None:
    assert normalize_latex("$x + 1$").text == "x + 1"
    assert normalize_latex(r"\[x\]").text == "x"
    assert normalize_latex("x  +   1" + chr(10)).text == "x + 1"
    assert normalize_latex("x\u200b+1").text == "x+1"
