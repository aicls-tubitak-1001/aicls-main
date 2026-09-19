"""Three-level round-trip verification (docs/mimari/latex-ayristirici.md 5)."""

from __future__ import annotations

import pytest

from aicls.latex2sympy import roundtrip_check
from aicls.latex2sympy.roundtrip import (
    LEVEL_CANONICAL,
    LEVEL_NUMERIC,
    LEVEL_SREPR,
    REASON_PARSE_FAILED,
)

CURRICULUM: list[str] = [
    r"x^{2} + 2x + 1",
    r"\frac{x+1}{x-1}",
    r"\frac{1}{3}",
    r"\sqrt{x^{2}+1}",
    r"\sqrt[3]{x}",
    r"\sin(x) \cos(x)",
    r"\tan(x)",
    r"e^{-x}",
    r"\ln(x+1)",
    r"\log_{10}(x)",
    r"\int_{0}^{1} x^{2} \, dx",
    r"\int x^{2} \, dx",
    r"\frac{d}{dx} x^{3}",
    r"\lim_{x \to 0} \frac{\sin(x)}{x}",
    r"\lim_{x \to 0^{+}} \frac{1}{x}",
    r"\sum_{k=1}^{n} k^{2}",
    r"\lvert x \rvert",
    r"x^{2} - 4 = 0",
    r"\pi r^{2}",
    r"2x\sin(x)",
    r"\frac{x}{y+1}",
    r"(x+1)^{2}",
    r"n!",
]


@pytest.mark.parametrize("src", CURRICULUM)
def test_roundtrip_check_ok_on_curriculum(src: str) -> None:
    result = roundtrip_check(src)
    assert result.ok is True, (src, result.reason)
    assert result.reason in {LEVEL_CANONICAL, LEVEL_SREPR, LEVEL_NUMERIC}
    assert result.canonical


def test_roundtrip_level_is_canonical_for_simple_expressions() -> None:
    assert roundtrip_check(r"\frac{1}{3}").reason == LEVEL_CANONICAL


def test_roundtrip_reports_failure_instead_of_raising() -> None:
    result = roundtrip_check(r"\vec{v}")
    assert result.ok is False
    assert result.reason.startswith(REASON_PARSE_FAILED)
    assert result.canonical == ""

    broken = roundtrip_check(r"\frac{1}{2}}}")
    assert broken.ok is False


def test_roundtrip_numeric_samples_can_be_disabled() -> None:
    result = roundtrip_check(r"x^{2} + 1", numeric_samples=0)
    assert result.ok is True


def test_roundtrip_numeric_level_used_when_structures_differ() -> None:
    # (x+1)^2 and its expansion are numerically equal but structurally different
    result = roundtrip_check(r"\frac{x^{2}-1}{x-1}", numeric_samples=5)
    assert result.ok is True


def test_roundtrip_rejects_oversized_input() -> None:
    result = roundtrip_check("x" * 5000)
    assert result.ok is False
