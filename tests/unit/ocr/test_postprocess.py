"""Postprocessing tests: brace repair, whitespace canonicalisation, grammar precheck."""

from __future__ import annotations

import pytest

from aicls.ocr import postprocess_latex
from aicls.ocr.postprocess import canonicalize_whitespace, grammar_precheck, repair_braces


class TestRepairBraces:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("\\frac{a}{b}", "\\frac{a}{b}"),
            ("\\frac{a}{b", "\\frac{a}{b}"),
            ("x^{2", "x^{2}"),
            ("x^2}", "x^2"),
            ("{{a}", "{{a}}"),
            ("\\{a\\}", "\\{a\\}"),
            ("\\{a", "\\{a"),
            ("", ""),
        ],
    )
    def test_repair(self, raw: str, expected: str) -> None:
        assert repair_braces(raw) == expected

    def test_dangling_backslash_is_dropped_before_closing(self) -> None:
        assert repair_braces("{\\") == "{}"
        assert repair_braces("\\frac{a\\\\}{b}") == "\\frac{a\\\\}{b}"

    def test_repaired_output_is_balanced(self) -> None:
        repaired = repair_braces("\\frac{\\sqrt{x}{2")
        assert repaired.count("{") == repaired.count("}")

    def test_repair_is_idempotent(self) -> None:
        once = repair_braces("\\sqrt{x")
        assert repair_braces(once) == once


class TestWhitespace:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  x  +  1 ", "x+1"),
            ("\\frac { a } { b }", "\\frac{a}{b}"),
            ("\\sin   x", "\\sin x"),
            ("a\t\nb", "ab"),
        ],
    )
    def test_canonicalize(self, raw: str, expected: str) -> None:
        assert canonicalize_whitespace(raw) == expected

    def test_canonicalize_is_idempotent(self) -> None:
        once = canonicalize_whitespace("\\lim _ { x \\to 0 } x")
        assert canonicalize_whitespace(once) == once


class TestGrammarPrecheck:
    def test_valid_expression_passes(self) -> None:
        report = grammar_precheck("\\frac{x^{2}}{2}+1")
        assert report.ok is True
        assert report.reasons == []

    @pytest.mark.parametrize(
        ("raw", "reason"),
        [
            ("", "empty"),
            ("x^{2", "unbalanced_braces"),
            ("\\left(x", "unbalanced_delimiters"),
            ("x+", "trailing_operator"),
            ("\\evil{x}", "disallowed_commands"),
        ],
    )
    def test_failures_are_reported(self, raw: str, reason: str) -> None:
        report = grammar_precheck(raw)
        assert report.ok is False
        assert reason in report.reasons

    def test_report_is_serialisable(self) -> None:
        report = grammar_precheck("x+")
        assert report.as_dict() == {"ok": False, "reasons": ["trailing_operator"]}


class TestPostprocessLatex:
    def test_pipeline_repairs_and_sanitizes(self) -> None:
        assert postprocess_latex("  \\frac { x } { 2 ") == "\\frac{x}{2}"

    def test_forbidden_commands_are_removed_by_the_core_sanitizer(self) -> None:
        assert "\\href" not in postprocess_latex("\\href{http://x}{y}+1")

    def test_math_delimiters_are_stripped(self) -> None:
        assert postprocess_latex("$x^{2}$") == "x^{2}"

    def test_sanitize_can_be_disabled(self) -> None:
        assert postprocess_latex("\\unknowncmd{x}", sanitize=False) == "\\unknowncmd{x}"

    def test_empty_input_stays_empty(self) -> None:
        assert postprocess_latex("   ") == ""
