"""The curriculum compiler itself: schema refusals, report arithmetic, output files, CLI.

``tests/unit/datasets/test_curriculum.py`` is about the *corpus*; this module is about
``scripts/build_curriculum_testset.py``. Every refusal path is exercised, because a compiler
that silently accepts a broken card would let a wrong number reach the İP-4 report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import sympy
import yaml  # type: ignore[import-untyped]  # PyYAML ships no stubs, types-PyYAML is absent

from aicls.core.errors import InputValidationError
from aicls.mathengine import MathEngine

REPO_ROOT = Path(__file__).resolve().parents[3]
CARDS_DIR = REPO_ROOT / "datasets" / "curriculum" / "cards"


@pytest.fixture(scope="module")
def build(load_script: Any) -> Any:
    """The compiler module, imported by path."""
    return load_script("scripts/build_curriculum_testset.py")


@pytest.fixture(scope="module")
def good_row(build: Any) -> dict[str, Any]:
    """A minimal, valid card used as the base of every negative case."""
    return {
        "id": "limit-0001",
        "topic": "limit",
        "subtopic": "polinom_limiti",
        "chapter_ref": build.CHAPTER_BY_TOPIC["limit"],
        "difficulty": 1,
        "input": {"kind": "latex", "value": "\\lim_{x \\to 2} \\left(x^{2}\\right)"},
        "command": {"op": "limit", "expr": "x^{2}", "var": "x", "to": "2"},
        "expected": {"latex": "4", "srepr": "Integer(4)"},
        "checker": "symbolic",
        "checker_args": {},
        "source": "sablon:test/limit",
        "author": "test",
        "reviewed_by": "",
        "status": "frozen",
        "notes_tr": "Yerine koyma ile limit 4 olur.",
    }


def _cards_dir(tmp_path: Path, rows: list[Any]) -> Path:
    directory = tmp_path / "cards"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "limit.yaml").write_text(
        yaml.safe_dump(rows, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n"
    )
    return directory


def _mutate(row: dict[str, Any], **changes: Any) -> dict[str, Any]:
    copied: dict[str, Any] = json.loads(json.dumps(row))
    copied.update(changes)
    return copied


# --------------------------------------------------------------------------------------------
# schema refusals
# --------------------------------------------------------------------------------------------
def test_a_valid_card_loads(build: Any, good_row: dict[str, Any], tmp_path: Path) -> None:
    cards = build.load_cards(_cards_dir(tmp_path, [good_row]))
    assert [card.id for card in cards] == ["limit-0001"]
    assert cards[0].topic == "limit"
    assert cards[0].path.name == "limit.yaml"


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"checker": "elle"}, "unknown_checker"),
        ({"status": "yayinda"}, "unknown_status"),
        ({"difficulty": 9}, "difficulty_out_of_range"),
        ({"difficulty": "1"}, "difficulty_out_of_range"),
        ({"chapter_ref": "aicls:bolum-9"}, "chapter_mismatch"),
        ({"id": "turev-0001"}, "id_topic_mismatch"),
        ({"topic": "olmayan"}, "unknown_topic"),
        ({"command": {"op": "hesapla", "expr": "x"}}, "unknown_op"),
        ({"subtopic": "  "}, "empty_field"),
        ({"notes_tr": ""}, "empty_field"),
        ({"source": ""}, "empty_field"),
        ({"author": ""}, "empty_field"),
        ({"input": {"kind": "latex", "value": "   "}}, "empty_input"),
        ({"input": {"kind": "ses", "value": "x"}}, "unknown_input_kind"),
        ({"expected": {"latex": "4", "srepr": ""}}, "empty_srepr"),
    ],
)
def test_a_broken_card_is_refused_with_a_named_reason(
    build: Any, good_row: dict[str, Any], tmp_path: Path, changes: dict[str, Any], reason: str
) -> None:
    with pytest.raises(InputValidationError) as info:
        build.load_cards(_cards_dir(tmp_path, [_mutate(good_row, **changes)]))
    assert info.value.details["reason"] == reason


def test_an_extra_field_is_refused(build: Any, good_row: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(InputValidationError) as info:
        build.load_cards(_cards_dir(tmp_path, [_mutate(good_row, extra_alan=1)]))
    assert info.value.details["reason"] == "unknown_fields"


def test_a_row_that_is_not_a_mapping_is_refused(build: Any, tmp_path: Path) -> None:
    with pytest.raises(InputValidationError) as info:
        build.load_cards(_cards_dir(tmp_path, ["sadece bir dize"]))
    assert info.value.details["reason"] == "row_not_a_mapping"


def test_a_card_file_that_is_not_a_list_is_refused(build: Any, tmp_path: Path) -> None:
    directory = tmp_path / "cards"
    directory.mkdir()
    (directory / "limit.yaml").write_text("id: limit-0001\n", encoding="utf-8", newline="\n")
    with pytest.raises(InputValidationError) as info:
        build.load_cards(directory)
    assert info.value.details["reason"] == "card_file_not_a_list"


def test_an_empty_cards_directory_is_refused(build: Any, tmp_path: Path) -> None:
    (tmp_path / "cards").mkdir()
    with pytest.raises(InputValidationError) as info:
        build.load_cards(tmp_path / "cards")
    assert info.value.details["reason"] == "no_card_files"


def test_an_unrebuildable_srepr_is_refused(build: Any) -> None:
    with pytest.raises(InputValidationError) as info:
        build.rebuild("Mul(Integer(2)")
    assert info.value.details["reason"] == "srepr_not_rebuildable"


def test_check_expected_reports_an_unrebuildable_srepr(
    build: Any, good_row: dict[str, Any]
) -> None:
    card = build.Card(
        path=Path("x.yaml"), row=_mutate(good_row, expected={"latex": "4", "srepr": "("})
    )
    assert "srepr_not_rebuildable" in build.check_expected(card)


# --------------------------------------------------------------------------------------------
# report arithmetic
# --------------------------------------------------------------------------------------------
def _verdict(build: Any, **kwargs: Any) -> Any:
    base = {
        "id": "limit-0001",
        "topic": "limit",
        "outcome": "correct",
        "reason": "",
        "parse_ok": True,
        "roundtrip_ok": True,
        "is_latex": True,
        "elapsed_ms": 1.0,
    }
    base.update(kwargs)
    return build.Verdict(**base)


def test_an_empty_report_reports_zeroes_rather_than_dividing_by_zero(build: Any) -> None:
    report = build.Report([])
    assert report.n == 0
    assert report.correct_rate == 0.0
    assert report.parse_success == 0.0
    assert report.parse_only_success == 0.0
    assert report.timing_ms == {"mean": 0.0, "p95": 0.0}
    assert report.by_topic == {}


def test_report_counts_every_outcome_separately(build: Any) -> None:
    report = build.Report(
        [
            _verdict(build, id="a", elapsed_ms=1.0),
            _verdict(build, id="b", outcome="wrong", reason="different", elapsed_ms=3.0),
            _verdict(build, id="c", outcome="unsupported", reason="x", elapsed_ms=2.0),
            _verdict(build, id="d", outcome="abstain", reason="y", elapsed_ms=4.0),
            _verdict(build, id="e", outcome="error", parse_ok=False, roundtrip_ok=False),
            _verdict(build, id="f", roundtrip_ok=False),
            _verdict(build, id="g", is_latex=False, topic="turev"),
        ]
    )
    assert report.n == 7
    assert report.correct == 3
    assert report.unsupported == 1
    assert report.abstained == 1
    assert report.n_latex == 6
    assert report.parse_success == pytest.approx(4 / 6)
    assert report.parse_only_success == pytest.approx(5 / 6)
    assert report.roundtrip_failures == ["f"]
    assert len(report.failures) == 4
    assert report.by_topic == {"limit": (2, 6), "turev": (1, 1)}
    assert report.timing_ms["p95"] >= report.timing_ms["mean"]


def test_report_payload_carries_the_provenance_sentences(build: Any) -> None:
    report = build.Report([_verdict(build)])
    payload = build.report_payload(report, dict.fromkeys(build.TOPIC_QUOTAS, 0))
    assert payload["n"] == 1
    assert payload["correct_rate"] == 1.0
    assert "ÖRNEKLENMEMİŞTİR" in payload["provenance_tr"]
    assert "not an independent" in payload["provenance_en"].lower()
    assert payload["sympy_version"] == sympy.__version__


# --------------------------------------------------------------------------------------------
# verification paths
# --------------------------------------------------------------------------------------------
def test_a_card_whose_input_does_not_parse_is_an_error_not_a_wrong_answer(
    build: Any, good_row: dict[str, Any]
) -> None:
    row = _mutate(good_row, input={"kind": "latex", "value": "\\vec{v} + 1"})
    verdict = build.verify_card(build.Card(path=Path("x.yaml"), row=row), MathEngine())
    assert verdict.outcome == "error"
    assert verdict.parse_ok is False
    assert "parse" in verdict.reason


def test_a_card_the_engine_declines_is_counted_as_unsupported(build: Any) -> None:
    row = {
        "id": "belirsiz_integral-9999",
        "topic": "belirsiz_integral",
        "subtopic": "kuvvet_kurali",
        "chapter_ref": build.CHAPTER_BY_TOPIC["belirsiz_integral"],
        "difficulty": 3,
        "input": {"kind": "latex", "value": "\\int e^{x^{2}} \\, dx"},
        "command": {"op": "integrate", "expr": "e^{x^{2}}", "var": "x"},
        "expected": {"latex": "0", "srepr": "Integer(0)"},
        "checker": "symbolic",
        "checker_args": {},
        "source": "sablon:test/unsupported",
        "author": "test",
        "reviewed_by": "",
        "status": "draft",
        "notes_tr": "SymPy bu ilkeli temel fonksiyonlarla veremez.",
    }
    verdict = build.verify_card(build.Card(path=Path("x.yaml"), row=row), MathEngine())
    assert verdict.outcome in {"unsupported", "wrong"}
    if verdict.outcome == "unsupported":
        assert "UnsupportedProblem" in verdict.reason


def test_a_wrong_expected_answer_is_caught(build: Any, good_row: dict[str, Any]) -> None:
    row = _mutate(good_row, expected={"latex": "5", "srepr": "Integer(5)"})
    verdict = build.verify_card(build.Card(path=Path("x.yaml"), row=row), MathEngine())
    assert verdict.outcome == "wrong"


def test_expected_latex_that_is_not_the_canonical_printing_is_caught(
    build: Any, good_row: dict[str, Any]
) -> None:
    row = _mutate(good_row, expected={"latex": "2 + 2", "srepr": "Integer(4)"})
    verdict = build.verify_card(build.Card(path=Path("x.yaml"), row=row), MathEngine())
    assert verdict.outcome == "wrong"
    assert "canonical" in verdict.reason


def test_an_antiderivative_that_is_not_one_is_caught(build: Any) -> None:
    """The ``up_to_constant`` path also checks d/dx(answer) against the integrand."""
    row = {
        "id": "belirsiz_integral-9998",
        "topic": "belirsiz_integral",
        "subtopic": "kuvvet_kurali",
        "chapter_ref": build.CHAPTER_BY_TOPIC["belirsiz_integral"],
        "difficulty": 1,
        "input": {"kind": "latex", "value": "\\int x \\, dx"},
        "command": {"op": "integrate", "expr": "x", "var": "x"},
        "expected": {"latex": "x^{3}", "srepr": "Pow(Symbol('x', real=True), Integer(3))"},
        "checker": "symbolic",
        "checker_args": {"up_to_constant": True, "var": "x"},
        "source": "sablon:test/antiderivative",
        "author": "test",
        "reviewed_by": "",
        "status": "draft",
        "notes_tr": "Kasıtlı olarak yanlış ilkel.",
    }
    verdict = build.verify_card(build.Card(path=Path("x.yaml"), row=row), MathEngine())
    assert verdict.outcome == "wrong"
    assert "constant" in verdict.reason


def test_a_text_row_is_verified_without_a_parse_denominator(
    build: Any, good_row: dict[str, Any]
) -> None:
    row = _mutate(good_row, input={"kind": "text_tr", "value": "$x^{2}$ limitini hesapla"})
    verdict = build.verify_card(build.Card(path=Path("x.yaml"), row=row), MathEngine())
    assert verdict.outcome == "correct"
    assert verdict.is_latex is False


def test_verify_uses_a_default_engine_when_none_is_given(
    build: Any, good_row: dict[str, Any]
) -> None:
    report = build.verify([build.Card(path=Path("x.yaml"), row=good_row)])
    assert report.n == 1
    assert report.correct_rate == 1.0


def test_plot_ok_requires_actual_png_bytes(build: Any, good_row: dict[str, Any]) -> None:
    from aicls.mathengine import Result

    row = _mutate(good_row, checker="plot_ok")
    card = build.Card(path=Path("x.yaml"), row=row)
    empty = Result(kind="expr", latex="4", plain="4", srepr="Integer(4)", plot_png=None)
    assert build._outcome_for_result(card, empty)[0] == "wrong"


def test_plot_ok_also_checks_that_the_plotted_expression_is_the_expected_one(
    build: Any, good_row: dict[str, Any]
) -> None:
    from aicls.mathengine import Result

    card = build.Card(path=Path("x.yaml"), row=_mutate(good_row, checker="plot_ok"))
    drawn = Result(
        kind="plot", latex="5", plain="5", srepr="Integer(5)", plot_png=b"\x89PNG-not-really"
    )
    outcome, reason = build._outcome_for_result(card, drawn)
    assert outcome == "wrong"
    assert reason


def test_a_command_the_parser_rejects_is_an_error_not_an_unsupported_problem(
    build: Any, good_row: dict[str, Any]
) -> None:
    """Only ``UnsupportedProblem`` counts as "unsupported"; a parse failure is an error."""
    row = _mutate(good_row, command={"op": "simplify", "expr": "\\vec{v} + 1"})
    verdict = build.verify_card(build.Card(path=Path("x.yaml"), row=row), MathEngine())
    assert verdict.outcome == "error"
    assert "UnsupportedConstruct" in verdict.reason or "ParseError" in verdict.reason


def test_numeric_checker_handles_a_pure_number(build: Any) -> None:
    args = {"tolerance": 1e-9, "domain": [-1.0, 1.0], "var": "x"}
    assert build.compare("numeric", sympy.Rational(1, 3), sympy.Rational(2, 6), args)[0]
    assert not build.compare("numeric", sympy.Integer(2), sympy.Integer(3), args)[0]


def test_numeric_checker_reports_when_no_probe_can_be_evaluated(build: Any) -> None:
    """Every probe sits on a pole: the checker must say so, not silently pass."""
    x = sympy.Symbol("x", real=True)
    args = {"tolerance": 1e-9, "domain": [0.0, 0.0], "var": "x"}
    ok, reason = build.compare("numeric", 1 / x, 2 / x, args)
    assert not ok
    assert "could not evaluate" in reason


def test_numeric_checker_skips_a_probe_on_a_pole_and_uses_the_rest(build: Any) -> None:
    x = sympy.Symbol("x", real=True)
    args = {"tolerance": 1e-9, "domain": [0.0, 1.0], "var": "x"}
    ok, reason = build.compare("numeric", 1 / x, 1 / x + 1, args)
    assert not ok
    assert "numeric gap" in reason


def test_numeric_checker_abstains_on_an_undefined_difference(build: Any) -> None:
    """``zoo - 1`` is NaN, not "different": the checker must say it could not decide."""
    args = {"tolerance": 1e-9, "domain": [-1.0, 1.0], "var": "x"}
    ok, reason = build.compare("numeric", sympy.zoo, sympy.Integer(1), args)
    assert not ok
    assert "could not evaluate" in reason


def test_a_roundtrip_mismatch_is_not_counted_as_a_wrong_answer(
    build: Any, good_row: dict[str, Any]
) -> None:
    """The strict parse metric records it; the CAS correctness rate must not."""
    expr = "\\frac{1 \\left(1 - \\left(\\frac{1}{3}\\right)^{n}\\right)}{1 - \\frac{1}{3}}"
    row = _mutate(
        good_row,
        input={"kind": "latex", "value": f"\\lim_{{n \\to \\infty}} {expr}"},
        expected={"latex": "\\frac{3}{2}", "srepr": "Rational(3, 2)"},
        command={"op": "limit", "expr": expr, "var": "n", "to": "\\infty"},
    )
    verdict = build.verify_card(build.Card(path=Path("x.yaml"), row=row), MathEngine())
    assert verdict.outcome == "correct"
    assert verdict.parse_ok is True
    assert verdict.roundtrip_ok is False
    report = build.Report([verdict])
    assert report.correct_rate == 1.0
    assert report.parse_success == 0.0
    assert report.roundtrip_failures == [verdict.id]


# --------------------------------------------------------------------------------------------
# output files and the CLI
# --------------------------------------------------------------------------------------------
def test_write_read_round_trip_of_problems_and_frozen(
    build: Any, good_row: dict[str, Any], tmp_path: Path
) -> None:
    rows = [good_row, _mutate(good_row, id="limit-0002")]
    problems = build.write_problems(rows, tmp_path / "problems.jsonl")
    frozen = build.write_frozen(tmp_path / "FROZEN.sha256", problems)

    read_back = build.read_problems(problems)
    assert [row["id"] for row in read_back] == [build.HEADER_ID, "limit-0001", "limit-0002"]
    assert read_back[0]["n_problems"] == 2
    assert read_back[0]["counts"]["limit"] == 2

    digests = build.read_frozen(frozen)
    assert set(digests) == {build.HEADER_ID, "limit-0001", "limit-0002"}
    assert frozen.read_text(encoding="utf-8").startswith("#")
    assert build.DATASET_VERSION in frozen.read_text(encoding="utf-8")

    # rewriting the same rows must produce the same bytes (determinism)
    before = problems.read_bytes()
    build.write_problems(list(reversed(rows)), problems)
    assert problems.read_bytes() == before


def test_main_check_writes_nothing_and_returns_zero(
    build: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "problems.jsonl"
    code = build.main(
        ["--check", "--sample", "97", "--out", str(out), "--frozen", str(tmp_path / "F.sha256")]
    )
    assert code == 0
    assert not out.exists()
    printed = capsys.readouterr().out
    assert "CAS doğruluğu" in printed
    assert "kota" in printed


def test_main_writes_the_two_output_files_and_a_report(
    build: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "problems.jsonl"
    frozen = tmp_path / "FROZEN.sha256"
    report = tmp_path / "metrics" / "cas.json"
    code = build.main(
        [
            "--skip-verify",
            "--out",
            str(out),
            "--frozen",
            str(frozen),
            "--report-json",
            str(report),
        ]
    )
    assert code == 0
    assert out.is_file() and frozen.is_file() and report.is_file()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["dataset_version"] == build.DATASET_VERSION
    assert payload["n"] == 0  # --skip-verify means nothing was executed
    assert "yazıldı" in capsys.readouterr().out
    # the freshly written file must equal the committed one
    committed = (REPO_ROOT / "datasets" / "curriculum" / "problems.jsonl").read_bytes()
    assert out.read_bytes() == committed


def test_main_fails_loudly_on_a_quota_violation(
    build: Any, good_row: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cards = _cards_dir(tmp_path, [good_row])
    code = build.main(["--check", "--cards-dir", str(cards)])
    assert code == 1
    captured = capsys.readouterr()
    assert "SORUN" in captured.err
    assert "limit: 1 < 120" in captured.err


def test_main_fails_when_the_parse_success_rate_drops(
    build: Any, good_row: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unparsable input must fail the build, not quietly lower a reported rate."""
    broken = _mutate(good_row, input={"kind": "latex", "value": "\\vec{v} + 1"})
    code = build.main(["--check", "--cards-dir", str(_cards_dir(tmp_path, [broken]))])
    assert code == 1
    assert "parse_success" in capsys.readouterr().err


def test_topic_counts_ignores_the_header_row(build: Any, good_row: dict[str, Any]) -> None:
    header = build.build_header([good_row])
    counts = build.topic_counts([header, good_row])
    assert counts["limit"] == 1
    assert sum(counts.values()) == 1
