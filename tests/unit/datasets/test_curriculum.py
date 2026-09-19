"""The İP-4 curriculum evaluation set: schema, quotas, freeze and the compiler's checkers.

The corpus itself lives in ``datasets/curriculum/`` and is compiled by
``scripts/build_curriculum_testset.py``. These tests fix the three properties the progress
report depends on:

* the cards and the compiled ``problems.jsonl`` say the same thing, and the topic distribution
  matches the İP-4 quota table (``test_curriculum_cards_compile_and_quotas_match``);
* a frozen row's bytes never change (``test_frozen_curriculum_rows_unchanged``);
* the compiler's comparison strategies actually distinguish a right answer from a wrong one --
  otherwise "> 99 % CAS accuracy" would be a tautology.

The full 1200-row engine run is a separate, ``slow``-marked test at the bottom of this file.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pytest
import sympy

from aicls.core.errors import AiclsError, InputValidationError
from aicls.core.hashing import sha256_json
from aicls.data import load_curriculum
from aicls.latex2sympy import latex_to_sympy, parse_latex, roundtrip_check, to_sympy
from aicls.mathengine import MathEngine

REPO_ROOT = Path(__file__).resolve().parents[3]
CURRICULUM_DIR = REPO_ROOT / "datasets" / "curriculum"
CARDS_DIR = CURRICULUM_DIR / "cards"
PROBLEMS_PATH = CURRICULUM_DIR / "problems.jsonl"
FROZEN_PATH = CURRICULUM_DIR / "FROZEN.sha256"
README_PATH = CURRICULUM_DIR / "README.md"

ID_RE = re.compile(r"^[a-z_]+-\d{4}$")
NAME_RE = re.compile(r"^[a-z0-9_]+$")

#: How many rows the engine is really run on in the fast test (deterministic stride).
SAMPLE_STRIDE = 19


@pytest.fixture(scope="module")
def build(load_script: Any) -> Any:
    """The compiler module, imported by path (``scripts/`` is not a package)."""
    return load_script("scripts/build_curriculum_testset.py")


@pytest.fixture(scope="module")
def cards(build: Any) -> list[Any]:
    """Every YAML card, validated and in compile order."""
    loaded: list[Any] = build.load_cards(CARDS_DIR)
    return loaded


@pytest.fixture(scope="module")
def compiled_rows() -> list[dict[str, Any]]:
    """The compiled JSONL rows, header included."""
    return [
        json.loads(line)
        for line in PROBLEMS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def problem_rows(build: Any, compiled_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The compiled rows without the provenance header."""
    return [row for row in compiled_rows if row["id"] != build.HEADER_ID]


# --------------------------------------------------------------------------------------------
# contractual tests (docs/mimari/latex-ayristirici.md §7.2)
# --------------------------------------------------------------------------------------------
def test_curriculum_cards_compile_and_quotas_match(
    build: Any, cards: list[Any], problem_rows: list[dict[str, Any]]
) -> None:
    """Cards compile 1:1 into problems.jsonl and every topic meets its İP-4 quota."""
    assert len(problem_rows) >= build.TOTAL_QUOTA, "the İP-4 set must hold at least 1000 problems"

    card_rows = {card.row["id"]: card.row for card in cards}
    compiled = {row["id"]: row for row in problem_rows}
    assert set(card_rows) == set(compiled), "cards and problems.jsonl disagree about the id set"
    differing = [row_id for row_id in card_rows if card_rows[row_id] != compiled[row_id]]
    assert differing == [], f"compiled rows drifted from their cards: {differing[:5]}"

    counts = build.topic_counts(problem_rows)
    assert set(counts) == set(build.TOPIC_QUOTAS), "the topic set is fixed by the quota table"
    problems = build.check_quotas(counts)
    assert problems == [], f"quota violations: {problems}"

    total = sum(counts.values())
    for topic, quota in build.TOPIC_QUOTAS.items():
        target_share = quota / build.TOTAL_QUOTA
        actual_share = counts[topic] / total
        assert abs(actual_share - target_share) <= build.SHARE_TOLERANCE, (
            f"{topic}: share {actual_share:.4f} is far from the documented {target_share:.4f}"
        )


def test_frozen_curriculum_rows_unchanged(build: Any, compiled_rows: list[dict[str, Any]]) -> None:
    """Every digest in FROZEN.sha256 still matches the row it names, byte for byte."""
    expected = build.read_frozen(FROZEN_PATH)
    rows = {row["id"]: row for row in compiled_rows}
    assert expected, "FROZEN.sha256 must not be empty"
    missing = sorted(set(expected) - set(rows))
    assert missing == [], f"frozen rows disappeared: {missing[:10]}"
    changed = [row_id for row_id, digest in expected.items() if sha256_json(rows[row_id]) != digest]
    assert changed == [], f"frozen rows changed: {changed[:10]}"
    unfrozen = sorted(set(rows) - set(expected))
    assert unfrozen == [], f"compiled rows missing from FROZEN.sha256: {unfrozen[:10]}"


# --------------------------------------------------------------------------------------------
# schema and identity
# --------------------------------------------------------------------------------------------
def test_every_card_carries_the_full_row_schema(build: Any, cards: list[Any]) -> None:
    """No card may rely on a loader default: all fifteen fields are written out."""
    for card in cards:
        row = card.row
        assert set(row) == set(build.REQUIRED_FIELDS), f"{row.get('id')}: field set differs"
        assert ID_RE.match(row["id"]), row["id"]
        assert NAME_RE.match(row["topic"]) and NAME_RE.match(row["subtopic"])
        assert row["chapter_ref"] == build.CHAPTER_BY_TOPIC[row["topic"]]
        assert 1 <= row["difficulty"] <= 3
        assert row["input"]["kind"] in {"latex", "text_tr"}
        assert row["input"]["value"].strip()
        assert row["command"]["op"] in build.SUPPORTED_OPS
        assert row["checker"] in build.CHECKERS
        assert row["expected"]["latex"].strip() and row["expected"]["srepr"].strip()
        assert row["source"].startswith("sablon:")
        assert row["author"]
        assert row["status"] in {"draft", "reviewed", "frozen"}
        assert row["notes_tr"].strip()


def test_ids_are_unique_across_every_card_file(cards: list[Any]) -> None:
    ids = [card.row["id"] for card in cards]
    assert len(set(ids)) == len(ids)


def test_row_ids_are_prefixed_with_their_topic(cards: list[Any]) -> None:
    for card in cards:
        assert card.row["id"].startswith(card.row["topic"] + "-")


def test_problems_jsonl_loads_through_aicls_data() -> None:
    """The compiled file is exactly what ``aicls.data.load_curriculum`` accepts."""
    problems = load_curriculum(PROBLEMS_PATH)
    assert len(problems) >= 1000
    assert len({p.id for p in problems}) == len(problems)


def test_every_card_file_loads_through_aicls_data() -> None:
    for path in sorted(CARDS_DIR.glob("*.yaml")):
        assert load_curriculum(path), path.name


def test_header_row_states_the_provenance(build: Any, compiled_rows: list[dict[str, Any]]) -> None:
    """Line 1 of the compiled file says the set was authored, not sampled."""
    header = compiled_rows[0]
    assert header["id"] == build.HEADER_ID
    assert header["_header"] is True
    assert header["dataset_version"] == build.DATASET_VERSION
    assert header["n_problems"] == len(compiled_rows) - 1
    for needle in ("proje içinde", "ÖRNEKLENMEMİŞTİR"):
        assert needle in header["provenance_tr"]
    assert "authored" in header["provenance_en"].lower()
    assert "not an independent" in header["provenance_en"].lower()


def test_readme_documents_provenance_quotas_and_freeze_rules() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    for needle in ("KÖKEN", "ÖRNEKLENMEMİŞTİR", "FROZEN.sha256", "up_to_constant", "_header"):
        assert needle in readme, needle
    for field in ("chapter_ref", "checker_args", "reviewed_by", "notes_tr"):
        assert field in readme, field


# --------------------------------------------------------------------------------------------
# the answers themselves
# --------------------------------------------------------------------------------------------
def test_expected_srepr_rebuilds_into_expected_latex(build: Any, cards: list[Any]) -> None:
    """``expected.latex`` is the canonical printing of ``expected.srepr`` -- no drift."""
    problems = [message for card in cards if (message := build.check_expected(card))]
    assert problems == [], problems[:5]


def test_latex_inputs_parse_and_round_trip(cards: list[Any]) -> None:
    """> 99 % of the LaTeX inputs parse *and* survive the level-2 round trip."""
    latex_cards = [card for card in cards if card.row["input"]["kind"] == "latex"]
    assert len(latex_cards) >= 1000
    failures: list[str] = []
    for card in latex_cards:
        source = card.row["input"]["value"]
        try:
            to_sympy(parse_latex(source))
            check = roundtrip_check(source, numeric_samples=3)
        except AiclsError as exc:
            failures.append(f"{card.row['id']}: {type(exc).__name__}")
            continue
        if not check.ok:
            failures.append(f"{card.row['id']}: roundtrip {check.reason}")
    rate = 1 - len(failures) / len(latex_cards)
    assert rate > 0.99, f"parse success {rate:.4f}; first failures {failures[:10]}"


def test_text_rows_are_turkish_sentences_carrying_an_executable_command(
    cards: list[Any],
) -> None:
    text_cards = [card for card in cards if card.row["input"]["kind"] == "text_tr"]
    assert text_cards, "the set must exercise the NLU path as well"
    for card in text_cards:
        value = card.row["input"]["value"]
        assert "$" in value and len(value.split()) >= 3
        assert card.row["command"]["op"]


def test_a_deterministic_sample_verifies_against_the_engine(build: Any, cards: list[Any]) -> None:
    """Run every ``SAMPLE_STRIDE``-th card through the real engine and check the answer."""
    sample = cards[::SAMPLE_STRIDE]
    report = build.verify(sample, engine=MathEngine(timeout_s=30.0))
    assert report.n == len(sample)
    assert report.failures == [], report.failures[:5]
    assert report.correct_rate == 1.0


# --------------------------------------------------------------------------------------------
# the checkers themselves must be able to fail
# --------------------------------------------------------------------------------------------
def test_symbolic_checker_accepts_an_equivalent_form(build: Any) -> None:
    x = sympy.Symbol("x", real=True)
    assert build.compare("symbolic", (x + 1) ** 2, x**2 + 2 * x + 1, {}) == (True, "")


def test_symbolic_checker_rejects_a_wrong_answer(build: Any) -> None:
    x = sympy.Symbol("x", real=True)
    ok, reason = build.compare("symbolic", 2 * x, 3 * x, {})
    assert not ok and reason


def test_symbolic_checker_accepts_an_antiderivative_up_to_a_constant(build: Any) -> None:
    """``sin^2(x)/2`` and ``-cos(2x)/4`` are the same antiderivative; the checker must say so."""
    x = sympy.Symbol("x", real=True)
    args = {"up_to_constant": True, "var": "x"}
    assert build.compare("symbolic", sympy.sin(x) ** 2 / 2, -sympy.cos(2 * x) / 4, args)[0]
    assert not build.compare("symbolic", sympy.sin(x) ** 2 / 2, -sympy.cos(2 * x) / 4 + x, args)[0]


def test_set_checker_ignores_element_order(build: Any) -> None:
    assert build.compare("set", sympy.FiniteSet(2, -2), sympy.FiniteSet(-2, 2), {})[0]
    assert not build.compare("set", sympy.FiniteSet(2, -2), sympy.FiniteSet(-2, 3), {})[0]


def test_numeric_checker_uses_its_tolerance(build: Any) -> None:
    x = sympy.Symbol("x", real=True)
    args = {"tolerance": 1e-6, "domain": [-2.0, 2.0], "var": "x"}
    assert build.compare("numeric", x**2, x * x, args)[0]
    assert not build.compare("numeric", x**2, x**2 + 1, args)[0]


def test_bool_checker_compares_boolean_atoms(build: Any) -> None:
    assert build.compare("bool", sympy.true, sympy.true, {})[0]
    assert not build.compare("bool", sympy.true, sympy.false, {})[0]


def test_symbolic_checker_treats_infinities_as_equal(build: Any) -> None:
    """``symbolic_equivalent(oo, oo)`` abstains, so the compiler needs an identity pre-check."""
    assert build.compare("symbolic", sympy.oo, sympy.oo, {})[0]
    assert not build.compare("symbolic", sympy.oo, -sympy.oo, {})[0]


def test_exact_latex_checker_is_a_string_comparison(build: Any) -> None:
    assert build.compare_latex("\\infty", "\\infty")[0]
    assert not build.compare_latex("\\infty", "- \\infty")[0]


# --------------------------------------------------------------------------------------------
# the compiler refuses a broken corpus
# --------------------------------------------------------------------------------------------
def _write_cards(directory: Path, rows: list[dict[str, Any]]) -> Path:
    import yaml  # type: ignore[import-untyped]  # PyYAML ships no stubs, types-PyYAML is absent

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "limit.yaml"
    path.write_text(
        yaml.safe_dump(rows, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n"
    )
    return path


def test_duplicate_ids_across_card_files_are_rejected(
    build: Any, cards: list[Any], tmp_path: Path
) -> None:
    row = json.loads(json.dumps(cards[0].row))
    _write_cards(tmp_path / "cards", [row, dict(row)])
    with pytest.raises(InputValidationError):
        build.load_cards(tmp_path / "cards")


def test_a_missing_field_is_rejected(build: Any, cards: list[Any], tmp_path: Path) -> None:
    row = json.loads(json.dumps(cards[0].row))
    del row["notes_tr"]
    _write_cards(tmp_path / "cards", [row])
    with pytest.raises(InputValidationError):
        build.load_cards(tmp_path / "cards")


def test_an_unknown_topic_is_rejected(build: Any, cards: list[Any], tmp_path: Path) -> None:
    row = json.loads(json.dumps(cards[0].row))
    row["topic"] = "olmayan_konu"
    _write_cards(tmp_path / "cards", [row])
    with pytest.raises(InputValidationError):
        build.load_cards(tmp_path / "cards")


def test_quota_violations_are_reported_not_hidden(build: Any) -> None:
    short = dict.fromkeys(build.TOPIC_QUOTAS, 1)
    problems = build.check_quotas(short)
    # one message per topic, plus one for the total
    assert len(problems) == len(build.TOPIC_QUOTAS) + 1
    assert any(message.startswith("total:") for message in problems)
    assert build.check_quotas(dict(build.TOPIC_QUOTAS)) == []


def test_expected_latex_drift_is_detected(build: Any, cards: list[Any]) -> None:
    """A card whose LaTeX no longer matches its srepr must be caught by the compiler."""
    broken = build.Card(
        path=cards[0].path,
        row={
            **json.loads(json.dumps(cards[0].row)),
            "expected": {"latex": "x", "srepr": "Integer(2)"},
        },
    )
    assert build.check_expected(broken)


def test_frozen_file_header_names_the_generator_and_the_version(build: Any) -> None:
    text = FROZEN_PATH.read_text(encoding="utf-8")
    assert text.startswith("#")
    assert build.DATASET_VERSION in text
    assert "build_curriculum_testset.py" in text


# --------------------------------------------------------------------------------------------
# the whole corpus (slow)
# --------------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_full_curriculum_cas_accuracy_above_0_99(build: Any, cards: list[Any]) -> None:
    """The contractual İP-4 number, measured on the authored set (see README provenance)."""
    report = build.verify(cards, engine=MathEngine(timeout_s=60.0))
    assert report.n == len(cards)
    assert report.correct_rate > 0.99, (
        f"CAS accuracy {report.correct_rate:.4f}; failures {report.failures[:10]}"
    )
    assert report.parse_success > 0.99


def test_sympy_can_rebuild_every_expected_srepr(cards: list[Any]) -> None:
    """A cheap guard on the machine-checkable half of the answer."""
    for card in cards[::7]:
        rebuilt = sympy.sympify(card.row["expected"]["srepr"])
        assert rebuilt is not None
        if card.row["checker"] != "exact_latex":
            assert not isinstance(rebuilt, str)


def test_expected_latex_stays_inside_the_grammar_except_for_solution_sets(
    cards: list[Any],
) -> None:
    """Only ``\\left\\{a, b\\right\\}`` answers fall outside the v1 grammar (§3.2, finite sets).

    This is a real İP-4 finding, not a corpus defect: the platform can *compute* a solution set
    but cannot read its own printed form back, so a user who copies a result back into the
    input box hits a parse error. Recorded here so the gap cannot regress unnoticed.
    """
    checker_of = {card.row["id"]: card.row["checker"] for card in cards}
    outside: list[str] = []
    for card in cards:
        try:
            latex_to_sympy(card.row["expected"]["latex"])
        except AiclsError:
            outside.append(card.row["id"])
    unexpected = [row_id for row_id in outside if checker_of[row_id] != "set"]
    assert unexpected == [], f"answers outside the grammar that are not sets: {unexpected[:10]}"
    non_set = [card for card in cards if card.row["checker"] != "set"]
    assert len(outside) < len(cards) - len(non_set) + 1
    assert len(non_set) / len(cards) > 0.9
