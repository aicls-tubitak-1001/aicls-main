"""Curriculum problem loader (docs/mimari/latex-ayristirici.md §7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aicls.core.errors import InputValidationError, NotFoundError
from aicls.data import Problem, load_curriculum

ROW = {
    "id": "turev-0042",
    "topic": "turev",
    "subtopic": "zincir_kurali",
    "chapter_ref": "aicls:bolum-3",
    "difficulty": 2,
    "input": {"kind": "latex", "value": "\\frac{d}{dx} \\sin(x^{2})"},
    "command": {"op": "diff", "expr": "\\sin(x^{2})", "var": "x", "order": 1},
    "expected": {
        "latex": "2 x \\cos{\\left(x^{2} \\right)}",
        "srepr": "Mul(Integer(2), Symbol('x'), cos(Pow(Symbol('x'), Integer(2))))",
    },
    "checker": "symbolic",
    "checker_args": {"tolerance": 1.0e-9},
    "source": "kitap:3.4",
    "author": "mmturhan",
    "reviewed_by": "csahin",
    "status": "reviewed",
    "notes_tr": "",
}


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    """Write ``rows`` as UTF-8 JSONL with LF endings."""
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def test_load_curriculum_parses_problem_rows(tmp_path: Path) -> None:
    second = {**ROW, "id": "limit-0001", "topic": "limit", "checker": "numeric", "status": "frozen"}
    path = write_jsonl(tmp_path / "problems.jsonl", [ROW, second])

    problems = load_curriculum(path)

    assert len(problems) == 2
    assert all(isinstance(p, Problem) for p in problems)
    first = problems[0]
    assert first.id == "turev-0042"
    assert first.topic == "turev"
    assert first.difficulty == 2
    assert first.input.kind == "latex"
    assert first.input.value.startswith("\\frac")
    assert first.command["op"] == "diff"
    assert first.expected.latex.endswith("\\right)}")
    assert first.checker == "symbolic"
    assert first.status == "reviewed"
    assert problems[1].is_frozen
    assert not first.is_frozen


def test_load_curriculum_skips_blank_lines_and_keeps_order(tmp_path: Path) -> None:
    path = tmp_path / "problems.jsonl"
    rows = [json.dumps({**ROW, "id": f"p-{i}"}, ensure_ascii=False) for i in range(3)]
    path.write_text("\n".join([rows[0], "", rows[1], "   ", rows[2], ""]), encoding="utf-8")

    problems = load_curriculum(path)
    assert [p.id for p in problems] == ["p-0", "p-1", "p-2"]


def test_load_curriculum_reads_yaml_cards(tmp_path: Path) -> None:
    import yaml

    path = tmp_path / "turev.yaml"
    path.write_text(yaml.safe_dump([ROW], allow_unicode=True), encoding="utf-8")
    problems = load_curriculum(path)
    assert [p.id for p in problems] == ["turev-0042"]


def test_load_curriculum_rejects_missing_file_and_bad_rows(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError):
        load_curriculum(tmp_path / "nope.jsonl")

    broken = write_jsonl(tmp_path / "broken.jsonl", [{"id": "x"}])
    with pytest.raises(InputValidationError) as excinfo:
        load_curriculum(broken)
    assert excinfo.value.details["line"] == 1

    invalid_json = tmp_path / "invalid.jsonl"
    invalid_json.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(InputValidationError):
        load_curriculum(invalid_json)


def test_load_curriculum_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "problems.jsonl", [ROW, ROW])
    with pytest.raises(InputValidationError):
        load_curriculum(path)
