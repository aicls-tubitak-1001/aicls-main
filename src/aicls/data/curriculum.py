"""Curriculum problem loader (docs/mimari/latex-ayristirici.md §7, İP-4 test set).

``datasets/curriculum/problems.jsonl`` is the ≥ 1000-problem set the CAS accuracy target
(> 99 %) is measured on. It is compiled from reviewed YAML cards, so both shapes are accepted
here: the compiled JSONL (one problem per line) and a card file (a YAML list of the same rows).

The loader validates every row and refuses duplicate ids — a duplicate would silently double a
problem's weight in a reported percentage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
import yaml  # type: ignore[import-untyped]  # PyYAML ships no stubs and types-PyYAML is absent

from aicls.core.logging import get_logger
from aicls.data.errors import invalid_data, missing_path

_log = get_logger(__name__)

#: Comparison strategies a problem can request (docs/ml/degerlendirme.md §2).
CheckerName = Literal["symbolic", "numeric", "set", "bool", "exact_latex", "plot_ok"]

#: Review status; ``frozen`` rows may never change (``datasets/curriculum/FROZEN.sha256``).
ProblemStatus = Literal["draft", "reviewed", "frozen"]

#: File suffixes read as YAML cards instead of JSONL.
YAML_SUFFIXES = (".yaml", ".yml")


class ProblemInput(BaseModel):
    """What the user would type or write: LaTeX, or a Turkish sentence (which also tests NLU)."""

    model_config = ConfigDict(frozen=True, extra="allow")

    kind: Literal["latex", "text_tr"] = "latex"
    value: str


class ProblemExpected(BaseModel):
    """The expected answer, as LaTeX and as a SymPy ``srepr`` (the machine-checkable form)."""

    model_config = ConfigDict(frozen=True, extra="allow")

    latex: str = ""
    srepr: str = ""


class Problem(BaseModel):
    """One curriculum problem.

    Attributes:
        id: Stable identifier (``turev-0042``); also the pytest id of the generated case.
        topic: Curriculum topic used for the quota table and the per-topic breakdown.
        input: The user-visible input.
        command: The CAS command the platform is expected to build.
        expected: The expected answer.
        checker: How the answer is compared.
        status: ``draft``/``reviewed``/``frozen``.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    id: str
    topic: str = ""
    subtopic: str = ""
    chapter_ref: str = ""
    difficulty: int = Field(default=1, ge=0)
    input: ProblemInput
    command: dict[str, Any] = Field(default_factory=dict)
    expected: ProblemExpected = Field(default_factory=ProblemExpected)
    checker: CheckerName = "symbolic"
    checker_args: dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    author: str = ""
    reviewed_by: str = ""
    status: ProblemStatus = "draft"
    notes_tr: str = ""

    @property
    def is_frozen(self) -> bool:
        """Whether this row is frozen (its bytes may never change)."""
        return self.status == "frozen"

    @property
    def latex(self) -> str:
        """The input value when it is LaTeX, otherwise ``""``."""
        return self.input.value if self.input.kind == "latex" else ""


def load_curriculum(path: Path) -> list[Problem]:
    """Read a curriculum problem set.

    Args:
        path: ``problems.jsonl`` (one problem per line) or a ``*.yaml`` card file (a list).

    Returns:
        The problems, in file order.

    Raises:
        NotFoundError: If ``path`` does not exist.
        InputValidationError: On malformed JSON/YAML, an invalid row (the offending 1-based
            ``line`` is in ``details``) or a duplicate id.
    """
    path = Path(path)
    if not path.is_file():
        raise missing_path(path, "curriculum_missing")
    rows = _read_yaml_rows(path) if path.suffix.lower() in YAML_SUFFIXES else _read_jsonl_rows(path)

    problems: list[Problem] = []
    seen: set[str] = set()
    for line_no, row in rows:
        try:
            problem = Problem.model_validate(row)
        except ValueError as exc:
            raise invalid_data(
                "curriculum_row_invalid", path=str(path), line=line_no, error=str(exc)
            ) from exc
        if problem.id in seen:
            raise invalid_data(
                "curriculum_duplicate_id", path=str(path), line=line_no, id=problem.id
            )
        seen.add(problem.id)
        problems.append(problem)
    _log.info("data.curriculum.loaded", path=str(path), n=len(problems))
    return problems


def _read_jsonl_rows(path: Path) -> list[tuple[int, Any]]:
    """Parse a JSONL file into ``(line number, row)`` pairs, skipping blank lines."""
    rows: list[tuple[int, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append((line_no, json.loads(line)))
        except json.JSONDecodeError as exc:
            raise invalid_data(
                "curriculum_row_not_json", path=str(path), line=line_no, error=str(exc)
            ) from exc
    return rows


def _read_yaml_rows(path: Path) -> list[tuple[int, Any]]:
    """Parse a YAML card file into ``(index, row)`` pairs."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise invalid_data("curriculum_not_yaml", path=str(path), error=str(exc)) from exc
    if not isinstance(payload, list):
        raise invalid_data("curriculum_cards_not_a_list", path=str(path))
    return list(enumerate(payload, start=1))


__all__ = [
    "YAML_SUFFIXES",
    "CheckerName",
    "Problem",
    "ProblemExpected",
    "ProblemInput",
    "ProblemStatus",
    "load_curriculum",
]
