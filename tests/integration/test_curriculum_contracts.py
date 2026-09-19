"""Cross-module contracts of the İP-4 curriculum set (wave-1 lesson: test the boundary).

``datasets/curriculum/`` is data, but it sits on three real module boundaries and every one of
them is exercised here with the peer module itself, never a fake:

1. **datasets -> ``aicls.data``.** ``load_curriculum`` is how every other package reads the
   set. The compiled ``problems.jsonl`` (header row included) and each YAML card file must both
   load through it, and the fields must survive as the typed :class:`aicls.data.Problem` the
   evaluation harness codes against -- ``checker`` and ``status`` are ``Literal`` types there,
   so a value the compiler allows but the loader rejects would only surface at report time.
2. **datasets -> ``aicls.latex2sympy``.** Every ``input.kind == "latex"`` row is a string the
   parser must accept; ``expected.srepr`` must rebuild and print back to ``expected.latex``
   through ``sympy_to_latex``.
3. **datasets -> ``aicls.mathengine``.** Every ``command`` mapping must validate as a typed
   :data:`aicls.mathengine.CasCommand` (the same ``COMMAND_ADAPTER`` the API uses on the
   request body) and execute; the answer is compared with the card's authored answer.

The heavy end of (3) -- all 1204 rows -- lives in ``tests/unit/datasets``; here a deterministic
slice covering every ``op`` and every ``checker`` is enough to keep the boundary honest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import sympy

from aicls.data import Problem, load_curriculum
from aicls.latex2sympy import latex_to_sympy, sympy_to_latex
from aicls.mathengine import COMMAND_ADAPTER, SUPPORTED_OPS, MathEngine

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CURRICULUM_DIR = REPO_ROOT / "datasets" / "curriculum"
CARDS_DIR = CURRICULUM_DIR / "cards"
PROBLEMS_PATH = CURRICULUM_DIR / "problems.jsonl"

HEADER_ID = "_header"


@pytest.fixture(scope="module")
def problems() -> list[Problem]:
    """The compiled set as ``aicls.data`` sees it (the header row is kept on purpose)."""
    return load_curriculum(PROBLEMS_PATH)


@pytest.fixture(scope="module")
def build(load_script: Any) -> Any:
    """The compiler module."""
    return load_script("scripts/build_curriculum_testset.py")


def test_compiled_set_loads_through_the_data_loader(problems: list[Problem]) -> None:
    """``aicls.data.load_curriculum`` reads the file whole -- header row included."""
    assert len(problems) >= 1001  # 1000 problems plus the provenance header
    assert problems[0].id == HEADER_ID
    assert problems[0].topic == "_meta"
    assert all(isinstance(problem, Problem) for problem in problems)


def test_header_row_is_a_harmless_problem_for_a_consumer_that_forgets_to_filter(
    problems: list[Problem],
) -> None:
    """A consumer that runs every row must still get the right answer for the header."""
    header = problems[0]
    engine = MathEngine(timeout_s=10.0)
    result = engine.execute(dict(header.command))
    assert result.latex == header.expected.latex
    assert result.srepr == header.expected.srepr


def test_every_card_file_loads_through_the_data_loader() -> None:
    """The YAML cards are the same shape as the compiled rows (``load_curriculum`` reads both)."""
    total = 0
    for path in sorted(CARDS_DIR.glob("*.yaml")):
        rows = load_curriculum(path)
        assert rows, path.name
        total += len(rows)
    assert total >= 1000


def test_every_command_validates_as_a_typed_cas_command(problems: list[Problem]) -> None:
    """``command`` is exactly what ``POST /compute/latex`` accepts (``COMMAND_ADAPTER``)."""
    ops: set[str] = set()
    for problem in problems:
        command = COMMAND_ADAPTER.validate_python(dict(problem.command))
        assert command.op in SUPPORTED_OPS
        ops.add(command.op)
    # the set must exercise the whole union except the ones with no curriculum use yet
    assert ops >= {
        "diff",
        "integrate",
        "limit",
        "solve",
        "simplify",
        "expand",
        "factor",
        "series",
        "evaluate",
        "plot",
    }


def test_checker_and_status_values_are_inside_the_loader_literals(
    problems: list[Problem],
) -> None:
    """``CheckerName`` / ``ProblemStatus`` are ``Literal``s: a stray value must not exist."""
    assert {p.checker for p in problems} <= {
        "symbolic",
        "numeric",
        "set",
        "bool",
        "exact_latex",
        "plot_ok",
    }
    assert {p.status for p in problems} <= {"draft", "reviewed", "frozen"}


def test_expected_srepr_and_latex_agree_through_the_printer(problems: list[Problem]) -> None:
    """``sympy_to_latex(sympify(srepr)) == expected.latex`` on a deterministic slice."""
    for problem in problems[::37]:
        rebuilt = sympy.sympify(problem.expected.srepr)
        assert sympy_to_latex(rebuilt) == problem.expected.latex, problem.id


def test_latex_inputs_reach_sympy_through_the_parser(problems: list[Problem]) -> None:
    """Every ``latex`` input is a string ``aicls.latex2sympy`` accepts."""
    checked = 0
    for problem in problems[::23]:
        if problem.input.kind != "latex":
            continue
        assert latex_to_sympy(problem.input.value) is not None, problem.id
        checked += 1
    assert checked >= 20


def test_a_slice_covering_every_op_and_checker_runs_end_to_end(
    build: Any, problems: list[Problem]
) -> None:
    """One card per (op, checker) pair is executed and its answer verified."""
    rows = [json.loads(line) for line in PROBLEMS_PATH.read_text(encoding="utf-8").splitlines()]
    chosen: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["command"]["op"], row["checker"])
        chosen.setdefault(key, row)
    cards = [build.Card(path=PROBLEMS_PATH, row=row) for row in chosen.values()]
    report = build.verify(cards, engine=MathEngine(timeout_s=30.0))
    assert report.failures == [], report.failures
    assert report.n == len(chosen) >= 10
    assert len({key[0] for key in chosen}) == len(SUPPORTED_OPS)


def test_the_set_is_registered_in_the_datasets_readme() -> None:
    """The corpus is discoverable from ``datasets/README.md`` with its owner and freeze rule."""
    readme = (REPO_ROOT / "datasets" / "README.md").read_text(encoding="utf-8")
    assert "curriculum/" in readme
    assert "FROZEN.sha256" in readme
