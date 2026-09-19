"""Corpus level guarantees: parse success rate, unsupported set, frozen curriculum rows.

The strict success definition (docs/mimari/latex-ayristirici.md 6) is
``parse_latex`` + ``to_sympy`` without an exception **and** a successful round trip; a
silent misparse therefore counts as a failure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aicls.core.errors import AiclsError
from aicls.core.hashing import sha256_json
from aicls.latex2sympy import (
    ParseError,
    SemanticError,
    UnsupportedConstruct,
    parse_latex,
    roundtrip_check,
    to_sympy,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
LATEX_DIR = REPO_ROOT / "datasets" / "latex"
VALID_PATH = LATEX_DIR / "valid_inputs.jsonl"
UNSUPPORTED_PATH = LATEX_DIR / "unsupported.jsonl"
CURRICULUM_DIR = REPO_ROOT / "datasets" / "curriculum"

TARGET_RATE = 0.99
KNOWN_SOURCES = {"curriculum", "sympy_latex", "ocr_output", "mathwriting_label"}
ERROR_BY_KIND: dict[str, type[AiclsError]] = {
    "unsupported": UnsupportedConstruct,
    "parse_error": ParseError,
    "semantic": SemanticError,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@pytest.fixture(scope="module")
def valid_rows() -> list[dict[str, Any]]:
    return _read_jsonl(VALID_PATH)


@pytest.fixture(scope="module")
def unsupported_rows() -> list[dict[str, Any]]:
    return _read_jsonl(UNSUPPORTED_PATH)


def test_valid_inputs_schema(valid_rows: list[dict[str, Any]]) -> None:
    assert len(valid_rows) >= 2000
    ids = [row["id"] for row in valid_rows]
    assert len(set(ids)) == len(ids)
    for row in valid_rows:
        assert row["source"] in KNOWN_SOURCES
        assert row["expect_ok"] is True
        assert isinstance(row["latex"], str) and row["latex"].strip()
        assert len(row["latex"]) <= 2048
    # every declared source must actually be present
    assert {row["source"] for row in valid_rows} == KNOWN_SOURCES


def test_curriculum_and_valid_inputs_parse_success_rate_above_0_99(
    valid_rows: list[dict[str, Any]],
) -> None:
    failures: list[tuple[str, str]] = []
    by_source: dict[str, list[int]] = {source: [0, 0] for source in KNOWN_SOURCES}
    for row in valid_rows:
        source = row["source"]
        by_source[source][1] += 1
        try:
            node = parse_latex(row["latex"])
            to_sympy(node)
            check = roundtrip_check(row["latex"], numeric_samples=3)
            if not check.ok:
                failures.append((row["id"], f"roundtrip: {check.reason}"))
                continue
        except AiclsError as exc:
            failures.append((row["id"], f"{type(exc).__name__}: {exc}"))
            continue
        by_source[source][0] += 1
    total = len(valid_rows)
    passed = total - len(failures)
    rate = passed / total
    breakdown = {src: f"{ok}/{n}" for src, (ok, n) in by_source.items()}
    assert rate > TARGET_RATE, (
        f"parse success rate {rate:.4f} <= {TARGET_RATE}; breakdown={breakdown}; "
        f"first failures={failures[:10]}"
    )


def test_unsupported_inputs_are_rejected_with_a_named_construct(
    unsupported_rows: list[dict[str, Any]],
) -> None:
    assert len(unsupported_rows) >= 100
    ids = [row["id"] for row in unsupported_rows]
    assert len(set(ids)) == len(ids)
    wrong: list[tuple[str, str]] = []
    for row in unsupported_rows:
        expected = ERROR_BY_KIND[row["expect_error"]]
        try:
            to_sympy(parse_latex(row["latex"]))
        except expected as exc:
            wanted = row.get("construct")
            if isinstance(exc, UnsupportedConstruct) and wanted and exc.construct != wanted:
                wrong.append((row["id"], f"construct {exc.construct!r}"))
        except AiclsError as exc:
            wrong.append((row["id"], f"{type(exc).__name__}"))
        else:
            wrong.append((row["id"], "parsed successfully"))
    assert wrong == []


def test_datasets_readme_documents_the_row_schema() -> None:
    readme = (LATEX_DIR / "README.md").read_text(encoding="utf-8")
    for field in ("id", "latex", "source", "expect_ok", "expect_error", "construct"):
        assert field in readme


def test_frozen_curriculum_rows_unchanged() -> None:
    frozen_path = CURRICULUM_DIR / "FROZEN.sha256"
    problems_path = CURRICULUM_DIR / "problems.jsonl"
    if not frozen_path.is_file() or not problems_path.is_file():
        pytest.skip("datasets/curriculum is produced by the IP-4 curriculum work package")
    expected: dict[str, str] = {}
    for line in frozen_path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, row_id = line.split(maxsplit=1)
            expected[row_id.strip()] = digest
    rows = {row["id"]: row for row in _read_jsonl(problems_path)}
    missing = sorted(set(expected) - set(rows))
    assert missing == [], f"frozen rows disappeared: {missing}"
    changed = [
        row_id
        for row_id, digest in expected.items()
        if sha256_json(rows[row_id]).removeprefix("sha256:") != digest.removeprefix("sha256:")
    ]
    assert changed == []


def test_sympy_latex_output_is_within_grammar_on_curriculum() -> None:
    import sympy

    from aicls.latex2sympy import SymbolTable, latex_to_sympy, sympy_to_latex

    symbols = SymbolTable()
    symbols.declare("n", integer=True)
    x, y = sympy.symbols("x y", real=True)
    n = sympy.Symbol("n", integer=True)
    exprs: list[sympy.Basic] = [
        x**2 - 4,
        (x + 1) * (x - 1),
        sympy.Rational(1, 3),
        sympy.sqrt(x + 1),
        sympy.root(x, 3),
        sympy.sin(x) + sympy.cos(x),
        sympy.tan(x),
        sympy.asin(x),
        sympy.exp(2 * x),
        sympy.log(x),
        sympy.log(x, 10),
        sympy.Abs(x - 1),
        x / (y + 1),
        sympy.pi * x,
        sympy.Integral(x**2, x),
        sympy.Integral(sympy.sin(x), (x, 0, sympy.pi)),
        sympy.Derivative(x**3, x),
        sympy.Derivative(x**3, (x, 2)),
        sympy.Limit(sympy.sin(x) / x, x, 0),
        sympy.Limit(1 / x, x, 0, "+"),
        sympy.Sum(n**2, (n, 1, 10)),
        sympy.Product(n, (n, 1, 10)),
        sympy.Eq(x**2, 4),
        sympy.factorial(n),
        sympy.Matrix([[1, 2], [3, 4]]),
        sympy.Interval(0, 1),
        sympy.Interval.open(0, 1),
        -sympy.Rational(3, 4) * x**2,
        x ** sympy.Rational(1, 2),
        sympy.cosh(x),
    ]
    problems: list[str] = []
    for expr in exprs:
        rendered = sympy_to_latex(expr)
        try:
            back = latex_to_sympy(rendered, symbols=symbols)
        except AiclsError as exc:
            problems.append(f"{rendered!r}: {exc}")
            continue
        if not (back == expr or sympy.simplify(back - expr) == 0):
            problems.append(f"{rendered!r}: {back} != {expr}")
    assert problems == []


def test_differential_against_sympy_parse_latex_subset() -> None:
    """Differential oracle: sympy.parsing.latex on the overlapping subset (needs antlr4)."""
    pytest.importorskip("antlr4", reason="sympy.parsing.latex needs the antlr4 runtime")
    import sympy
    from sympy.parsing.latex import parse_latex as sympy_parse_latex

    from aicls.latex2sympy import latex_to_sympy

    subset = [
        r"x^{2} + 2x + 1",
        r"\frac{x+1}{x-1}",
        r"\sqrt{x}",
        r"\sin(x)",
        r"\int_{0}^{1} x^{2} dx",
        r"\frac{1}{3}",
    ]
    for src in subset:
        ours = latex_to_sympy(src)
        theirs = sympy_parse_latex(src)
        assert sympy.simplify(ours - theirs) == 0, src
