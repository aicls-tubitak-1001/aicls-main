"""Verify the İP deliverable → repository artefact table (ADR-0009, CI job ``docs-map``).

Reads ``docs/raporlama/ip-eslestirme.md`` §1. For every row, the backtick-quoted repository paths
in the "Artefakt" column are checked:

* a row passes when it is marked ``(planlı: ...)`` / ``(planlı)`` or when at least one of its
  repository paths exists; rows with no repository path at all (external dependencies, git tags,
  database tables) are skipped;
* every individual missing path is reported as a warning so it stays visible in CI logs;
* ``--strict`` (used at ``report-YYYY-MM`` tags) requires every path to exist and forbids any
  remaining ``planlı`` marker.

Also warns (never fails) about bare percentages in ``docs/raporlama/*.md`` report pages, which
must come from metric macros instead (``{{ metrics.<kind>.<metric> }}``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import re
import sys

TABLE_PATH = Path("docs/raporlama/ip-eslestirme.md")
PLANNED_RE = re.compile(r"\(planlı[^)]*\)")
BACKTICK_RE = re.compile(r"`([^`]+)`")
ROW_RE = re.compile(r"^\|\s*İP-\d+\s*\|")
PERCENT_RE = re.compile(r"(?<![`{\w])%\s?\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s?%(?![\w}`])")
NON_PATH_PREFIXES = (
    "AICLS_",
    "http://",
    "https://",
    "/",
    "GET ",
    "POST ",
    "PUT ",
    "DELETE ",
    "v0.",
    "v1.",
)


@dataclass(slots=True)
class Row:
    """One deliverable row of the mapping table."""

    ip: str
    deliverable: str
    artefact_cell: str
    line_no: int
    paths: list[str] = field(default_factory=list)
    planned: bool = False
    missing: list[str] = field(default_factory=list)
    existing: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Default-mode verdict."""
        return self.planned or not self.paths or bool(self.existing)


def _looks_like_repo_path(token: str) -> bool:
    token = token.strip()
    if (
        not token
        or " " in token
        or "*" in token
        or "<" in token
        or token.startswith(NON_PATH_PREFIXES)
    ):
        return False
    if "/" not in token:
        return False
    return re.fullmatch(r"[A-Za-z0-9_./\-#ğüşöçıİĞÜŞÖÇ]+", token) is not None


def _normalise(token: str) -> str:
    return token.split("#", 1)[0].strip()


def parse_rows(text: str) -> list[Row]:
    """Extract deliverable rows from the markdown table in §1."""
    rows: list[Row] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if not ROW_RE.match(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        row = Row(ip=cells[0], deliverable=cells[1], artefact_cell=cells[2], line_no=i)
        row.planned = PLANNED_RE.search(row.artefact_cell) is not None
        for tok in BACKTICK_RE.findall(row.artefact_cell):
            if _looks_like_repo_path(tok):
                p = _normalise(tok)
                if p and p not in row.paths:
                    row.paths.append(p)
        rows.append(row)
    return rows


def check_rows(rows: list[Row], root: Path) -> None:
    """Fill ``existing``/``missing`` for each row relative to *root*."""
    for row in rows:
        for p in row.paths:
            if (root / p).exists():
                row.existing.append(p)
            else:
                row.missing.append(p)


def scan_hand_typed_percentages(root: Path) -> list[str]:
    """Return warnings for bare percentages in report pages (ip-eslestirme.md is exempt)."""
    warnings: list[str] = []
    rep_dir = root / "docs" / "raporlama"
    if not rep_dir.is_dir():
        return warnings
    for md in sorted(rep_dir.glob("*.md")):
        if md.name == "ip-eslestirme.md":
            continue
        for i, line in enumerate(md.read_text(encoding="utf-8").splitlines(), start=1):
            if "{{" in line or line.lstrip().startswith("|"):  # macro lines / target tables
                continue
            if PERCENT_RE.search(line):
                warnings.append(
                    f"{md.relative_to(root).as_posix()}:{i}: bare percentage — use a metrics macro"
                )
    return warnings


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--table", type=Path, default=TABLE_PATH)
    parser.add_argument(
        "--strict", action="store_true", help="every path must exist; no planlı left"
    )
    args = parser.parse_args(argv)
    root: Path = args.root.resolve()
    table = args.table if args.table.is_absolute() else root / args.table
    if not table.is_file():
        print(f"check_docs_map: table not found: {table}")
        return 1

    rows = parse_rows(table.read_text(encoding="utf-8"))
    if not rows:
        print("check_docs_map: no İP rows parsed — table format changed?")
        return 1
    check_rows(rows, root)

    failures: list[str] = []
    warnings: list[str] = []
    for row in rows:
        for p in row.missing:
            warnings.append(
                f"line {row.line_no} ({row.ip} — {row.deliverable}): missing path `{p}`"
            )
        if args.strict:
            if row.planned:
                failures.append(f"line {row.line_no} ({row.ip}): still marked planlı")
            if row.missing:
                failures.append(f"line {row.line_no} ({row.ip}): missing {row.missing}")
        elif not row.ok:
            failures.append(
                f"line {row.line_no} ({row.ip} — {row.deliverable}): none of {row.paths} exists "
                "and the row is not marked (planlı: ...)"
            )
    warnings.extend(scan_hand_typed_percentages(root))

    for w in warnings:
        print(f"WARN {w}")
    n_paths = sum(len(r.paths) for r in rows)
    n_existing = sum(len(r.existing) for r in rows)
    if failures:
        print(f"check_docs_map: FAIL — {len(failures)} row(s):")
        for f in failures:
            print(f"  {f}")
        return 1
    print(
        f"check_docs_map: OK — {len(rows)} rows, {n_existing}/{n_paths} artefact paths present, "
        f"{sum(r.planned for r in rows)} planned, {len(warnings)} warning(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
