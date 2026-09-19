"""Flag raw SQL outside Alembic migrations (docs/guvenlik-gizlilik.md §5, is-akisi.md §5).

Rules (Python files under ``src/`` by default; ``src/aicls/db/migrations/`` is exempt):

* ``sqlalchemy.text(...)`` / ``text("SELECT ...")`` calls
* ``session.execute("...")`` / ``connection.execute("...")`` / ``cursor.execute(`` with a string
* string literals that start with an SQL verb (``SELECT``, ``INSERT``, ``UPDATE``, ``DELETE``,
  ``CREATE``, ``DROP``, ``ALTER``) followed by whitespace
* f-strings / ``%`` / ``.format`` building such statements

Only the typed SQLAlchemy 2.0 API (``select()``, ``Mapped[...]``) is allowed in application code.
A reviewed exception may end the line with ``# raw-sql-ok: <reason>``.
Usable as a pre-commit hook (file names as arguments) or standalone (scans ``src/``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

ALLOW_MARK = "# raw-sql-ok:"
EXEMPT_PARTS = ("migrations",)
SQL_VERBS = r"(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|TRUNCATE|GRANT|REVOKE)"
RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("text()", re.compile(r"(?<![\w.])(sqlalchemy\.)?text\(\s*[frb]*[\"']")),
    ("execute(str)", re.compile(r"\.execute\(\s*[frb]*[\"']")),
    # Upper-case verbs only (SQL convention) so English log messages are not flagged.
    ("sql-literal", re.compile(r"[\"']\s*" + SQL_VERBS + r"\s+[^\"']*[\"']")),
    ("sql-fstring", re.compile(r"\bf[\"']\s*" + SQL_VERBS + r"\s")),
)


def is_exempt(path: Path) -> bool:
    """Migrations and this linter itself are exempt."""
    return any(part in EXEMPT_PARTS for part in path.parts) or path.name == "lint_no_raw_sql.py"


def scan_text(text: str, path: Path) -> list[str]:
    """Return ``path:line: [rule] code`` strings for each violation."""
    out: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("#") or ALLOW_MARK in line:
            continue
        for rule, pattern in RULES:
            if pattern.search(line):
                out.append(f"{path.as_posix()}:{i}: [{rule}] {s}")
                break
    return out


def scan_paths(paths: list[Path], root: Path) -> list[str]:
    """Scan files (or directories recursively) below *root*."""
    problems: list[str] = []
    for arg in paths:
        p = arg if arg.is_absolute() else root / arg
        files = sorted(p.rglob("*.py")) if p.is_dir() else [p]
        for f in files:
            if f.suffix != ".py" or is_exempt(f):
                continue
            try:
                rel = f.relative_to(root)
            except ValueError:
                rel = f
            problems.extend(scan_text(f.read_text(encoding="utf-8"), rel))
    return problems


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exit 1 on any violation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path, help="files/dirs (default: src/)")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    root: Path = args.root.resolve()
    paths = args.paths or [root / "src"]
    problems = scan_paths(paths, root)
    if problems:
        print(f"lint_no_raw_sql: {len(problems)} raw SQL usage(s) outside migrations:")
        print("\n".join(problems))
        print(
            "Use SQLAlchemy 2.0 typed API (select(), Mapped[...]); "
            "see docs/guvenlik-gizlilik.md §5."
        )
        return 1
    print("lint_no_raw_sql: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
