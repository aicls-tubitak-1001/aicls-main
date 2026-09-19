"""Create an experiment directory from ``experiments/_template/`` (docs/deneme-alani.md §1).

Usage::

    python scripts/new_experiment.py <slug> [--date YYYY-MM-DD] [--person "Ad"]

Creates ``experiments/YYYY-MM-DD-<slug>/`` with ``NOTE.md``, ``run.py`` and ``config.yaml``
copied from the template and header fields pre-filled. Approval gates apply inside experiments
too: synthetic data, stub models, CPU only.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import re
import shutil
import sys

EXP_DIR = Path("experiments")
TEMPLATE = EXP_DIR / "_template"
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def fill_note(text: str, *, date: dt.date, person: str, slug: str) -> str:
    """Pre-fill the NOTE.md header fields."""
    out = text.replace("# <Deneme başlığı>", f"# {slug.replace('-', ' ')}")
    out = out.replace("YYYY-MM-DD", date.isoformat())
    out = out.replace("<ad>", person)
    out = out.replace("exp/<slug>", f"exp/{slug}")
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slug")
    parser.add_argument("--date", type=dt.date.fromisoformat, default=dt.date.today())
    parser.add_argument("--person", default="<ad>")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    if not SLUG_RE.match(args.slug):
        print(f"new_experiment: slug must be kebab-case ascii: {args.slug!r}")
        return 2
    root: Path = args.root
    src = root / TEMPLATE
    if not src.is_dir():
        print(f"new_experiment: template missing: {src}")
        return 1
    dest = root / EXP_DIR / f"{args.date.isoformat()}-{args.slug}"
    if dest.exists():
        print(f"new_experiment: {dest} already exists")
        return 1
    shutil.copytree(src, dest)
    note = dest / "NOTE.md"
    note.write_text(
        fill_note(
            note.read_text(encoding="utf-8"), date=args.date, person=args.person, slug=args.slug
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(f"new_experiment: created {dest.relative_to(root).as_posix()}")
    print(
        f"Next: git switch -c exp/{args.slug}; edit NOTE.md; keep results small "
        "(big outputs -> AICLS_DATA_ROOT/runs)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
