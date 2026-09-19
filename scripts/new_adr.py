"""Create a new ADR from ``docs/adr/0000-sablon.md`` (docs/adr/README.md süreç §1).

Usage::

    python scripts/new_adr.py <slug> [--title "Karar tek cümle"] [--ip 4]

Picks the next number, writes ``docs/adr/NNNN-<slug>.md`` with status ``önerildi`` and today's
date, and appends a row to the index table in ``docs/adr/README.md``.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import re
import sys

ADR_DIR = Path("docs/adr")
TEMPLATE = ADR_DIR / "0000-sablon.md"
INDEX = ADR_DIR / "README.md"
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def next_number(adr_dir: Path) -> int:
    """Highest existing ``NNNN-*.md`` + 1 (template 0000 counts as 0)."""
    numbers = [int(p.name[:4]) for p in adr_dir.glob("[0-9][0-9][0-9][0-9]-*.md")]
    return (max(numbers) if numbers else 0) + 1


def render(template: str, *, number: int, title: str, date: dt.date, ip: str, authors: str) -> str:
    """Fill the MADR template header fields."""
    out = template.replace("ADR-NNNN: <Başlık: karar tek cümle>", f"ADR-{number:04d}: {title}")
    out = out.replace(
        "- **Durum:** önerildi | kabul edildi | yerini aldı (ADR-XXXX) | reddedildi",
        "- **Durum:** önerildi",
    )
    out = out.replace("- **Tarih:** YYYY-MM-DD", f"- **Tarih:** {date.isoformat()}")
    out = out.replace("- **Karar vericiler:** <adlar>", f"- **Karar vericiler:** {authors}")
    out = out.replace("- **İlgili İP:** İP-N", f"- **İlgili İP:** İP-{ip}")
    return out


def append_index_row(index_text: str, *, number: int, slug: str, title: str) -> str:
    """Append ``| [NNNN](file) | title | önerildi |`` after the last table row of the index."""
    row = f"| [{number:04d}]({number:04d}-{slug}.md) | {title} | önerildi |"
    lines = index_text.splitlines()
    last_row = max(
        (i for i, ln in enumerate(lines) if re.match(r"^\| \[\d{4}\]", ln)), default=None
    )
    if last_row is None:
        return index_text.rstrip("\n") + "\n\n" + row + "\n"
    lines.insert(last_row + 1, row)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slug", help="kebab-case slug, e.g. redis-rate-limit")
    parser.add_argument("--title", default=None, help="decision in one sentence (Turkish)")
    parser.add_argument("--ip", default="N", help="related work package number")
    parser.add_argument("--authors", default="<adlar>")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    if not SLUG_RE.match(args.slug):
        print(f"new_adr: slug must be kebab-case ascii: {args.slug!r}")
        return 2
    root: Path = args.root
    adr_dir = root / ADR_DIR
    template = (root / TEMPLATE).read_text(encoding="utf-8")
    number = next_number(adr_dir)
    title = args.title or args.slug.replace("-", " ")
    target = adr_dir / f"{number:04d}-{args.slug}.md"
    if target.exists():
        print(f"new_adr: {target} already exists")
        return 1
    target.write_text(
        render(
            template,
            number=number,
            title=title,
            date=dt.date.today(),
            ip=args.ip,
            authors=args.authors,
        ),
        encoding="utf-8",
        newline="\n",
    )
    index_path = root / INDEX
    index_path.write_text(
        append_index_row(
            index_path.read_text(encoding="utf-8"), number=number, slug=args.slug, title=title
        ),
        encoding="utf-8",
        newline="\n",
    )
    rel = target.relative_to(root).as_posix()
    print(f"new_adr: created {rel} and indexed it in docs/adr/README.md")
    print(f"Next: fill in 'Öneri metniyle uyum' and open a PR on branch docs/adr-{number:04d}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
