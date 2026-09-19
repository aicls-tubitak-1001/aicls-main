"""Generate an entity-relationship diagram (Mermaid ``erDiagram``) from the SQLAlchemy metadata.

Stub until ``aicls.db.base`` lands: it imports ``aicls.db.base.Base`` lazily, walks
``Base.metadata.sorted_tables`` and writes ``docs/mimari/assets/erd.mmd`` (Mermaid) which
``docs/mimari/veri-modeli.md`` embeds with a ```` ```mermaid ```` fence. No external tools
(Graphviz etc.) are required — Mermaid renders in the MkDocs site.

Usage::

    python tools/gen_erd.py [--out docs/mimari/assets/erd.mmd]

Exit code 0 with a notice when the db package is not implemented yet, so ``poe erd`` and CI do
not break during the scaffold phase.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

DEFAULT_OUT = Path("docs/mimari/assets/erd.mmd")


def _mermaid_type(col: Any) -> str:
    try:
        return str(col.type).split("(")[0].lower().replace(" ", "_")
    except Exception:
        return "unknown"


def render_erd(metadata: Any) -> str:
    """Render SQLAlchemy ``MetaData`` as a Mermaid ``erDiagram`` string."""
    lines = ["erDiagram"]
    for table in metadata.sorted_tables:
        lines.append(f"    {table.name} {{")
        for col in table.columns:
            flags = []
            if col.primary_key:
                flags.append("PK")
            if col.foreign_keys:
                flags.append("FK")
            flag = " " + ",".join(flags) if flags else ""
            lines.append(f"        {_mermaid_type(col)} {col.name}{flag}")
        lines.append("    }")
    for table in metadata.sorted_tables:
        for fk in table.foreign_keys:
            parent = fk.column.table.name
            lines.append(f'    {parent} ||--o{{ {table.name} : "{fk.parent.name}"')
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        from aicls.db.base import Base  # lazy: db package may not exist yet
    except (ImportError, AttributeError) as exc:
        print(f"gen_erd: aicls.db.base.Base not available yet ({exc}); nothing generated.")
        return 0
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_erd(Base.metadata), encoding="utf-8", newline="\n")
    print(f"gen_erd: wrote {out} ({len(Base.metadata.sorted_tables)} tables)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
