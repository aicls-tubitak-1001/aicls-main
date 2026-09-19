"""Export the FastAPI OpenAPI schema (CI job ``openapi``, api.md §5).

Two files are written from one in-process ``create_app()``:

* ``docs/api/openapi.json`` — the machine-readable half of the İP-2 "API Dokümantasyonu"
  deliverable, checked into git so a route or schema change is always a visible diff;
* ``web/src/api/generated/openapi.json`` — the input ``openapi-typescript`` turns into
  ``web/src/api/generated/schema.d.ts`` (``npm run api:types`` in ``web/``). The generator is a
  Node tool and is not run from here; this script only keeps its input in step, which is what
  KNOWN_ISSUES asked for.

The document is always generated on the ``full`` profile, so a developer whose ``.env`` says
``AICLS_PROFILE=collect`` cannot silently commit a truncated schema. ``--check`` exits 1 when
either file on disk differs from the freshly generated one (drift detection; CI also runs
``git diff --exit-code``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

#: Where the checked-in schema lives.
DEFAULT_OUT = Path("docs/api/openapi.json")

#: Input file of ``openapi-typescript`` (``web/src/api/generated/schema.d.ts``).
DEFAULT_WEB_OUT = Path("web/src/api/generated/openapi.json")

#: Repository root, derived from this file's location.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_src_on_path() -> None:
    """Make ``src/`` importable when the script runs from a plain checkout (no editable install)."""
    src = REPO_ROOT / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def build_schema() -> dict[str, Any] | None:
    """Return the OpenAPI dict of the ``full`` profile, or ``None`` when ``aicls.api`` is absent."""
    _ensure_src_on_path()
    try:
        from aicls.api.app import create_app
        from aicls.core.settings import settings_override
    except (ImportError, AttributeError):
        return None
    with settings_override(profile="full") as settings:
        app = create_app(settings)
        schema: dict[str, Any] = app.openapi()
    return schema


def dump(schema: dict[str, Any]) -> str:
    """Stable JSON text (sorted keys, 2-space indent, trailing newline)."""
    return json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` with LF endings, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _is_stale(path: Path, text: str) -> bool:
    """Whether the file on disk differs from ``text``."""
    current = path.read_text(encoding="utf-8") if path.is_file() else ""
    return current != text


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--web-out", type=Path, default=DEFAULT_WEB_OUT)
    parser.add_argument("--no-web", action="store_true", help="skip the web/ input file")
    parser.add_argument("--check", action="store_true", help="fail if a file on disk is stale")
    parser.add_argument("--require", action="store_true", help="fail if aicls.api is missing")
    args = parser.parse_args(argv)

    schema = build_schema()
    if schema is None:
        print("export_openapi: aicls.api.app.create_app not available yet; nothing exported.")
        return 1 if args.require else 0

    text = dump(schema)
    targets: list[Path] = [args.out]
    if not args.no_web:
        targets.append(args.web_out)

    if args.check:
        stale = [target for target in targets if _is_stale(target, text)]
        if stale:
            names = ", ".join(str(path) for path in stale)
            print(f"export_openapi: {names} stale — run `poe openapi` and commit.")
            return 1
        print(f"export_openapi: {len(targets)} file(s) up to date")
        return 0

    for target in targets:
        _write(target, text)
    print(
        f"export_openapi: wrote {', '.join(str(path) for path in targets)} "
        f"({len(schema.get('paths', {}))} paths)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
