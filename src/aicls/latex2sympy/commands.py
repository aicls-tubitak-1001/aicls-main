"""Supported-command table loaded from ``grammar/commands.yaml`` (ADR-0004).

The table has three jobs:

* it is the allow-list of the strict pre-check in :mod:`aicls.latex2sympy.parser`, so an
  unknown macro is reported as :class:`~aicls.latex2sympy.errors.UnsupportedConstruct` **with
  its name** instead of turning into an anonymous syntax error;
* it names every construct that is deliberately out of scope in v1 together with the release
  that will bring it (``UNSUPPORTED_CONSTRUCTS``);
* :func:`report_rows` renders it as the command -> AST node -> SymPy mapping table required by
  the IP-4 technical report.

``tests/unit/latex2sympy/test_commands_table.py`` keeps the file in sync with the grammar.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any, Literal, cast

import yaml  # type: ignore[import-untyped]  # PyYAML ships no stubs and types-PyYAML is absent

from aicls.core.paths import package_resource

#: Result of :func:`classify`.
Classification = Literal["supported", "unsupported", "decorative", "unknown"]

_RESOURCE_PACKAGE = "aicls.latex2sympy"
_RESOURCE_NAME = "grammar/commands.yaml"

_COMMAND_RE = re.compile(r"\\(?:begin|end)\s*\{[A-Za-z*]+\}|\\[A-Za-z]+|\\[,;:!>]")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class CommandEntry:
    """One row of the command table."""

    latex: str
    kind: str
    node: str
    sympy: str
    terminal: str | None = None
    handled_by: str = "grammar"
    note: str = ""


@dataclass(frozen=True, slots=True)
class CommandTable:
    """The parsed ``commands.yaml``."""

    version: str
    entries: tuple[CommandEntry, ...]
    by_latex: dict[str, CommandEntry]
    decorative: frozenset[str]
    unsupported: dict[str, str]
    lenient_drop: frozenset[str]


def _as_str(value: Any, default: str = "") -> str:
    return default if value is None else str(value)


@lru_cache(maxsize=1)
def command_table() -> CommandTable:
    """Load (and cache for the process) the command table shipped with the package.

    Returns:
        The parsed :class:`CommandTable`.

    Raises:
        aicls.core.errors.NotFoundError: If the packaged resource is missing.
    """
    path = package_resource(_RESOURCE_PACKAGE, _RESOURCE_NAME)
    raw = cast(dict[str, Any], yaml.safe_load(path.read_text(encoding="utf-8")))
    entries: list[CommandEntry] = []
    for row in raw.get("commands", []):
        entries.append(
            CommandEntry(
                latex=str(row["latex"]),
                kind=str(row["kind"]),
                node=_as_str(row.get("node"), "-"),
                sympy=_as_str(row.get("sympy"), "-"),
                terminal=None if row.get("terminal") is None else str(row["terminal"]),
                handled_by=_as_str(row.get("handled_by"), "grammar"),
                note=_as_str(row.get("note")),
            )
        )
    unsupported: dict[str, str] = {}
    lenient: set[str] = set()
    for row in raw.get("unsupported", []):
        name = str(row["latex"])
        unsupported[name] = _as_str(row.get("note"))
        if row.get("lenient_drop"):
            lenient.add(name)
    return CommandTable(
        version=str(raw.get("version", "0")),
        entries=tuple(entries),
        by_latex={entry.latex: entry for entry in entries},
        decorative=frozenset(str(item) for item in raw.get("decorative", [])),
        unsupported=unsupported,
        lenient_drop=frozenset(lenient),
    )


#: Commands the grammar (or the normaliser) understands.
SUPPORTED_COMMANDS: frozenset[str] = frozenset(command_table().by_latex)
#: Layout-only commands: silently removed, never an error.
DECORATIVE_COMMANDS: frozenset[str] = command_table().decorative
#: Constructs deliberately out of scope in v1, mapped to their roadmap note.
UNSUPPORTED_CONSTRUCTS: dict[str, str] = dict(command_table().unsupported)
#: Unsupported constructs that ``parse_latex(strict=False)`` strips instead of rejecting.
LENIENT_DROP_COMMANDS: frozenset[str] = command_table().lenient_drop


def classify(command: str) -> Classification:
    """Classify a command token such as ``\\sin`` or ``\\begin{cases}``.

    Args:
        command: The command exactly as it appears in the input.

    Returns:
        ``"decorative"`` for layout-only macros, ``"unsupported"`` for named out-of-scope
        constructs, ``"supported"`` for everything in the table and ``"unknown"`` otherwise.
    """
    if command in DECORATIVE_COMMANDS:
        return "decorative"
    if command in UNSUPPORTED_CONSTRUCTS:
        return "unsupported"
    if command in SUPPORTED_COMMANDS:
        return "supported"
    return "unknown"


def scan_commands(src: str) -> list[str]:
    """Return the LaTeX commands in ``src``, de-duplicated, in order of first appearance.

    ``\\begin``/``\\end`` are returned together with their environment
    (``\\begin{cases}``) so the caller can name the construct precisely.
    """
    found: list[str] = []
    seen: set[str] = set()
    for match in _COMMAND_RE.finditer(src):
        token = _WS_RE.sub("", match.group(0))
        if token not in seen:
            seen.add(token)
            found.append(token)
    return found


def report_rows() -> list[dict[str, str]]:
    """Render the table as report rows sorted by ``(kind, latex)`` (IP-4 mapping table)."""
    rows = [
        {
            "latex": entry.latex,
            "kind": entry.kind,
            "node": entry.node,
            "sympy": entry.sympy,
            "terminal": entry.terminal or "-",
            "handled_by": entry.handled_by,
            "note": entry.note,
        }
        for entry in command_table().entries
    ]
    return sorted(rows, key=lambda row: (row["kind"], row["latex"]))


__all__ = [
    "DECORATIVE_COMMANDS",
    "LENIENT_DROP_COMMANDS",
    "SUPPORTED_COMMANDS",
    "UNSUPPORTED_CONSTRUCTS",
    "Classification",
    "CommandEntry",
    "CommandTable",
    "classify",
    "command_table",
    "report_rows",
    "scan_commands",
]
