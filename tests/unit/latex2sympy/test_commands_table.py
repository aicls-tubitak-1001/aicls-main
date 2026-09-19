"""The supported-command table drives both the strict pre-check and the IP-4 report table."""

from __future__ import annotations

import re

from aicls.latex2sympy import commands as cmd
from aicls.latex2sympy.parser import grammar_source


def test_command_table_loads_and_is_cached() -> None:
    table = cmd.command_table()
    assert table is cmd.command_table()
    assert len(table.entries) >= 60
    entry = table.by_latex[r"\sin"]
    assert entry.kind == "function"
    assert entry.sympy == "sin"


def test_supported_commands_cover_the_grammar_terminals() -> None:
    """Every alternative inside FUNC_NAME/GREEK must be documented in commands.yaml."""
    source = grammar_source()
    for terminal in ("FUNC_NAME", "GREEK"):
        match = re.search(rf"^{terminal}\.\d+: /\\\\\((.+?)\)\(\?!", source, re.MULTILINE)
        assert match is not None, terminal
        names = match.group(1).split("|")
        for name in names:
            assert f"\\{name}" in cmd.SUPPORTED_COMMANDS, f"{terminal}: \\{name}"


def test_every_table_command_is_known_to_the_grammar_or_normaliser() -> None:
    source = grammar_source()
    for entry in cmd.command_table().entries:
        if entry.handled_by == "normalize":
            assert entry.terminal is None, entry.latex
            continue
        assert entry.terminal is not None, entry.latex
        assert f"{entry.terminal}." in source or f"{entry.terminal}:" in source, entry.latex


def test_unsupported_constructs_are_named() -> None:
    assert r"\begin{cases}" in cmd.UNSUPPORTED_CONSTRUCTS
    assert r"\vec" in cmd.UNSUPPORTED_CONSTRUCTS
    assert cmd.UNSUPPORTED_CONSTRUCTS[r"\vec"].startswith("v1.")


def test_classify_command() -> None:
    assert cmd.classify(r"\sin") == "supported"
    assert cmd.classify(r"\vec") == "unsupported"
    assert cmd.classify(r"\notacommand") == "unknown"
    assert cmd.classify(r"\displaystyle") == "decorative"


def test_scan_commands_finds_environments_and_macros() -> None:
    found = cmd.scan_commands(r"\begin{cases} \vec{v} \end{cases} \sin x")
    assert r"\begin{cases}" in found
    assert r"\vec" in found
    assert r"\sin" in found


def test_report_rows_are_sorted_and_complete() -> None:
    rows = cmd.report_rows()
    assert rows == sorted(rows, key=lambda row: (row["kind"], row["latex"]))
    assert all({"latex", "kind", "node", "sympy"} <= set(row) for row in rows)
