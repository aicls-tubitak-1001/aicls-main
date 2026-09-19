"""The shared CLI plumbing: JSON encoding, input parsing and the error boundary.

These helpers are what makes ``--json`` the same shape in every command and what turns a typed
:class:`aicls.core.errors.AiclsError` into a documented exit code, so they are tested directly
rather than only through the commands that use them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aicls.cli.common import (
    EXIT_ERROR,
    EXIT_GATE,
    dump_json,
    jsonable,
    parse_ratios,
    read_jsonl,
    render_error,
    split_csv,
)
from aicls.core.errors import AiclsError, InputValidationError, NotFoundError
from aicls.core.gates import GateError


def test_exit_codes_match_the_error_hierarchy() -> None:
    """Exit code 3 means "a gate refused" and nothing else; every other typed error is 1."""
    assert GateError.exit_code == EXIT_GATE
    assert AiclsError.exit_code == EXIT_ERROR
    assert render_error(GateError()) == EXIT_GATE
    assert render_error(NotFoundError()) == EXIT_ERROR


def test_split_csv_drops_empty_parts() -> None:
    """``--stratify university,gender`` and a trailing comma mean the same thing."""
    assert split_csv("university, gender,") == ("university", "gender")
    assert split_csv("") == ()
    assert split_csv(None) == ()


def test_parse_ratios_defaults_and_validates() -> None:
    """Three numbers or nothing; a typo must not silently become a different split."""
    default = (0.8, 0.1, 0.1)
    assert parse_ratios("", default) == default
    assert parse_ratios("0.7,0.2,0.1", default) == (0.7, 0.2, 0.1)
    with pytest.raises(InputValidationError) as two:
        parse_ratios("0.9,0.1", default)
    assert two.value.details["reason"] == "ratios_need_three_values"
    with pytest.raises(InputValidationError) as text:
        parse_ratios("a,b,c", default)
    assert text.value.details["reason"] == "ratios_not_numeric"


def test_read_jsonl_reports_the_offending_line(tmp_path: Path) -> None:
    """A manifest with a broken row names the row, so it can be fixed."""
    good = tmp_path / "good.jsonl"
    good.write_text('{"id": "a"}\n\n{"id": "b"}\n', encoding="utf-8", newline="\n")
    assert [row["id"] for row in read_jsonl(good)] == ["a", "b"]

    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"id": "a"}\n{oops\n', encoding="utf-8", newline="\n")
    with pytest.raises(InputValidationError) as exc:
        read_jsonl(broken)
    assert exc.value.details == {
        "reason": "jsonl_not_json",
        "path": str(broken),
        "line": 2,
    }

    scalar = tmp_path / "scalar.jsonl"
    scalar.write_text("42\n", encoding="utf-8", newline="\n")
    with pytest.raises(InputValidationError) as row_exc:
        read_jsonl(scalar)
    assert row_exc.value.details["reason"] == "jsonl_row_not_object"

    with pytest.raises(NotFoundError):
        read_jsonl(tmp_path / "absent.jsonl")


def test_jsonable_never_raises_on_an_unknown_object() -> None:
    """A CLI must not fail while reporting a result, so unknown values degrade to text."""

    class Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    assert jsonable(Opaque()) == "<opaque>"
    assert json.loads(dump_json({"value": Opaque()})) == {"value": "<opaque>"}


def test_jsonable_uses_as_dict_when_a_record_provides_one() -> None:
    """``aicls.collect`` records expose ``as_dict``; the encoder prefers it over a repr."""

    class Record:
        def as_dict(self) -> dict[str, int]:
            return {"n": 1}

    assert jsonable(Record()) == {"n": 1}
