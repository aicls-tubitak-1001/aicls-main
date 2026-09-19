"""Errors raised by the LaTeX front end (docs/mimari/latex-ayristirici.md 4).

``parse_latex`` and everything below it raise **only** subclasses of
:class:`aicls.core.errors.AiclsError`:

* :class:`ParseError` - lexer/parser failure, carries ``line``, ``col``, ``expected`` and a
  ``snippet`` of the offending input; no partial result is ever returned;
* :class:`UnsupportedConstruct` - a construct that is outside the v1 grammar scope, named
  explicitly (``\\begin{cases}``) so the UI can tell the user what to avoid;
* :class:`SemanticError` - syntactically valid but meaningless input (an integral without a
  differential, a subscript on a number).

All Turkish wording comes from :mod:`aicls.core.messages_tr`; this module never spells out
user-facing text of its own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from aicls.core import messages_tr
from aicls.core.errors import AiclsError, ErrorCode


class ParseError(AiclsError):
    """The input is not a sentence of the grammar (``PARSE_ERROR``, HTTP 422).

    Args:
        line: 1-based line of the first offending character.
        col: 1-based column of the first offending character (also exposed as
            ``details["position"]``, the field name used by ``docs/mimari/api.md`` 2).
        expected: Terminal names the parser could have accepted at that point.
        snippet: A short excerpt of the input around the error position.
        message_tr: Optional override for the Turkish detail text.
        details: Extra structured fields merged into the problem document.
    """

    default_code = ErrorCode.PARSE_ERROR

    line: int
    col: int
    expected: list[str]
    snippet: str

    def __init__(
        self,
        *,
        line: int = 1,
        col: int = 1,
        expected: Sequence[str] | None = None,
        snippet: str = "",
        message_tr: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.line = line
        self.col = col
        self.expected = list(expected or ())
        self.snippet = snippet
        merged: dict[str, Any] = {
            "line": line,
            "col": col,
            "position": col,
            "expected": self.expected,
            "snippet": snippet,
        }
        if details:
            merged.update(details)
        text = message_tr
        if text is None:
            text = messages_tr.msg(
                "PARSE_ERROR_AT", position=col, reason=messages_tr.MSG_PARSE_ERROR
            )
        super().__init__(self.default_code, text, merged)


class UnsupportedConstruct(AiclsError):  # noqa: N818 - name fixed by the module contract
    """A LaTeX construct outside the v1 grammar scope (``UNSUPPORTED_CONSTRUCT``, HTTP 422).

    Args:
        construct: The construct exactly as written in the input (``\\vec``,
            ``\\begin{cases}``); it is repeated in the Turkish message and in
            ``details["construct"]``.
        note: Optional roadmap note (``"v1.1: ..."``).
        details: Extra structured fields.
    """

    default_code = ErrorCode.UNSUPPORTED_CONSTRUCT

    construct: str

    def __init__(
        self,
        construct: str,
        *,
        note: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.construct = construct
        merged: dict[str, Any] = {"construct": construct}
        if note:
            merged["note"] = note
        if details:
            merged.update(details)
        super().__init__(
            self.default_code,
            messages_tr.msg("UNSUPPORTED_CONSTRUCT_NAMED", construct=construct),
            merged,
        )


class SemanticError(AiclsError):
    """Syntactically valid input with no meaning (``PARSE_ERROR``, HTTP 422).

    ``ErrorCode`` has no dedicated semantic code; ``PARSE_ERROR`` is used so the API keeps
    returning 422 for every "your formula is wrong" case. ``details["reason"]`` carries a
    stable machine-readable reason such as ``integral_without_differential``.

    Args:
        reason: Stable reason slug.
        details: Extra structured fields.
    """

    default_code = ErrorCode.PARSE_ERROR

    reason: str

    def __init__(self, reason: str, *, details: Mapping[str, Any] | None = None) -> None:
        self.reason = reason
        merged: dict[str, Any] = {"reason": reason}
        if details:
            merged.update(details)
        super().__init__(self.default_code, messages_tr.MSG_PARSE_ERROR, merged)


__all__ = ["ParseError", "SemanticError", "UnsupportedConstruct"]
