"""Errors raised by the Turkish text-to-command layer (docs/mimari/metin-komut-katmani.md).

Everything that leaves :mod:`aicls.nlu` is a subclass of :class:`aicls.core.errors.AiclsError`,
which is what ``test_property_parse_command_never_raises_non_aicls_error`` pins down: a pydantic
``ValidationError`` or a ``lark`` exception must never reach the API layer.

Three failure families exist:

* :class:`InventoryError` -- the YAML command inventory shipped inside the package is malformed
  (a broken pattern, an unknown slot type, an intent without examples). This is a *developer*
  error: it can only happen when the resource or the schema changed, so it is raised at load
  time and never caught.
* :class:`SpokenMathError` -- a spoken-mathematics fragment ("x kare artı iki x") is not a
  sentence of ``spoken_math/grammar.lark``. The slot extractor catches it and marks the slot
  ``unresolved`` instead of failing the whole command, so the user gets a clarification question
  rather than a stack trace.
* :class:`CommandBuildError` -- intent and slots resolved but the typed
  :mod:`aicls.mathengine` model refused the values (a variable name outside ``VAR_PATTERN``, a
  derivative order above ``MAX_ORDER``). The resolver turns it into ``NoMatch`` so that an
  unbuildable command can never reach SymPy (ADR-0005).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aicls.core import messages_tr
from aicls.core.errors import AiclsError, ErrorCode


class NluError(AiclsError):
    """Base class of every error raised by :mod:`aicls.nlu` (``NLU_NO_INTENT``, HTTP 422)."""

    default_code = ErrorCode.NLU_NO_INTENT


class InventoryError(NluError):
    """The packaged command inventory is malformed (``VALIDATION_ERROR``, HTTP 422).

    Args:
        reason: Stable machine-readable slug (``"unknown_slot_type"``, ``"bad_pattern"``).
        details: Extra structured fields (``intent``, ``slot``, ``pattern`` ...).
    """

    default_code = ErrorCode.VALIDATION_ERROR

    reason: str

    def __init__(self, reason: str, *, details: Mapping[str, Any] | None = None) -> None:
        self.reason = reason
        merged: dict[str, Any] = {"reason": reason}
        if details:
            merged.update(details)
        super().__init__(self.default_code, messages_tr.MSG_VALIDATION_ERROR, merged)


class SpokenMathError(NluError):
    """A spoken-mathematics fragment did not parse (``PARSE_ERROR``, HTTP 422).

    Args:
        text: The normalised fragment that failed.
        reason: Stable slug (``"lex_error"``, ``"parse_error"``, ``"empty"``).
        position: 1-based column of the first offending character, when known.
        details: Extra structured fields.
    """

    default_code = ErrorCode.PARSE_ERROR

    text: str
    reason: str
    position: int

    def __init__(
        self,
        text: str,
        *,
        reason: str = "parse_error",
        position: int = 1,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.text = text
        self.reason = reason
        self.position = position
        merged: dict[str, Any] = {"text": text[:200], "reason": reason, "position": position}
        if details:
            merged.update(details)
        super().__init__(
            self.default_code,
            messages_tr.msg(
                "PARSE_ERROR_AT", position=position, reason=messages_tr.MSG_PARSE_ERROR
            ),
            merged,
        )


class CommandBuildError(NluError):
    """Resolved slots did not build a valid typed command (``VALIDATION_ERROR``, HTTP 422).

    Args:
        intent: Inventory intent id the slots came from.
        reason: Stable slug (``"model_rejected"``, ``"unknown_intent"``).
        details: Extra structured fields (the pydantic error list, the offending slot).
    """

    default_code = ErrorCode.VALIDATION_ERROR

    intent: str
    reason: str

    def __init__(
        self,
        intent: str,
        *,
        reason: str = "model_rejected",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.intent = intent
        self.reason = reason
        merged: dict[str, Any] = {"intent": intent, "reason": reason}
        if details:
            merged.update(details)
        super().__init__(self.default_code, messages_tr.MSG_VALIDATION_ERROR, merged)


__all__ = ["CommandBuildError", "InventoryError", "NluError", "SpokenMathError"]
