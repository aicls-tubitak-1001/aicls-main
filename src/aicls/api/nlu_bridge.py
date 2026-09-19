"""Helpers that turn an NLU outcome into the shapes ``ComputeOut`` needs (api.md §3.4).

Two jobs live here:

* :func:`command_for_intent` — re-resolve one *named* intent against the same utterance. The
  layer's :func:`aicls.nlu.parse_command` returns a :class:`~aicls.nlu.types.Clarification`
  whose candidates carry only an intent id and a score, but the UI needs a command preview for
  each of them, and ``confirm_intent`` must be able to say "I meant ``series``". Both are
  answered by running the package's own :class:`~aicls.nlu.SlotExtractor` and
  :func:`~aicls.nlu.build_command` for that one intent — the rule-based path, never a second
  guess: an intent the caller names still has to fill its required slots from the text.
* :func:`primary_latex` — which LaTeX field of a typed command is "the input" for the history
  row and for ``ComputeOut.input_latex``.
"""

from __future__ import annotations

from typing import Any

from aicls.core.errors import AiclsError
from aicls.core.logging import get_logger
from aicls.nlu import (
    IntentMatcher,
    SlotExtractor,
    build_command,
    load_inventory,
    normalize_text,
)
from aicls.nlu.inventory.schema import Inventory
from aicls.nlu.types import Command, ParseContext

#: LaTeX-bearing fields of the CAS commands, in the order they identify an input.
LATEX_FIELDS: tuple[str, ...] = ("expr", "equation")

_logger = get_logger(__name__)


def command_for_intent(
    text: str,
    intent_id: str,
    *,
    context: ParseContext | None = None,
    inventory: Inventory | None = None,
) -> tuple[Command | None, dict[str, Any]]:
    """Build the command ``intent_id`` would produce for ``text``.

    Args:
        text: The utterance the user typed or spoke.
        intent_id: Inventory intent to force (from ``confirm_intent`` or a clarification
            candidate).
        context: Session context (``last_latex``, ``active_var``).
        inventory: Inventory to resolve against; the packaged one when omitted.

    Returns:
        ``(command, slots)``. ``command`` is ``None`` when the intent is unknown or its required
        slots cannot be filled from this text — the caller then keeps asking rather than
        computing something the user did not say.
    """
    active = inventory or load_inventory()
    intent = active.by_id.get(intent_id)
    if intent is None:
        return None, {}
    analysed = normalize_text(text)
    matches = next(
        (
            candidate.matches
            for candidate in IntentMatcher(active).rank(analysed)
            if candidate.intent.id == intent_id
        ),
        (),
    )
    resolution = SlotExtractor(active).best(intent, analysed, matches, context=context)
    if not resolution.complete:
        return None, dict(resolution.values)
    try:
        return build_command(intent, resolution.values), dict(resolution.values)
    except AiclsError as exc:
        _logger.info("api.nlu.command_build_failed", intent=intent_id, error=str(exc)[:120])
        return None, dict(resolution.values)


def primary_latex(command: Command | None) -> str | None:
    """Return the LaTeX a command was built from (``ComputeOut.input_latex``)."""
    if command is None:
        return None
    for field in LATEX_FIELDS:
        value = getattr(command, field, None)
        if isinstance(value, str) and value:
            return value
    return None


def command_payload(command: Command | None) -> dict[str, Any] | None:
    """JSON-serialisable form of a typed command (``ComputeOut.command``)."""
    if command is None:
        return None
    payload: dict[str, Any] = command.model_dump(mode="json")
    return payload


__all__ = ["LATEX_FIELDS", "command_for_intent", "command_payload", "primary_latex"]
