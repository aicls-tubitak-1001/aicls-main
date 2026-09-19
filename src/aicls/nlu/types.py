"""Public value types of the text-to-command layer.

Three groups live here:

* the **non-CAS commands** the layer can produce -- :class:`BookFetch` (İP-8 content lookup) and
  :class:`UiAction` (client-side actions). Together with
  :data:`aicls.mathengine.models.CasCommand` they form :data:`Command`, everything
  :class:`Matched` may carry.
* the **parse context** :class:`ParseContext`, which is how "türevini al" (no expression) resolves
  against the expression already on the board.
* the **outcome union** :data:`ParseOutcome` = :class:`Matched` | :class:`Clarification` |
  :class:`NoMatch`, plus :class:`Alternative` for runner-up intents.

:class:`UnitType` mirrors :class:`aicls.book.schema.UnitType` value for value. It is duplicated
rather than imported because ADR-0002 forbids ``aicls.nlu`` from importing ``aicls.book`` (the
book package sits *above* the NLU layer); ``tests/integration/test_nlu_contracts.py`` asserts the
two enumerations stay in step.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field

from aicls.mathengine.models import CasCommand

#: How the text reached the layer; ``voice`` makes the UI ask for a ``CommandPreview`` (ADR-0005).
Modality: TypeAlias = Literal["text", "voice", "ink"]

#: Client-side actions the layer may emit; they never reach SymPy.
UiActionName: TypeAlias = Literal["clear", "undo", "repeat_last", "export"]

#: Where a resolved slot value came from, recorded in the trace.
SlotSource: TypeAlias = Literal["pattern", "scan", "context", "default", "inferred"]


class UnitType(StrEnum):
    """Type of a textbook content unit (mirror of :class:`aicls.book.schema.UnitType`)."""

    DEFINITION = "Definition"
    THEOREM = "Theorem"
    LEMMA = "Lemma"
    COROLLARY = "Corollary"
    PROOF = "Proof"
    EXAMPLE = "Example"
    EXERCISE = "Exercise"
    FIGURE = "Figure"
    TABLE = "Table"
    REMARK = "Remark"
    SECTION = "Section"


class BookFetch(BaseModel):
    """Fetch textbook content (İP-8); executed by ``aicls.book``, never by the CAS.

    Attributes:
        op: Discriminator, always ``"book_fetch"``.
        unit_type: Requested unit type, or ``None`` for a free search over every type.
        query: The Turkish phrase to resolve against unit titles and aliases.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    op: Literal["book_fetch"] = "book_fetch"
    unit_type: UnitType | None = None
    query: str = Field(min_length=1, max_length=200)


class UiAction(BaseModel):
    """A client-side action (clear the board, undo, repeat, export).

    Attributes:
        op: Discriminator, always ``"ui"``.
        action: Which action the client must perform.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    op: Literal["ui"] = "ui"
    action: UiActionName


#: Everything :class:`Matched` may carry: a typed CAS command, a book lookup or a UI action.
Command: TypeAlias = CasCommand | BookFetch | UiAction


@dataclass(frozen=True, slots=True)
class ParseContext:
    """What the session already knows, used to fill ``from_context`` slots.

    Attributes:
        last_latex: The expression currently on the board (from handwriting, typing or the
            previous result). ``"türevini al"`` resolves its ``expr`` slot from it.
        active_var: Variable the session is working with; wins over the inventory default when
            it occurs in the expression.
        modality: How the text arrived (``text``, ``voice``, ``ink``).
    """

    last_latex: str | None = None
    active_var: str | None = None
    modality: Modality = "text"


#: Modality values a caller may send; anything else falls back to ``text``.
MODALITIES: frozenset[str] = frozenset(("text", "voice", "ink"))


def context_from_mapping(mapping: Mapping[str, str] | None) -> ParseContext | None:
    """Build a :class:`ParseContext` from loosely typed data (YAML, JSON, a request body).

    Args:
        mapping: ``last_latex`` / ``active_var`` / ``modality`` keys, any of them absent.

    Returns:
        The context, or ``None`` when ``mapping`` is empty, so callers can pass the result
        straight to :func:`aicls.nlu.parse_command`.
    """
    if not mapping:
        return None
    raw = str(mapping.get("modality", "text"))
    modality: Modality = cast("Modality", raw) if raw in MODALITIES else "text"
    return ParseContext(
        last_latex=mapping.get("last_latex"),
        active_var=mapping.get("active_var"),
        modality=modality,
    )


@dataclass(frozen=True, slots=True)
class Alternative:
    """A runner-up intent, shown in a clarification question or in ``Matched.alternatives``.

    Attributes:
        intent: Inventory intent id.
        score: Final score of that candidate in ``[0, 1]``.
        label_tr: Turkish one-line description of the intent (from the inventory).
    """

    intent: str
    score: float
    label_tr: str = ""


@dataclass(frozen=True, slots=True)
class Matched:
    """The text was understood and a typed command was built.

    Attributes:
        command: The typed command; only :class:`aicls.mathengine.models.CasCommand` members
            ever reach SymPy.
        intent: Inventory intent id that won.
        slots: Resolved slot values, keyed by slot name.
        score: Final score of the winning candidate in ``[0, 1]``.
        alternatives: Runner-up intents, best first.
        trace: Human-readable "how I understood you" lines (normalised text, matched pattern,
            slot resolutions) for the UI panel and the İP-9 error analysis.
        source: ``"rules"`` for the deterministic path, ``"llm"`` when an
            :class:`~aicls.nlu.fallback.IntentFallback` suggested the intent id.
    """

    command: Command
    intent: str
    slots: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    alternatives: list[Alternative] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    source: Literal["rules", "llm"] = "rules"


@dataclass(frozen=True, slots=True)
class Clarification:
    """The intent is known (or nearly) but something must be asked before computing.

    Attributes:
        question_tr: The Turkish question to show the user.
        missing_slot: Name of the slot that could not be resolved, or ``None`` when the
            ambiguity is between intents.
        candidates: Up to three candidate intents, best first.
        intent: Best-scoring intent id, or ``""`` when several tied.
        trace: Same trace lines as :class:`Matched`.
    """

    question_tr: str
    missing_slot: str | None = None
    candidates: list[Alternative] = field(default_factory=list)
    intent: str = ""
    trace: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class NoMatch:
    """Nothing in the inventory covers the request (the İP-4 "guide, do not fail" path).

    Attributes:
        suggestions_tr: Example commands the platform *does* understand, in Turkish.
        reason: Stable slug (``"no_intent"``, ``"below_min_score"``, ``"command_invalid"``,
            ``"empty_input"``).
        trace: Same trace lines as :class:`Matched`.
    """

    suggestions_tr: list[str] = field(default_factory=list)
    reason: str = "no_intent"
    trace: list[str] = field(default_factory=list)


#: What :func:`aicls.nlu.parse_command` returns.
ParseOutcome: TypeAlias = Matched | Clarification | NoMatch


__all__ = [
    "MODALITIES",
    "Alternative",
    "BookFetch",
    "Clarification",
    "Command",
    "Matched",
    "Modality",
    "NoMatch",
    "ParseContext",
    "ParseOutcome",
    "SlotSource",
    "UiAction",
    "UiActionName",
    "UnitType",
    "context_from_mapping",
]
