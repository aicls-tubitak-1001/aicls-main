"""Pydantic schema of the packaged command inventory (``inventory/commands.yaml``).

The inventory is the *contract surface* of the NLU layer: seventeen v1 intents, each with its
Turkish utterance patterns, keyword weights, slot definitions, clarification questions and at
least ten worked examples. Validating it with pydantic at load time means a typo in the YAML is
an :class:`~aicls.nlu.errors.InventoryError` at import, not a silently missing intent in the
classroom.

:meth:`Inventory.validate_examples` closes the loop: it re-runs every example through
``parse_command`` and reports the ones that do not resolve to their own intent and slots. That is
simultaneously the inventory's documentation, its regression suite
(``test_inventory_every_example_resolves_to_own_intent``) and the seed of
``datasets/nlu/commands_test.jsonl``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aicls.nlu.errors import InventoryError
from aicls.nlu.patterns import CompiledPattern, compile_patterns

#: Slot data types the extractor implements (docs/mimari/metin-komut-katmani.md §2).
SlotTypeName: TypeAlias = Literal[
    "EXPR", "VAR", "BOUNDS", "POINT", "DIRECTION", "ORDER", "UNIT_TYPE", "QUERY"
]

#: What an intent ultimately builds.
IntentKind: TypeAlias = Literal["cas", "book", "ui"]

#: How a slot value is found.
SlotFill: TypeAlias = Literal["pattern", "scan"]

#: Minimum number of examples every intent must carry (``test_inventory_each_intent_...``).
MIN_EXAMPLES_PER_INTENT = 10


class SlotType(BaseModel):
    """Declaration of one slot data type.

    Attributes:
        kind: Machine-readable family (``latex_expr``, ``symbol``, ``expr_or_infty``, ``enum``,
            ``int``, ``text``).
        values: Allowed values for ``kind="enum"``.
        of: Element type for composite kinds (``BOUNDS`` is a pair ``of: EXPR``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    values: tuple[str, ...] = ()
    of: str | None = None


class SlotSpec(BaseModel):
    """Declaration of one slot inside an intent.

    Attributes:
        type: Slot type name; must be a key of :attr:`Inventory.slot_types`.
        required: Whether the command cannot be built without it.
        default: Value used when nothing else fills the slot.
        from_context: :class:`~aicls.nlu.types.ParseContext` attribute that may fill it
            (``last_latex``, ``active_var``).
        fill: ``pattern`` (captured by an utterance pattern) or ``scan`` (found anywhere in the
            text by a lexicon scan rule).
        scan_key: Key into ``lexicon.yaml``'s ``scan`` section when ``fill="scan"``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: SlotTypeName
    required: bool = False
    default: Any = None
    from_context: Literal["last_latex", "active_var"] | None = None
    fill: SlotFill = "pattern"
    scan_key: str | None = None


class Example(BaseModel):
    """A worked example: the utterance and what it must resolve to.

    Attributes:
        text: The Turkish utterance.
        slots: Expected slot values; only the listed slots are checked.
        context: Optional ``ParseContext`` keyword arguments (``last_latex``, ``active_var``).
        tags: Free labels carried into ``datasets/nlu/commands_test.jsonl``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    slots: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, str] = Field(default_factory=dict)
    tags: tuple[str, ...] = ()


class IntentSpec(BaseModel):
    """One intent of the inventory.

    Attributes:
        id: Stable intent id (``differentiate``, ``book_fetch``, ``ui_clear``).
        kind: What it builds: a CAS command, a book lookup or a UI action.
        label_tr: One-line Turkish description, shown in clarification questions.
        cas_op: ``op`` discriminator of the :mod:`aicls.mathengine` model (``kind="cas"``).
        ui_action: :class:`~aicls.nlu.types.UiAction` action (``kind="ui"``).
        priority: Tie-breaker; higher wins when scores are equal.
        tags: ``core`` marks the İP-3 risk-plan subset of most-used commands.
        slots: Slot declarations by name.
        keywords: Token → weight; the strongest weight normalises the keyword score.
        patterns: Utterance patterns (see :mod:`aicls.nlu.patterns`).
        clarify_tr: Turkish question per slot, asked when the slot cannot be resolved.
        examples: At least :data:`MIN_EXAMPLES_PER_INTENT` worked examples.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    kind: IntentKind
    label_tr: str = Field(min_length=1)
    cas_op: str | None = None
    ui_action: str | None = None
    priority: int = 0
    tags: tuple[str, ...] = ()
    slots: dict[str, SlotSpec] = Field(default_factory=dict)
    keywords: dict[str, int] = Field(default_factory=dict)
    patterns: tuple[str, ...] = ()
    clarify_tr: dict[str, str] = Field(default_factory=dict)
    examples: tuple[Example, ...] = ()

    @model_validator(mode="after")
    def _target_matches_kind(self) -> IntentSpec:  # noqa: N804 - pydantic 'after' validator
        """A CAS intent needs ``cas_op``, a UI intent needs ``ui_action``."""
        if self.kind == "cas" and not self.cas_op:
            raise ValueError(f"intent {self.id}: kind='cas' requires cas_op")
        if self.kind == "ui" and not self.ui_action:
            raise ValueError(f"intent {self.id}: kind='ui' requires ui_action")
        return self

    @property
    def max_keyword_weight(self) -> int:
        """Largest keyword weight, used to normalise the keyword score (never 0)."""
        return max(self.keywords.values(), default=1) or 1

    @property
    def required_slots(self) -> tuple[str, ...]:
        """Names of the slots without which the command cannot be built."""
        return tuple(name for name, spec in self.slots.items() if spec.required)


class Weights(BaseModel):
    """Scoring weights (``defaults.weights`` in the YAML).

    Attributes:
        pattern: Weight of the pattern match (1.0 full, 0.5 partial).
        keyword: Weight of the normalised keyword score.
        coverage: Weight of the token coverage of the match.
        conflict: Penalty per foreign strong keyword found in the text.
        slots: Bonus added when every required slot of the candidate resolved.
        slots_missing: Penalty applied when a required slot did not resolve.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    pattern: float = 0.45
    keyword: float = 0.35
    coverage: float = 0.20
    conflict: float = 0.15
    slots: float = 0.20
    slots_missing: float = 0.40


class Defaults(BaseModel):
    """Inventory-wide defaults.

    Attributes:
        var: Variable assumed when the expression does not determine one.
        min_score: Score a candidate must reach before it can win.
        margin: Score gap the winner needs over the runner-up.
        max_candidates: Upper bound on the candidates offered in a clarification.
        weights: Scoring weights.
        ambiguous_question_tr: Turkish question asked when candidates tie.
        suggestions_tr: Example commands offered by ``NoMatch``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    var: str = "x"
    min_score: float = Field(default=0.55, ge=0.0, le=1.0)
    margin: float = Field(default=0.15, ge=0.0, le=1.0)
    max_candidates: int = Field(default=3, ge=1, le=10)
    weights: Weights = Field(default_factory=Weights)
    ambiguous_question_tr: str = ""
    suggestions_tr: tuple[str, ...] = ()


class Inventory(BaseModel):
    """The validated command inventory.

    Attributes:
        version: Inventory version string.
        defaults: Inventory-wide defaults and scoring weights.
        slot_types: Slot type declarations by name.
        intents: The intents, in file order.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    defaults: Defaults = Field(default_factory=Defaults)
    slot_types: dict[str, SlotType] = Field(default_factory=dict)
    intents: tuple[IntentSpec, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> Inventory:  # noqa: N804 - pydantic 'after' validator
        """Unique ids, known slot types, and a clarification text per required slot."""
        seen: set[str] = set()
        for intent in self.intents:
            if intent.id in seen:
                raise ValueError(f"duplicate intent id: {intent.id}")
            seen.add(intent.id)
            for name, spec in intent.slots.items():
                if spec.type not in self.slot_types:
                    raise ValueError(f"intent {intent.id}: unknown slot type {spec.type}")
                if spec.fill == "scan" and not spec.scan_key:
                    raise ValueError(f"intent {intent.id}: slot {name} needs scan_key")
                if spec.required and name not in intent.clarify_tr:
                    raise ValueError(
                        f"intent {intent.id}: required slot {name} has no clarify_tr text"
                    )
        return self

    @cached_property
    def by_id(self) -> Mapping[str, IntentSpec]:
        """Intents keyed by id."""
        return {intent.id: intent for intent in self.intents}

    @cached_property
    def intent_ids(self) -> tuple[str, ...]:
        """Every intent id, in file order (the only values a fallback may return)."""
        return tuple(intent.id for intent in self.intents)

    @cached_property
    def compiled(self) -> Mapping[str, tuple[CompiledPattern, ...]]:
        """Compiled utterance patterns per intent id."""
        from aicls.nlu.lexicon import load_lexicon

        suffixes = load_lexicon().suffixes
        return {
            intent.id: compile_patterns(
                intent.patterns,
                suffixes=suffixes,
                slot_types={name: spec.type for name, spec in intent.slots.items()},
                intent=intent.id,
            )
            for intent in self.intents
        }

    @cached_property
    def strong_keywords(self) -> Mapping[str, str]:
        """Keyword → the single intent that owns it, for keywords of maximal weight.

        A keyword is *strong* when its weight is the highest weight used anywhere in the
        inventory and exactly one intent declares it. Finding a foreign strong keyword in an
        utterance is evidence *against* a candidate, which is the ``conflict`` term of the score.
        """
        return {
            keyword: next(iter(ids))
            for keyword, ids in self.command_keywords.items()
            if len(ids) == 1
        }

    @cached_property
    def top_keyword_weight(self) -> int:
        """The highest keyword weight used anywhere in the inventory."""
        return max(
            (weight for intent in self.intents for weight in intent.keywords.values()),
            default=0,
        )

    @cached_property
    def command_keywords(self) -> Mapping[str, frozenset[str]]:
        """Keyword → every intent that claims it at :attr:`top_keyword_weight`.

        Unlike :attr:`strong_keywords` this keeps the words *two* intents share (``integral``
        belongs to both integral intents). The resolver needs that wider set to notice that an
        unread part of the utterance names a second command:
        ``"x karenin türevini al ve integralini al"`` is two commands, and answering only the
        first is a silent truncation.
        """
        top = self.top_keyword_weight
        owners: dict[str, set[str]] = {}
        for intent in self.intents:
            for keyword, weight in intent.keywords.items():
                if weight >= top:
                    owners.setdefault(keyword, set()).add(intent.id)
        return {keyword: frozenset(ids) for keyword, ids in owners.items()}

    def examples(self) -> tuple[tuple[str, Example], ...]:
        """Every ``(intent_id, example)`` pair in the inventory."""
        return tuple((intent.id, example) for intent in self.intents for example in intent.examples)

    def validate_examples(self) -> list[str]:
        """Re-resolve every example and report the ones that do not come back correctly.

        Returns:
            A list of human-readable problem descriptions; empty when the inventory is
            self-consistent. Each line names the example text, the expected intent and what was
            resolved instead.
        """
        from aicls.nlu.resolver import parse_command
        from aicls.nlu.types import Matched, context_from_mapping

        problems: list[str] = []
        for intent_id, example in self.examples():
            context = context_from_mapping(example.context)
            outcome = parse_command(example.text, context=context, inventory=self)
            if not isinstance(outcome, Matched):
                problems.append(
                    f"{intent_id}: {example.text!r} -> {type(outcome).__name__} (expected Matched)"
                )
                continue
            if outcome.intent != intent_id:
                problems.append(f"{intent_id}: {example.text!r} -> intent {outcome.intent!r}")
                continue
            for slot, expected in example.slots.items():
                actual = outcome.slots.get(slot)
                if actual != expected:
                    problems.append(
                        f"{intent_id}: {example.text!r} slot {slot}={actual!r},"
                        f" expected {expected!r}"
                    )
        return problems

    def check_example_counts(self, minimum: int = MIN_EXAMPLES_PER_INTENT) -> list[str]:
        """Report intents carrying fewer than ``minimum`` examples."""
        return [
            f"{intent.id}: {len(intent.examples)} examples (minimum {minimum})"
            for intent in self.intents
            if len(intent.examples) < minimum
        ]


def build_inventory(data: Mapping[str, Any], *, path: str = "") -> Inventory:
    """Validate a raw YAML mapping into an :class:`Inventory`.

    Args:
        data: The parsed ``commands.yaml`` mapping.
        path: Source path, reported in the error details.

    Returns:
        The validated inventory.

    Raises:
        aicls.nlu.errors.InventoryError: If validation fails.
    """
    from pydantic import ValidationError

    try:
        inventory = Inventory.model_validate(data)
    except ValidationError as exc:
        raise InventoryError(
            "schema_invalid", details={"path": path, "errors": exc.errors()[:5]}
        ) from exc
    # Compiling now turns a malformed pattern into a load-time InventoryError.
    _ = inventory.compiled
    return inventory


def slot_type_of(inventory: Inventory, intent: IntentSpec, slot: str) -> SlotType:
    """Return the :class:`SlotType` declaration behind ``intent.slots[slot]``."""
    return inventory.slot_types[intent.slots[slot].type]


def example_texts(inventory: Inventory, tags: Sequence[str] = ()) -> tuple[str, ...]:
    """Example utterances, optionally restricted to intents carrying every tag in ``tags``."""
    wanted = frozenset(tags)
    return tuple(
        example.text
        for intent in inventory.intents
        if wanted.issubset(intent.tags)
        for example in intent.examples
    )


__all__ = [
    "MIN_EXAMPLES_PER_INTENT",
    "Defaults",
    "Example",
    "IntentKind",
    "IntentSpec",
    "Inventory",
    "SlotFill",
    "SlotSpec",
    "SlotType",
    "SlotTypeName",
    "Weights",
    "build_inventory",
    "example_texts",
    "slot_type_of",
]
