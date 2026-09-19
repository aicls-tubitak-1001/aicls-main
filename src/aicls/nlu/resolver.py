"""``parse_command``: the decision procedure of the text-to-command layer.

The pipeline is normalise → rank → resolve slots → decide → map, and the *decision* step is
where the İP-4 risk plan is implemented:

* the best candidate wins only when its base score clears ``min_score`` **and** every required
  slot resolved **and** no other complete candidate sits within ``margin`` of it;
* a candidate whose required slot did not resolve becomes a
  :class:`~aicls.nlu.types.Clarification` naming that slot, in Turkish, from the inventory;
* two complete candidates within ``margin`` become a :class:`~aicls.nlu.types.Clarification`
  offering both -- "açılım" is genuinely both ``expand`` and ``series`` in Turkish;
* nothing above ``min_score`` becomes a :class:`~aicls.nlu.types.NoMatch` carrying example
  commands the platform *does* understand, never a wrong command. This is the proposal's
  "kapsam dışı istekler süzülür" requirement, and the reason the layer can be used in a
  classroom while the inventory is still growing.

An :class:`~aicls.nlu.fallback.IntentFallback` (off by default) may only re-enter the pipeline at
the *intent* step; its suggestion is re-run through the same rule-based slot extractor, so an LLM
can never fill a slot or write LaTeX.
"""

from __future__ import annotations

from dataclasses import dataclass

from aicls.core.errors import AiclsError, InternalError
from aicls.core.logging import get_logger
from aicls.nlu.errors import CommandBuildError, SpokenMathError
from aicls.nlu.fallback import IntentFallback, fallback_enabled, sanitize_suggestion
from aicls.nlu.inventory import load_inventory
from aicls.nlu.inventory.schema import IntentSpec, Inventory
from aicls.nlu.mapper import build_command
from aicls.nlu.matcher import IntentCandidate, IntentMatcher, PatternMatch, score_candidate
from aicls.nlu.normalize import NormalizedText, normalize_text
from aicls.nlu.slots import LATEX_SLOT_TYPES, SlotExtractor, SlotResolution
from aicls.nlu.spoken_math.parser import spoken_to_latex
from aicls.nlu.types import (
    Alternative,
    Clarification,
    Matched,
    NoMatch,
    ParseContext,
    ParseOutcome,
)

_logger = get_logger(__name__)

#: How many ranked candidates get their slots resolved (the rest cannot win anyway).
MAX_RESOLVED_CANDIDATES = 5


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """A candidate after slot resolution.

    Attributes:
        candidate: The ranked intent and its evidence.
        resolution: The slot resolution for that intent.
        score: Final score: adopted score plus the slot bonus or minus the missing-slot penalty.
        adopted_score: Score of the reading that was adopted, before the slot bonus.
    """

    candidate: IntentCandidate
    resolution: SlotResolution
    score: float
    adopted_score: float = 0.0

    @property
    def intent(self) -> IntentSpec:
        """The inventory intent."""
        return self.candidate.intent

    def alternative(self) -> Alternative:
        """This candidate as an :class:`~aicls.nlu.types.Alternative`."""
        return Alternative(
            intent=self.intent.id, score=round(self.score, 4), label_tr=self.intent.label_tr
        )


def _score_candidates(
    candidates: list[IntentCandidate],
    analysed: NormalizedText,
    extractor: SlotExtractor,
    inventory: Inventory,
    context: ParseContext | None,
) -> list[ScoredCandidate]:
    """Resolve slots for the top candidates and re-rank them."""
    weights = inventory.defaults.weights
    scored: list[ScoredCandidate] = []
    for candidate in candidates[:MAX_RESOLVED_CANDIDATES]:
        resolution = extractor.best(candidate.intent, analysed, candidate.matches, context=context)
        bonus = weights.slots if resolution.complete else -weights.slots_missing
        adopted = score_candidate(candidate, resolution.match, weights)
        scored.append(
            ScoredCandidate(
                candidate=candidate,
                resolution=resolution,
                score=max(0.0, min(1.0, adopted + bonus)),
                adopted_score=adopted,
            )
        )
    scored.sort(key=lambda item: (item.score, item.intent.priority), reverse=True)
    return scored


def _trace(analysed: NormalizedText, scored: ScoredCandidate) -> list[str]:
    """Machine-readable "how I understood you" lines."""
    lines = [f"normalized: {analysed.text}", f"intent: {scored.intent.id} score={scored.score:.3f}"]
    match = scored.resolution.match
    if match is not None:
        lines.append(f"pattern: {match.pattern.source} ({'full' if match.full else 'partial'})")
    if scored.candidate.conflicts:
        lines.append(f"conflicts: {', '.join(scored.candidate.conflicts)}")
    lines.extend(scored.resolution.trace)
    return lines


def _no_match(inventory: Inventory, analysed: NormalizedText, reason: str) -> NoMatch:
    return NoMatch(
        suggestions_tr=list(inventory.defaults.suggestions_tr),
        reason=reason,
        trace=[f"normalized: {analysed.text}", f"reason: {reason}"],
    )


def _clarify_missing(
    scored: ScoredCandidate, analysed: NormalizedText, inventory: Inventory
) -> Clarification:
    slot = scored.resolution.missing[0]
    question = scored.intent.clarify_tr.get(slot) or inventory.defaults.ambiguous_question_tr
    return Clarification(
        question_tr=question,
        missing_slot=slot,
        candidates=[scored.alternative()],
        intent=scored.intent.id,
        trace=_trace(analysed, scored),
    )


def _clarify_ambiguous(
    scored: list[ScoredCandidate], analysed: NormalizedText, inventory: Inventory
) -> Clarification:
    limit = inventory.defaults.max_candidates
    return Clarification(
        question_tr=inventory.defaults.ambiguous_question_tr,
        missing_slot=None,
        candidates=[item.alternative() for item in scored[:limit]],
        intent="",
        trace=_trace(analysed, scored[0]),
    )


def unaccounted_content(
    analysed: NormalizedText,
    match: PatternMatch | None,
    intent: IntentSpec,
    inventory: Inventory,
) -> tuple[str, ...]:
    """Fragments outside the adopted reading that carry meaning the command dropped.

    A partial pattern match means part of the utterance was *not* read. Most of the time the
    remainder is harmless -- a stray verb ("... açılımını **al**"), a word the lexicon does not
    know ("**hepsini** temizle") -- and refusing there would make the layer brittle. Two kinds of
    remainder are not harmless:

    * one the **spoken-math grammar accepts**: that is an expression the command silently
      dropped, and computing an answer from the rest answers a question nobody asked;
    * one naming **another intent's strong keyword**: "x karenin türevini al ve integralini al"
      is two commands, and executing the first while dropping the second is a silent truncation.

    Args:
        analysed: The normalised utterance.
        match: The adopted pattern match, or ``None``.
        intent: The winning intent (its own keywords are not foreign).
        inventory: Inventory the strong keywords come from.

    Returns:
        The unread fragments, in reading order; empty when the reading accounts for everything
        that matters.
    """
    if match is None or match.full:
        return ()
    first, last = analysed.token_range(*match.match.span())
    commanding = inventory.command_keywords
    out: list[str] = []
    for start, end in ((0, first), (last, len(analysed.tokens))):
        fragment = _longest_math_fragment(analysed, start, end)
        if not fragment:
            fragment = next(
                (
                    token
                    for token, kind in zip(
                        analysed.tokens[start:end], analysed.kinds[start:end], strict=True
                    )
                    if kind != "suffix"
                    and token in commanding
                    and intent.id not in commanding[token]
                    and token not in intent.keywords
                ),
                "",
            )
        if fragment:
            out.append(fragment)
    return tuple(out)


def _longest_math_fragment(analysed: NormalizedText, start: int, end: int) -> str:
    """Longest sub-run of ``tokens[start:end]`` the spoken-math grammar accepts.

    The unread remainder is rarely mathematics from end to end -- "al 2 y küp" is a leftover verb
    followed by an expression -- so the whole run parsing is the wrong question. What matters is
    whether *some* of it is an expression the command did not read.
    """
    for size in range(end - start, 0, -1):
        for begin in range(start, end - size + 1):
            fragment = analysed.slice_math(begin, begin + size).strip()
            if not fragment:
                continue
            try:
                spoken_to_latex(fragment, normalized=True)
            except SpokenMathError:
                continue
            return fragment
    return ""


def _refuse_unaccounted(
    scored: ScoredCandidate,
    analysed: NormalizedText,
    inventory: Inventory,
    leftovers: tuple[str, ...],
) -> ParseOutcome:
    """Turn an unread mathematical fragment into a question, or into a refusal."""
    trace = _trace(analysed, scored)
    trace.append(f"unaccounted: {' | '.join(leftovers)}")
    intent = scored.intent
    slot = next(
        (
            name
            for name in intent.required_slots
            if intent.slots[name].type in LATEX_SLOT_TYPES and name in intent.clarify_tr
        ),
        None,
    )
    if slot is None:
        return NoMatch(
            suggestions_tr=list(inventory.defaults.suggestions_tr),
            reason="unaccounted_tokens",
            trace=trace,
        )
    return Clarification(
        question_tr=intent.clarify_tr[slot],
        missing_slot=slot,
        candidates=[scored.alternative()],
        intent=intent.id,
        trace=trace,
    )


def _matched(
    scored: ScoredCandidate,
    others: list[ScoredCandidate],
    analysed: NormalizedText,
    inventory: Inventory,
    source: str,
) -> ParseOutcome:
    try:
        command = build_command(scored.intent, scored.resolution.values)
    except CommandBuildError as exc:
        _logger.warning("nlu.command_build_failed", intent=scored.intent.id, detail=exc.details)
        return _no_match(inventory, analysed, "command_invalid")
    trace = _trace(analysed, scored)
    trace.append(f"command: {command.op}")
    limit = inventory.defaults.max_candidates
    return Matched(
        command=command,
        intent=scored.intent.id,
        slots=dict(scored.resolution.values),
        score=round(scored.score, 4),
        alternatives=[item.alternative() for item in others[: limit - 1]],
        trace=trace,
        source="llm" if source == "llm" else "rules",
    )


def _try_fallback(
    text: str,
    analysed: NormalizedText,
    inventory: Inventory,
    extractor: SlotExtractor,
    context: ParseContext | None,
    fallback: IntentFallback | None,
    candidates: list[IntentCandidate],
) -> ParseOutcome | None:
    """Ask the fallback for an intent id and re-run the rule-based slot extractor on it.

    The model contributes exactly one thing: which inventory intent to try. Slots are still
    filled by :class:`~aicls.nlu.slots.SlotExtractor`, and an incomplete resolution still becomes
    a clarification question rather than a guessed command.
    """
    if fallback is None or not fallback_enabled():
        return None
    suggestion = sanitize_suggestion(
        fallback.suggest(text, inventory.intent_ids), inventory.intent_ids
    )
    if suggestion is None:
        return None
    intent = inventory.by_id[suggestion]
    matches = next(
        (item.matches for item in candidates if item.intent.id == suggestion),
        (),
    )
    resolution = extractor.best(intent, analysed, matches, context=context)
    scored = ScoredCandidate(
        candidate=IntentCandidate(intent=intent, base_score=inventory.defaults.min_score),
        resolution=resolution,
        score=inventory.defaults.min_score,
    )
    if not resolution.complete:
        return _clarify_missing(scored, analysed, inventory)
    return _matched(scored, [], analysed, inventory, source="llm")


def parse_command(
    text: str,
    *,
    context: ParseContext | None = None,
    inventory: Inventory | None = None,
    fallback: IntentFallback | None = None,
) -> ParseOutcome:
    """Turn Turkish text into a typed command, a clarification question or a refusal.

    Args:
        text: The utterance, typed or straight out of the ASR decoder.
        context: Session context; ``last_latex`` is what makes "türevini al" work.
        inventory: Inventory to resolve against; the packaged one when omitted.
        fallback: Intent-only fallback; ignored unless ``AICLS_LLM_FALLBACK`` is on.

    Returns:
        :class:`~aicls.nlu.types.Matched`, :class:`~aicls.nlu.types.Clarification` or
        :class:`~aicls.nlu.types.NoMatch`.

    Raises:
        aicls.core.errors.AiclsError: Only for a malformed inventory or an internal failure;
            ordinary "I did not understand" is a ``NoMatch``, not an exception.
    """
    try:
        return _parse_command(text, context=context, inventory=inventory, fallback=fallback)
    except AiclsError:
        raise
    except Exception as exc:  # the package contract is AiclsError only
        _logger.error("nlu.parse_failed", error=str(exc)[:200], kind=type(exc).__name__)
        raise InternalError(details={"stage": "nlu.parse_command"}) from exc


def _parse_command(
    text: str,
    *,
    context: ParseContext | None,
    inventory: Inventory | None,
    fallback: IntentFallback | None,
) -> ParseOutcome:
    """Body of :func:`parse_command` (see there); errors are wrapped by the caller."""
    active = inventory or load_inventory()
    analysed = normalize_text(text)
    if not analysed.tokens:
        return _no_match(active, analysed, "empty_input")

    extractor = SlotExtractor(active)
    candidates = IntentMatcher(active).rank(analysed)
    scored = _score_candidates(candidates, analysed, extractor, active, context)
    plausible = [item for item in scored if item.candidate.base_score >= active.defaults.min_score]
    if not plausible:
        return _try_fallback(
            text, analysed, active, extractor, context, fallback, candidates
        ) or _no_match(active, analysed, "no_intent")

    best = plausible[0]
    if not best.resolution.complete:
        return _clarify_missing(best, analysed, active)

    # A reading that leaves mathematics unread is not a reading of *this* utterance.
    leftovers = unaccounted_content(analysed, best.resolution.match, best.intent, active)
    if leftovers:
        return _refuse_unaccounted(best, analysed, active, leftovers)

    margin = active.defaults.margin
    # A rival must be a *complete* reading, close in score, and not outranked by the winner's
    # inventory priority: "sıfırdan bire kadar ... integrali" is close to the indefinite reading
    # but ``integrate_definite`` outranks it by design, while ``expand`` and ``series`` share a
    # priority precisely because "açılım" really is ambiguous in Turkish.
    rivals = [
        item
        for item in plausible[1:]
        if item.resolution.complete
        and best.score - item.score < margin
        and item.intent.priority >= best.intent.priority
    ]
    if rivals:
        return _clarify_ambiguous([best, *rivals], analysed, active)
    return _matched(best, plausible[1:], analysed, active, source="rules")


def explain(text: str, *, context: ParseContext | None = None) -> list[str]:
    """Return only the trace of parsing ``text`` (used by ``aicls nlu try``)."""
    outcome = parse_command(text, context=context)
    return list(outcome.trace)


__all__: list[str] = [
    "MAX_RESOLVED_CANDIDATES",
    "ScoredCandidate",
    "explain",
    "parse_command",
    "unaccounted_content",
]
