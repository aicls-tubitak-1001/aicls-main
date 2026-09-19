"""Intent ranking over the inventory (docs/mimari/metin-komut-katmani.md §5).

The score of a candidate intent is

``score = w_p·pattern + w_k·keyword + w_c·coverage − w_x·conflict``

where

* **pattern** is 1.0 when one of the intent's patterns matches the whole normalised text and 0.5
  when it only matches part of it (a partial match is real evidence -- "türev" inside a longer
  sentence -- but never as strong as a full reading);
* **keyword** is the sum of the matched keyword weights divided by the intent's strongest weight,
  capped at 1.0, so a single decisive word ("türev") already scores 1.0 while a pile of weak ones
  cannot outrun it;
* **coverage** is the fraction of the text the match spans, which is what separates "türevini al"
  (whole sentence) from "türev" buried in "türev tanımını getir";
* **conflict** counts *foreign strong keywords*: words that only one other intent claims at the
  highest weight in the whole inventory. Finding "tanım" is evidence against ``differentiate``
  exactly as finding "türev" is evidence against ``book_fetch``.

Slot resolution then adds a bonus or a penalty (:mod:`aicls.nlu.slots`), because a candidate
whose required slots do not resolve is not a candidate -- that is what keeps
``integrate_indefinite`` from stealing "sıfırdan bire kadar ... integrali" from
``integrate_definite``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import re

from aicls.nlu.inventory.schema import IntentSpec, Inventory, Weights
from aicls.nlu.normalize import NormalizedText
from aicls.nlu.patterns import CompiledPattern

#: Score of a pattern that matches the whole text, and of one that matches only a part.
FULL_MATCH = 1.0
PARTIAL_MATCH = 0.5


@dataclass(frozen=True, slots=True)
class PatternMatch:
    """One regex match of one inventory pattern.

    Attributes:
        pattern: The pattern that matched.
        match: The regex match object (named groups hold the raw slot text).
        full: Whether the whole normalised text was matched.
        coverage: Fraction of the normalised text the match spans.
    """

    pattern: CompiledPattern
    match: re.Match[str]
    full: bool
    coverage: float

    @property
    def quality(self) -> float:
        """:data:`FULL_MATCH` or :data:`PARTIAL_MATCH`."""
        return FULL_MATCH if self.full else PARTIAL_MATCH


@dataclass(frozen=True, slots=True)
class IntentCandidate:
    """A ranked intent with the evidence behind its score.

    Attributes:
        intent: The inventory intent.
        base_score: Score before slot resolution, clamped to ``[0, 1]``.
        matches: Pattern matches, best first.
        keyword_hits: Keywords found in the text, with their weights.
        conflicts: Foreign strong keywords found in the text.
    """

    intent: IntentSpec
    base_score: float
    matches: tuple[PatternMatch, ...] = ()
    keyword_hits: dict[str, int] = field(default_factory=dict)
    conflicts: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        """The intent id."""
        return self.intent.id


def _match_patterns(patterns: Sequence[CompiledPattern], text: str) -> tuple[PatternMatch, ...]:
    """Match every pattern of one intent against ``text``, best first."""
    found: list[PatternMatch] = []
    length = max(len(text), 1)
    for pattern in patterns:
        match = pattern.fullmatch(text)
        if match is not None:
            found.append(PatternMatch(pattern=pattern, match=match, full=True, coverage=1.0))
            continue
        partial = pattern.search(text)
        if partial is not None and partial.end() > partial.start():
            span = (partial.end() - partial.start()) / length
            found.append(PatternMatch(pattern=pattern, match=partial, full=False, coverage=span))
    found.sort(key=lambda item: (item.full, item.coverage), reverse=True)
    return tuple(found)


def score_candidate(
    candidate: IntentCandidate, match: PatternMatch | None, weights: Weights
) -> float:
    """Score ``candidate`` using ``match`` as the adopted reading.

    :meth:`IntentMatcher.rank` scores every intent by its *best-looking* pattern, but the slot
    extractor may end up adopting a different one -- typically a weaker, partial match whose
    slots actually resolve. The score must describe the reading that was adopted, otherwise a
    full pattern match whose slots all failed would keep propping the candidate up.
    """
    intent = candidate.intent
    pattern_score = match.quality if match is not None else 0.0
    coverage = match.coverage if match is not None else 0.0
    keyword_score = min(1.0, sum(candidate.keyword_hits.values()) / intent.max_keyword_weight)
    score = (
        weights.pattern * pattern_score
        + weights.keyword * keyword_score
        + weights.coverage * coverage
        - weights.conflict * min(len(candidate.conflicts), 2)
    )
    return max(0.0, min(1.0, score))


class IntentMatcher:
    """Ranks the inventory's intents against a normalised utterance.

    Args:
        inventory: The inventory to rank over.
    """

    def __init__(self, inventory: Inventory) -> None:
        self._inventory = inventory

    @property
    def inventory(self) -> Inventory:
        """The inventory this matcher ranks over."""
        return self._inventory

    def rank(self, analysed: NormalizedText) -> list[IntentCandidate]:
        """Score every intent against ``analysed`` and return the candidates, best first.

        Args:
            analysed: The normalised utterance.

        Returns:
            Candidates with a non-zero score, sorted by score then by intent priority.
        """
        text = analysed.text
        if not text:
            return []
        words = frozenset(analysed.words)
        weights = self._inventory.defaults.weights
        strong = self._inventory.strong_keywords
        out: list[IntentCandidate] = []
        for intent in self._inventory.intents:
            matches = _match_patterns(self._inventory.compiled[intent.id], text)
            hits = {word: weight for word, weight in intent.keywords.items() if word in words}
            conflicts = tuple(
                sorted(
                    word
                    for word in words
                    if word in strong and strong[word] != intent.id and word not in intent.keywords
                )
            )
            candidate = IntentCandidate(
                intent=intent,
                base_score=0.0,
                matches=matches,
                keyword_hits=hits,
                conflicts=conflicts,
            )
            score = score_candidate(candidate, matches[0] if matches else None, weights)
            if score <= 0.0:
                continue
            out.append(
                IntentCandidate(
                    intent=intent,
                    base_score=score,
                    matches=matches,
                    keyword_hits=hits,
                    conflicts=conflicts,
                )
            )
        out.sort(key=lambda item: (item.base_score, item.intent.priority), reverse=True)
        return out


__all__ = [
    "FULL_MATCH",
    "PARTIAL_MATCH",
    "IntentCandidate",
    "IntentMatcher",
    "PatternMatch",
    "score_candidate",
]
