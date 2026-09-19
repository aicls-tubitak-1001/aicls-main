"""Command resolver: ``"türev tanımını getir"`` → ``tanim-2.4`` (İP-8, ADR-0019).

The proposal's İP-8 task is a command that fetches book content ("x tanımını ekrana getir"), and
its criterion is *command-fetch accuracy > 95 %*, measured by
``datasets/book/fetch_test.jsonl``. The resolver is the layer that turns the phrasing into a
unit:

1. normalise the phrase (:func:`aicls.book.query.normalize_query`);
2. drop the command verbs and fillers ("getir", "göster", "nedir", "lütfen", ...);
3. read the unit type off the phrase when it names one — the stem of "tanımını" is "tanım", so
   ``UnitType.DEFINITION`` — and remove that word from the concept;
4. try the surviving phrasings against the indexed alias table, then fall back to
   :func:`aicls.book.query.search`.

Results are memoised in a bounded LRU (``cache_size``, default 2048 as ADR-0019 requires); the
cached values are detached pydantic models, so nothing ORM-bound escapes a closed session.
"""

from __future__ import annotations

from collections import OrderedDict

from aicls.book.aliases import TYPE_WORDS, harden
from aicls.book.query import DEFAULT_LIMIT, normalize_query, search, unit_from_row
from aicls.book.schema import ResolvedUnit, UnitType
from aicls.core.turkish import strip_diacritics, strip_suffix
from aicls.db.repositories.book import BookRepository

#: Default size of the alias → unit LRU (ADR-0019: "LRU 2048").
DEFAULT_CACHE_SIZE = 2048

#: Command verbs and fillers removed before a phrase is matched against the alias index.
COMMAND_STOPWORDS: frozenset[str] = frozenset(
    {
        "aç",
        "açıkla",
        "anlat",
        "bak",
        "bana",
        "bul",
        "de",
        "demek",
        "ekran",
        "ekranda",
        "ekrana",
        "getir",
        "göster",
        "hakkında",
        "hatırlat",
        "hemen",
        "ile",
        "ilgili",
        "istiyorum",
        "kitapta",
        "kitaptaki",
        "konusunda",
        "konusundaki",
        "listele",
        "lütfen",
        "misin",
        "misiniz",
        "mısın",
        "musun",
        "musunuz",
        "nasıl",
        "ne",
        "nedir",
        "oku",
        "olan",
        "söyle",
        "ver",
        "yaz",
    }
)

#: The same stop words without Turkish diacritics, so "goster" is dropped like "göster".
COMMAND_STOPWORDS_ASCII: frozenset[str] = frozenset(
    strip_diacritics(word) for word in COMMAND_STOPWORDS
)

#: Stem of every type word → the unit type it names ("tanım" → Definition).
TYPE_BY_WORD: dict[str, UnitType] = {
    word: unit_type for unit_type, words in TYPE_WORDS.items() for word in words
}


def word_stem(token: str) -> str:
    """Strip one Turkish suffix and undo consonant softening (``"örneğini"`` → ``"örnek"``)."""
    return harden(strip_suffix(token))


def detect_unit_type(tokens: list[str]) -> tuple[UnitType | None, list[str]]:
    """Split a normalised token list into the unit type it names and the remaining concept.

    Args:
        tokens: Normalised query tokens.

    Returns:
        ``(unit_type, concept_tokens)``; ``unit_type`` is ``None`` when no token names one and
        ``concept_tokens`` then equals ``tokens``.
    """
    for index, token in enumerate(tokens):
        unit_type = TYPE_BY_WORD.get(word_stem(token))
        if unit_type is not None:
            return unit_type, tokens[:index] + tokens[index + 1 :]
    return None, list(tokens)


def is_stopword(token: str) -> bool:
    """True for a command verb or filler that carries no content."""
    return token in COMMAND_STOPWORDS or strip_diacritics(token) in COMMAND_STOPWORDS_ASCII


def content_tokens(query: str) -> list[str]:
    """Normalise ``query`` and drop its command verbs and fillers."""
    return [token for token in normalize_query(query).split() if not is_stopword(token)]


def _phrase_variants(tokens: list[str]) -> list[str]:
    """The phrase itself plus the same phrase with its last word reduced to its stem."""
    if not tokens:
        return []
    phrase = " ".join(tokens)
    stemmed = " ".join([*tokens[:-1], word_stem(tokens[-1])])
    return [phrase] if stemmed == phrase else [phrase, stemmed]


def _windows(tokens: list[str]) -> list[str]:
    """Contiguous sub-phrases of ``tokens``, longest and leftmost first.

    A student adds words the book never used ("türev tanımının **kendisini** yaz"). Trying the
    sub-phrases recovers the alias without loosening the exact-match rule.
    """
    out: list[str] = []
    for size in range(len(tokens) - 1, 0, -1):
        for start in range(len(tokens) - size + 1):
            out.extend(_phrase_variants(tokens[start : start + size]))
    return out


def resolution_plan(
    query: str, unit_type: UnitType | None
) -> tuple[list[str], list[str], UnitType | None]:
    """Split a phrase into the phrasings to try, in two tiers.

    Args:
        query: Raw user phrase.
        unit_type: Caller-supplied type filter; when ``None`` the type is read off the phrase.

    Returns:
        ``(primary, ladder, unit_type)``. ``primary`` holds the whole phrase and the phrase
        without its type word — the only ones worth handing to the fuzzy search. ``ladder`` adds
        the progressively shorter sub-phrases and is used for the exact alias lookups only.
    """
    tokens = content_tokens(query)
    detected, concept = detect_unit_type(tokens)
    effective = unit_type if unit_type is not None else detected
    primary = _dedupe([*_phrase_variants(tokens), *_phrase_variants(concept)])
    ladder = _dedupe([*primary, *_windows(tokens), *_windows(concept)])
    return primary, ladder, effective


def _dedupe(values: list[str]) -> list[str]:
    """Drop empties and repeats while keeping order."""
    return [value for value in dict.fromkeys(values) if value]


def query_variants(query: str, unit_type: UnitType | None) -> tuple[list[str], UnitType | None]:
    """Return every phrasing tried against the alias index, most specific first.

    Thin wrapper over :func:`resolution_plan` kept for callers that only need the flat list.
    """
    _, ladder, effective = resolution_plan(query, unit_type)
    return ladder, effective


class BookResolver:
    """Resolve Turkish fetch commands to book units, with a bounded result cache.

    Args:
        repo: Book repository bound to an open session.
        cache_size: Maximum number of distinct ``(query, type, k)`` keys memoised; ``0``
            disables caching.
    """

    def __init__(self, repo: BookRepository, *, cache_size: int = DEFAULT_CACHE_SIZE) -> None:
        self.repo = repo
        self.cache_size = max(0, int(cache_size))
        self._cache: OrderedDict[tuple[str, str, int], list[ResolvedUnit]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def resolve(
        self,
        query: str,
        *,
        unit_type: UnitType | None = None,
        k: int = DEFAULT_LIMIT,
    ) -> list[ResolvedUnit]:
        """Return the units that answer ``query``, best first.

        Args:
            query: The phrase as typed or transcribed ("türevin tanımını getir").
            unit_type: Optional explicit type filter; otherwise inferred from the phrase.
            k: Maximum number of results.

        Returns:
            Up to ``k`` :class:`~aicls.book.schema.ResolvedUnit` records; empty when nothing
            matches. An exact alias hit scores 1.0.
        """
        size = max(1, k)
        key = (normalize_query(query), unit_type.value if unit_type else "", size)
        cached = self._cache.get(key)
        if cached is not None:
            self._hits += 1
            self._cache.move_to_end(key)
            return list(cached)
        self._misses += 1
        results = self._resolve_uncached(query, unit_type, size)
        if self.cache_size:
            self._cache[key] = list(results)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return results

    def _resolve_uncached(
        self, query: str, unit_type: UnitType | None, size: int
    ) -> list[ResolvedUnit]:
        """Run the alias → full-text → fuzzy ladder for one phrase."""
        primary, variants, effective = resolution_plan(query, unit_type)
        if not variants:
            return []
        for variant in variants:
            rows = self.repo.search_alias(variant, effective)
            if not rows and effective is not None:
                rows = self.repo.search_alias(variant, None)
            if rows:
                return [
                    ResolvedUnit(
                        unit=unit_from_row(row),
                        score=1.0,
                        matched_alias=variant,
                        stage="alias",
                        query_normalized=primary[0],
                        unit_type=effective,
                    )
                    for row in rows[:size]
                ]

        for filter_type in (effective, None) if effective is not None else (None,):
            for variant in primary:
                hits = search(self.repo, variant, unit_type=filter_type, limit=size)
                if hits:
                    return [
                        ResolvedUnit(
                            unit=hit.unit,
                            score=hit.score,
                            matched_alias=hit.matched_alias,
                            stage=hit.stage,
                            query_normalized=primary[0],
                            unit_type=effective,
                        )
                        for hit in hits
                    ]
        return []

    def resolve_one(self, query: str, *, unit_type: UnitType | None = None) -> ResolvedUnit | None:
        """Return the single best match for ``query``, or ``None``."""
        results = self.resolve(query, unit_type=unit_type, k=1)
        return results[0] if results else None

    def cache_clear(self) -> None:
        """Drop every memoised result and reset the counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def cache_info(self) -> dict[str, int]:
        """Return ``{"hits", "misses", "size", "maxsize"}`` for diagnostics and tests."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "maxsize": self.cache_size,
        }


__all__ = [
    "COMMAND_STOPWORDS",
    "COMMAND_STOPWORDS_ASCII",
    "DEFAULT_CACHE_SIZE",
    "TYPE_BY_WORD",
    "BookResolver",
    "content_tokens",
    "detect_unit_type",
    "is_stopword",
    "query_variants",
    "resolution_plan",
    "word_stem",
]
