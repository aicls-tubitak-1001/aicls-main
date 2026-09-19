"""Book search: alias index → full-text/LIKE → ``difflib`` (docs/mimari/veri-modeli.md §5.4).

Three stages, cheapest first, exactly as ADR-0019 prescribes:

1. **alias** — the normalised query is looked up in the indexed ``book_aliases`` table. This is
   the hot path (``"türev tanımı"`` was expanded at load time) and returns ``score = 1.0``.
2. **fulltext** — MySQL ``MATCH ... AGAINST`` / SQLite ``LIKE`` over ``search_text``, re-ranked
   by :class:`difflib.SequenceMatcher`. The repository owns the dialect branch.
3. **fuzzy** — a last ``difflib`` pass over every title and stored alias, keeping candidates at
   or above :data:`FUZZY_THRESHOLD`, so a typo (``"türve tanımı"``) still finds its unit.

Nothing here builds a query; every read goes through :class:`
aicls.db.repositories.book.BookRepository`.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from aicls.book.schema import AssetRef, ContentUnit, Hit, UnitType
from aicls.core.turkish import tr_casefold
from aicls.db.models import BookUnit
from aicls.db.repositories.book import BookRepository

#: Minimum ``difflib`` ratio for a fuzzy candidate to be returned (ADR-0019: "difflib ≥ 0.8").
FUZZY_THRESHOLD = 0.8
#: Default number of hits.
DEFAULT_LIMIT = 5
#: Number of stored aliases of one unit compared during the fuzzy pass.
FUZZY_ALIAS_LIMIT = 40


def normalize_query(text: str) -> str:
    """Turkish-casefold ``text`` and collapse punctuation/whitespace into single spaces."""
    folded = tr_casefold(text)
    cleaned = "".join(char if char.isalnum() or char.isspace() else " " for char in folded)
    return " ".join(cleaned.split())


def unit_from_row(row: BookUnit) -> ContentUnit:
    """Convert a ``book_units`` row into a :class:`~aicls.book.schema.ContentUnit`.

    Aliases and figure references are read back out of the stored JSON-LD document, which is the
    unit's complete representation (docs/mimari/veri-modeli.md §5.2).
    """
    document = dict(row.jsonld or {})
    latex = row.latex_json or {}
    blocks = latex.get("blocks", []) if isinstance(latex, dict) else list(latex)
    aliases = [str(alias) for alias in document.get("altLabel", [])]
    figures = [str(ref) for ref in document.get("figure", [])]
    return ContentUnit(
        id=row.id,
        type=UnitType(row.unit_type.value),
        chapter=row.chapter,
        number=row.number,
        title_tr=row.title_tr,
        aliases=aliases,
        body_md=row.body_md,
        latex_blocks=[str(block) for block in blocks],
        assets=[AssetRef(id=ref.removeprefix("aicls:")) for ref in figures],
        source_page=row.source_page,
        jsonld=document,
        order_in_chapter=row.order_in_chapter,
    )


def _fuzzy_score(needle: str, unit: ContentUnit) -> tuple[float, str | None]:
    """Best ratio of ``needle`` against the unit's title and stored aliases."""
    best = SequenceMatcher(None, needle, tr_casefold(unit.title_tr)).ratio()
    matched: str | None = None
    for alias in unit.aliases[:FUZZY_ALIAS_LIMIT]:
        ratio = SequenceMatcher(None, needle, alias).ratio()
        if ratio > best:
            best, matched = ratio, alias
    return best, matched


def search(
    repo: BookRepository,
    q: str,
    *,
    unit_type: UnitType | None = None,
    limit: int = DEFAULT_LIMIT,
) -> list[Hit]:
    """Search the book graph for ``q`` and return the best hits.

    Args:
        repo: Book repository bound to an open session.
        q: Free-text query; normalised with :func:`normalize_query`.
        unit_type: Optional type filter (``"türev *tanımı*"`` → ``UnitType.DEFINITION``).
        limit: Maximum number of hits (at least 1).

    Returns:
        Hits ordered by descending score. An exact alias match short-circuits the later stages
        and scores 1.0.
    """
    needle = normalize_query(q)
    size = max(1, limit)
    if not needle:
        return []

    exact = repo.search_alias(needle, unit_type)
    if exact:
        return [
            Hit(unit=unit_from_row(row), score=1.0, matched_alias=needle, stage="alias")
            for row in exact[:size]
        ]

    hits = [
        Hit(unit=unit_from_row(row), score=score, matched_alias=None, stage="fulltext")
        for row, score in repo.search_text(needle, unit_type, size)
    ]
    if hits:
        return hits

    scored: list[Hit] = []
    for row in repo.units(unit_type=unit_type):
        unit = unit_from_row(row)
        score, matched = _fuzzy_score(needle, unit)
        if score >= FUZZY_THRESHOLD:
            scored.append(Hit(unit=unit, score=score, matched_alias=matched, stage="fuzzy"))
    scored.sort(key=lambda hit: (-hit.score, hit.unit.id))
    return scored[:size]


def related(
    repo: BookRepository, unit_id: str, *, predicate: str | None = None
) -> list[ContentUnit]:
    """Return the units one hop away from ``unit_id`` (``GET /book/units/{id}/related``)."""
    return [unit_from_row(row) for row in repo.related(unit_id, predicate)]


def get_unit(repo: BookRepository, unit_id: str) -> ContentUnit | None:
    """Return one unit by identifier, or ``None`` (``GET /book/units/{id}``)."""
    row = repo.get(unit_id)
    return None if row is None else unit_from_row(row)


__all__ = [
    "DEFAULT_LIMIT",
    "FUZZY_ALIAS_LIMIT",
    "FUZZY_THRESHOLD",
    "get_unit",
    "normalize_query",
    "related",
    "search",
    "unit_from_row",
]
