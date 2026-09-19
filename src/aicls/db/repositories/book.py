"""Textbook knowledge-graph repository (İP-8, ADR-0019).

This is the only query surface :mod:`aicls.book` uses. Unit, triple and alias inputs may be
pydantic models, dataclasses or plain mappings — :func:`aicls.db.repositories.base.field_of`
reads whichever shape arrives, so the loader and the repository can evolve independently.

Search follows the ADR: exact alias match first (indexed), then a full-text pass that uses
MySQL ``MATCH ... AGAINST`` and falls back to ``LIKE`` on SQLite, then a ``difflib`` similarity
score used to order the candidates.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher
from typing import Any

from sqlalchemy import delete, func, select

from aicls.core.turkish import tr_casefold
from aicls.db.enums import TagStatus, UnitType, coerce_enum
from aicls.db.models import BookAlias, BookTriple, BookUnit
from aicls.db.repositories.base import Repository, clamp_limit, field_of
from aicls.db.results import LoadReport

#: Default number of hits returned by :meth:`BookRepository.search_text`.
DEFAULT_SEARCH_LIMIT = 5
#: Candidate rows fetched before similarity re-ranking.
SEARCH_CANDIDATE_FACTOR = 10


def _unit_type_of(value: Any) -> UnitType | None:
    """Coerce an optional unit-type filter, passing ``None`` through."""
    return None if value is None else coerce_enum(UnitType, value)


class BookRepository(Repository):
    """Queries over ``book_units``, ``book_aliases`` and ``book_triples``."""

    # -- ingest -------------------------------------------------------------------------------
    def upsert_units(
        self,
        units: Iterable[Any],
        triples: Iterable[Any] = (),
        aliases: Iterable[Any] = (),
        *,
        tagged_ratio: float = 1.0,
    ) -> LoadReport:
        """Insert or update units, then replace their triples and aliases.

        Reloading a chapter is idempotent: a unit that already exists is updated in place and its
        aliases and outgoing triples are rewritten, so ``aicls book load`` may be run repeatedly.

        Args:
            units: Content units. Recognised fields: ``id``, ``unit_type``/``type``, ``chapter``,
                ``number``, ``order_in_chapter``, ``title_tr``/``title``, ``title_normalized``,
                ``body_md``, ``latex_blocks``/``latex_json``, ``search_text``, ``jsonld``,
                ``source_page``, ``tag_status``, ``aliases``.
            triples: RDF edges. Recognised fields: ``subject``/``subject_id``, ``predicate``,
                ``object``/``object_id``, ``provenance``.
            aliases: Extra aliases as ``(unit_id, alias)`` pairs or mappings with ``unit_id``,
                ``alias``/``alias_normalized`` and optional ``weight``. Aliases carried on a unit
                are loaded as well.
            tagged_ratio: Share of source headings that became units; reported back so the İP-8
                "100 % tagged" criterion is measured at load time.

        Returns:
            A :class:`~aicls.db.results.LoadReport` counting the rows written.
        """
        unit_rows = list(units)
        unit_ids: list[str] = []
        alias_pairs: list[tuple[str, str, float]] = []
        for item in unit_rows:
            unit_id = str(field_of(item, "id", "unit_id", default=""))
            if not unit_id:
                continue
            unit_ids.append(unit_id)
            self._upsert_unit(unit_id, item)
            for alias in field_of(item, "aliases", default=()) or ():
                alias_pairs.append((unit_id, str(alias), 1.0))
        for entry in aliases:
            alias_pairs.append(self._alias_tuple(entry))
        self.session.flush()

        if unit_ids:
            self.session.execute(delete(BookAlias).where(BookAlias.unit_id.in_(unit_ids)))
            self.session.execute(delete(BookTriple).where(BookTriple.subject_id.in_(unit_ids)))

        written_aliases = self._write_aliases(alias_pairs)
        written_triples = self._write_triples(triples)
        self.session.flush()
        return LoadReport(
            units=len(unit_ids),
            triples=written_triples,
            aliases=written_aliases,
            tagged_ratio=tagged_ratio,
        )

    def _upsert_unit(self, unit_id: str, item: Any) -> BookUnit:
        """Create or update one ``book_units`` row from an arbitrary unit shape."""
        title = str(field_of(item, "title_tr", "title", "name", default=""))
        normalized = str(field_of(item, "title_normalized", default="") or tr_casefold(title))
        body = str(field_of(item, "body_md", "bodyMarkdown", default=""))
        latex_blocks = field_of(item, "latex_json", "latex_blocks", "latex", default=[])
        unit = self.session.get(BookUnit, unit_id)
        if unit is None:
            unit = BookUnit(id=unit_id)
            self.session.add(unit)
        else:
            unit.version += 1
        unit.unit_type = coerce_enum(
            UnitType, field_of(item, "unit_type", "type", default="Remark")
        )
        unit.chapter = str(field_of(item, "chapter", default=""))
        unit.number = str(field_of(item, "number", default=""))
        unit.order_in_chapter = int(field_of(item, "order_in_chapter", default=0))
        unit.title_tr = title
        unit.title_normalized = normalized
        unit.body_md = body
        unit.latex_json = (
            dict(latex_blocks) if isinstance(latex_blocks, dict) else {"blocks": list(latex_blocks)}
        )
        unit.search_text = str(
            field_of(item, "search_text", default="") or f"{normalized} {tr_casefold(body)}".strip()
        )
        unit.jsonld = dict(field_of(item, "jsonld", default={}) or {})
        unit.source_page = field_of(item, "source_page")
        tag_status = field_of(item, "tag_status")
        if tag_status is not None:
            unit.tag_status = coerce_enum(TagStatus, tag_status)
        return unit

    @staticmethod
    def _alias_tuple(entry: Any) -> tuple[str, str, float]:
        """Normalise an alias entry into ``(unit_id, alias_normalized, weight)``."""
        if isinstance(entry, tuple | list) and len(entry) >= 2:
            weight = float(entry[2]) if len(entry) > 2 else 1.0
            return str(entry[0]), str(entry[1]), weight
        return (
            str(field_of(entry, "unit_id", "unit", default="")),
            str(field_of(entry, "alias_normalized", "alias", default="")),
            float(field_of(entry, "weight", default=1.0)),
        )

    def _write_aliases(self, pairs: Sequence[tuple[str, str, float]]) -> int:
        """Insert de-duplicated normalised aliases; return how many rows were written."""
        seen: set[tuple[str, str]] = set()
        written = 0
        for unit_id, alias, weight in pairs:
            normalized = tr_casefold(alias).strip()
            if not unit_id or not normalized or (unit_id, normalized) in seen:
                continue
            seen.add((unit_id, normalized))
            self.session.add(BookAlias(unit_id=unit_id, alias_normalized=normalized, weight=weight))
            written += 1
        return written

    def _write_triples(self, triples: Iterable[Any]) -> int:
        """Insert de-duplicated triples; return how many rows were written."""
        seen: set[tuple[str, str, str]] = set()
        written = 0
        for triple in triples:
            subject = str(field_of(triple, "subject_id", "subject", default=""))
            predicate = str(field_of(triple, "predicate", default=""))
            obj = str(field_of(triple, "object_id", "object", default=""))
            key = (subject, predicate, obj)
            if not all(key) or key in seen:
                continue
            seen.add(key)
            self.session.add(
                BookTriple(
                    subject_id=subject,
                    predicate=predicate,
                    object_id=obj,
                    provenance=str(field_of(triple, "provenance", default="")),
                )
            )
            written += 1
        return written

    # -- queries ------------------------------------------------------------------------------
    def get(self, unit_id: str) -> BookUnit | None:
        """Return the unit with ``unit_id`` (``"tanim-2.4"``) or ``None``."""
        return self.session.get(BookUnit, unit_id)

    def count(self) -> int:
        """Number of stored units."""
        return int(self.session.scalar(select(func.count(BookUnit.id))) or 0)

    def chapters(self) -> list[str]:
        """Distinct chapter identifiers, ordered (``GET /book/chapters``)."""
        stmt = select(BookUnit.chapter).distinct().order_by(BookUnit.chapter)
        return [row for row in self.session.scalars(stmt) if row]

    def search_alias(
        self, alias_norm: str, unit_type: UnitType | str | None = None
    ) -> list[BookUnit]:
        """Return units whose normalised alias matches exactly (indexed lookup, step 1).

        Args:
            alias_norm: Already normalised alias; normalised again defensively.
            unit_type: Optional type filter ("türev *tanımı*" → ``Definition``).
        """
        normalized = tr_casefold(alias_norm).strip()
        stmt = (
            select(BookUnit)
            .join(BookAlias, BookAlias.unit_id == BookUnit.id)
            .where(BookAlias.alias_normalized == normalized)
        )
        filter_type = _unit_type_of(unit_type)
        if filter_type is not None:
            stmt = stmt.where(BookUnit.unit_type == filter_type)
        stmt = stmt.order_by(BookAlias.weight.desc(), BookUnit.id)
        return list(self.session.scalars(stmt))

    def search_text(
        self,
        q: str,
        unit_type: UnitType | str | None = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[tuple[BookUnit, float]]:
        """Full-text search over ``search_text``, re-ranked by similarity (steps 2–3).

        MySQL uses the ``FULLTEXT`` index created by the migration
        (``MATCH ... AGAINST``); SQLite falls back to ``LIKE`` — the two paths are covered by
        ``test_book_search_like_fallback_on_sqlite``. Both return the same shape.

        Args:
            q: Free-text query (normalised with :func:`aicls.core.turkish.tr_casefold`).
            unit_type: Optional type filter.
            limit: Maximum number of hits.

        Returns:
            ``(unit, score)`` pairs ordered by descending score, where ``score`` is a
            ``difflib`` similarity ratio in ``[0, 1]``.
        """
        needle = tr_casefold(q).strip()
        size = clamp_limit(limit)
        if not needle:
            return []
        stmt = select(BookUnit)
        if self.dialect == "mysql":
            stmt = stmt.where(BookUnit.search_text.match(needle))
        else:
            stmt = stmt.where(BookUnit.search_text.like(f"%{needle}%"))
        filter_type = _unit_type_of(unit_type)
        if filter_type is not None:
            stmt = stmt.where(BookUnit.unit_type == filter_type)
        candidates = list(
            self.session.scalars(stmt.order_by(BookUnit.id).limit(size * SEARCH_CANDIDATE_FACTOR))
        )
        scored = [(unit, self._similarity(needle, unit)) for unit in candidates]
        scored.sort(key=lambda pair: (-pair[1], pair[0].id))
        return scored[:size]

    @staticmethod
    def _similarity(needle: str, unit: BookUnit) -> float:
        """Best ``difflib`` ratio of the query against the unit's title and search text."""
        title_ratio = SequenceMatcher(None, needle, unit.title_normalized or "").ratio()
        text_ratio = SequenceMatcher(None, needle, (unit.search_text or "")[:200]).ratio()
        return max(title_ratio, text_ratio)

    def related(self, unit_id: str, predicate: str | None = None) -> list[BookUnit]:
        """Return units one hop away from ``unit_id`` (graph walk, ADR-0019).

        Args:
            unit_id: Subject unit.
            predicate: Optional predicate filter (``"aicls:requires"``); all edges when ``None``.
        """
        stmt = (
            select(BookUnit)
            .join(BookTriple, BookTriple.object_id == BookUnit.id)
            .where(BookTriple.subject_id == unit_id)
        )
        if predicate is not None:
            stmt = stmt.where(BookTriple.predicate == str(getattr(predicate, "value", predicate)))
        return list(self.session.scalars(stmt.order_by(BookTriple.predicate, BookUnit.id)))

    def triples(self, *, limit: int | None = None) -> list[BookTriple]:
        """Return stored triples, ordered — the source of the RDF/Turtle export."""
        stmt = select(BookTriple).order_by(
            BookTriple.subject_id, BookTriple.predicate, BookTriple.object_id
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self.session.scalars(stmt))

    def units(self, *, unit_type: UnitType | str | None = None) -> list[BookUnit]:
        """Return every unit (optionally of one type), ordered by chapter then position."""
        stmt = select(BookUnit)
        filter_type = _unit_type_of(unit_type)
        if filter_type is not None:
            stmt = stmt.where(BookUnit.unit_type == filter_type)
        return list(
            self.session.scalars(
                stmt.order_by(BookUnit.chapter, BookUnit.order_in_chapter, BookUnit.id)
            )
        )


__all__ = ["DEFAULT_SEARCH_LIMIT", "BookRepository"]
