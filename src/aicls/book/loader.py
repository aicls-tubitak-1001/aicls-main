"""Loader: units, triples and generated aliases into the relational store (ADR-0019).

The loader is the only writer of the book graph. It

1. refuses a chapter whose tagging is incomplete (:class:`
   ~aicls.book.ingest.validators.UntaggedHeadingError` lists the headings — İP-8 100 % criterion),
2. expands every title into its Turkish suffix forms (:mod:`aicls.book.aliases`) and weights them,
3. fills ``book_units.jsonld`` with the JSON-LD document and ``search_text`` with the text the
   SQLite ``LIKE`` / MySQL ``FULLTEXT`` pass searches, and
4. drops edges whose endpoints exist neither in the batch nor in the database, because
   ``book_triples`` carries real foreign keys — a dangling edge would abort the whole load.

All writing goes through :class:`aicls.db.repositories.book.BookRepository`; there is no SQL here
(docs/mimari/veri-modeli.md §1 rule 8).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
import re
from typing import Any

from aicls.book.aliases import generate_alias_entries
from aicls.book.export import unit_to_jsonld
from aicls.book.ingest.markdown_tagger import tag_markdown
from aicls.book.ingest.validators import UntaggedHeadingError, coverage, require_full_coverage
from aicls.book.query import unit_from_row
from aicls.book.schema import (
    ContentUnit,
    Predicate,
    TagResult,
    Triple,
    UnitType,
    UntaggedSpan,
)
from aicls.core.errors import InputValidationError
from aicls.core.logging import get_logger
from aicls.core.messages_tr import msg
from aicls.core.turkish import tr_casefold
from aicls.db.repositories.book import BookRepository
from aicls.db.results import LoadReport

_LOGGER = get_logger(__name__)

#: Characters of the body copied into ``search_text`` (the rest is not worth indexing).
SEARCH_BODY_CHARS = 600
#: Suffix of a Markdown chapter file picked up by :func:`load_markdown_dir`.
MARKDOWN_SUFFIXES = (".md", ".markdown")


def search_text_for(unit: ContentUnit, aliases: Sequence[str]) -> str:
    """Build the normalised text the full-text/LIKE pass searches for one unit."""
    parts = [tr_casefold(unit.title_tr), " ".join(aliases), tr_casefold(unit.body_md)]
    text = " ".join(part for part in parts if part)
    return " ".join(text.split())[: SEARCH_BODY_CHARS + 2000]


def unit_payload(unit: ContentUnit, triples: Sequence[Triple]) -> dict[str, Any]:
    """Render a unit into the mapping :meth:`BookRepository.upsert_units` stores.

    The author-supplied aliases stay on the unit (weight 1.0 in the repository); the generated
    suffix forms are passed separately so they can carry their own weights.
    """
    aliases = [entry.alias for entry in generate_alias_entries(unit.title_tr, unit.type)]
    authored = [tr_casefold(alias) for alias in unit.aliases]
    document = unit.jsonld or unit_to_jsonld(unit, triples)
    return {
        "id": unit.id,
        "unit_type": unit.type.value,
        "chapter": unit.chapter,
        "number": unit.number,
        "order_in_chapter": unit.order_in_chapter,
        "title_tr": unit.title_tr,
        "title_normalized": tr_casefold(unit.title_tr),
        "body_md": unit.body_md,
        "latex_blocks": list(unit.latex_blocks),
        "search_text": search_text_for(unit, authored + aliases),
        "jsonld": document,
        "source_page": unit.source_page,
        "aliases": authored,
    }


def _alias_rows(units: Sequence[ContentUnit]) -> list[tuple[str, str, float]]:
    """Generate ``(unit_id, alias, weight)`` rows for every unit."""
    rows: list[tuple[str, str, float]] = []
    for unit in units:
        for entry in generate_alias_entries(unit.title_tr, unit.type):
            rows.append((unit.id, entry.alias, entry.weight))
    return rows


def resolve_triples(
    triples: Iterable[Triple], known: set[str], repo: BookRepository
) -> tuple[list[Triple], list[Triple]]:
    """Split ``triples`` into those whose endpoints exist and those that dangle.

    Args:
        triples: Candidate edges.
        known: Identifiers being written in this batch.
        repo: Repository used to look up endpoints already stored.

    Returns:
        ``(kept, dropped)``. A dropped edge points at a unit that is neither in the batch nor in
        the database; keeping it would violate the ``book_triples`` foreign keys.
    """
    kept: list[Triple] = []
    dropped: list[Triple] = []
    seen: dict[str, bool] = {identifier: True for identifier in known}

    def exists(identifier: str) -> bool:
        if identifier not in seen:
            seen[identifier] = repo.get(identifier) is not None
        return seen[identifier]

    for triple in triples:
        if exists(triple.subject) and exists(triple.object):
            kept.append(triple)
        else:
            dropped.append(triple)
    return kept, dropped


def load_units(
    units: Sequence[ContentUnit],
    triples: Sequence[Triple],
    repo: BookRepository,
    *,
    tagged_ratio: float = 1.0,
    untagged: Sequence[UntaggedSpan] = (),
) -> LoadReport:
    """Upsert units, triples and generated aliases through :class:`BookRepository`.

    Reloading the same chapter is idempotent: the repository updates each unit in place and
    rewrites its aliases and outgoing edges.

    Args:
        units: Content units to store.
        triples: Edges between them; dangling edges are dropped and logged.
        repo: Book repository bound to an open session (the caller commits).
        tagged_ratio: Share of source headings that became units. Anything below 1.0 refuses
            the load (İP-8 criterion).
        untagged: The untagged headings behind a ``tagged_ratio`` below 1.0; they are listed in
            the raised error.

    Returns:
        A :class:`~aicls.db.results.LoadReport` with ``units``, ``triples``, ``aliases`` and
        ``tagged_ratio``.

    Raises:
        UntaggedHeadingError: If ``untagged`` is non-empty or ``tagged_ratio`` is below 1.0.
        InputValidationError: If two units in the batch share an identifier.
    """
    if untagged or tagged_ratio < 1.0:
        raise UntaggedHeadingError(list(untagged), tagged_ratio=tagged_ratio)

    identifiers = [unit.id for unit in units]
    duplicates = sorted({key for key in identifiers if identifiers.count(key) > 1})
    if duplicates:
        raise InputValidationError(
            msg("VALIDATION_ERROR"), details={"field": "unit.id", "duplicates": duplicates}
        )

    edges = list(triples)
    kept, dropped = resolve_triples(edges, set(identifiers), repo)
    if dropped:
        _LOGGER.warning(
            "book_dangling_triples",
            count=len(dropped),
            edges=[list(triple.key()) for triple in dropped],
        )
    payloads = [unit_payload(unit, kept) for unit in units]
    report = repo.upsert_units(
        payloads,
        triples=[triple.model_dump(mode="json") for triple in kept],
        aliases=_alias_rows(units),
        tagged_ratio=tagged_ratio,
    )
    _LOGGER.info(
        "book_loaded",
        units=report.units,
        triples=report.triples,
        aliases=report.aliases,
        tagged_ratio=report.tagged_ratio,
        dropped_triples=len(dropped),
    )
    return report


def load_tag_result(result: TagResult, repo: BookRepository) -> LoadReport:
    """Load one tagger output, refusing it when a heading went untagged."""
    return load_units(
        result.nodes,
        result.triples,
        repo,
        tagged_ratio=coverage(result),
        untagged=result.untagged_spans,
    )


def chapter_id_for(path: Path, fallback_index: int = 1) -> str:
    """Derive a chapter identifier from a file name (``bolum-02-turev.md`` → ``bolum-2``).

    The first group of digits in the stem is the chapter number; a stem without digits falls
    back to ``fallback_index`` (the file's position in the sorted directory listing).
    """
    match = re.search(r"\d+", path.stem)
    number = int(match.group()) if match else fallback_index
    return f"bolum-{number}"


def tag_markdown_dir(directory: Path) -> list[TagResult]:
    """Tag every Markdown chapter of ``directory``, sorted by file name.

    Raises:
        NotFoundError: If ``directory`` does not exist.
    """
    from aicls.core.errors import NotFoundError

    if not directory.is_dir():
        raise NotFoundError(
            msg("FILE_NOT_FOUND", path=str(directory)), details={"path": str(directory)}
        )
    results: list[TagResult] = []
    files = sorted(path for path in directory.iterdir() if path.suffix.lower() in MARKDOWN_SUFFIXES)
    for index, path in enumerate(files, start=1):
        results.append(
            tag_markdown(
                path.read_text(encoding="utf-8"),
                chapter=chapter_id_for(path, index),
                source=path.name,
            )
        )
    return results


def load_markdown_dir(directory: Path, repo: BookRepository) -> LoadReport:
    """Tag and load every Markdown chapter of ``directory`` in one transaction-safe batch.

    All files are tagged first so cross-file ``requires`` edges resolve, then the combined
    coverage is checked (``aicls book load <dir>``, docs/mimari/moduller.md).

    Raises:
        UntaggedHeadingError: If any chapter carries an untagged heading.
    """
    results = tag_markdown_dir(directory)
    ratio = require_full_coverage(*results)
    units = [node for result in results for node in result.nodes]
    edges = [triple for result in results for triple in result.triples]
    return load_units(units, edges, repo, tagged_ratio=ratio)


def units_from_repo(
    repo: BookRepository, *, unit_type: UnitType | None = None
) -> list[ContentUnit]:
    """Read every stored unit back as a :class:`~aicls.book.schema.ContentUnit`."""
    return [unit_from_row(row) for row in repo.units(unit_type=unit_type)]


def triples_from_repo(repo: BookRepository) -> list[Triple]:
    """Read every stored edge back as a :class:`~aicls.book.schema.Triple`.

    Raises:
        InputValidationError: If a stored predicate is outside the :class:`Predicate` vocabulary
            (only reachable if a row was written by something other than this loader).
    """
    triples: list[Triple] = []
    for row in repo.triples():
        try:
            predicate = Predicate(row.predicate)
        except ValueError as exc:
            raise InputValidationError(
                msg("VALIDATION_ERROR"),
                details={
                    "field": "book_triples.predicate",
                    "value": row.predicate,
                    "allowed": [member.value for member in Predicate],
                },
            ) from exc
        triples.append(
            Triple(
                subject=row.subject_id,
                predicate=predicate,
                object=row.object_id,
                provenance=row.provenance,
            )
        )
    return triples


__all__ = [
    "MARKDOWN_SUFFIXES",
    "SEARCH_BODY_CHARS",
    "chapter_id_for",
    "load_markdown_dir",
    "load_tag_result",
    "load_units",
    "resolve_triples",
    "search_text_for",
    "tag_markdown_dir",
    "triples_from_repo",
    "unit_payload",
    "units_from_repo",
]
