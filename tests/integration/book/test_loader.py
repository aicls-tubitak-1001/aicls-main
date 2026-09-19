"""Loader behaviour against the real schema: refusal, idempotence, aliases and foreign keys."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import func, select

from aicls.book import (
    ContentUnit,
    Predicate,
    TagResult,
    Triple,
    UnitType,
    UntaggedHeadingError,
    coverage,
    load_markdown_dir,
    load_tag_result,
    load_units,
)
from aicls.book.loader import chapter_id_for, resolve_triples, search_text_for, unit_payload
from aicls.core.errors import InputValidationError
from aicls.db import BookAlias, BookRepository, BookTriple, BookUnit

pytestmark = pytest.mark.integration


def test_loader_rejects_untagged_heading_with_list(
    repo: BookRepository, untagged_chapter: TagResult
) -> None:
    """A chapter with an untagged heading is refused and the headings are listed.

    This is the İP-8 "100 % tagged" criterion: nothing of the chapter reaches the database.
    """
    assert coverage(untagged_chapter) < 1.0
    with pytest.raises(UntaggedHeadingError) as excinfo:
        load_tag_result(untagged_chapter, repo)
    error = excinfo.value
    assert error.headings == ["Tanım: Belirsiz İntegral", "Teorem: Analizin Temel Teoremi"]
    assert error.details["untagged_headings"] == error.headings
    assert error.tagged_ratio == pytest.approx(0.5)
    assert error.http_status == 422
    assert repo.count() == 0


def test_loader_reports_units_triples_and_aliases(
    repo: BookRepository, sample_units: list[ContentUnit], sample_triples: list[Triple]
) -> None:
    """The load report counts exactly what reached the three tables."""
    report = load_units(sample_units, sample_triples, repo)
    assert report.tagged_ratio == 1.0
    assert report.units == len(sample_units) == repo.count()
    assert report.triples == len(sample_triples)
    session = repo.session
    assert session.scalar(select(func.count(BookTriple.id))) == report.triples
    assert session.scalar(select(func.count(BookAlias.id))) == report.aliases
    assert report.aliases > 10 * report.units


def test_reloading_the_same_chapter_is_idempotent(
    repo: BookRepository, sample_units: list[ContentUnit], sample_triples: list[Triple]
) -> None:
    """``aicls book load`` may be run repeatedly; rows are updated, never duplicated."""
    first = load_units(sample_units, sample_triples, repo)
    second = load_units(sample_units, sample_triples, repo)
    assert (second.units, second.triples, second.aliases) == (
        first.units,
        first.triples,
        first.aliases,
    )
    assert repo.count() == first.units
    session = repo.session
    assert session.scalar(select(func.count(BookAlias.id))) == first.aliases
    assert session.get(BookUnit, "tanim-2.4").version == 2


def test_dangling_edges_are_dropped_instead_of_breaking_the_load(
    repo: BookRepository,
) -> None:
    """``book_triples`` carries real foreign keys, so an edge to an unknown unit is dropped."""
    unit = ContentUnit(id="tanim-9.1", type=UnitType.DEFINITION, chapter="bolum-9", title_tr="A")
    edges = [
        Triple(subject="tanim-9.1", predicate=Predicate.REQUIRES, object="tanim-9.9"),
        Triple(subject="tanim-9.1", predicate=Predicate.NEXT, object="tanim-9.1"),
    ]
    kept, dropped = resolve_triples(edges, {"tanim-9.1"}, repo)
    assert [triple.object for triple in dropped] == ["tanim-9.9"]
    assert len(kept) == 1
    report = load_units([unit], edges, repo)
    assert report.units == 1
    assert report.triples == 1


def test_cross_file_edges_resolve_once_both_chapters_are_loaded(
    repo: BookRepository, sample_units: list[ContentUnit], sample_triples: list[Triple]
) -> None:
    """An edge pointing at a unit already in the database survives a second, separate load."""
    chapter_one = [unit for unit in sample_units if unit.chapter == "bolum-1"]
    chapter_two = [unit for unit in sample_units if unit.chapter == "bolum-2"]
    load_units(chapter_one, [], repo)
    extra = Triple(
        subject="tanim-2.4", predicate=Predicate.RELATED, object="tanim-1.1", provenance="manual"
    )
    edges = [*[t for t in sample_triples if t.subject.endswith(("2.1", "2.4"))], extra]
    load_units(chapter_two, edges, repo)
    related_ids = [unit.id for unit in repo.related("tanim-2.4", Predicate.RELATED.value)]
    assert related_ids == ["tanim-1.1"]


def test_duplicate_identifiers_are_rejected(repo: BookRepository) -> None:
    """Two units with the same id in one batch is an authoring error, not an overwrite."""
    unit = ContentUnit(id="tanim-9.1", type=UnitType.DEFINITION, title_tr="A")
    with pytest.raises(InputValidationError) as excinfo:
        load_units([unit, unit], [], repo)
    assert excinfo.value.details["duplicates"] == ["tanim-9.1"]
    assert repo.count() == 0


def test_load_markdown_dir_loads_every_chapter(repo: BookRepository, book_datasets: Path) -> None:
    """``aicls book load <dir>`` tags all files first, so cross-file edges resolve."""
    report = load_markdown_dir(book_datasets / "sample", repo)
    assert report.tagged_ratio == 1.0
    assert repo.count() == report.units == 20
    assert sorted(repo.chapters()) == ["bolum-1", "bolum-2"]


def test_load_markdown_dir_refuses_a_directory_with_an_untagged_heading(
    repo: BookRepository, book_datasets: Path
) -> None:
    """One broken chapter refuses the whole directory (all-or-nothing tagging)."""
    with pytest.raises(UntaggedHeadingError):
        load_markdown_dir(book_datasets / "invalid", repo)
    assert repo.count() == 0


def test_search_text_and_payload_carry_the_generated_aliases() -> None:
    """The LIKE/FULLTEXT column holds the title, the aliases and the body."""
    unit = ContentUnit(
        id="tanim-2.4",
        type=UnitType.DEFINITION,
        chapter="bolum-2",
        number="2.4",
        title_tr="Türev",
        aliases=["Diferansiyel Katsayı"],
        body_md="Limit varsa ...",
    )
    payload = unit_payload(unit, [])
    assert payload["title_normalized"] == "türev"
    assert payload["aliases"] == ["diferansiyel katsayı"]
    assert payload["jsonld"]["@id"] == "aicls:tanim-2.4"
    text = search_text_for(unit, ["türev tanımı"])
    assert "türev tanımı" in text
    assert "limit varsa" in text


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("bolum-02-turev.md", "bolum-2"),
        ("bolum-11-integral.md", "bolum-11"),
        ("giris.md", "bolum-7"),
    ],
)
def test_chapter_id_for(name: str, expected: str) -> None:
    """The chapter id comes from the first number in the file name, else from its position."""
    assert chapter_id_for(Path(name), 7) == expected
