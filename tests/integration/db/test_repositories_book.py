"""Behaviour of the book knowledge-graph repository (İP-8, ADR-0019).

The units are the ones the proposal uses as examples ("türev tanımı" → ``tanim-2.4``), and the
inputs deliberately arrive in three different shapes (mapping, attribute object, tuple) because
``aicls.book`` passes pydantic models while the CLI passes dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from sqlalchemy.orm import Session

from aicls.core.errors import InputValidationError
from aicls.db import BookRepository, TagStatus, UnitType

pytestmark = pytest.mark.integration

UNITS: list[dict[str, object]] = [
    {
        "id": "tanim-2.1",
        "unit_type": "Definition",
        "chapter": "bolum-2",
        "number": "2.1",
        "order_in_chapter": 1,
        "title_tr": "Limit",
        "body_md": "Bir fonksiyonun limiti ...",
        "aliases": ["limit", "limitin tanımı"],
    },
    {
        "id": "tanim-2.4",
        "type": "Definition",
        "chapter": "bolum-2",
        "number": "2.4",
        "order_in_chapter": 2,
        "title_tr": "Türev",
        "body_md": "$f$ fonksiyonu $a$ noktasında türevlenebilir ...",
        "latex": ["f'(a) = \\lim_{h \\to 0} \\frac{f(a+h)-f(a)}{h}"],
        "aliases": ["türev", "türevin tanımı", "türev tanımı"],
        "source_page": 58,
        "tag_status": "reviewed",
        "jsonld": {"@id": "aicls:tanim-2.4"},
    },
    {
        "id": "teorem-3.1",
        "unit_type": "Theorem",
        "chapter": "bolum-3",
        "number": "3.1",
        "title_tr": "Ortalama Değer Teoremi",
        "body_md": "Türevlenebilir bir fonksiyon için ...",
    },
]


@dataclass
class TripleIn:
    """A triple in the attribute shape ``aicls.book`` uses."""

    subject: str
    predicate: str
    object: str
    provenance: str = "markdown_tagger"


@dataclass
class UnitIn:
    """A unit in the attribute shape ``aicls.book.ContentUnit`` uses."""

    id: str
    type: str
    title_tr: str
    chapter: str = "bolum-4"
    aliases: list[str] = field(default_factory=list)


TRIPLES = [
    TripleIn("tanim-2.4", "aicls:requires", "tanim-2.1"),
    TripleIn("tanim-2.4", "aicls:partOf", "teorem-3.1"),
    TripleIn("teorem-3.1", "aicls:proves", "tanim-2.4"),
]


@pytest.fixture
def book(session: Session) -> BookRepository:
    """A repository preloaded with the three sample units and their triples."""
    repo = BookRepository(session)
    repo.upsert_units(UNITS, TRIPLES)
    return repo


def test_upsert_units_writes_units_aliases_and_triples(session: Session) -> None:
    """The load report counts every row written (İP-8 tagged-ratio criterion)."""
    repo = BookRepository(session)
    report = repo.upsert_units(UNITS, TRIPLES, [("teorem-3.1", "ortalama değer teoremi")])
    assert report.units == 3
    assert report.triples == 3
    assert report.aliases == 6
    assert report.tagged_ratio == 1.0
    assert repo.count() == 3
    assert repo.chapters() == ["bolum-2", "bolum-3"]


def test_upsert_units_is_idempotent_and_bumps_the_version(session: Session) -> None:
    """Reloading a chapter updates in place and replaces aliases and outgoing triples."""
    repo = BookRepository(session)
    repo.upsert_units(UNITS, TRIPLES)
    second = repo.upsert_units(UNITS, TRIPLES)
    assert repo.count() == 3
    assert second.aliases == 5  # the five unit-carried aliases, without the extra one above
    assert len(repo.triples()) == 3
    unit = repo.get("tanim-2.4")
    assert unit is not None and unit.version == 2
    assert len(repo.search_alias("türev tanımı")) == 1


def test_get_and_normalised_columns(book: BookRepository) -> None:
    """Denormalised columns are filled from the source document."""
    unit = book.get("tanim-2.4")
    assert unit is not None
    assert unit.unit_type is UnitType.DEFINITION
    assert unit.title_tr == "Türev"
    assert unit.title_normalized == "türev"
    assert unit.tag_status is TagStatus.REVIEWED
    assert unit.source_page == 58
    assert unit.jsonld == {"@id": "aicls:tanim-2.4"}
    assert unit.latex_json["blocks"][0].startswith("f'(a)")
    assert book.get("yok-1.1") is None


def test_book_resolve_turevin_tanimi_and_turev_tanimi_same_unit(book: BookRepository) -> None:
    """Both Turkish suffix forms of the proposal example resolve to the same unit."""
    first = book.search_alias("türevin tanımı")
    second = book.search_alias("Türev Tanımı")
    assert [u.id for u in first] == [u.id for u in second] == ["tanim-2.4"]


def test_search_alias_filters_by_unit_type(book: BookRepository, session: Session) -> None:
    """The type filter separates "türev tanımı" from a theorem of the same name."""
    BookRepository(session).upsert_units(
        [UnitIn(id="teorem-9.9", type="Theorem", title_tr="Türev", aliases=["türev"])]
    )
    assert {u.id for u in book.search_alias("türev")} == {"tanim-2.4", "teorem-9.9"}
    assert [u.id for u in book.search_alias("türev", "Definition")] == ["tanim-2.4"]
    assert [u.id for u in book.search_alias("türev", UnitType.THEOREM)] == ["teorem-9.9"]
    with pytest.raises(InputValidationError):
        book.search_alias("türev", "Sarki")


def test_book_search_like_fallback_on_sqlite(book: BookRepository) -> None:
    """On SQLite the full-text step is a LIKE scan re-ranked by difflib similarity."""
    assert book.dialect == "sqlite"
    hits = book.search_text("türevlenebilir")
    assert hits and hits[0][0].id in {"tanim-2.4", "teorem-3.1"}
    assert all(0.0 <= score <= 1.0 for _, score in hits)
    assert [h[0].id for h in book.search_text("türevlenebilir", "Theorem")] == ["teorem-3.1"]
    assert book.search_text("") == []
    assert book.search_text("kuantum kromodinamiği") == []


def test_search_text_respects_the_limit(book: BookRepository) -> None:
    """The limit is applied after re-ranking."""
    assert len(book.search_text("t", limit=1)) == 1


def test_related_walks_one_hop_and_filters_by_predicate(book: BookRepository) -> None:
    """``related`` is the single-join graph walk of ADR-0019."""
    assert {u.id for u in book.related("tanim-2.4")} == {"tanim-2.1", "teorem-3.1"}
    assert [u.id for u in book.related("tanim-2.4", "aicls:requires")] == ["tanim-2.1"]
    assert book.related("tanim-2.1") == []


def test_units_listing_is_ordered_and_filterable(book: BookRepository) -> None:
    """``units`` returns chapter order, which the export and the chapter tree rely on."""
    assert [u.id for u in book.units()] == ["tanim-2.1", "tanim-2.4", "teorem-3.1"]
    assert [u.id for u in book.units(unit_type="Theorem")] == ["teorem-3.1"]
    assert len(book.triples(limit=2)) == 2


def test_upsert_skips_rows_without_an_id_and_deduplicates(session: Session) -> None:
    """Malformed rows are skipped rather than raising, and duplicates collapse."""
    repo = BookRepository(session)
    report = repo.upsert_units(
        [{"title_tr": "kimliksiz"}, {"id": "ornek-1.1", "unit_type": "Example", "title_tr": "Ör"}],
        [TripleIn("ornek-1.1", "aicls:next", ""), TripleIn("ornek-1.1", "aicls:next", "ornek-1.1")],
        [("ornek-1.1", "ör"), {"unit_id": "ornek-1.1", "alias": "ÖR", "weight": 0.8}],
    )
    assert report.units == 1
    assert report.triples == 1
    assert report.aliases == 1
