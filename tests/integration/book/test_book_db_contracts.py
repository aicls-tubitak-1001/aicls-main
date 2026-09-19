"""Contract between ``aicls.book`` and ``aicls.db`` — the boundary this module introduces.

Wave 1 lost a nested field between two packages because each side only tested its own fixtures
(KNOWN_ISSUES, "Cross-module contract testing"). Everything asserted here is a promise the two
packages make to each other:

* the field names ``BookRepository`` reads (``unit_type``/``type``, ``latex_blocks``,
  ``subject``/``object``) match what ``aicls.book`` writes — a pydantic ``ContentUnit`` and
  ``Triple`` must be storable *without* the loader's payload conversion;
* the vocabularies agree (``aicls.book.UnitType`` values, ``Predicate`` values vs.
  ``aicls.db.enums.BOOK_PREDICATES``);
* a unit survives the full round trip ``ContentUnit → book_units → ContentUnit``;
* the alias weights the loader assigns really order ``BookRepository.search_alias``;
* the column limits of ``book_units`` (id 64 chars, title 255) are respected by minted ids.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from aicls.book import ContentUnit, Predicate, Triple, UnitType, load_units, unit_from_row
from aicls.book.aliases import WEIGHT_BY_KIND, generate_alias_entries
from aicls.book.loader import triples_from_repo, unit_payload, units_from_repo
from aicls.db import BookAlias, BookRepository, BookUnit
from aicls.db.enums import BOOK_PREDICATES, TagStatus
from aicls.db.enums import UnitType as DbUnitType
from aicls.db.models.book import UNIT_ID_LENGTH

pytestmark = pytest.mark.integration

UNIT = ContentUnit(
    id="tanim-2.4",
    type=UnitType.DEFINITION,
    chapter="bolum-2",
    number="2.4",
    title_tr="Türev",
    aliases=["türevin tanımı", "diferansiyel katsayı"],
    body_md="$f$ fonksiyonu $a$ noktasında ...",
    latex_blocks=["f'(a) = \\lim_{h \\to 0} \\frac{f(a+h)-f(a)}{h}"],
    source_page=58,
    order_in_chapter=3,
)


def test_repository_accepts_a_pydantic_content_unit_directly(repo: BookRepository) -> None:
    """``BookRepository`` reads a ``ContentUnit`` without the loader translating field names."""
    report = repo.upsert_units([UNIT])
    assert report.units == 1
    row = repo.get("tanim-2.4")
    assert row is not None
    assert row.unit_type is DbUnitType.DEFINITION
    assert row.chapter == "bolum-2"
    assert row.number == "2.4"
    assert row.title_tr == "Türev"
    assert row.order_in_chapter == 3
    assert row.source_page == 58
    assert row.latex_json == {"blocks": UNIT.latex_blocks}
    assert row.title_normalized == "türev"
    assert row.tag_status is TagStatus.AUTO
    aliases = repo.session.scalars(
        select(BookAlias.alias_normalized).where(BookAlias.unit_id == "tanim-2.4")
    ).all()
    assert set(aliases) == {"türevin tanımı", "diferansiyel katsayı"}


def test_repository_accepts_a_pydantic_triple_directly(repo: BookRepository) -> None:
    """A ``Triple`` stores its predicate as the CURIE ``book_triples.predicate`` expects."""
    other = UNIT.model_copy(update={"id": "tanim-2.1", "number": "2.1", "title_tr": "Hız"})
    repo.upsert_units(
        [UNIT, other],
        triples=[Triple(subject="tanim-2.4", predicate=Predicate.REQUIRES, object="tanim-2.1")],
    )
    stored = repo.triples()
    assert [(row.subject_id, row.predicate, row.object_id) for row in stored] == [
        ("tanim-2.4", "aicls:requires", "tanim-2.1")
    ]
    assert stored[0].predicate in BOOK_PREDICATES


def test_content_unit_round_trips_through_the_database(repo: BookRepository) -> None:
    """Everything ``ContentUnit`` carries survives ``load_units`` → ``unit_from_row``."""
    load_units([UNIT], [], repo)
    row = repo.get("tanim-2.4")
    assert row is not None
    restored = unit_from_row(row)
    assert restored.id == UNIT.id
    assert restored.type is UNIT.type
    assert restored.chapter == UNIT.chapter
    assert restored.number == UNIT.number
    assert restored.title_tr == UNIT.title_tr
    assert restored.body_md == UNIT.body_md
    assert restored.latex_blocks == UNIT.latex_blocks
    assert restored.source_page == UNIT.source_page
    assert restored.order_in_chapter == UNIT.order_in_chapter
    assert restored.aliases == UNIT.aliases
    assert restored.jsonld["@id"] == "aicls:tanim-2.4"


def test_alias_weights_order_the_index(repo: BookRepository) -> None:
    """A weight the generator assigns really decides which unit wins a shared alias.

    "Türev" is both the title of the chapter (Section) and of the definition; the definition
    must win, and it does because a section alias is written with a lower weight.
    """
    section = ContentUnit(id="bolum-2", type=UnitType.SECTION, chapter="bolum-2", title_tr="Türev")
    load_units([section, UNIT], [], repo)
    assert [row.id for row in repo.search_alias("türev")] == ["tanim-2.4", "bolum-2"]
    weights = {
        (row.unit_id, row.alias_normalized): row.weight
        for row in repo.session.scalars(select(BookAlias)).all()
        if row.alias_normalized == "türev"
    }
    assert weights[("tanim-2.4", "türev")] == WEIGHT_BY_KIND["title"]
    assert weights[("bolum-2", "türev")] == WEIGHT_BY_KIND["section"]


def test_generated_aliases_fit_the_alias_column(
    repo: BookRepository, sample_units: list[ContentUnit]
) -> None:
    """No generated alias or minted id overflows the column widths of ``book_units``."""
    for unit in sample_units:
        assert len(unit.id) <= UNIT_ID_LENGTH
        assert len(unit.title_tr) <= 255
        for entry in generate_alias_entries(unit.title_tr, unit.type):
            assert len(entry.alias) <= 255


def test_repo_readers_return_detached_models(
    loaded_repo: BookRepository, sample_units: list[ContentUnit], sample_triples: list[Triple]
) -> None:
    """``units_from_repo``/``triples_from_repo`` hand back plain models, not ORM rows."""
    units = units_from_repo(loaded_repo)
    triples = triples_from_repo(loaded_repo)
    assert len(units) == len(sample_units)
    assert len(triples) == len(sample_triples)
    assert all(isinstance(unit, ContentUnit) for unit in units)
    assert all(isinstance(triple, Triple) for triple in triples)
    assert {triple.predicate for triple in triples} <= set(Predicate)
    definitions = units_from_repo(loaded_repo, unit_type=UnitType.DEFINITION)
    assert {unit.type for unit in definitions} == {UnitType.DEFINITION}


def test_payload_keys_are_all_understood_by_the_repository(repo: BookRepository) -> None:
    """Every key the loader emits is one the repository actually reads."""
    payload = unit_payload(UNIT, [])
    repo.upsert_units([payload])
    row = repo.session.get(BookUnit, "tanim-2.4")
    assert row is not None
    assert row.search_text == payload["search_text"]
    assert row.jsonld == payload["jsonld"]
    assert row.title_normalized == payload["title_normalized"]
    assert row.latex_json == {"blocks": payload["latex_blocks"]}
