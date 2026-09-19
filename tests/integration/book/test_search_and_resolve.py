"""Search ladder and command resolution over the loaded sample book."""

from __future__ import annotations

import pytest

from aicls.book import BookResolver, Predicate, UnitType, get_unit, related, search
from aicls.book.query import FUZZY_THRESHOLD
from aicls.db import BookRepository

pytestmark = pytest.mark.integration


def test_book_resolve_turevin_tanimi_and_turev_tanimi_same_unit(resolver: BookResolver) -> None:
    """The contract example: every phrasing of "türev tanımı" resolves to ``tanim-2.4``."""
    phrasings = [
        "türev tanımı",
        "türevin tanımı",
        "türev tanımını getir",
        "türevin tanımını getir",
        "türev tanımını ekrana getir",
        "turev tanimi",
    ]
    resolved = [resolver.resolve(phrase) for phrase in phrasings]
    assert all(hits for hits in resolved)
    assert {hits[0].unit.id for hits in resolved} == {"tanim-2.4"}
    assert {hits[0].stage for hits in resolved} == {"alias"}
    assert all(hits[0].score == 1.0 for hits in resolved)
    assert resolved[0][0].unit_type is UnitType.DEFINITION
    assert resolved[0][0].unit.title_tr == "Türev"


def test_book_search_like_fallback_on_sqlite(loaded_repo: BookRepository) -> None:
    """A phrase that is no alias but occurs in the body is found by the SQLite LIKE pass."""
    assert loaded_repo.dialect == "sqlite"
    hits = search(loaded_repo, "kesen doğruların limit konumu")
    assert hits
    assert hits[0].unit.id == "sekil-2.3"
    assert hits[0].stage == "fulltext"
    assert 0.0 < hits[0].score <= 1.0


def test_search_prefers_the_alias_index(loaded_repo: BookRepository) -> None:
    """An exact alias short-circuits the later stages and scores 1.0."""
    hits = search(loaded_repo, "türev tanımı")
    assert [hit.unit.id for hit in hits] == ["tanim-2.4"]
    assert hits[0].stage == "alias"
    assert hits[0].matched_alias == "türev tanımı"


def test_search_falls_through_to_difflib_on_a_typo(loaded_repo: BookRepository) -> None:
    """A misspelling reaches neither the alias index nor LIKE, so difflib decides."""
    hits = search(loaded_repo, "türve", unit_type=UnitType.DEFINITION)
    assert hits
    assert hits[0].unit.id == "tanim-2.4"
    assert hits[0].stage == "fuzzy"
    assert hits[0].score >= FUZZY_THRESHOLD


def test_search_honours_the_type_filter(loaded_repo: BookRepository) -> None:
    """Asking for a Theorem never returns the Definition of the same name."""
    hits = search(loaded_repo, "türev", unit_type=UnitType.THEOREM)
    assert all(hit.unit.type is UnitType.THEOREM for hit in hits)
    assert "tanim-2.4" not in {hit.unit.id for hit in hits}


def test_search_respects_the_limit_and_empty_query(loaded_repo: BookRepository) -> None:
    """``limit`` clamps the result list; an empty query yields nothing."""
    assert search(loaded_repo, "   ") == []
    assert len(search(loaded_repo, "türev", limit=1)) <= 1


def test_resolve_reads_the_unit_type_off_the_phrase(resolver: BookResolver) -> None:
    """ "…teoremini" selects a Theorem, "…örneğini" an Example, without an explicit filter."""
    assert resolver.resolve("ara değer teoremini getir")[0].unit.id == "teorem-1.5"
    assert resolver.resolve("kuvvet kuralı örneğini göster")[0].unit.id == "ornek-2.7"
    assert resolver.resolve("teğet doğrusu şeklini getir")[0].unit.id == "sekil-2.3"
    assert resolver.resolve("temel türev kuralları tablosunu aç")[0].unit.id == "tablo-2.11"


def test_resolve_one_and_unknown_query(resolver: BookResolver) -> None:
    """``resolve_one`` returns the best hit; an unrelated phrase returns nothing."""
    best = resolver.resolve_one("türev tanımını getir")
    assert best is not None
    assert best.unit.id == "tanim-2.4"
    assert resolver.resolve("kuantum kromodinamiği laboratuvarı") == []
    assert resolver.resolve_one("lütfen getir") is None


def test_resolver_cache_serves_the_second_call(loaded_repo: BookRepository) -> None:
    """The LRU (ADR-0019: 2048 entries) answers a repeated phrase without touching the store."""
    resolver = BookResolver(loaded_repo, cache_size=4)
    assert resolver.cache_info() == {"hits": 0, "misses": 0, "size": 0, "maxsize": 4}
    first = resolver.resolve("türev tanımını getir")
    second = resolver.resolve("türev tanımını getir")
    assert first == second
    info = resolver.cache_info()
    assert (info["hits"], info["misses"], info["size"]) == (1, 1, 1)
    for phrase in ("limit tanımı", "süreklilik tanımı", "zincir kuralı", "kuvvet kuralı"):
        resolver.resolve(phrase)
    assert resolver.cache_info()["size"] <= 4
    resolver.cache_clear()
    assert resolver.cache_info() == {"hits": 0, "misses": 0, "size": 0, "maxsize": 4}


def test_resolver_without_cache(loaded_repo: BookRepository) -> None:
    """``cache_size=0`` disables memoisation but not resolution."""
    resolver = BookResolver(loaded_repo, cache_size=0)
    assert resolver.resolve("türev tanımı")[0].unit.id == "tanim-2.4"
    assert resolver.cache_info()["size"] == 0


def test_graph_walk_and_get_unit(loaded_repo: BookRepository) -> None:
    """One-hop relation queries and identifier lookup return detached content units."""
    unit = get_unit(loaded_repo, "tanim-2.4")
    assert unit is not None
    assert unit.title_tr == "Türev"
    assert unit.number == "2.4"
    assert unit.source_page == 58
    assert unit.latex_blocks
    assert "türevin tanımı" in unit.aliases
    # The stored JSON-LD keeps the figure reference, so it comes back as an asset ref.
    assert [asset.id for asset in unit.assets] == ["sekil-2.3"]
    assert get_unit(loaded_repo, "yok-1.1") is None
    requires = related(loaded_repo, "tanim-2.4", predicate=Predicate.REQUIRES.value)
    assert [item.id for item in requires] == ["tanim-2.1"]
    figures = related(loaded_repo, "tanim-2.4", predicate=Predicate.HAS_FIGURE.value)
    assert [item.id for item in figures] == ["sekil-2.3"]
