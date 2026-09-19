"""Command-phrase parsing of the resolver (no database involved)."""

from __future__ import annotations

import pytest

from aicls.book import UnitType, detect_unit_type
from aicls.book.query import normalize_query
from aicls.book.resolver import (
    COMMAND_STOPWORDS,
    content_tokens,
    is_stopword,
    query_variants,
    resolution_plan,
    word_stem,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Türev tanımını getir!", "türev tanımını getir"),
        ("  TÜREV   TANIMI  ", "türev tanımı"),
        ("Işık, İzmir", "ışık izmir"),
        ("", ""),
    ],
)
def test_normalize_query_is_turkish_aware(raw: str, expected: str) -> None:
    """Normalisation casefolds the Turkish way (İ→i, I→ı) and strips punctuation."""
    assert normalize_query(raw) == expected


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("tanımını", "tanım"),
        ("tanımı", "tanım"),
        ("teoremi", "teorem"),
        ("örneğini", "örnek"),
        ("şeklini", "şekil"),
        ("sonucunu", "sonuç"),
        ("tablosunu", "tablo"),
        ("bölümünü", "bölüm"),
        ("türev", "türev"),
    ],
)
def test_word_stem_undoes_suffix_and_softening(token: str, expected: str) -> None:
    """One suffix is removed and consonant softening undone, so type words are recognisable."""
    assert word_stem(token) == expected


@pytest.mark.parametrize(
    ("phrase", "expected_type", "expected_concept"),
    [
        ("türev tanımını", UnitType.DEFINITION, ["türev"]),
        ("ara değer teoremini", UnitType.THEOREM, ["ara", "değer"]),
        ("kuvvet kuralı örneğini", UnitType.EXAMPLE, ["kuvvet", "kuralı"]),
        ("teğet doğrusu grafiğini", UnitType.FIGURE, ["teğet", "doğrusu"]),
        ("türev", None, ["türev"]),
    ],
)
def test_detect_unit_type_reads_the_type_word(
    phrase: str, expected_type: UnitType | None, expected_concept: list[str]
) -> None:
    """The type word is recognised through its suffixes and removed from the concept."""
    unit_type, concept = detect_unit_type(phrase.split())
    assert unit_type is expected_type
    assert concept == expected_concept


def test_command_verbs_are_dropped() -> None:
    """Verbs and fillers never reach the alias index, with or without diacritics."""
    assert content_tokens("lütfen türev tanımını ekrana getir") == ["türev", "tanımını"]
    assert content_tokens("limit tanimini goster") == ["limit", "tanimini"]
    assert is_stopword("göster") and is_stopword("goster")
    assert not is_stopword("türev")
    assert "getir" in COMMAND_STOPWORDS


def test_resolution_plan_orders_from_specific_to_general() -> None:
    """The full phrase is tried first, the type-free concept next, sub-phrases last."""
    primary, ladder, unit_type = resolution_plan("türev tanımını getir", None)
    assert unit_type is UnitType.DEFINITION
    assert primary[0] == "türev tanımını"
    assert "türev" in primary
    assert ladder[: len(primary)] == primary
    assert len(ladder) >= len(primary)


def test_window_fallback_covers_extra_words() -> None:
    """A phrase padded with words the book never used still offers the exact sub-phrase."""
    ladder, _ = query_variants("türev tanımının kendisini yaz", None)
    assert "türev tanımının" in ladder


def test_explicit_unit_type_overrides_the_detected_one() -> None:
    """A caller-supplied filter wins over the type read off the phrase."""
    _, _, unit_type = resolution_plan("türev tanımını getir", UnitType.EXAMPLE)
    assert unit_type is UnitType.EXAMPLE


def test_empty_query_has_no_variants() -> None:
    """A query made only of stop words yields nothing to look up."""
    primary, ladder, unit_type = resolution_plan("lütfen getir", None)
    assert primary == []
    assert ladder == []
    assert unit_type is None
