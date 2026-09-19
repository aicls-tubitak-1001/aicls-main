"""Turkish suffix-table alias generation (``türev`` → ``türevin``, ``türev tanımı``, ...)."""

from __future__ import annotations

import pytest

from aicls.book import UnitType, generate_alias_entries, generate_aliases
from aicls.book.aliases import (
    ASCII_WEIGHT_FACTOR,
    WEIGHT_BY_KIND,
    genitive,
    harden,
    head_stem,
    inflect,
    possessive_forms,
    soften,
)


def test_generate_aliases_turkish_suffixes_turev() -> None:
    """The contract example: "Türev" expands to its case forms and its "tanım" combinations."""
    aliases = generate_aliases("Türev", UnitType.DEFINITION)
    for expected in (
        "türev",
        "türevi",
        "türevin",
        "türeve",
        "türevde",
        "türevden",
        "türevini",
        "türevinin",
        "türevine",
        "türevler",
        "türevleri",
        "türevlerin",
        "türev tanımı",
        "türevin tanımı",
        "türev tanımını",
        "türevin tanımını",
        "türev tanımının",
        "türev tanımına",
    ):
        assert expected in aliases, f"{expected} missing from {aliases}"
    assert aliases[0] == "türev"
    assert len(aliases) == len(set(aliases))


def test_ascii_variants_are_generated_at_half_weight() -> None:
    """A query typed without Turkish characters still hits the index, ranked below the original."""
    entries = {entry.alias: entry for entry in generate_alias_entries("Türev", UnitType.DEFINITION)}
    assert "turev" in entries
    assert "turev tanimi" in entries
    assert entries["turev"].kind == "ascii"
    assert entries["turev"].weight == pytest.approx(entries["türev"].weight * ASCII_WEIGHT_FACTOR)


def test_title_alias_outranks_every_generated_form() -> None:
    """The normalised title is the first, highest-weighted alias."""
    entries = generate_alias_entries("Zincir Kuralı", UnitType.THEOREM)
    assert entries[0].alias == "zincir kuralı"
    assert entries[0].weight == WEIGHT_BY_KIND["title"]
    assert all(entry.weight <= entries[0].weight for entry in entries)


def test_type_word_suffix_is_dropped_from_the_core_phrase() -> None:
    """ "Sıkıştırma Teoremi" is addressable as "sıkıştırma" and as "sıkıştırma teoremi"."""
    aliases = generate_aliases("Sıkıştırma Teoremi", UnitType.THEOREM)
    assert "sıkıştırma teoremi" in aliases
    assert "sıkıştırma" in aliases
    assert "sıkıştırmanın teoremi" in aliases


def test_possessive_title_is_declined_from_its_stem() -> None:
    """ "Ortalama Değişim Hızı" yields "hızını"/"hızının", not only the nominative."""
    aliases = generate_aliases("Ortalama Değişim Hızı", UnitType.DEFINITION)
    assert "ortalama değişim hızı" in aliases
    assert "ortalama değişim hızını" in aliases
    assert "ortalama değişim hızının" in aliases
    assert "hız tanımı" in aliases


def test_section_gets_no_bare_forms() -> None:
    """A section titled "Türev" must not claim the bare alias of the definition "Türev"."""
    aliases = generate_aliases("Türev", UnitType.SECTION)
    entries = generate_alias_entries("Türev", UnitType.SECTION)
    assert "türev bölümü" in aliases
    assert "türevin bölümü" in aliases
    assert "türevi" not in aliases
    assert entries[0].alias == "türev"
    assert entries[0].weight == WEIGHT_BY_KIND["section"]
    assert entries[0].weight < WEIGHT_BY_KIND["title"]


def test_multi_word_title_gets_no_bare_head_alias() -> None:
    """Only single-word concepts get bare head aliases, so heads cannot collide across units."""
    aliases = generate_aliases("Türevlenebilirlik Teoremi Kanıtı", UnitType.PROOF)
    assert "türevlenebilirlik teoremi kanıtı" in aliases
    assert "teorem kanıtı" in aliases
    assert "teorem" not in aliases


def test_empty_title_generates_nothing() -> None:
    """A unit without a title contributes no aliases (its id still resolves it)."""
    assert generate_aliases("   ", UnitType.REMARK) == []


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("türev", ["türev", "türevi", "türevin", "türeve", "türevde", "türevden"]),
        ("masa", ["masa", "masayı", "masanın", "masaya", "masada", "masadan"]),
        ("örnek", ["örnek", "örneği", "örneğin", "örneğe", "örnekte", "örnekten"]),
        ("tanım", ["tanım", "tanımı", "tanımın", "tanıma", "tanımda", "tanımdan"]),
    ],
)
def test_inflect_applies_vowel_harmony_and_assimilation(stem: str, expected: list[str]) -> None:
    """Harmony (ı/i/u/ü, a/e), buffer consonants and d→t after a voiceless stop."""
    assert inflect(stem)[: len(expected)] == expected


def test_inflect_declines_only_the_last_word() -> None:
    """A multi-word phrase keeps its prefix and declines its head."""
    assert inflect("ara değer")[:3] == ["ara değer", "ara değeri", "ara değerin"]


def test_soften_and_harden_round_trip() -> None:
    """Softening is applied before vowel-initial suffixes and undone when stemming back."""
    assert soften("örnek") == "örneğ"
    assert soften("kitap") == "kitab"
    assert soften("şekil") == "şekl"
    assert soften("kanıt") == "kanıt"
    assert soften("ek") == "ek"
    assert harden("örneğ") == "örnek"
    assert harden("şekl") == "şekil"
    assert harden("türev") == "türev"


def test_head_stem_and_genitive() -> None:
    """The head stem drops one suffix and undoes softening; the genitive re-attaches one."""
    assert head_stem("fonksiyonun limiti") == "limit"
    assert head_stem("temel türev kuralları") == "kural"
    assert genitive("türev") == "türevin"
    assert genitive("masa") == "masanın"
    assert genitive("") == ""


def test_possessive_forms_of_a_type_word() -> None:
    """The four possessive forms used to build "<konu> tanımı" combinations."""
    assert possessive_forms("tanım") == ["tanımı", "tanımını", "tanımının", "tanımına"]
    assert possessive_forms("tablo") == ["tablosu", "tablosunu", "tablosunun", "tablosuna"]
