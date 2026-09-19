"""The pattern compiler: slots, suffix tolerance, macros and word classes."""

from __future__ import annotations

import pytest

from aicls.nlu.errors import InventoryError
from aicls.nlu.lexicon import load_lexicon
from aicls.nlu.patterns import (
    MAX_TOLERATED_SUFFIXES,
    SUFFIX_MACROS,
    WORD_CLASSES,
    compile_pattern,
    compile_patterns,
)


@pytest.fixture(scope="module")
def suffixes() -> tuple[str, ...]:
    return load_lexicon().suffixes


def compile_one(source: str, suffixes: tuple[str, ...], **kwargs: object):  # type: ignore[no-untyped-def]
    return compile_pattern(source, suffixes=suffixes, **kwargs)  # type: ignore[arg-type]


class TestLiterals:
    def test_plain_literal_matches(self, suffixes: tuple[str, ...]) -> None:
        pattern = compile_one("temizle", suffixes)
        assert pattern.fullmatch("temizle")
        assert not pattern.fullmatch("temizle şimdi")

    def test_literal_tolerates_a_following_suffix_token(self, suffixes: tuple[str, ...]) -> None:
        pattern = compile_one("türev al", suffixes)
        assert pattern.fullmatch("türev ini al")
        assert pattern.fullmatch("türev i al")
        assert pattern.fullmatch("türev al")

    def test_literal_tolerates_two_stacked_suffix_tokens(self, suffixes: tuple[str, ...]) -> None:
        assert MAX_TOLERATED_SUFFIXES == 2
        pattern = compile_one("ifade sadeleştir", suffixes)
        assert pattern.fullmatch("ifade si nin sadeleştir")

    def test_optional_literal(self, suffixes: tuple[str, ...]) -> None:
        pattern = compile_one("türev al?", suffixes)
        assert pattern.fullmatch("türev")
        assert pattern.fullmatch("türev al")

    def test_alternation_group(self, suffixes: tuple[str, ...]) -> None:
        pattern = compile_one("türev (al|bul|hesapla)", suffixes)
        assert pattern.fullmatch("türev bul")
        assert not pattern.fullmatch("türev çiz")


class TestSlots:
    def test_slot_captures_and_is_named(self, suffixes: tuple[str, ...]) -> None:
        pattern = compile_one(
            "{expr} türev", suffixes, slot_types={"expr": "EXPR"}, intent="differentiate"
        )
        assert pattern.slots == ("expr",)
        match = pattern.fullmatch("x kare nin türev")
        assert match is not None
        assert match["expr"] == "x kare nin"

    def test_var_slot_captures_one_token_only(self, suffixes: tuple[str, ...]) -> None:
        pattern = compile_one(
            "{expr} {var} <DAT>? göre türev",
            suffixes,
            slot_types={"expr": "EXPR", "var": "VAR"},
        )
        match = pattern.fullmatch("x kare nin x e göre türev")
        assert match is not None
        assert match["var"] == "x"

    def test_point_slot_takes_at_most_two_tokens(self, suffixes: tuple[str, ...]) -> None:
        pattern = compile_one("{point} <DAT> giderken", suffixes, slot_types={"point": "POINT"})
        match = pattern.fullmatch("eksi sonsuz a giderken")
        assert match is not None
        assert match["point"] == "eksi sonsuz"

    def test_optional_slot_may_be_absent(self, suffixes: tuple[str, ...]) -> None:
        pattern = compile_one("türev {expr}?", suffixes, slot_types={"expr": "EXPR"})
        match = pattern.fullmatch("türev")
        assert match is not None
        assert match["expr"] is None


class TestMacros:
    def test_suffix_macro_matches_its_family(self, suffixes: tuple[str, ...]) -> None:
        pattern = compile_one("{lower} <ABL> kadar", suffixes, slot_types={"lower": "BOUNDS"})
        assert pattern.fullmatch("0 dan kadar")
        assert pattern.fullmatch("0 ten kadar")
        assert not pattern.fullmatch("0 xyz kadar")

    def test_any_macro_matches_any_suffix(self) -> None:
        assert set(SUFFIX_MACROS["ANY"]) >= set(SUFFIX_MACROS["GEN"])

    def test_word_class_macro(self, suffixes: tuple[str, ...]) -> None:
        pattern = compile_one("{expr} <OBJ>? türev", suffixes, slot_types={"expr": "EXPR"})
        assert pattern.fullmatch("x kare ifade sinin türev")
        assert "OBJ" in WORD_CLASSES

    def test_ordinal_word_class(self, suffixes: tuple[str, ...]) -> None:
        pattern = compile_one("<ORD> türev", suffixes)
        assert pattern.fullmatch("ikinci türev")


class TestErrors:
    @pytest.mark.parametrize(
        "source",
        ["", "   ", "<NOPE> temizle", "türev ()", "türev {expr} {expr}", "türev ((a|b))"],
    )
    def test_malformed_patterns_are_refused(self, source: str, suffixes: tuple[str, ...]) -> None:
        with pytest.raises(InventoryError):
            compile_one(source, suffixes, slot_types={"expr": "EXPR"})

    def test_optional_first_unit_is_refused(self, suffixes: tuple[str, ...]) -> None:
        with pytest.raises(InventoryError):
            compile_one("(lütfen)? temizle", suffixes)

    def test_compile_patterns_keeps_order(self, suffixes: tuple[str, ...]) -> None:
        compiled = compile_patterns(("temizle", "sil"), suffixes=suffixes)
        assert [item.source for item in compiled] == ["temizle", "sil"]

    def test_compiles_without_a_suffix_inventory(self) -> None:
        pattern = compile_pattern("temizle", suffixes=())
        assert pattern.fullmatch("temizle")
