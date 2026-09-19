"""Turkish text helpers: casefold with dotted/dotless i, lower, diacritics, suffix table."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st
import pytest

from aicls.core.turkish import (
    TR_SUFFIXES,
    strip_diacritics,
    strip_suffix,
    tr_casefold,
    tr_lower,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("İ", "i"),
        ("I", "ı"),
        ("i", "i"),
        ("ı", "ı"),
        ("İSTANBUL", "istanbul"),
        ("ISPARTA", "ısparta"),
        ("DİYARBAKIR", "diyarbakır"),
        ("Türev", "türev"),
        ("İNTEGRAL AL", "integral al"),
        ("ÇĞÖŞÜ", "çğöşü"),
    ],
)
def test_tr_casefold_dotted_and_dotless_i(raw: str, expected: str) -> None:
    assert tr_casefold(raw) == expected


def test_tr_casefold_differs_from_unicode_casefold() -> None:
    assert "İ".casefold() != "i"  # plain casefold yields i + combining dot
    assert tr_casefold("İ") == "i"
    assert "I".casefold() == "i"  # plain casefold loses the dotless i
    assert tr_casefold("I") == "ı"


def test_tr_casefold_normalises_nfc_and_decomposed_input() -> None:
    decomposed = "İ"  # I + COMBINING DOT ABOVE == İ after NFC
    assert tr_casefold(decomposed) == "i"
    assert tr_casefold("ü") == "ü"


def test_tr_lower_keeps_letters_but_uses_turkish_i_rules() -> None:
    assert tr_lower("TÜREVİ AL") == "türevi al"
    assert tr_lower("KISMİ") == "kısmi"
    assert tr_lower("ß") == "ß"  # lower, not casefold
    assert tr_casefold("ß") == "ss"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("türev", "turev"),
        ("İntegral", "Integral"),
        ("ışık", "isik"),
        ("ÇĞÖŞÜ çğöşü", "CGOSU cgosu"),
        ("café", "cafe"),
        ("x^2", "x^2"),
    ],
)
def test_strip_diacritics(raw: str, expected: str) -> None:
    assert strip_diacritics(raw) == expected


def test_suffix_table_shape() -> None:
    assert isinstance(TR_SUFFIXES, dict)
    for family, forms in TR_SUFFIXES.items():
        assert isinstance(family, str) and family
        assert forms and all(isinstance(f, str) and f == tr_casefold(f) for f in forms)
    # vowel-harmony variants for the accusative and genitive are present
    assert {"ı", "i", "u", "ü"} <= set(TR_SUFFIXES["acc"])
    assert {"ın", "in", "un", "ün"} <= set(TR_SUFFIXES["gen"])
    assert {"ını", "ini", "unu", "ünü"} <= set(TR_SUFFIXES["poss3_acc"])


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("türevi", "türev"),
        ("türevini", "türev"),
        ("integralin", "integral"),
        ("limitinin", "limit"),
        ("fonksiyonu", "fonksiyon"),
        ("teoremler", "teorem"),
        ("tanımı", "tanım"),
        ("türev", "türev"),
        ("al", "al"),  # too short to strip
        ("TÜREVİNİ", "türev"),  # casefolded first
    ],
)
def test_strip_suffix(word: str, expected: str) -> None:
    assert strip_suffix(word) == expected


def test_strip_suffix_respects_min_stem_and_families() -> None:
    assert strip_suffix("türevi", families=("gen",)) == "türevi"
    assert strip_suffix("türevi", min_stem=6) == "türevi"


@given(st.text())
def test_tr_casefold_is_idempotent(s: str) -> None:
    once = tr_casefold(s)
    assert tr_casefold(once) == once


@given(st.text())
def test_tr_casefold_never_yields_uppercase_i_forms(s: str) -> None:
    out = tr_casefold(s)
    assert "İ" not in out
    assert "I" not in out
