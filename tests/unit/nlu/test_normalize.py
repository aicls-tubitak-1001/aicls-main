"""Normalisation: casefold, suffix splitting, phrase folding, numbers, spans."""

from __future__ import annotations

import pytest

from aicls.nlu.lexicon import load_lexicon
from aicls.nlu.normalize import normalize_text, normalized_words


class TestCasefold:
    def test_nlu_normalizer_handles_turkish_i_casefold(self) -> None:
        """``İ`` becomes ``i`` and ``I`` becomes ``ı`` -- the default Unicode mapping does not."""
        analysed = normalize_text("İNTEGRALİNİ AL")
        assert analysed.tokens[0] == "integral"
        assert "̇" not in analysed.text  # never i + combining dot above
        assert normalize_text("Türevini Al").text == normalize_text("türevini al").text

    def test_dotless_capital_i_lowercases_to_dotless(self) -> None:
        assert normalize_text("Işık").tokens[0].startswith("ı")

    def test_nfc_normalisation_merges_decomposed_letters(self) -> None:
        decomposed = "x kareṡi"  # s + combining dot, not a Turkish letter
        assert normalize_text(decomposed).tokens[0] == "x"


class TestSuffixSplitting:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("x karenin türevini al", ("x", "kare", "nin", "türev", "ini", "al")),
            ("x'in", ("x", "in")),
            ("integralini", ("integral", "ini")),
            ("sonucunu kaydet", ("sonuç", "unu", "kaydet")),
            ("grafiğini çiz", ("grafik", "ini", "çiz")),
            ("köklerini bul", ("kök", "lerini", "bul")),
        ],
    )
    def test_splits_known_stems(self, text: str, expected: tuple[str, ...]) -> None:
        assert normalize_text(text).tokens == expected

    def test_unknown_word_is_not_split(self) -> None:
        # "kadar" must survive: "ka" is not a word, so there is no morpheme boundary to find.
        assert "kadar" in normalize_text("sıfırdan bire kadar").tokens

    def test_detached_suffix_is_marked_as_a_suffix(self) -> None:
        analysed = normalize_text("x kare nin türevini al")
        assert analysed.kinds[analysed.tokens.index("nin")] == "suffix"

    def test_detached_suffix_is_not_read_as_a_function(self) -> None:
        # "sini" must stay an ending; splitting it into sin + i would invent a sine.
        analysed = normalize_text("x kare ifade sini sadeleştir")
        assert "sin" not in analysed.tokens


class TestLexiconFolding:
    @pytest.mark.parametrize(
        ("text", "token"),
        [
            ("sinüs x", "sin"),
            ("kosinüs x", "cos"),
            ("türef al", "türev"),
            ("entegral al", "integral"),
            ("sifir", "0"),
            ("esittir", "eşittir"),
        ],
    )
    def test_variants_and_ascii_fallback(self, text: str, token: str) -> None:
        assert token in normalize_text(text).tokens

    def test_phrases_are_folded(self) -> None:
        assert normalize_text("mutlak değer x").tokens == ("mutlakdeğer", "x")
        assert normalize_text("parantez aç x parantez kapa").tokens == ("(", "x", ")")

    def test_fillers_are_dropped(self) -> None:
        assert normalize_text("yani tahtayı lütfen temizle").tokens == ("tahta", "yı", "temizle")

    def test_ascii_index_is_unambiguous(self) -> None:
        lexicon = load_lexicon()
        assert lexicon.ascii_index["sifir"] == "sıfır"
        assert all(key == key.lower() for key in lexicon.ascii_index)


class TestNumbers:
    def test_number_word_runs_become_digits(self) -> None:
        assert normalize_text("yüz yirmi üç x").tokens == ("123", "x")

    def test_suffix_breaks_a_number_run(self) -> None:
        assert normalize_text("sıfırdan bire").tokens == ("0", "dan", "1", "e")

    def test_decimal_literal_survives(self) -> None:
        assert normalize_text("örnek 2.7").tokens == ("örnek", "2.7")

    def test_turkish_decimal_comma_becomes_a_dot(self) -> None:
        assert normalize_text("2,5 x").tokens == ("2.5", "x")


class TestSpansAndSlices:
    def test_spans_point_into_the_original_text(self) -> None:
        text = "x karenin türevini al"
        analysed = normalize_text(text)
        for token, (start, end) in zip(analysed.tokens, analysed.spans, strict=True):
            assert 0 <= start <= end <= len(analysed.original)
            assert analysed.original[start:end]
            assert token  # every token keeps a non-empty source span

    def test_token_range_maps_a_character_range_back(self) -> None:
        analysed = normalize_text("x karenin türevini al")
        first, last = analysed.token_range(0, len("x kare nin"))
        assert analysed.slice_text(first, last) == "x kare nin"
        assert analysed.slice_math(first, last) == "x kare"

    def test_token_range_of_an_empty_range_is_empty(self) -> None:
        analysed = normalize_text("temizle")
        assert analysed.token_range(50, 60) == (0, 0)
        assert analysed.source_span(0, 0) == (0, 0)

    def test_items_and_words_expose_the_same_tokens(self) -> None:
        analysed = normalize_text("x karenin türevini al")
        assert tuple(item.text for item in analysed.items) == analysed.tokens
        assert analysed.words == tuple(
            token
            for token, kind in zip(analysed.tokens, analysed.kinds, strict=True)
            if kind != "suffix"
        )

    def test_empty_input_yields_no_tokens(self) -> None:
        analysed = normalize_text("   ")
        assert analysed.tokens == ()
        assert analysed.text == ""

    def test_normalized_words_helper(self) -> None:
        assert normalized_words("x karenin türevini al") == ("x", "kare", "türev", "al")
