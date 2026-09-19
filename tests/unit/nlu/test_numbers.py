"""Turkish numerals: cardinals, fractions, decimals and ordinals."""

from __future__ import annotations

import pytest

from aicls.core.errors import InputValidationError
from aicls.nlu.errors import InventoryError
from aicls.nlu.numbers import format_number, load_numbers, ordinal_value, to_number


class TestToNumber:
    def test_nlu_number_words_yuz_yirmi_uc_to_123(self) -> None:
        """The documented example: ``to_number("yüz yirmi üç") == 123``."""
        assert to_number("yüz yirmi üç") == 123

    @pytest.mark.parametrize(
        ("phrase", "value"),
        [
            ("sıfır", 0),
            ("bir", 1),
            ("on", 10),
            ("on iki", 12),
            ("yirmi beş", 25),
            ("yüz", 100),
            ("iki yüz", 200),
            ("iki yüz otuz dört", 234),
            ("bin", 1000),
            ("üç bin dört yüz", 3400),
            ("bir milyon", 1_000_000),
            ("iki buçuk", 2.5),
            ("yarım", 0.5),
            ("çeyrek", 0.25),
            ("iki virgül beş", 2.5),
            ("12", 12),
            ("2.5", 2.5),
        ],
    )
    def test_reads_turkish_numerals(self, phrase: str, value: float) -> None:
        assert to_number(phrase) == value

    def test_accepts_a_token_sequence(self) -> None:
        assert to_number(["yüz", "yirmi", "üç"]) == 123

    def test_integral_values_come_back_as_int(self) -> None:
        assert isinstance(to_number("iki"), int)
        assert isinstance(to_number("iki buçuk"), float)

    @pytest.mark.parametrize("phrase", ["", "   ", "kadar", "türev iki"])
    def test_rejects_non_numerals(self, phrase: str) -> None:
        with pytest.raises(InputValidationError):
            to_number(phrase)

    @pytest.mark.parametrize(
        "phrase",
        [
            "dört sıfır",  # the KNOWN_ISSUES PRIORITY 1 phrase: two units in a row
            "üç beş",
            "sıfır dört",
            "beş on",  # a unit may not be followed by a tens word
            "yirmi yüz",  # only a unit may multiply "yüz"
            "yüz yüz",
            "iki bin milyon",  # scale words must decrease
            "iki buçuk yarım",  # a fraction closes the phrase
            "4 0",  # what a decoder writes when it runs two numerals together
            "2 3",
        ],
    )
    def test_rejects_an_ill_formed_numeral_phrase(self, phrase: str) -> None:
        """Turkish names magnitude classes in decreasing order; anything else is not a numeral.

        Summing them regardless is how ``"x kare eksi dört sıfır denklemini çöz"`` lost its last
        content word and solved a different equation (KNOWN_ISSUES PRIORITY 1).
        """
        with pytest.raises(InputValidationError) as info:
            to_number(phrase)
        assert info.value.details["reason"] == "numeral_out_of_order"

    @pytest.mark.parametrize(
        ("phrase", "value"),
        [("iki bin dört yüz elli altı", 2456), ("on iki bin", 12000), ("üç buçuk", 3.5)],
    )
    def test_still_reads_the_well_formed_ones(self, phrase: str, value: float) -> None:
        assert to_number(phrase) == value


class TestOrdinals:
    @pytest.mark.parametrize(
        ("word", "value"),
        [("birinci", 1), ("ikinci", 2), ("üçüncü", 3), ("beşinci", 5), ("2", 2), ("3.", 3)],
    )
    def test_reads_ordinals(self, word: str, value: int) -> None:
        assert ordinal_value(word) == value

    def test_returns_none_for_a_non_ordinal(self) -> None:
        assert ordinal_value("türev") is None


class TestTable:
    def test_is_number_word_covers_words_and_digits(self) -> None:
        table = load_numbers()
        assert table.is_number_word("yüz")
        assert table.is_number_word("12")
        assert not table.is_number_word("türev")

    def test_format_number_matches_the_normaliser(self) -> None:
        assert format_number(3) == "3"
        assert format_number(3.0) == "3"
        assert format_number(2.5) == "2.5"

    def test_malformed_table_is_an_inventory_error(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "numbers.yaml"
        path.write_text("units: [1, 2]\n", encoding="utf-8")
        with pytest.raises(InventoryError):
            load_numbers(str(path))
