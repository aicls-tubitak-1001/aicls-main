"""Adversarial sweep over the command inventory.

The worst defect class for this project is a *wrong-but-confident* command: the layer resolves,
the CAS computes, and the platform renders a correct-looking answer to a question nobody asked
(KNOWN_ISSUES PRIORITY 1). Unit tests written from the happy path cannot find those, so this
module attacks the inventory from two sides:

* :class:`TestMeaningPreservingVariants` rewrites **every worked example** the inventory carries
  -- filler words, shortened or detached case endings, numerals as digits, Turkish capitalisation,
  a lost apostrophe -- and requires the resolved command to be *the same command* or an honest
  question. A different command is a wrong answer waiting to happen.
* :class:`TestMeaningChangingVariants` feeds phrasings that must **not** resolve: another command
  hidden in the tail, an expression the reading dropped, an order the speaker never said, word
  order Turkish does not use.

Findings are recorded in docs/mimari/metin-komut-katmani.md §9.5; the ones this module leaves as
``xfail``-free "known limitation" tests are pinned so they cannot silently get worse.
"""

from __future__ import annotations

from collections.abc import Iterator
import re

import pytest

from aicls.mathengine.models import Diff
from aicls.nlu import (
    Clarification,
    Matched,
    NoMatch,
    ParseContext,
    load_inventory,
    parse_command,
)
from aicls.nlu.inventory.schema import Inventory

#: Hesitation words a decoder transcribes; the normaliser must drop them.
FILLERS = ("yani", "şey")

#: Cardinal spellings a speaker may replace with digits without changing the meaning.
CARDINALS = {
    "bir": "1",
    "iki": "2",
    "üç": "3",
    "dört": "4",
    "beş": "5",
    "altı": "6",
    "yedi": "7",
    "sekiz": "8",
    "dokuz": "9",
    "on": "10",
}

#: Case endings a fast speaker (or a decoder) shortens.
SUFFIX_SWAPS = (("ini", "i"), ("ını", "ı"), ("unu", "u"), ("ünü", "ü"))


def _filler_variants(words: list[str]) -> Iterator[str]:
    for position in range(1, len(words)):
        for filler in FILLERS:
            yield " ".join([*words[:position], filler, *words[position:]])


def _suffix_variants(words: list[str]) -> Iterator[str]:
    for long, short in SUFFIX_SWAPS:
        hit = [word.endswith(long) and len(word) > len(long) + 2 for word in words]
        if any(hit):
            yield " ".join(
                word[: -len(long)] + short if flag else word
                for word, flag in zip(words, hit, strict=True)
            )


def _numeral_variants(words: list[str]) -> Iterator[str]:
    for index, word in enumerate(words):
        digits = CARDINALS.get(word.rstrip("'’"))
        if digits is None:
            continue
        swapped = list(words)
        swapped[index] = digits
        yield " ".join(swapped)


def meaning_preserving(text: str) -> list[str]:
    """Rewrites of ``text`` a Turkish speaker would call the same command."""
    words = text.split()
    out = [
        *_filler_variants(words),
        *_suffix_variants(words),
        *_numeral_variants(words),
    ]
    if "'" in text:
        out.append(text.replace("'", " "))
    capitalised = re.sub(r"^(\w)", lambda m: m.group(1).upper().replace("I", "İ"), text)
    if capitalised != text:
        out.append(capitalised)
    return out


@pytest.fixture(scope="module")
def inventory() -> Inventory:
    return load_inventory()


class TestMeaningPreservingVariants:
    """A rewrite that keeps the meaning must keep the command -- or ask."""

    def test_no_variant_of_any_example_produces_a_different_command(
        self, inventory: Inventory
    ) -> None:
        wrong: list[str] = []
        variants = 0
        for _, example in inventory.examples():
            context = ParseContext(**example.context) if example.context else None
            base = parse_command(example.text, context=context)
            if not isinstance(base, Matched):
                continue
            for variant in meaning_preserving(example.text):
                variants += 1
                outcome = parse_command(variant, context=context)
                if isinstance(outcome, Matched) and outcome.command != base.command:
                    wrong.append(f"{example.text!r} -> {variant!r}: {outcome.command!r}")
        assert variants > 2000, variants
        assert wrong == [], wrong

    def test_a_filler_never_changes_the_command(self) -> None:
        base = parse_command("x karenin türevini al")
        assert isinstance(base, Matched)
        for text in (
            "x karenin yani türevini al",
            "x karenin türevini al lütfen",
            "şimdi x karenin türevini al bakalım",
        ):
            outcome = parse_command(text)
            assert isinstance(outcome, Matched), text
            assert outcome.command == base.command, text


class TestMeaningChangingVariants:
    """Phrasings that carry something the command would drop must not resolve."""

    @pytest.mark.parametrize(
        "text",
        [
            "x karenin türevini al ve integralini al",
            "x karenin türevini ve integralini al",
            "tahtayı temizle ve x karenin türevini al",
            "türev teoremini getir ve x karenin türevini al",
        ],
    )
    def test_two_commands_in_one_sentence_are_never_half_executed(self, text: str) -> None:
        """Answering the first command and dropping the second is a silent truncation."""
        assert not isinstance(parse_command(text), Matched), text

    @pytest.mark.parametrize(
        "text",
        [
            "x karenin türevini al iki y küp",
            "x karenin türevini al x küp",
            "x karenin integralini al artı bir",
            "tahtayı temizle x kare artı bir",
        ],
    )
    def test_dropped_mathematics_refuses_the_command(self, text: str) -> None:
        assert not isinstance(parse_command(text), Matched), text

    @pytest.mark.parametrize(
        "text",
        [
            "x kare eksi dört sıfır denklemini çöz",
            "x kare eksi dört sıfır ifadesini çarpanlarına ayır",
            "iki üç x in türevini al",
            "x kare üç beş ifadesinin türevini al",
            "sıfırdan bir iki ye kadar x karenin integralini al",
        ],
    )
    def test_an_unreadable_numeral_run_refuses_the_command(self, text: str) -> None:
        assert not isinstance(parse_command(text), Matched), text

    @pytest.mark.parametrize(
        "text",
        [
            "türevini al x karenin",
            "al x karenin türevini",
            "hesapla sıfırdan bire kadar x karenin integralini",
            "çöz x kare eksi dört eşittir sıfır denklemini",
        ],
    )
    def test_verb_first_word_order_is_refused_not_guessed(self, text: str) -> None:
        """Turkish is SOV; the inventory deliberately has no trailing-slot patterns."""
        outcome = parse_command(text)
        assert isinstance(outcome, Clarification | NoMatch), outcome

    @pytest.mark.parametrize(
        "text",
        [
            "x karenin sağdan türevini al",
            "x karenin soldan integralini al",
            "sağdan x kare grafiğini çiz",
        ],
    )
    def test_a_direction_word_where_it_makes_no_sense_is_refused(self, text: str) -> None:
        assert not isinstance(parse_command(text), Matched), text


class TestOrderSweep:
    """Every cardinal, every ordinal, on the intent the PRIORITY 1 bug lived in."""

    @pytest.mark.parametrize(
        "word",
        ["iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz", "on"],
    )
    def test_no_exponent_word_ever_reaches_the_order_slot(self, word: str) -> None:
        outcome = parse_command(f"x üzeri {word} türevini al")
        assert isinstance(outcome, Matched), outcome
        assert isinstance(outcome.command, Diff)
        assert outcome.command.order == 1, outcome.command

    @pytest.mark.parametrize(
        ("phrase", "order"),
        [
            ("ikinci", 2),
            ("üçüncü", 3),
            ("dördüncü", 4),
            ("beşinci", 5),
            ("onikinci", 12),
        ],
    )
    def test_every_ordinal_word_reaches_the_order_slot(self, phrase: str, order: int) -> None:
        outcome = parse_command(f"x karenin {phrase} türevini al")
        assert isinstance(outcome, Matched), outcome
        assert isinstance(outcome.command, Diff)
        assert outcome.command.order == order

    @pytest.mark.parametrize(
        "text",
        [
            "x karenin 2 mertebeden türevini al",
            "x karenin iki mertebeden türevini al",
            "x karenin iki dereceden türevini al",
            "ikinci mertebeden x karenin türevini al",
        ],
    )
    def test_an_explicit_order_phrase_is_read_wherever_it_sits(self, text: str) -> None:
        outcome = parse_command(text)
        assert isinstance(outcome, Matched), outcome
        assert isinstance(outcome.command, Diff)
        assert outcome.command.order == 2
        assert outcome.command.expr == "x^{2}"


class TestKnownLimitations:
    """Findings the sweep produced that this wave does *not* fix, pinned so they stay visible."""

    @pytest.mark.parametrize(
        "text",
        ["x karenin türevini alma", "x karenin türevini almayı bırak"],
    )
    def test_negation_is_not_understood(self, text: str) -> None:
        """The layer has no negation model; "alma" still reaches ``diff``.

        Low stakes (nobody tells a CAS *not* to differentiate) and a naive "-ma/-me" rule would
        break ordinary nouns, so v1 leaves it. Documented in §9.5 as a v1.1 item.
        """
        assert isinstance(parse_command(text), Matched)

    def test_ordinals_above_twelve_are_not_in_the_table(self) -> None:
        """``numbers.yaml`` stops at ``onikinci``; ``MAX_ORDER`` is 12 anyway, so this asks."""
        outcome = parse_command("x karenin on üçüncü türevini al")
        assert isinstance(outcome, Clarification | NoMatch), outcome

    def test_word_boundary_recovery_cannot_reach_into_free_text(self) -> None:
        """Resegmentation is lexicon-driven, so a joined *query* term stays joined."""
        assert not isinstance(parse_command("rolleteoremini getir"), Matched)
