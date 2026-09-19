"""Gold-set measurements on ``datasets/nlu`` (İP-2 "komut iletimi > %95", ADR-0005).

**Provenance, stated once and quoted in every report that uses these numbers:**
``commands_test.jsonl`` and ``asr_noisy_tr.jsonl`` were *authored by the project team* -- the
first from the packaged inventory examples plus the İP-3 reading-scenario library, the second by
applying synthetic ASR-like corruptions to the first. They are a **regression suite**, not an
independent test set, and the noisy set's error mix is an assumption rather than a measurement
because no fine-tuned decoder exists yet. The 2027-05-01 progress report must say so and must
re-measure against real ASR output once the training gate is approved
(KNOWN_ISSUES.md, "Metric provenance").
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.nlu.metrics import (
    STRICT_UNDECLARED_IS_FALSE_POSITIVE,
    EvalReport,
    GoldRow,
    evaluate,
    filter_rows,
    load_gold,
)

DATASET_DIR = Path(__file__).resolve().parents[3] / "datasets" / "nlu"
COMMANDS_TEST = DATASET_DIR / "commands_test.jsonl"
ASR_NOISY = DATASET_DIR / "asr_noisy_tr.jsonl"

#: Contract targets (docs/mimari/metin-komut-katmani.md §8).
INTENT_ACCURACY_TARGET = 0.95
NOISY_INTENT_ACCURACY_TARGET = 0.90
OUT_OF_SCOPE_TARGET = 0.95
MIN_GOLD_ROWS = 300

#: Per-perturbation floor. ``join_words`` used to sit at 0.276 while every other perturbation was
#: >= 0.99 (KNOWN_ISSUES PRIORITY 2); word-boundary recovery lifted it, and this floor keeps it up.
PERTURBATION_ACCURACY_TARGET = 0.85


@pytest.fixture(scope="module")
def gold() -> list[GoldRow]:
    return load_gold(COMMANDS_TEST)


@pytest.fixture(scope="module")
def noisy() -> list[GoldRow]:
    return load_gold(ASR_NOISY)


@pytest.fixture(scope="module")
def gold_report(gold: list[GoldRow]) -> EvalReport:
    return evaluate(gold)


class TestDatasetShape:
    def test_gold_set_has_at_least_300_rows(self, gold: list[GoldRow]) -> None:
        assert len(gold) >= MIN_GOLD_ROWS

    def test_row_ids_are_unique(self, gold: list[GoldRow], noisy: list[GoldRow]) -> None:
        for rows in (gold, noisy):
            identifiers = [row.id for row in rows]
            assert len(identifiers) == len(set(identifiers))

    def test_every_intent_is_represented(self, gold: list[GoldRow]) -> None:
        from aicls.nlu import load_inventory

        covered = {row.intent for row in gold if row.intent}
        assert covered == set(load_inventory().intent_ids)

    def test_out_of_scope_rows_exist(self, gold: list[GoldRow]) -> None:
        assert len(filter_rows(gold, tag="out_of_scope")) >= 10

    def test_rows_with_gold_slots_exist(self, gold: list[GoldRow]) -> None:
        assert sum(1 for row in gold if row.slots) >= 100


class TestIntentAccuracy:
    def test_nlu_command_test_set_intent_accuracy_above_0_95(self, gold_report: EvalReport) -> None:
        assert gold_report.intent_accuracy >= INTENT_ACCURACY_TARGET, gold_report.errors

    def test_nlu_asr_noisy_set_intent_accuracy_above_0_90(self, noisy: list[GoldRow]) -> None:
        report = evaluate(noisy)
        assert report.intent_accuracy >= NOISY_INTENT_ACCURACY_TARGET, report.errors

    def test_out_of_scope_rows_are_refused(self, gold: list[GoldRow]) -> None:
        """The proposal's filtering requirement, measured rather than asserted anecdotally."""
        report = evaluate(filter_rows(gold, tag="out_of_scope"))
        assert report.intent_accuracy >= OUT_OF_SCOPE_TARGET, report.errors

    def test_no_out_of_scope_row_ever_reaches_the_cas(self, gold: list[GoldRow]) -> None:
        """The hard half of the filter: nothing out of scope may become a SymPy command.

        One known miss exists and is deliberately kept in the set: "logaritma x fonksiyonunun
        tanım kümesini söyle" resolves to a ``book_fetch`` because "tanım" is a strong book
        keyword. That routes to the textbook resolver, not to the CAS, so no wrong answer can be
        computed -- but it is a real limitation and v1.1 should add a ``domain`` intent
        (see docs/mimari/metin-komut-katmani.md, "Uygulama notları").
        """
        from aicls.nlu import BookFetch, Matched, ParseContext, UiAction, parse_command

        for row in filter_rows(gold, tag="out_of_scope"):
            context = ParseContext(**row.context) if row.context else None
            outcome = parse_command(row.text, context=context)
            if isinstance(outcome, Matched):
                assert isinstance(outcome.command, BookFetch | UiAction), row.text


class TestSlotQuality:
    def test_slot_f1_on_the_rows_that_declare_slots(self, gold: list[GoldRow]) -> None:
        report = evaluate([row for row in gold if row.slots])
        assert report.slot_f1 >= 0.95, report.errors
        assert report.slot_precision >= 0.95
        assert report.slot_recall >= 0.95

    def test_the_strict_definition_is_the_one_in_force(self) -> None:
        """A resolved-but-undeclared slot must count against the layer.

        This is the metric that hid KNOWN_ISSUES PRIORITY 1: with undeclared slots ignored,
        ``differentiate`` rows said nothing about ``order`` and a third derivative of ``x³``
        scored ``slot_f1 = 1.0000``.
        """
        assert STRICT_UNDECLARED_IS_FALSE_POSITIVE

    def test_every_gold_row_with_slots_declares_the_whole_command(
        self, gold: list[GoldRow]
    ) -> None:
        """Under the strict definition an under-annotated gold row is a *wrong* gold row.

        A row that declares ``expr`` but not ``order`` asserts that ``order`` must stay unfilled,
        which is never what the annotator meant. Keeping the rows complete is what makes
        ``slot_f1`` mean "every slot of every command is right" rather than "the slots someone
        bothered to write down are right".
        """
        from aicls.nlu import Matched, ParseContext, parse_command

        incomplete: list[str] = []
        for row in gold:
            if not row.slots:
                continue
            context = ParseContext(**row.context) if row.context else None
            outcome = parse_command(row.text, context=context)
            if not isinstance(outcome, Matched):
                continue
            undeclared = sorted(name for name in outcome.slots if name not in row.slots)
            if undeclared:
                incomplete.append(f"{row.id}: {row.text!r} does not declare {undeclared}")
        assert incomplete == [], incomplete

    def test_the_order_slot_is_declared_on_every_differentiate_row(
        self, gold: list[GoldRow]
    ) -> None:
        """The specific annotation gap the wrong-answer bug hid behind."""
        missing = [
            row.id
            for row in gold
            if row.intent == "differentiate" and row.slots and "order" not in row.slots
        ]
        assert missing == [], missing

    def test_report_dictionary_is_report_ready(self, gold_report: EvalReport) -> None:
        payload = gold_report.as_dict()
        assert set(payload) == {
            "rows",
            "intent_accuracy",
            "slot_precision",
            "slot_recall",
            "slot_f1",
            "clarification_rate",
            "matched",
            "clarifications",
            "no_matches",
        }
        assert payload["rows"] == gold_report.total


class TestPerturbations:
    def test_every_noisy_row_names_its_perturbation(self) -> None:
        import json

        with ASR_NOISY.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        assert rows
        assert all(row["perturbations"] for row in rows)
        assert all(row["origin_id"] for row in rows)

    def test_perturbations_are_deterministic(self) -> None:
        from tests.unit.nlu.perturb import perturb

        first = perturb("x karenin türevini al", seed=1234)
        second = perturb("x karenin türevini al", seed=1234)
        assert first == second

    def test_perturbations_change_the_text(self) -> None:
        from tests.unit.nlu.perturb import perturb_variants

        variants = perturb_variants("x karenin türevini al", seed=1234, n=3)
        assert len(variants) == 3
        assert all(text != "x karenin türevini al" for text, _ in variants)

    def test_every_perturbation_type_stays_above_the_floor(self, noisy: list[GoldRow]) -> None:
        """The per-perturbation breakdown, not just the average.

        The average hid the real weakness: ``join_words`` was at 0.276 while the overall noisy
        accuracy read 0.9355 (KNOWN_ISSUES PRIORITY 2). A floor per perturbation type is what
        makes a single collapsing failure mode visible.
        """
        import json

        with ASR_NOISY.open(encoding="utf-8") as handle:
            kinds = {
                row["id"]: tuple(row["perturbations"])
                for row in (json.loads(line) for line in handle if line.strip())
            }
        totals: dict[str, int] = {}
        correct: dict[str, int] = {}
        by_id = {row.id: row for row in noisy}
        for row_id, perturbations in kinds.items():
            report = evaluate([by_id[row_id]])
            for name in perturbations:
                totals[name] = totals.get(name, 0) + 1
                correct[name] = correct.get(name, 0) + report.intent_correct
        low = {
            name: round(correct[name] / totals[name], 4)
            for name in totals
            if correct[name] / totals[name] < PERTURBATION_ACCURACY_TARGET
        }
        assert low == {}, low

    def test_turkish_capitalisation_is_used(self) -> None:
        """A Turkish decoder writes ``İ``, not ``I`` -- the case ``tr_casefold`` exists for."""
        import random

        from tests.unit.nlu.perturb import capitalise

        assert capitalise("integralini al", random.Random(0)) in {
            "İntegralini al",
            "integralini Al",
        }
