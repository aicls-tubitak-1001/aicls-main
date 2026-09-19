"""Gold-set evaluation helpers for :mod:`aicls.nlu` (test-support, not library code).

Definitions used by ``test_nlu_command_test_set_intent_accuracy_above_0_95`` and by the report
tables. They are stated here rather than in a docstring somewhere else because the numbers are
quoted in TÜBİTAK progress reports and must mean exactly one thing:

**intent_accuracy** -- a row is correct when the resolved intent equals the gold intent. A row
whose gold intent is the empty string is an *out-of-scope* prompt: it is correct only when the
layer returns :class:`~aicls.nlu.types.NoMatch`. Clarifications count as wrong for in-scope rows
(they are not the requested command) and as wrong for out-of-scope rows (they still offer one).

**slot_f1** -- computed over the *union* of the slots the gold row declares and the slots the
layer resolved. For each declared ``(name, value)``: a matching resolved value is a true
positive, a differing one is both a false positive and a false negative, and a missing one is a
false negative. **A slot the layer resolved but the gold row does not declare is a false
positive** (:data:`STRICT_UNDECLARED_IS_FALSE_POSITIVE`).

That last rule is new, and it exists because the old one hid a wrong-answer bug. Under the old
definition an undeclared slot was simply ignored, so ``differentiate`` rows that declared only
``expr`` said nothing at all about ``order`` -- and ``"x üzeri üçün türevini al"`` could resolve
``order=3``, compute the third derivative of ``x³`` and still score ``slot_f1 = 1.0000``
(KNOWN_ISSUES PRIORITY 1). A gold row is now an assertion about the *whole* command: a slot the
row does not mention is a slot the layer must not fill beyond nothing.

The price is that the metric also measures annotation completeness, which is the honest trade:
an under-annotated gold row is now a *wrong* gold row and has to be fixed in the corpus rather
than forgiven by the metric.

Provenance is the caller's job to report: ``datasets/nlu/commands_test.jsonl`` was authored by
the project team, so these numbers are a *regression* measurement, not evidence from an
independent test set (KNOWN_ISSUES.md, "Metric provenance").
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from aicls.nlu import Clarification, Matched, ParseContext, parse_command
from aicls.nlu.inventory.schema import Inventory


@dataclass(frozen=True, slots=True)
class GoldRow:
    """One row of ``commands_test.jsonl`` / ``asr_noisy_tr.jsonl``."""

    id: str
    text: str
    intent: str
    slots: dict[str, Any] = field(default_factory=dict)
    context: dict[str, str] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    @property
    def out_of_scope(self) -> bool:
        """Whether the correct answer is "no command at all"."""
        return not self.intent


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Aggregate metrics over a gold set."""

    total: int
    intent_correct: int
    slot_tp: int
    slot_fp: int
    slot_fn: int
    matched: int
    clarifications: int
    no_matches: int
    errors: tuple[tuple[str, str, str], ...] = ()

    @property
    def intent_accuracy(self) -> float:
        """Fraction of rows whose intent was resolved correctly."""
        return self.intent_correct / self.total if self.total else 0.0

    @property
    def slot_precision(self) -> float:
        """Precision over declared gold slots."""
        denominator = self.slot_tp + self.slot_fp
        return self.slot_tp / denominator if denominator else 1.0

    @property
    def slot_recall(self) -> float:
        """Recall over declared gold slots."""
        denominator = self.slot_tp + self.slot_fn
        return self.slot_tp / denominator if denominator else 1.0

    @property
    def slot_f1(self) -> float:
        """Harmonic mean of :attr:`slot_precision` and :attr:`slot_recall`."""
        precision, recall = self.slot_precision, self.slot_recall
        if precision + recall == 0.0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @property
    def clarification_rate(self) -> float:
        """Fraction of rows that ended in a clarification question."""
        return self.clarifications / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, float | int]:
        """Report-ready mapping (the shape ``reports/metrics/nlu_commands_*.json`` uses)."""
        return {
            "rows": self.total,
            "intent_accuracy": round(self.intent_accuracy, 4),
            "slot_precision": round(self.slot_precision, 4),
            "slot_recall": round(self.slot_recall, 4),
            "slot_f1": round(self.slot_f1, 4),
            "clarification_rate": round(self.clarification_rate, 4),
            "matched": self.matched,
            "clarifications": self.clarifications,
            "no_matches": self.no_matches,
        }


def load_gold(path: Path) -> list[GoldRow]:
    """Read a JSONL gold file into :class:`GoldRow` records."""
    rows: list[GoldRow] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            rows.append(
                GoldRow(
                    id=str(data["id"]),
                    text=str(data["text"]),
                    intent=str(data.get("intent", "")),
                    slots=dict(data.get("slots") or {}),
                    context=dict(data.get("context") or {}),
                    tags=tuple(data.get("tags") or ()),
                )
            )
    return rows


#: Whether a resolved-but-undeclared slot counts as a false positive. Kept as a named constant
#: so a report can state which definition produced a number instead of guessing.
STRICT_UNDECLARED_IS_FALSE_POSITIVE = True


def _slot_counts(gold: Mapping[str, Any], predicted: Mapping[str, Any]) -> tuple[int, int, int]:
    """``(tp, fp, fn)`` for one row under the strict definition documented above."""
    true_positive = false_positive = false_negative = 0
    for name, value in gold.items():
        if name not in predicted:
            false_negative += 1
        elif predicted[name] == value:
            true_positive += 1
        else:
            false_positive += 1
            false_negative += 1
    if STRICT_UNDECLARED_IS_FALSE_POSITIVE:
        false_positive += sum(1 for name in predicted if name not in gold)
    return true_positive, false_positive, false_negative


def evaluate(
    rows: Iterable[GoldRow], *, inventory: Inventory | None = None, max_errors: int = 25
) -> EvalReport:
    """Run ``parse_command`` over ``rows`` and aggregate the metrics.

    Args:
        rows: The gold rows.
        inventory: Inventory to resolve against; the packaged one when omitted.
        max_errors: How many mistakes to keep in :attr:`EvalReport.errors`.

    Returns:
        The :class:`EvalReport`.
    """
    total = correct = matched = clarifications = no_matches = 0
    tp = fp = fn = 0
    errors: list[tuple[str, str, str]] = []
    for row in rows:
        total += 1
        context = ParseContext(**row.context) if row.context else None
        outcome = parse_command(row.text, context=context, inventory=inventory)
        if isinstance(outcome, Matched):
            matched += 1
            predicted = outcome.intent
        elif isinstance(outcome, Clarification):
            clarifications += 1
            predicted = f"<clarification:{outcome.intent or 'ambiguous'}>"
        else:
            no_matches += 1
            predicted = ""
        if predicted == row.intent:
            correct += 1
        elif len(errors) < max_errors:
            errors.append((row.id, row.text, predicted))
        if row.slots and isinstance(outcome, Matched):
            row_tp, row_fp, row_fn = _slot_counts(row.slots, outcome.slots)
            tp += row_tp
            fp += row_fp
            fn += row_fn
        elif row.slots:
            fn += len(row.slots)
    return EvalReport(
        total=total,
        intent_correct=correct,
        slot_tp=tp,
        slot_fp=fp,
        slot_fn=fn,
        matched=matched,
        clarifications=clarifications,
        no_matches=no_matches,
        errors=tuple(errors),
    )


def filter_rows(rows: Sequence[GoldRow], *, tag: str | None = None) -> list[GoldRow]:
    """Return the rows carrying ``tag`` (or all of them when ``tag`` is ``None``)."""
    if tag is None:
        return list(rows)
    return [row for row in rows if tag in row.tags]


__all__ = [
    "STRICT_UNDECLARED_IS_FALSE_POSITIVE",
    "EvalReport",
    "GoldRow",
    "evaluate",
    "filter_rows",
    "load_gold",
]
