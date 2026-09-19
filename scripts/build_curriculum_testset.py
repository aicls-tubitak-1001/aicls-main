"""Compile the İP-4 curriculum cards into ``problems.jsonl`` and ``FROZEN.sha256``.

Run from the repository root with the project venv::

    python scripts/build_curriculum_testset.py            # compile and write
    python scripts/build_curriculum_testset.py --check    # verify only, write nothing
    python scripts/build_curriculum_testset.py --sample 60 --check    # quick smoke run

What the compiler does with every card (docs/mimari/latex-ayristirici.md §7.2):

1. **schema** -- all fifteen fields present, id unique across every card file, topic/subtopic
   in the documented vocabulary, ``chapter_ref`` consistent with the topic, and the row also
   validates as an :class:`aicls.data.Problem` (the loader other packages read the set with);
2. **input** -- a ``latex`` input must parse (``parse_latex`` + ``to_sympy``) *and* survive the
   level-2 round trip; a ``text_tr`` input is a Turkish sentence and is only checked
   structurally, because ``aicls.nlu`` does not exist yet;
3. **answer** -- ``expected.srepr`` is rebuilt with SymPy and must print back to
   ``expected.latex`` exactly, so the two halves of the answer can never drift apart;
4. **CAS** -- the card's command is executed by :class:`aicls.mathengine.MathEngine` and the
   result is compared with the *authored* answer through the row's ``checker``. This is the
   only step that produces a number, and it is a real comparison: the cards were written from
   curriculum rules by ``datasets/curriculum/generate_cards.py``, never from the engine;
5. **quotas** -- the per-topic counts must meet the İP-4 quota table and the distribution must
   stay within :data:`SHARE_TOLERANCE` of the documented shares.

Only then are ``problems.jsonl`` (id-sorted, header row first) and ``FROZEN.sha256``
(``sha256_json`` per row) written.

**Provenance.** The compiled set is *authored inside the project*. It is a strong regression
and coverage instrument, but it is not an independent test set, and the 2027-05-01 progress
report must say so wherever it quotes a number measured here
(``datasets/curriculum/README.md``, KNOWN_ISSUES "Metric provenance").
"""

from __future__ import annotations

import argparse
import cmath
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import sympy
import yaml  # type: ignore[import-untyped]  # PyYAML ships no stubs and types-PyYAML is absent

from aicls.core.errors import AiclsError, InputValidationError
from aicls.core.hashing import sha256_json
from aicls.data import Problem
from aicls.latex2sympy import (
    latex_to_sympy,
    parse_latex,
    roundtrip_check,
    sympy_to_latex,
    to_sympy,
)
from aicls.mathengine import (
    SUPPORTED_OPS,
    MathEngine,
    Result,
    UnsupportedProblem,
    symbolic_equivalent,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CURRICULUM_DIR = REPO_ROOT / "datasets" / "curriculum"
CARDS_DIR = CURRICULUM_DIR / "cards"
PROBLEMS_PATH = CURRICULUM_DIR / "problems.jsonl"
FROZEN_PATH = CURRICULUM_DIR / "FROZEN.sha256"

#: Row schema version; bump when a field is added or removed.
SCHEMA_VERSION = "curriculum-row-1"

#: Dataset version; bump (and record in the README changelog) whenever a frozen row changes.
DATASET_VERSION = "curriculum-2026.09-v1"

#: Id of the provenance header, which is line 1 of ``problems.jsonl``.
HEADER_ID = "_header"

#: Topic of the header row, so a consumer can filter it out by topic as well as by id.
HEADER_TOPIC = "_meta"

#: İP-4 quota table (docs/mimari/latex-ayristirici.md §7.1); these are minimums.
TOPIC_QUOTAS: dict[str, int] = {
    "limit": 120,
    "sureklilik": 60,
    "turev": 200,
    "turev_uygulamalari": 120,
    "belirsiz_integral": 150,
    "belirli_integral": 120,
    "integral_uygulamalari": 80,
    "diziler_seriler": 80,
    "denklem_esitsizlik": 50,
    "grafik": 20,
}

#: The contractual size of the set (the quota table sums to this).
TOTAL_QUOTA = sum(TOPIC_QUOTAS.values())

#: How far a topic's share may drift from its documented share.
SHARE_TOLERANCE = 0.02

#: Chapter each topic belongs to (mirrors ``datasets/curriculum/generate_cards.py``).
CHAPTER_BY_TOPIC: dict[str, str] = {
    "limit": "aicls:bolum-1",
    "sureklilik": "aicls:bolum-1",
    "turev": "aicls:bolum-2",
    "turev_uygulamalari": "aicls:bolum-3",
    "belirsiz_integral": "aicls:bolum-4",
    "belirli_integral": "aicls:bolum-4",
    "integral_uygulamalari": "aicls:bolum-5",
    "diziler_seriler": "aicls:bolum-6",
    "denklem_esitsizlik": "aicls:bolum-0",
    "grafik": "aicls:bolum-0",
}

#: Comparison strategies, fixed by ``aicls.data.curriculum.CheckerName``.
CHECKERS: frozenset[str] = frozenset(
    {"symbolic", "numeric", "set", "bool", "exact_latex", "plot_ok"}
)

#: Every field a card must write out; a loader default may not stand in for one.
REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "topic",
    "subtopic",
    "chapter_ref",
    "difficulty",
    "input",
    "command",
    "expected",
    "checker",
    "checker_args",
    "source",
    "author",
    "reviewed_by",
    "status",
    "notes_tr",
)

#: Review states, fixed by ``aicls.data.curriculum.ProblemStatus``.
STATUSES: frozenset[str] = frozenset({"draft", "reviewed", "frozen"})

#: Default sampling grid for the ``numeric`` checker.
NUMERIC_SAMPLES = 20

#: Default absolute tolerance for the ``numeric`` checker.
NUMERIC_TOLERANCE = 1e-9

PROVENANCE_TR = (
    "KÖKEN: Bu 1000+ problemlik müfredat test seti proje içinde yazılmıştır; şablonlardan "
    "makine ile üretilmiş, bağımsız bir derlemden ÖRNEKLENMEMİŞTİR ve bir ders kitabından "
    "alıntı değildir. Beklenen yanıtlar müfredat kuralından türetilmiş, CAS'a sorulmamıştır; "
    "derleyici motorun yanıtını bu yanıtla karşılaştırır. Burada ölçülen CAS doğruluk oranı "
    "bir regresyon ölçüsüdür: ilerleme raporunda bu köken cümlesi ile birlikte verilmelidir."
)

PROVENANCE_EN = (
    "PROVENANCE: this curriculum set was authored inside the project (machine-expanded from "
    "hand-written templates). It is not an independent test set and no row is copied from a "
    "textbook. Expected answers come from the curriculum rule, not from the CAS; the compiler "
    "compares the engine against them. Any number measured here must be reported with this "
    "sentence."
)


@dataclass(frozen=True, slots=True)
class Card:
    """One validated card: where it came from and its raw mapping.

    Attributes:
        path: The YAML file the row was read from.
        row: The card mapping, exactly as it will be written to ``problems.jsonl``.
    """

    path: Path
    row: dict[str, Any]

    @property
    def id(self) -> str:
        """The card's stable identifier."""
        return str(self.row["id"])

    @property
    def topic(self) -> str:
        """The card's curriculum topic."""
        return str(self.row["topic"])


@dataclass(frozen=True, slots=True)
class Verdict:
    """What happened to one card when the engine ran it.

    Attributes:
        id: The card id.
        topic: Its topic.
        outcome: ``correct``, ``wrong``, ``error``, ``unsupported`` or ``abstain``.
        reason: Empty when correct; otherwise why it is not.
        parse_ok: Whether a LaTeX input parsed and lowered to SymPy.
        roundtrip_ok: Whether the level-2 round trip succeeded.
        is_latex: Whether the input was LaTeX (the parse-success denominator).
        elapsed_ms: Milliseconds the engine spent.
    """

    id: str
    topic: str
    outcome: str
    reason: str = ""
    parse_ok: bool = True
    roundtrip_ok: bool = True
    is_latex: bool = True
    elapsed_ms: float = 0.0


@dataclass
class Report:
    """The measurement produced by :func:`verify`."""

    verdicts: list[Verdict] = field(default_factory=list)

    @property
    def n(self) -> int:
        """Number of cards verified."""
        return len(self.verdicts)

    @property
    def correct(self) -> int:
        """Cards whose engine answer matched the authored answer."""
        return sum(1 for v in self.verdicts if v.outcome == "correct")

    @property
    def correct_rate(self) -> float:
        """Correct answers over all cards (the İP-4 CAS accuracy definition)."""
        return self.correct / self.n if self.n else 0.0

    @property
    def unsupported(self) -> int:
        """Cards the engine declined with ``UnsupportedProblem`` (reported separately)."""
        return sum(1 for v in self.verdicts if v.outcome == "unsupported")

    @property
    def abstained(self) -> int:
        """Cards where the equivalence check could not decide."""
        return sum(1 for v in self.verdicts if v.outcome == "abstain")

    @property
    def n_latex(self) -> int:
        """Cards whose input is LaTeX (the parse-success denominator)."""
        return sum(1 for v in self.verdicts if v.is_latex)

    @property
    def parse_success(self) -> float:
        """Parse + level-2 round-trip success over the LaTeX inputs (the strict definition)."""
        latex = [v for v in self.verdicts if v.is_latex]
        if not latex:
            return 0.0
        return sum(1 for v in latex if v.parse_ok and v.roundtrip_ok) / len(latex)

    @property
    def parse_only_success(self) -> float:
        """Parse + ``to_sympy`` success, without the round-trip requirement."""
        latex = [v for v in self.verdicts if v.is_latex]
        if not latex:
            return 0.0
        return sum(1 for v in latex if v.parse_ok) / len(latex)

    @property
    def failures(self) -> list[str]:
        """``"id: reason"`` for every card that is not ``correct``."""
        return [f"{v.id}: {v.outcome}: {v.reason}" for v in self.verdicts if v.outcome != "correct"]

    @property
    def roundtrip_failures(self) -> list[str]:
        """Ids whose LaTeX input parsed but did not survive the level-2 round trip."""
        return [v.id for v in self.verdicts if v.is_latex and v.parse_ok and not v.roundtrip_ok]

    @property
    def by_topic(self) -> dict[str, tuple[int, int]]:
        """``topic -> (correct, total)``."""
        out: dict[str, tuple[int, int]] = {}
        for verdict in self.verdicts:
            ok, total = out.get(verdict.topic, (0, 0))
            out[verdict.topic] = (ok + (verdict.outcome == "correct"), total + 1)
        return out

    @property
    def timing_ms(self) -> dict[str, float]:
        """Mean and p95 engine time in milliseconds."""
        samples = sorted(v.elapsed_ms for v in self.verdicts)
        if not samples:
            return {"mean": 0.0, "p95": 0.0}
        index = min(len(samples) - 1, round(0.95 * (len(samples) - 1)))
        return {"mean": round(statistics.fmean(samples), 3), "p95": round(samples[index], 3)}


# --------------------------------------------------------------------------------------------
# loading and schema validation
# --------------------------------------------------------------------------------------------
def _reject(reason: str, **details: Any) -> InputValidationError:
    """Build the single error type the compiler raises."""
    return InputValidationError(
        f"Müfredat kartı geçersiz: {reason}", details={"reason": reason, **details}
    )


def validate_row(row: Any, path: Path, seen: set[str]) -> dict[str, Any]:
    """Validate one raw card mapping and return it unchanged.

    Args:
        row: The mapping read from YAML.
        path: The card file, for error messages.
        seen: Ids already used by earlier cards; updated in place.

    Returns:
        The same mapping, once it is known to be a well-formed card.

    Raises:
        aicls.core.errors.InputValidationError: On any schema violation.
    """
    if not isinstance(row, dict):
        raise _reject("row_not_a_mapping", path=str(path))
    missing = [name for name in REQUIRED_FIELDS if name not in row]
    if missing:
        raise _reject("missing_fields", path=str(path), id=row.get("id"), fields=missing)
    extra = sorted(set(row) - set(REQUIRED_FIELDS))
    if extra:
        raise _reject("unknown_fields", path=str(path), id=row.get("id"), fields=extra)

    row_id = str(row["id"])
    if row_id in seen:
        raise _reject("duplicate_id", path=str(path), id=row_id)
    seen.add(row_id)

    topic = str(row["topic"])
    if topic not in TOPIC_QUOTAS:
        raise _reject("unknown_topic", path=str(path), id=row_id, topic=topic)
    if not row_id.startswith(f"{topic}-"):
        raise _reject("id_topic_mismatch", path=str(path), id=row_id, topic=topic)
    if row["chapter_ref"] != CHAPTER_BY_TOPIC[topic]:
        raise _reject("chapter_mismatch", path=str(path), id=row_id)
    if not (isinstance(row["difficulty"], int) and 1 <= row["difficulty"] <= 3):
        raise _reject("difficulty_out_of_range", path=str(path), id=row_id)
    if row["checker"] not in CHECKERS:
        raise _reject("unknown_checker", path=str(path), id=row_id, checker=row["checker"])
    if row["status"] not in STATUSES:
        raise _reject("unknown_status", path=str(path), id=row_id, status=row["status"])
    if row["command"].get("op") not in SUPPORTED_OPS:
        raise _reject("unknown_op", path=str(path), id=row_id, op=row["command"].get("op"))
    for name in ("subtopic", "source", "author", "notes_tr"):
        if not str(row[name]).strip():
            raise _reject("empty_field", path=str(path), id=row_id, field=name)
    if not str(row["input"].get("value", "")).strip():
        raise _reject("empty_input", path=str(path), id=row_id)
    if row["input"].get("kind") not in {"latex", "text_tr"}:
        raise _reject("unknown_input_kind", path=str(path), id=row_id)
    if not str(row["expected"].get("srepr", "")).strip():
        raise _reject("empty_srepr", path=str(path), id=row_id)

    try:
        Problem.model_validate(row)
    except ValueError as exc:  # pragma: no cover - the checks above catch this first
        raise _reject("not_a_problem", path=str(path), id=row_id, error=str(exc)[:200]) from exc
    return row


def load_cards(cards_dir: Path = CARDS_DIR) -> list[Card]:
    """Read and validate every ``cards/*.yaml`` file, in id order.

    Args:
        cards_dir: Directory holding the topic card files.

    Returns:
        Every card, sorted by id (the compiled file's order).

    Raises:
        aicls.core.errors.InputValidationError: If a file is not a YAML list or a row is
            invalid, an id repeats, or the directory holds no cards.
    """
    paths = sorted(Path(cards_dir).glob("*.yaml"))
    if not paths:
        raise _reject("no_card_files", path=str(cards_dir))
    cards: list[Card] = []
    seen: set[str] = set()
    for path in paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise _reject("card_file_not_a_list", path=str(path))
        for row in payload:
            cards.append(Card(path=path, row=validate_row(row, path, seen)))
    cards.sort(key=lambda card: card.id)
    return cards


def topic_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count rows per topic, ignoring the provenance header."""
    counter = Counter(str(row["topic"]) for row in rows if row["id"] != HEADER_ID)
    return {topic: counter.get(topic, 0) for topic in TOPIC_QUOTAS}


def check_quotas(counts: dict[str, int]) -> list[str]:
    """Return one message per topic that misses its quota (empty when all are met)."""
    problems: list[str] = []
    for topic, quota in TOPIC_QUOTAS.items():
        have = counts.get(topic, 0)
        if have < quota:
            problems.append(f"{topic}: {have} < {quota}")
    total = sum(counts.get(topic, 0) for topic in TOPIC_QUOTAS)
    if total < TOTAL_QUOTA:
        problems.append(f"total: {total} < {TOTAL_QUOTA}")
    return problems


# --------------------------------------------------------------------------------------------
# the answer: srepr <-> latex consistency
# --------------------------------------------------------------------------------------------
def rebuild(srepr: str) -> Any:
    """Rebuild a SymPy object from an ``srepr`` string.

    Raises:
        aicls.core.errors.InputValidationError: If SymPy cannot rebuild it.
    """
    try:
        return sympy.sympify(srepr)
    except (SyntaxError, TypeError, ValueError, AttributeError) as exc:
        raise _reject("srepr_not_rebuildable", srepr=srepr[:120], error=str(exc)[:200]) from exc


def check_expected(card: Card) -> str:
    """Check that ``expected.srepr`` prints back to ``expected.latex``.

    Args:
        card: The card to check.

    Returns:
        An empty string when the two halves agree, otherwise the problem description.
    """
    try:
        value = rebuild(str(card.row["expected"]["srepr"]))
    except InputValidationError as exc:
        return f"{card.id}: {exc.details.get('reason')}"
    printed = sympy_to_latex(value)
    declared = str(card.row["expected"]["latex"])
    if printed != declared:
        return f"{card.id}: expected.latex {declared!r} != canonical {printed!r}"
    return ""


# --------------------------------------------------------------------------------------------
# checkers
# --------------------------------------------------------------------------------------------
def _symbol(args: dict[str, Any]) -> sympy.Symbol:
    """The variable a checker differentiates or samples over."""
    return sympy.Symbol(str(args.get("var", "x")), real=True)


def compare_latex(actual: str, expected: str) -> tuple[bool, str]:
    """``exact_latex``: the rendered answer must be the declared string, character for character."""
    if actual == expected:
        return True, ""
    return False, f"latex {actual!r} != {expected!r}"


def _compare_numeric(actual: Any, expected: Any, args: dict[str, Any]) -> tuple[bool, str]:
    """Sample the two answers on a deterministic grid inside ``domain``."""
    tolerance = float(args.get("tolerance", NUMERIC_TOLERANCE))
    lower, upper = (float(bound) for bound in args.get("domain", (-3.0, 3.0)))
    variable = _symbol(args)
    free = (
        (actual.free_symbols | expected.free_symbols) if hasattr(actual, "free_symbols") else set()
    )
    if not free:
        try:
            difference = complex(sympy.N(actual - expected, 25))
        except (TypeError, ValueError, ZeroDivisionError, AttributeError) as exc:
            return False, f"numeric evaluation failed: {type(exc).__name__}: {exc}"[:160]
        if cmath.isnan(difference) or cmath.isinf(difference):
            # ``zoo``/``nan``/``oo``: the metric abstains rather than claiming a difference.
            return False, "numeric checker could not evaluate the difference"
        gap = abs(difference)
        return (gap <= tolerance, "" if gap <= tolerance else f"numeric gap {gap:.3e}")
    evaluated = 0
    for index in range(NUMERIC_SAMPLES):
        point = lower + (upper - lower) * index / (NUMERIC_SAMPLES - 1)
        substitution = {variable: sympy.Rational(point).limit_denominator(1000)}
        try:
            left = complex(sympy.N((actual - expected).subs(substitution), 25))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if cmath.isnan(left) or cmath.isinf(left):  # a pole or a NaN: this probe abstains
            continue
        evaluated += 1
        if abs(left) > tolerance:
            return False, f"numeric gap {abs(left):.3e} at {point:.3f}"
    if evaluated == 0:
        return False, "numeric checker could not evaluate any probe"
    return True, ""


def compare(checker: str, actual: Any, expected: Any, args: dict[str, Any]) -> tuple[bool, str]:
    """Compare an engine answer with the authored answer under one strategy.

    Args:
        checker: ``symbolic``, ``numeric``, ``set``, ``bool`` or ``plot_ok``. ``exact_latex``
            works on strings and lives in :func:`compare_latex`.
        actual: What the engine produced (a SymPy object).
        expected: What the card says the answer is (a SymPy object).
        args: The card's ``checker_args``.

    Returns:
        ``(ok, reason)``; ``reason`` is empty when ``ok`` is True.
    """
    if actual == expected:
        # Identity first: ``symbolic_equivalent(oo, oo)`` abstains, and an exact match never
        # needs simplification anyway.
        return True, ""
    if checker == "bool":
        return False, f"bool {actual} != {expected}"
    if checker == "numeric":
        return _compare_numeric(actual, expected, args)
    if checker == "symbolic" and args.get("up_to_constant"):
        variable = _symbol(args)
        verdict = symbolic_equivalent(sympy.diff(actual, variable), sympy.diff(expected, variable))
        if verdict == "equal":
            return True, ""
        return False, f"antiderivatives differ by more than a constant ({verdict})"
    verdict = symbolic_equivalent(actual, expected)
    if verdict == "equal":
        return True, ""
    return False, f"{checker}: {verdict}"


def _check_antiderivative(card: Card, actual: Any) -> str:
    """For an indefinite integral, check that the answer really differentiates to the integrand."""
    variable = _symbol(card.row["checker_args"])
    try:
        integrand = latex_to_sympy(str(card.row["command"]["expr"]))
    except AiclsError as exc:  # pragma: no cover - the input check catches this first
        return f"integrand did not parse: {type(exc).__name__}: {exc}"[:160]
    verdict = symbolic_equivalent(sympy.diff(actual, variable), integrand)
    if verdict == "equal":
        return ""
    return f"d/d{variable} of the answer is not the integrand ({verdict})"


# --------------------------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------------------------
def _check_input(card: Card) -> tuple[bool, bool, str]:
    """Parse and round-trip a card's input; ``text_tr`` rows are only checked structurally."""
    row = card.row
    value = str(row["input"]["value"])
    if row["input"]["kind"] != "latex":
        return True, True, ""
    try:
        to_sympy(parse_latex(value))
    except AiclsError as exc:
        return False, False, f"parse: {type(exc).__name__}: {exc}"[:160]
    check = roundtrip_check(value, numeric_samples=3)
    if not check.ok:
        return True, False, f"roundtrip: {check.reason}"[:160]
    return True, True, ""


def _outcome_for_result(card: Card, result: Result) -> tuple[str, str]:
    """Compare one engine result with the card's authored answer."""
    row = card.row
    checker = str(row["checker"])
    args = dict(row["checker_args"])
    if checker == "exact_latex":
        ok, reason = compare_latex(result.latex, str(row["expected"]["latex"]))
        return ("correct" if ok else "wrong"), reason
    expected = rebuild(str(row["expected"]["srepr"]))
    actual = rebuild(result.srepr)
    if checker == "plot_ok":
        if result.kind != "plot" or not result.plot_png:
            return "wrong", f"no PNG produced (kind={result.kind})"
        ok, reason = compare("symbolic", actual, expected, args)
        return ("correct" if ok else "wrong"), reason
    ok, reason = compare(checker, actual, expected, args)
    if not ok:
        return "wrong", reason
    if checker == "symbolic" and args.get("up_to_constant"):
        problem = _check_antiderivative(card, actual)
        if problem:
            return "wrong", problem
    return "correct", ""


def verify_card(card: Card, engine: MathEngine) -> Verdict:
    """Run one card end to end and judge the answer.

    Args:
        card: The validated card.
        engine: The engine to execute the card's command on.

    Returns:
        The :class:`Verdict` for this card.
    """
    is_latex = card.row["input"]["kind"] == "latex"
    parse_ok, roundtrip_ok, input_problem = _check_input(card)
    consistency = check_expected(card)
    if consistency:
        return Verdict(card.id, card.topic, "wrong", consistency, parse_ok, roundtrip_ok, is_latex)
    if not parse_ok:
        # The platform could not even read the problem statement: an end-to-end failure.
        return Verdict(
            card.id, card.topic, "error", input_problem, parse_ok, roundtrip_ok, is_latex
        )
    try:
        result = engine.execute(dict(card.row["command"]))
    except AiclsError as exc:
        outcome = "unsupported" if isinstance(exc, UnsupportedProblem) else "error"
        return Verdict(
            card.id,
            card.topic,
            outcome,
            f"{type(exc).__name__}: {exc.details.get('reason', '')}",
            parse_ok,
            roundtrip_ok,
            is_latex,
        )
    # A round-trip mismatch is a *printer* observation, not a wrong answer: it is counted in
    # ``parse_success`` (the strict İP-4 definition) and reported separately, never folded into
    # the CAS correctness rate.
    outcome, reason = _outcome_for_result(card, result)
    return Verdict(
        card.id,
        card.topic,
        outcome,
        reason,
        parse_ok,
        roundtrip_ok,
        is_latex,
        result.elapsed_ms,
    )


def verify(cards: list[Card], *, engine: MathEngine | None = None) -> Report:
    """Verify every card against the engine.

    Args:
        cards: The cards to run.
        engine: The engine to use; a fresh in-process engine by default.

    Returns:
        The :class:`Report`.
    """
    runner = engine if engine is not None else MathEngine(timeout_s=30.0)
    return Report([verify_card(card, runner) for card in cards])


# --------------------------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------------------------
def build_header(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The provenance record written as line 1 of ``problems.jsonl``.

    It is a valid (and trivially correct) :class:`aicls.data.Problem` so that a consumer that
    forgets to filter it cannot be poisoned by it; ``_header`` and ``topic == "_meta"`` are the
    two ways to recognise it.
    """
    counts = topic_counts(rows)
    return {
        "id": HEADER_ID,
        "_header": True,
        "topic": HEADER_TOPIC,
        "subtopic": "provenance",
        "chapter_ref": "",
        "difficulty": 1,
        "input": {"kind": "text_tr", "value": PROVENANCE_TR},
        "command": {"op": "simplify", "expr": "1"},
        "expected": {"latex": "1", "srepr": "Integer(1)"},
        "checker": "symbolic",
        "checker_args": {},
        "source": "scripts/build_curriculum_testset.py",
        "author": "generator:datasets/curriculum/generate_cards.py",
        "reviewed_by": "",
        "status": "frozen",
        "notes_tr": PROVENANCE_TR,
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "provenance_tr": PROVENANCE_TR,
        "provenance_en": PROVENANCE_EN,
        "n_problems": len(rows),
        "quotas": dict(TOPIC_QUOTAS),
        "counts": counts,
        "sympy_version": sympy.__version__,
        "compiler": "scripts/build_curriculum_testset.py",
        "generator": "datasets/curriculum/generate_cards.py",
    }


def write_problems(rows: list[dict[str, Any]], path: Path = PROBLEMS_PATH) -> Path:
    """Write the header plus every row as JSONL, id-sorted, UTF-8 and LF."""
    ordered = sorted(rows, key=lambda row: str(row["id"]))
    lines = [build_header(ordered), *ordered]
    body = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in lines
    )
    path.write_text(body, encoding="utf-8", newline="\n")
    return path


def write_frozen(path: Path = FROZEN_PATH, problems_path: Path = PROBLEMS_PATH) -> Path:
    """Write a ``sha256_json`` digest for every compiled row, id-sorted.

    ``datasets/curriculum/README.md`` documents the rule: v1 freezes *all* rows, not only the
    ones whose ``status`` is ``frozen``, so that a silent change to any generated row is caught.
    """
    rows = read_problems(problems_path)
    lines = [
        f"# {path.name} -- {DATASET_VERSION} ({SCHEMA_VERSION})",
        "# scripts/build_curriculum_testset.py tarafından üretilir; elle düzenlemeyin.",
        "# Her satır: <sha256(canonical_json(row))>  <row id>. Dondurulmuş satır değişmez;",
        "# bir satır düzeltilecekse DATASET_VERSION yükseltilir ve README değişiklik",
        "# günlüğüne yazılır.",
        f"# {PROVENANCE_TR}",
    ]
    lines += [f"{sha256_json(row)}  {row['id']}" for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def read_problems(path: Path = PROBLEMS_PATH) -> list[dict[str, Any]]:
    """Read ``problems.jsonl`` (header row included) in file order."""
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def read_frozen(path: Path = FROZEN_PATH) -> dict[str, str]:
    """Read ``FROZEN.sha256`` into ``{row id: hex digest}``."""
    frozen: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, row_id = line.split(maxsplit=1)
        frozen[row_id.strip()] = digest.removeprefix("sha256:")
    return frozen


def report_payload(report: Report, counts: dict[str, int]) -> dict[str, Any]:
    """The machine-readable summary printed by ``--report-json`` (İP-4 analysis report input)."""
    return {
        "dataset_version": DATASET_VERSION,
        "schema_version": SCHEMA_VERSION,
        "n": report.n,
        "correct": report.correct,
        "correct_rate": round(report.correct_rate, 6),
        "unsupported": report.unsupported,
        "abstained": report.abstained,
        "parse_success": round(report.parse_success, 6),
        "parse_only_success": round(report.parse_only_success, 6),
        "n_latex": report.n_latex,
        "roundtrip_failures": report.roundtrip_failures[:50],
        "counts": counts,
        "by_topic": {topic: list(value) for topic, value in report.by_topic.items()},
        "timing_ms": report.timing_ms,
        "failures": report.failures[:50],
        "provenance_tr": PROVENANCE_TR,
        "provenance_en": PROVENANCE_EN,
        "sympy_version": sympy.__version__,
    }


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cards-dir", type=Path, default=CARDS_DIR)
    parser.add_argument("--out", type=Path, default=PROBLEMS_PATH)
    parser.add_argument("--frozen", type=Path, default=FROZEN_PATH)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--sample", type=int, default=0, help="verify only every Nth card")
    parser.add_argument("--check", action="store_true", help="verify only; write nothing")
    parser.add_argument("--skip-verify", action="store_true", help="write without running the CAS")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Compile, verify and freeze the curriculum set; exit non-zero on any problem."""
    args = _parse_args(argv)
    cards = load_cards(args.cards_dir)
    rows = [card.row for card in cards]
    counts = topic_counts(rows)

    quota_problems = check_quotas(counts)
    total = sum(counts.values())
    for topic, quota in TOPIC_QUOTAS.items():
        share = counts[topic] / total if total else 0.0
        if abs(share - quota / TOTAL_QUOTA) > SHARE_TOLERANCE:
            quota_problems.append(f"{topic}: share {share:.4f} off the quota table")

    consistency = [message for card in cards if (message := check_expected(card))]

    report = Report()
    if not args.skip_verify:
        selected = cards[:: args.sample] if args.sample > 1 else cards
        report = verify(selected, engine=MathEngine(timeout_s=args.timeout))

    print(f"kart sayısı     : {len(cards)} ({total} problem, kota {TOTAL_QUOTA})")
    for topic, quota in TOPIC_QUOTAS.items():
        ok, seen = report.by_topic.get(topic, (0, 0))
        rate = f"{ok}/{seen}" if seen else "-"
        print(f"  {topic:24s} {counts[topic]:5d}  (kota {quota:4d})  doğru {rate}")
    parse_problems: list[str] = []
    if report.n:
        print(f"CAS doğruluğu   : {report.correct_rate:.4f} ({report.correct}/{report.n})")
        print(
            f"ayrıştırma      : {report.parse_success:.4f} (sıkı)  "
            f"{report.parse_only_success:.4f} (yalnız parse), n={report.n_latex}"
        )
        print(f"desteklenmeyen  : {report.unsupported}   karar veremeyen: {report.abstained}")
        print(f"gidiş-dönüş     : {len(report.roundtrip_failures)} satır eşleşmedi")
        print(f"süre (ms)       : {report.timing_ms}")
        if report.parse_success <= 0.99:
            parse_problems.append(f"parse_success {report.parse_success:.4f} <= 0.99")

    problems = quota_problems + consistency + report.failures + parse_problems
    if problems:
        print(f"SORUN ({len(problems)}):", file=sys.stderr)
        for message in problems[:40]:
            print(f"  {message}", file=sys.stderr)
        return 1

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report_payload(report, counts), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"rapor           : {args.report_json}")

    if not args.check:
        write_problems(rows, args.out)
        write_frozen(args.frozen, args.out)
        print(f"yazıldı         : {args.out} , {args.frozen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
