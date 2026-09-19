"""End-to-end contract for the deterministic pipeline, assembled from every wave so far.

This is the test that would have caught the wave-3 integration risk that no module owner could
see alone: each package tests its own fixtures, but nobody drives
``Turkish text -> aicls.nlu -> aicls.mathengine -> LaTeX`` in one process. The three legs the
proposal's pipeline diagram promises are pinned here:

* **voice/text leg** -- ``parse_command`` builds a typed
  :class:`~aicls.mathengine.models.CasCommand` and SymPy renders the answer. The expected LaTeX
  is written out literally, because the contractual claim of the project is that a *result* is
  computed, never generated; a change in what the user is shown must fail a test, not pass
  silently.
* **handwriting leg** -- ``aicls.ocr`` (stub recogniser, no checkpoint) -> ``aicls.latex2sympy``
  -> ``aicls.mathengine``, including the ``\\frac{\\partial f}{\\partial x}`` form that
  ``KNOWN_ISSUES.md`` recorded as an end-to-end failure path.
* **book leg (İP-8)** -- a ``book_fetch`` intent never reaches the CAS; it is returned as a
  :class:`~aicls.nlu.types.BookFetch` command and executed by ``aicls.book`` against a migrated
  database, which is exactly the ADR-0002 boundary that forbids ``aicls.nlu`` from importing
  ``aicls.book``.

The "guide, do not fail" paths are pinned too: an under-specified utterance must come back as a
Turkish :class:`~aicls.nlu.types.Clarification` and an unknown one as a
:class:`~aicls.nlu.types.NoMatch` with suggestions -- never as an exception and never as an
invented answer.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from aicls.book import BookResolver, load_units
from aicls.book.ingest.validators import require_full_coverage
from aicls.core.errors import AiclsError
from aicls.db import BookRepository, make_engine, session_scope, upgrade_head
from aicls.latex2sympy import latex_to_sympy, parse_latex, to_latex
from aicls.mathengine import (
    CAS_COMMAND_TYPES,
    Diff,
    InProcessExecutor,
    MathEngine,
    Result,
)
from aicls.nlu import (
    BookFetch,
    Clarification,
    Matched,
    NoMatch,
    ParseContext,
    parse_command,
)
from aicls.nlu.types import Command
from aicls.ocr import Ink, Stroke, StubRecognizer
from tests.unit.book.sample_set import tag_markdown_samples

pytestmark = [pytest.mark.integration, pytest.mark.e2e]


@pytest.fixture(scope="module")
def engine() -> Iterator[MathEngine]:
    """One in-process CAS for the whole module (no worker pool: this is a wiring test)."""
    with InProcessExecutor() as executor:
        yield MathEngine(executor)


def _run(engine: MathEngine, command: Command) -> Result:
    """Execute a command that the NLU layer decided belongs to the CAS."""
    assert not isinstance(command, BookFetch), "book intents must not reach the CAS"
    return engine.execute(command)  # type: ignore[arg-type]  # UiAction excluded by the callers


# --------------------------------------------------------------------------------------------
# Leg 1: Turkish text -> nlu -> mathengine -> LaTeX
# --------------------------------------------------------------------------------------------

#: ``(utterance, intent, op, result LaTeX, result kind)``. The LaTeX column is the string the
#: web UI hands to KaTeX, captured from a real run -- see the module docstring on why it is
#: spelled out rather than recomputed.
SPOKEN_CASES: tuple[tuple[str, str, str, str, str], ...] = (
    ("x karenin türevini al", "differentiate", "diff", r"2 \cdot x", "expr"),
    (
        "sıfırdan bire x kare integralini hesapla",
        "integrate_definite",
        "integrate",
        r"\frac{1}{3}",
        "number",
    ),
    ("x sonsuza giderken bir bölü x limitini al", "limit", "limit", "0", "number"),
    (
        "x kare artı iki x ifadesini sadeleştir",
        "simplify",
        "simplify",
        r"x \cdot \left(x + 2\right)",
        "expr",
    ),
    (
        "x kare eksi dört eşittir sıfır denklemini çöz",
        "solve",
        "solve",
        r"\left\{-2, 2\right\}",
        "set",
    ),
)


@pytest.mark.parametrize(("text", "intent", "op", "latex", "kind"), SPOKEN_CASES)
def test_turkish_sentence_reaches_sympy_and_renders_the_expected_latex(
    engine: MathEngine, text: str, intent: str, op: str, latex: str, kind: str
) -> None:
    """A spoken-style Turkish command becomes a typed CAS command and a rendered result."""
    outcome = parse_command(text)
    assert isinstance(outcome, Matched), f"{text!r} did not match: {outcome}"
    assert outcome.intent == intent
    assert outcome.command.op == op
    assert outcome.trace, "a matched command must explain itself for the UI panel"

    result = _run(engine, outcome.command)
    assert result.latex == latex
    assert str(result.kind) == kind
    assert result.srepr, "eval compares answers by srepr; it must never be empty"


def test_definite_integral_slots_carry_the_spoken_bounds() -> None:
    """ "sıfırdan bire" must land in the bounds, not merely produce a right-looking answer."""
    outcome = parse_command("sıfırdan bire x kare integralini hesapla")
    assert isinstance(outcome, Matched)
    assert outcome.slots["lower"] == "0"
    assert outcome.slots["upper"] == "1"
    assert outcome.slots["expr"] == "x^{2}"


def test_limit_at_infinity_sets_the_point_and_a_two_sided_direction() -> None:
    """The limit point is the parsed ``\\infty`` symbol and the default direction is two-sided."""
    outcome = parse_command("x sonsuza giderken bir bölü x limitini al")
    assert isinstance(outcome, Matched)
    assert outcome.command.op == "limit"
    assert outcome.slots["point"] == r"\infty"
    assert outcome.slots["direction"] == "+-"
    assert outcome.slots["expr"] == r"\frac{1}{x}"


def test_one_sided_limit_keeps_the_spoken_direction(engine: MathEngine) -> None:
    """ "sağdan yaklaşırken" must survive all the way into the SymPy call."""
    outcome = parse_command("x sıfıra sağdan yaklaşırken bir bölü x limitini al")
    assert isinstance(outcome, Matched)
    assert outcome.slots["direction"] == "+"
    assert _run(engine, outcome.command).latex == r"\infty"


def test_follow_up_command_uses_the_expression_already_on_the_board(engine: MathEngine) -> None:
    """ "türevini al" with no expression resolves ``expr`` from the session context."""
    outcome = parse_command("türevini al", context=ParseContext(last_latex="x^{3}"))
    assert isinstance(outcome, Matched)
    assert outcome.command.op == "diff"
    assert outcome.slots["expr"] == "x^{3}"
    assert _run(engine, outcome.command).latex == r"3 \cdot x^{2}"


def test_chained_commands_feed_the_previous_result_back_in(engine: MathEngine) -> None:
    """The classroom loop: differentiate, then differentiate the answer again."""
    first = parse_command("x küpünün türevini al")
    assert isinstance(first, Matched)
    assert first.slots["expr"] == "x^{3}"
    once = _run(engine, first.command)
    assert once.latex == r"3 \cdot x^{2}"

    second = parse_command("türevini al", context=ParseContext(last_latex=once.latex))
    assert isinstance(second, Matched)
    twice = _run(engine, second.command)
    assert twice.plain == "6*x"


# --------------------------------------------------------------------------------------------
# "Guide, do not fail" -- the İP-4 requirement that nothing is ever invented
# --------------------------------------------------------------------------------------------


def test_missing_expression_asks_a_turkish_question_instead_of_guessing() -> None:
    """Without context, "türevini al" must ask which expression, not pick one."""
    outcome = parse_command("türevini al")
    assert isinstance(outcome, Clarification)
    assert outcome.missing_slot == "expr"
    assert outcome.question_tr.strip().endswith("?")


@pytest.mark.parametrize("text", ["bugün hava nasıl", "zamazingo yap", ""])
def test_unknown_utterances_return_suggestions_never_an_exception(text: str) -> None:
    """An utterance outside the inventory is a :class:`NoMatch` carrying Turkish examples."""
    outcome = parse_command(text)
    assert isinstance(outcome, NoMatch)
    assert outcome.reason in {"no_intent", "below_min_score", "command_invalid", "empty_input"}
    if outcome.reason != "empty_input":
        assert outcome.suggestions_tr, "a no-match must still guide the user"


# --------------------------------------------------------------------------------------------
# Leg 2: handwriting -> ocr -> latex2sympy -> mathengine
# --------------------------------------------------------------------------------------------


def _ink(label: str) -> Ink:
    """A tiny deterministic three-point stroke carrying ``label`` as its ground truth."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 10.0], [2.0, 0.5, 20.0]], dtype=np.float32)
    return Ink(strokes=[Stroke(points)], label=label, writer_id="e2e-writer")


def test_handwriting_stub_to_cas_produces_a_rendered_derivative(engine: MathEngine) -> None:
    """Ink -> stub recogniser -> parser -> SymPy -> LaTeX, with no checkpoint on disk."""
    recognition = StubRecognizer().recognize(_ink("x^{2}"))
    assert recognition.latex == "x^{2}"
    assert recognition.model_version == "stub-0.1.0"

    # The recognised string must be real, re-renderable LaTeX before it reaches the CAS.
    assert to_latex(parse_latex(recognition.latex)) == "x^{2}"
    assert str(latex_to_sympy(recognition.latex)) == "x**2"

    result = _run(engine, Diff(op="diff", expr=recognition.latex, var="x"))
    assert result.latex == r"2 \cdot x"
    assert result.plain == "2*x"


def test_partial_derivative_shorthand_survives_the_whole_chain() -> None:
    """Regression for KNOWN_ISSUES parser item 1, which ``aicls.ocr`` emits verbatim.

    ``\\frac{\\partial f}{\\partial x}`` used to raise ``ParseError`` at column 7 while the
    operator form parsed, so the handwriting leg had a dead end for a string the recogniser is
    trained to produce.
    """
    latex = r"\frac{\partial f}{\partial x}"
    recognition = StubRecognizer().recognize(_ink(latex))
    assert recognition.latex == latex
    assert str(latex_to_sympy(recognition.latex)) == "Derivative(f, x)"


# --------------------------------------------------------------------------------------------
# Leg 3: book fetch (İP-8) -- crosses the ADR-0002 boundary the NLU layer may not import
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def resolver(repo_root: Path) -> Iterator[BookResolver]:
    """A :class:`BookResolver` over the migrated sample book in an in-memory database."""
    tagged = tag_markdown_samples(repo_root / "datasets" / "book")
    units = [node for result in tagged for node in result.nodes]
    triples = [triple for result in tagged for triple in result.triples]
    db_engine = make_engine("sqlite+pysqlite:///:memory:")
    upgrade_head(db_engine)
    try:
        with session_scope(db_engine) as session:
            repository = BookRepository(session)
            load_units(units, triples, repository, tagged_ratio=require_full_coverage(*tagged))
            yield BookResolver(repository)
    finally:
        db_engine.dispose()


def test_turev_tanimini_getir_fetches_the_definition_from_the_book(
    resolver: BookResolver,
) -> None:
    """The proposal's own example: "türev tanımını getir" returns the Türev definition."""
    outcome = parse_command("türev tanımını getir")
    assert isinstance(outcome, Matched)
    command = outcome.command
    assert isinstance(command, BookFetch), "a book intent must not become a CAS command"
    assert command.unit_type is not None
    assert command.unit_type.value == "Definition"
    assert command.query == "türev"

    hits = resolver.resolve(command.query, unit_type=command.unit_type)
    assert hits, "the sample book must answer the proposal's own example"
    top = hits[0]
    assert top.unit.id == "tanim-2.4"
    assert top.unit.title_tr == "Türev"
    assert top.score == pytest.approx(1.0)
    assert "limit" in (top.unit.body_md or "").lower()


def test_book_fetch_is_refused_by_the_cas_as_a_named_validation_error(
    engine: MathEngine,
) -> None:
    """A book command handed to the CAS must fail as an ``AiclsError``, not an ``AttributeError``.

    ``aicls.nlu`` emits the wider :data:`aicls.nlu.types.Command` union, so the router above it
    can hand the engine something that is not a :data:`CasCommand`. Before the wave-3 gate this
    slipped through :meth:`MathEngine.validate` untyped and blew up inside the executor with
    ``'NoneType' object has no attribute 'timings'`` -- an HTTP 500 for an utterance the
    platform actually understood.
    """
    outcome = parse_command("türev tanımını getir")
    assert isinstance(outcome, Matched)
    book_command = outcome.command
    assert isinstance(book_command, BookFetch)
    assert not isinstance(book_command, CAS_COMMAND_TYPES)

    with pytest.raises(AiclsError) as excinfo:
        engine.execute(book_command)  # type: ignore[arg-type]  # the point of the test
    assert excinfo.value.details["reason"] == "not_a_cas_command"
    assert excinfo.value.details["op"] == "book_fetch"


def test_every_cas_command_class_is_listed_in_cas_command_types() -> None:
    """The ``isinstance`` tuple must not drift from the discriminated union it mirrors."""
    union_ops = {
        "diff",
        "integrate",
        "limit",
        "solve",
        "simplify",
        "expand",
        "factor",
        "series",
        "evaluate",
        "plot",
    }
    listed = {cls.model_fields["op"].default for cls in CAS_COMMAND_TYPES}
    assert listed == union_ops
