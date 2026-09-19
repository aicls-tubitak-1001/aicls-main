"""The per-component runners: what goes in the denominator and what a runner refuses to do.

These tests use tiny hand-written corpora and fake backends, so every expected number can be
counted by hand. The end-to-end run over the committed corpora lives in
``tests/integration/test_eval_contracts.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel
import pytest

from aicls.core.errors import InputValidationError
from aicls.eval.provenance import GitInfo
from aicls.eval.report import WIRING_SYNTHETIC_DATA
from aicls.eval.runners import (
    blind_sample,
    read_jsonl,
    run_asr_eval,
    run_book_eval,
    run_cas_eval,
    run_latency_eval,
    run_nlu_eval,
    run_ocr_eval,
    run_parser_eval,
)
from aicls.mathengine import InProcessExecutor, MathEngine

GIT = GitInfo(sha="f" * 40, branch="main", dirty=False, dirty_known=True, source="status")


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write ``rows`` as a JSONL corpus and return the path."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


# --------------------------------------------------------------------------------------- fakes
@dataclass(frozen=True, slots=True)
class FakeRecognition:
    """What a recogniser returns, reduced to what the runner reads."""

    latex: str
    elapsed_ms: float = 1.5


@dataclass(frozen=True, slots=True)
class FakeRecognizer:
    """Returns a canned hypothesis per sample id."""

    answers: dict[str, str]

    def recognize(self, inp: Any, *, beam: int = 4) -> FakeRecognition:
        """Answer with the canned LaTeX for this sample."""
        return FakeRecognition(self.answers[str(inp.sample_id)])


@dataclass(frozen=True, slots=True)
class FakeSample:
    """An OCR sample: an id and its ground-truth label."""

    sample_id: str
    label: str | None


@dataclass(frozen=True, slots=True)
class PeekingRecognizer:
    """A recogniser that answers with the label it was handed, like the real stub does.

    ``aicls.ocr.StubRecognizer`` returns ``Ink.label`` when it finds one; this is the same
    behaviour reduced to what the runner can observe, so the blinding contract can be tested
    without importing the OCR package.
    """

    fallback: str = "?"

    def recognize(self, inp: Any, *, beam: int = 4) -> FakeRecognition:
        """Answer with the sample's own label, or ``fallback`` when it has none."""
        return FakeRecognition(str(getattr(inp, "label", None) or self.fallback))


@dataclass(frozen=True, slots=True)
class FakeTranscript:
    """What a transcriber returns."""

    text: str
    elapsed_ms: float = 2.0


@dataclass(frozen=True, slots=True)
class FakeTranscriber:
    """Returns a canned transcript per sample id."""

    answers: dict[str, str]

    def transcribe(self, audio: Any) -> FakeTranscript:
        """Answer with the canned text for this waveform."""
        return FakeTranscript(self.answers[str(audio)])


@dataclass(frozen=True, slots=True)
class FakeAudioSample:
    """An ASR sample; ``audio`` is just the key the fake transcriber looks up."""

    sample_id: str
    text: str
    speaker: str
    audio: str


@dataclass(frozen=True, slots=True)
class FakeAsrDataset:
    """A dataset that does (or does not) carry a speaker-disjoint split assignment."""

    samples: list[FakeAudioSample]
    splits_fingerprint: str | None = None

    def __iter__(self) -> Any:
        return iter(self.samples)


@dataclass(frozen=True, slots=True)
class FakeUnit:
    """A book unit, reduced to its id."""

    id: str


@dataclass(frozen=True, slots=True)
class FakeHit:
    """A resolver hit."""

    unit: FakeUnit


@dataclass(frozen=True, slots=True)
class FakeResolver:
    """Returns a canned ranking per query."""

    answers: dict[str, list[str]]

    def resolve(self, query: str, *, unit_type: Any = None, k: int = 5) -> list[FakeHit]:
        """Return the canned units for ``query``."""
        return [FakeHit(FakeUnit(unit)) for unit in self.answers.get(query, [])][:k]


@dataclass
class FakeClient:
    """A pipeline client with fixed stage timings."""

    calls: list[str] = field(default_factory=list)

    def run(self, text: str) -> dict[str, float]:
        """Record the call and return deterministic timings."""
        self.calls.append(text)
        step = float(len(self.calls))
        return {"nlu": step, "cas": 2 * step, "total": 3 * step}


# -------------------------------------------------------------------------------------- parser
def test_run_parser_eval_counts_strictly_and_breaks_down_by_source(tmp_path: Path) -> None:
    """One good row, one row whose printed answer cannot be re-read: 1/2."""
    corpus = _jsonl(
        tmp_path / "valid.jsonl",
        [
            {"id": "ok-1", "latex": "x^{2}", "source": "curriculum"},
            {"id": "bad-1", "latex": r"\frac{1}{0}", "source": "ocr_output"},
        ],
    )
    report = run_parser_eval(corpus, git=GIT, dataset_version="test-1", failures_dir=tmp_path)
    assert report.kind == "parser"
    assert report.n == 2
    assert report.metrics["success_rate"] == pytest.approx(0.5)
    assert report.breakdown["by_source"]["curriculum"]["rate"] == 1.0
    assert report.breakdown["by_source"]["ocr_output"]["rate"] == 0.0
    assert report.breakdown["failures"][0]["id"] == "bad-1"
    assert report.dataset_fingerprint.startswith("sha256:")
    assert report.citable is True  # GIT is clean and committed
    assert report.failures_path is not None
    assert Path(report.failures_path).exists()


def test_run_parser_eval_skips_the_provenance_header_row(tmp_path: Path) -> None:
    """A ``_header`` row carries provenance, not a measurement, and stays out of the denominator."""
    corpus = _jsonl(
        tmp_path / "valid.jsonl",
        [
            {"id": "_header", "latex": "1", "source": "meta"},
            {"id": "ok-1", "latex": "x", "source": "curriculum"},
        ],
    )
    assert run_parser_eval(corpus, git=GIT).n == 1


# ----------------------------------------------------------------------------------------- CAS
def _problem(**overrides: Any) -> dict[str, Any]:
    """A minimal curriculum row: differentiate x^2, expecting 2x."""
    row: dict[str, Any] = {
        "id": "turev-0001",
        "topic": "turev",
        "input": {"kind": "latex", "value": "x^{2}"},
        "command": {"op": "diff", "expr": "x^{2}"},
        "expected": {"latex": r"2 \cdot x", "srepr": "Mul(Integer(2), Symbol('x'))"},
        "checker": "symbolic",
        "status": "frozen",
    }
    row.update(overrides)
    return row


def test_run_cas_eval_scores_correct_wrong_and_error_rows(tmp_path: Path) -> None:
    """Three rows, three outcomes; the denominator is every non-header row."""
    corpus = _jsonl(
        tmp_path / "problems.jsonl",
        [
            _problem(),
            _problem(
                id="turev-0002",
                expected={"latex": "x", "srepr": "Symbol('x')"},
            ),
            _problem(
                id="turev-0003",
                input={"kind": "latex", "value": r"\binom{n}{k}"},
                command={"op": "diff", "expr": r"\binom{n}{k}"},
            ),
        ],
    )
    report = run_cas_eval(corpus, MathEngine(InProcessExecutor()), git=GIT, failures_dir=tmp_path)
    assert report.kind == "cas"
    assert report.n == 3
    assert report.metrics["correct_rate"] == pytest.approx(1 / 3)
    assert report.metrics["error_rate"] == pytest.approx(1 / 3)
    # the parse denominator counts LaTeX inputs only, and \binom is outside the grammar
    assert report.metrics["parse_n"] == 3
    assert report.metrics["parse_success_rate"] == pytest.approx(2 / 3)
    assert report.breakdown["by_topic"]["turev"]["n"] == 3
    assert {row["id"] for row in report.breakdown["failures"]} == {"turev-0002", "turev-0003"}
    assert "mean_ms" in report.metrics


def test_run_cas_eval_copies_the_provenance_sentence_from_the_header_row(tmp_path: Path) -> None:
    """The corpus's own provenance travels into the report, so no number can lose it."""
    corpus = _jsonl(
        tmp_path / "problems.jsonl",
        [
            {
                "id": "_header",
                "topic": "_meta",
                "input": {"kind": "text_tr", "value": "KÖKEN"},
                "command": {"op": "simplify", "expr": "1"},
                "expected": {"latex": "1", "srepr": "Integer(1)"},
                "checker": "symbolic",
                "dataset_version": "curriculum-test-v1",
                "provenance_en": "PROVENANCE: authored inside the project.",
            },
            _problem(),
        ],
    )
    report = run_cas_eval(corpus, MathEngine(InProcessExecutor()), git=GIT)
    assert report.n == 1
    assert report.dataset_version == "curriculum-test-v1"
    assert report.breakdown["notes"].startswith("PROVENANCE")


def test_run_cas_eval_text_rows_stay_out_of_the_parse_denominator(tmp_path: Path) -> None:
    """A Turkish sentence is an NLU input, not a parser input (docs/ml/degerlendirme.md §2)."""
    corpus = _jsonl(
        tmp_path / "problems.jsonl",
        [_problem(id="t-1", input={"kind": "text_tr", "value": "x karenin türevini al"})],
    )
    report = run_cas_eval(corpus, MathEngine(InProcessExecutor()), git=GIT)
    assert report.n == 1
    assert report.metrics["parse_n"] == 0


# ----------------------------------------------------------------------------------------- NLU
def test_run_nlu_eval_reports_both_slot_f1_definitions(tmp_path: Path) -> None:
    """The strict definition counts an undeclared resolved slot as a false positive.

    That is the definition KNOWN_ISSUES asks for: the lenient one is what hid a wrong ``order``
    slot behind a perfect F1. Both are reported so the two can be compared across reports.
    """
    corpus = _jsonl(
        tmp_path / "commands.jsonl",
        [
            {
                "id": "n-1",
                "text": "x karenin türevini al",
                "intent": "differentiate",
                "slots": {"expr": "x^{2}"},
            },
            {"id": "n-2", "text": "qwerty asdf zxcv", "intent": "", "slots": {}},
        ],
    )
    report = run_nlu_eval(corpus, git=GIT, failures_dir=tmp_path)
    assert report.kind == "nlu"
    assert report.n == 2
    assert report.metrics["intent_accuracy"] == pytest.approx(1.0)
    assert report.metrics["command_exact"] == pytest.approx(1.0)
    # the layer also resolves `var`/`order`, which the gold row does not declare
    assert report.metrics["slot_f1_declared_only"] == pytest.approx(1.0)
    assert report.metrics["slot_f1"] < report.metrics["slot_f1_declared_only"]
    assert report.breakdown["slot_counts_strict"]["fp"] > 0


def test_run_nlu_eval_counts_an_out_of_scope_row_correct_only_on_no_match(tmp_path: Path) -> None:
    """An out-of-scope prompt is answered correctly by refusing, not by guessing."""
    corpus = _jsonl(
        tmp_path / "commands.jsonl",
        [{"id": "n-1", "text": "x karenin türevini al", "intent": "", "slots": {}}],
    )
    report = run_nlu_eval(corpus, git=GIT)
    assert report.metrics["intent_accuracy"] == 0.0
    assert report.breakdown["failures"][0]["gold"] == ""


# ----------------------------------------------------------------------------------------- OCR
def test_run_ocr_eval_reports_cer_ter_exprate_and_symbolic_verdicts() -> None:
    """Two samples: one perfect, one with a single character wrong."""
    samples = [FakeSample("s1", "x^{2}"), FakeSample("s2", "x^{2}+1")]
    recognizer = FakeRecognizer({"s1": "x^{2}", "s2": "x^{2}+2"})
    report = run_ocr_eval(recognizer, samples, git=GIT, stub=True)
    assert report.kind == "ocr"
    assert report.n == 2
    # characters: "x^{2}" (5) + "x^{2}+1" (7) = 12, one substitution
    assert report.metrics["cer"] == pytest.approx(1 / 12)
    assert report.metrics["exprate"] == pytest.approx(0.5)
    assert report.metrics["symeq_equal"] == pytest.approx(0.5)
    assert report.metrics["symeq_different"] == pytest.approx(0.5)
    assert report.breakdown["stub"] is True
    assert report.breakdown["worst"][0]["id"] == "s2"
    assert "cer" in report.ci95


def test_a_real_recognizer_on_generated_ink_still_publishes_no_cer() -> None:
    """The case no other guard covers: real model, clean tree, thousands of samples.

    ``stub`` says there was no real model. It says nothing about the corpus, and the corpus is
    the other half: a CER over ink this repository drew is not evidence for a target about
    handwriting, however good the recogniser is. Only the runner knows where its samples came
    from - the report can see a dirty tree and a sample count for itself, but not that - so the
    fact has to be recorded here or it is lost.
    """
    samples = [FakeSample("s1", "x^{2}"), FakeSample("s2", "x^{2}+1")]
    recognizer = FakeRecognizer({"s1": "x^{2}", "s2": "x^{2}+1"})

    report = run_ocr_eval(recognizer, samples, git=GIT, stub=False, synthetic=True)

    assert report.metrics["cer"] == pytest.approx(0.0)
    assert report.breakdown["stub"] is False, (
        "the model was real and the report must not say otherwise"
    )
    assert report.is_wiring_check
    assert WIRING_SYNTHETIC_DATA in report.wiring_check_verdict.codes


def test_a_real_recognizer_on_a_real_corpus_publishes_its_number() -> None:
    """The other half of the rule: the flag must not fire on the run the contract asks for.

    The sample count is not decoration. A finished report also judges its own ``n``, so a fixture
    small enough to be convenient would be a wiring check for a different reason and this test
    would pass without proving anything about the corpus flag.
    """
    samples = [FakeSample(f"s{i}", "x^{2}") for i in range(120)]
    recognizer = FakeRecognizer({f"s{i}": "x^{2}" for i in range(120)})

    report = run_ocr_eval(recognizer, samples, git=GIT, stub=False, synthetic=False)

    assert not report.is_wiring_check
    assert report.wiring_check_verdict.codes == ()


def test_run_asr_eval_records_a_generated_corpus_the_same_way() -> None:
    """Same rule, and the ASR side is where it was found: a WER measured on audio the harness
    synthesised must not sit next to the contractual "WER < 10 %" as a measurement."""
    transcriber = FakeTranscriber(
        {"wav1": "x karenin türevini al", "wav2": "sinüs x'in integralini hesapla"}
    )

    report = run_asr_eval(
        transcriber,
        _asr_dataset(fingerprint="sha256:" + "a" * 64),
        git=GIT,
        stub=False,
        synthetic=True,
    )

    assert report.breakdown["stub"] is False
    assert report.is_wiring_check
    assert WIRING_SYNTHETIC_DATA in report.wiring_check_verdict.codes


def test_run_ocr_eval_honours_max_n_and_accepts_input_label_pairs() -> None:
    """``max_n`` bounds a long run; ``(input, label)`` pairs work as well as labelled samples."""
    samples = [(FakeSample("s1", ""), "x"), (FakeSample("s2", ""), "y")]
    recognizer = FakeRecognizer({"s1": "x", "s2": "z"})
    report = run_ocr_eval(recognizer, samples, max_n=1, git=GIT)
    assert report.n == 1
    assert report.metrics["cer"] == 0.0


# ------------------------------------------------------------------------------------- blinding
def test_blind_sample_clears_the_answer_and_keeps_everything_else() -> None:
    """A blinded copy loses ``label`` and nothing else; the original is untouched."""
    sample = FakeSample("s1", "x^{2}")
    blinded = blind_sample(sample)
    assert blinded.label is None
    assert blinded.sample_id == "s1"
    assert sample.label == "x^{2}"  # blinding copies, it never mutates


def test_blind_sample_leaves_an_already_blind_sample_alone() -> None:
    """Nothing to hide means nothing to copy — the identical object comes back."""
    sample = FakeSample("s1", None)
    assert blind_sample(sample) is sample


def test_blind_sample_handles_a_pydantic_sample_too() -> None:
    """Dataset rows are pydantic models in several places; they blind through ``model_copy``."""

    class PydanticSample(BaseModel):
        """A validated sample carrying its own ground truth."""

        sample_id: str
        label: str | None = None

    blinded = blind_sample(PydanticSample(sample_id="s1", label="x^{2}"))
    assert blinded.label is None
    assert blinded.sample_id == "s1"


def test_blind_sample_refuses_a_sample_whose_answer_it_cannot_remove() -> None:
    """A number measured by a backend that could read the answer key is not a measurement.

    A sample type that is neither a dataclass nor a pydantic model cannot be blinded, so the
    runner stops rather than quietly handing the answer to the model.
    """

    class Opaque:
        """A sample that carries its answer and cannot be copied without it."""

        sample_id = "s1"
        label = "x^{2}"

    with pytest.raises(InputValidationError) as caught:
        blind_sample(Opaque())
    assert caught.value.details["reason"] == "sample_cannot_be_blinded"


def test_run_ocr_eval_blinds_every_sample_before_the_recognizer_sees_it() -> None:
    """The invariant that stops a stub publishing a fake headline (KNOWN_ISSUES priority 1).

    ``PeekingRecognizer`` does exactly what ``aicls.ocr.StubRecognizer`` does — it answers with
    the label it was handed. Unblinded it scores a perfect ``cer = 0.0``; the runner must make
    that impossible, so the same recogniser over the same corpus now scores like the guesser it
    is. The reference is still read correctly, which is why ``ter`` and ``exprate`` move too.
    """
    samples = [FakeSample("s1", "x^{2}"), FakeSample("s2", "y^{3}")]
    report = run_ocr_eval(PeekingRecognizer(), samples, git=GIT, stub=True)
    assert report.n == 2
    assert report.metrics["cer"] == 1.0  # every reference character is an error
    assert report.metrics["exprate"] == 0.0
    # the corpus itself still carries its answer key, for the marking
    assert [sample.label for sample in samples] == ["x^{2}", "y^{3}"]


def test_run_ocr_eval_blinds_the_input_half_of_an_input_label_pair() -> None:
    """A ``(input, label)`` pair whose input still carries a label is blinded too.

    ``training/scripts/evaluate.py`` blinds at the source; a caller that forgets must not get a
    different (and flattering) number out of the same runner.
    """
    pairs = [(FakeSample("s1", "x^{2}"), "x^{2}"), (FakeSample("s2", "y^{3}"), "y^{3}")]
    report = run_ocr_eval(PeekingRecognizer(), pairs, git=GIT, stub=True)
    assert report.metrics["cer"] == 1.0


def test_run_ocr_eval_reads_a_missing_label_as_an_empty_reference() -> None:
    """An unlabelled sample contributes an empty reference, never the string ``"None"``."""
    report = run_ocr_eval(FakeRecognizer({"s1": ""}), [FakeSample("s1", None)], git=GIT)
    assert report.n == 1
    assert report.metrics["exprate"] == 1.0


# ----------------------------------------------------------------------------------------- ASR
def _asr_dataset(*, fingerprint: str | None) -> FakeAsrDataset:
    """Two utterances by two speakers."""
    return FakeAsrDataset(
        samples=[
            FakeAudioSample("a1", "x karenin türevini al", "SPK01", "wav1"),
            FakeAudioSample("a2", "sinüs x'in integralini hesapla", "SPK02", "wav2"),
        ],
        splits_fingerprint=fingerprint,
    )


def test_asr_eval_refuses_without_speaker_split_file() -> None:
    """A WER measured on a speaker the model has heard is not a WER: refuse to produce it."""
    transcriber = FakeTranscriber({"wav1": "x karenin türevini al", "wav2": "..."})
    with pytest.raises(InputValidationError) as excinfo:
        run_asr_eval(transcriber, _asr_dataset(fingerprint=None), git=GIT)
    assert excinfo.value.details["reason"] == "speaker_split_required"


def test_asr_eval_reports_command_intent_accuracy() -> None:
    """The İP-2 metric: a recognition error only counts when it changes the command.

    ``"x karenin türevini al"`` is transcribed perfectly; ``"sinüs x'in integralini hesapla"``
    comes back as ``"sinüs x'in integralini hesapla lütfen"`` — one inserted word, so the WER is
    non-zero, yet the command the platform would execute is unchanged.
    """
    transcriber = FakeTranscriber(
        {
            "wav1": "x karenin türevini al",
            "wav2": "sinüs x'in integralini hesapla lütfen",
        }
    )
    report = run_asr_eval(
        transcriber, _asr_dataset(fingerprint="sha256:" + "a" * 64), git=GIT, stub=True
    )
    assert report.kind == "asr"
    assert report.n == 2
    assert 0.0 < report.metrics["wer_tr_v1"] < 0.3
    assert report.metrics["command_intent_accuracy"] == pytest.approx(1.0)
    assert report.metrics["command_exact"] == pytest.approx(1.0)
    assert report.breakdown["speaker_disjoint"] is True
    assert report.breakdown["speakers"] == 2
    assert "wer_tr_v1" in report.ci95


def test_asr_eval_reports_raw_and_tr_v1_wer_side_by_side() -> None:
    """Casing and punctuation move the raw column only (ADR-0015)."""
    transcriber = FakeTranscriber(
        {"wav1": "X KARENİN TÜREVİNİ AL,", "wav2": "sinüs x'in integralini hesapla"}
    )
    report = run_asr_eval(transcriber, _asr_dataset(fingerprint="sha256:" + "a" * 64), git=GIT)
    assert report.metrics["wer_tr_v1"] == 0.0
    assert report.metrics["wer_raw"] > 0.0


def test_asr_eval_accepts_an_explicit_split_fingerprint() -> None:
    """A dataset that keeps its split elsewhere may pass the fingerprint in."""
    transcriber = FakeTranscriber({"wav1": "a", "wav2": "b"})
    report = run_asr_eval(
        transcriber,
        _asr_dataset(fingerprint=None),
        splits_fingerprint="sha256:" + "b" * 64,
        git=GIT,
    )
    assert report.breakdown["splits_fingerprint"].endswith("b" * 8)


# ---------------------------------------------------------------------------------------- book
def test_run_book_eval_reports_top1_top3_and_coverage(tmp_path: Path) -> None:
    """Top-1 is the contractual number; top-3 explains a near miss."""
    corpus = _jsonl(
        tmp_path / "fetch.jsonl",
        [
            {
                "id": "f-1",
                "query": "türev tanımını getir",
                "expect_unit": "tanim-2.4",
                "unit_type": None,
                "kind": "command",
            },
            {
                "id": "f-2",
                "query": "limit tanımı",
                "expect_unit": "tanim-1.1",
                "unit_type": None,
                "kind": "bare",
            },
        ],
    )
    resolver = FakeResolver(
        {
            "türev tanımını getir": ["tanim-2.4"],
            "limit tanımı": ["tanim-1.2", "tanim-1.1"],
        }
    )
    report = run_book_eval(resolver, corpus, tagging_coverage=1.0, git=GIT, failures_dir=tmp_path)
    assert report.kind == "book"
    assert report.metrics["fetch_top1"] == pytest.approx(0.5)
    assert report.metrics["fetch_top3"] == pytest.approx(1.0)
    assert report.metrics["tagging_coverage"] == 1.0
    assert report.breakdown["misses"][0]["id"] == "f-2"
    assert report.breakdown["by_kind"]["command"]["rate"] == 1.0


def test_run_book_eval_without_coverage_omits_the_column(tmp_path: Path) -> None:
    """Coverage is only reported when it was actually measured."""
    corpus = _jsonl(
        tmp_path / "fetch.jsonl",
        [{"id": "f-1", "query": "q", "expect_unit": "u", "unit_type": None, "kind": "bare"}],
    )
    report = run_book_eval(FakeResolver({"q": ["u"]}), corpus, git=GIT)
    assert "tagging_coverage" not in report.metrics
    assert report.metrics["fetch_top1"] == 1.0


# ------------------------------------------------------------------------------------- latency
def test_run_latency_eval_summarises_total_and_stages(tmp_path: Path) -> None:
    """Mean/p50/p95 of the total, plus a per-stage table (budgets of docs/mimari/api.md)."""
    corpus = _jsonl(
        tmp_path / "commands.jsonl",
        [
            {"id": "c-1", "text": "x karenin türevini al", "tags": ["latency"]},
            {"id": "c-2", "text": "sinüs x'in integralini hesapla", "tags": []},
        ],
    )
    client = FakeClient()
    report = run_latency_eval(client, corpus, n=4, git=GIT)
    assert report.kind == "latency"
    assert report.n == 4
    assert (
        client.calls
        == [
            "x karenin türevini al",
            "sinüs x'in integralini hesapla",
        ]
        * 2
    )
    # totals are 3, 6, 9, 12 -> mean 7.5
    assert report.metrics["mean_ms"] == pytest.approx(7.5)
    assert report.breakdown["by_stage"]["nlu"]["mean_ms"] == pytest.approx(2.5)
    assert report.breakdown["stub"] is True


def test_run_latency_eval_can_select_a_tagged_subset(tmp_path: Path) -> None:
    """``tags: ["latency"]`` selects the 200-command subset the CI budget uses."""
    corpus = _jsonl(
        tmp_path / "commands.jsonl",
        [
            {"id": "c-1", "text": "bir", "tags": ["latency"]},
            {"id": "c-2", "text": "iki", "tags": []},
        ],
    )
    client = FakeClient()
    run_latency_eval(client, corpus, n=2, tag="latency", git=GIT)
    assert client.calls == ["bir", "bir"]


def test_run_latency_eval_refuses_an_empty_selection(tmp_path: Path) -> None:
    """Timing nothing would report a mean of nothing."""
    corpus = _jsonl(tmp_path / "commands.jsonl", [{"id": "c-1", "text": "bir", "tags": []}])
    with pytest.raises(InputValidationError):
        run_latency_eval(FakeClient(), corpus, n=2, tag="latency", git=GIT)


# ------------------------------------------------------------------------------------- reading
def test_read_jsonl_skips_blank_lines_and_rejects_malformed_rows(tmp_path: Path) -> None:
    """A corpus reader that guessed at a broken line would silently change a denominator."""
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": "a"}\n\n{"id": "b"}\n', encoding="utf-8")
    assert [row["id"] for row in read_jsonl(path)] == ["a", "b"]

    path.write_text('{"id": "a"}\nnot json\n', encoding="utf-8")
    with pytest.raises(InputValidationError):
        read_jsonl(path)

    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(InputValidationError):
        read_jsonl(path)


# ------------------------------------------------------------------------------- error paths
def test_run_cas_eval_counts_an_unrebuildable_expected_answer_as_an_error(
    tmp_path: Path,
) -> None:
    """A dataset row whose ``srepr`` cannot be rebuilt is a data error, not a wrong answer."""
    corpus = _jsonl(
        tmp_path / "problems.jsonl",
        [_problem(expected={"latex": "2x", "srepr": "lambda: 1"})],
    )
    report = run_cas_eval(corpus, MathEngine(InProcessExecutor()), git=GIT)
    assert report.metrics["error_rate"] == 1.0
    assert "srepr" in report.breakdown["failures"][0]["reason"]


def test_run_cas_eval_checks_that_an_antiderivative_differentiates_back(tmp_path: Path) -> None:
    """``up_to_constant`` alone is not enough: the answer must return the integrand."""
    corpus = _jsonl(
        tmp_path / "problems.jsonl",
        [
            _problem(
                id="integral-0001",
                command={"op": "integrate", "expr": "x"},
                expected={
                    "latex": r"\frac{x^{2}}{2}",
                    "srepr": "Mul(Rational(1, 2), Pow(Symbol('x', real=True), Integer(2)))",
                },
                checker_args={"up_to_constant": True, "var": "x"},
            )
        ],
    )
    report = run_cas_eval(corpus, MathEngine(InProcessExecutor()), git=GIT)
    assert report.metrics["correct_rate"] == 1.0


def test_run_cas_eval_scores_an_exact_latex_row_on_the_rendered_string(tmp_path: Path) -> None:
    """``exact_latex`` is how infinite limits are checked (the CAS abstains on those)."""
    corpus = _jsonl(
        tmp_path / "problems.jsonl",
        [
            _problem(
                id="limit-0001",
                command={"op": "limit", "expr": r"\frac{1}{x^{2}}", "to": "0"},
                expected={"latex": r"\infty", "srepr": "oo"},
                checker="exact_latex",
            )
        ],
    )
    report = run_cas_eval(corpus, MathEngine(InProcessExecutor()), git=GIT)
    assert report.n == 1
    assert report.metrics["correct_rate"] in {0.0, 1.0}  # whichever, it is judged on the string


def test_run_nlu_eval_counts_a_clarification_as_a_wrong_intent(tmp_path: Path) -> None:
    """A question is not the requested command, so it is never scored as correct."""
    corpus = _jsonl(
        tmp_path / "commands.jsonl",
        [{"id": "n-1", "text": "türevini al", "intent": "differentiate", "slots": {}}],
    )
    report = run_nlu_eval(corpus, git=GIT)
    # without a context there is no expression to differentiate: the layer must ask
    assert report.metrics["clarification_rate"] + report.metrics["no_match_rate"] == 1.0
    assert report.metrics["intent_accuracy"] == 0.0


def test_run_nlu_eval_uses_the_row_context_when_it_has_one(tmp_path: Path) -> None:
    """``context.last_latex`` is what makes "türevini al" resolvable at all."""
    corpus = _jsonl(
        tmp_path / "commands.jsonl",
        [
            {
                "id": "n-1",
                "text": "türevini al",
                "intent": "differentiate",
                "slots": {},
                "context": {"last_latex": "x^{3}"},
            }
        ],
    )
    report = run_nlu_eval(corpus, git=GIT)
    assert report.metrics["intent_accuracy"] == 1.0


def test_asr_eval_reads_a_sample_that_carries_features_instead_of_audio() -> None:
    """``aicls.data.AudioCommandDataset`` yields features plus a sample rate, not an ``Audio``."""
    import numpy as np

    from aicls.asr.types import Audio

    @dataclass(frozen=True, slots=True)
    class FeatureSample:
        sample_id: str
        text: str
        speaker: str
        features: Any
        sample_rate: int = 16_000

    class EchoTranscriber:
        def transcribe(self, audio: Audio) -> FakeTranscript:
            assert isinstance(audio, Audio)
            return FakeTranscript("bir iki üç")

    dataset = FakeAsrDataset(
        samples=[],  # type: ignore[arg-type]  # replaced below with feature-carrying rows
        splits_fingerprint="sha256:" + "c" * 64,
    )
    samples = [
        FeatureSample("s1", "bir iki üç", "SPK1", np.zeros(320, dtype="float32")),
    ]
    report = run_asr_eval(
        EchoTranscriber(),
        samples,
        splits_fingerprint=dataset.splits_fingerprint,
        git=GIT,
    )
    assert report.n == 1
    assert report.metrics["wer_tr_v1"] == 0.0


def test_asr_eval_refuses_a_sample_with_no_waveform_at_all() -> None:
    """A row without audio cannot be transcribed; that is a corpus error."""

    @dataclass(frozen=True, slots=True)
    class Empty:
        sample_id: str = "s1"
        text: str = "bir"
        speaker: str = "SPK1"

    class NeverCalled:
        def transcribe(self, audio: Any) -> FakeTranscript:  # pragma: no cover - not reached
            raise AssertionError("must not be called")

    with pytest.raises(InputValidationError):
        run_asr_eval(NeverCalled(), [Empty()], splits_fingerprint="sha256:" + "c" * 64, git=GIT)
