"""Cross-module contracts of ``aicls.eval`` (KNOWN_ISSUES: "Cross-module contract testing").

``aicls.eval`` is the module with the most neighbours in the repository: it drives the parser,
the CAS, the NLU layer, the OCR and ASR backends, the book resolver and the dataset layer, and
it hands its output to ``aicls.db`` and to the MkDocs macros. Each of those boundaries is an
attribute name or a field name that no unit test on either side would notice breaking — the
class of bug that reached the wave-1 gate. Every one of them is pinned here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from aicls.eval import (
    NORMALIZER_VERSIONS,
    EvalReport,
    check_report,
    load_latest,
    promote,
    run_all,
    run_asr_eval,
    run_book_eval,
    run_ocr_eval,
)
from aicls.eval.harness import book_resolver, build_asr_dataset, build_ocr_samples, default_paths
from aicls.eval.provenance import GitInfo

pytestmark = pytest.mark.integration

CLEAN = GitInfo(sha="d" * 40, branch="main", dirty=False, dirty_known=True, source="status")


# --------------------------------------------------------------- eval <-> ocr / asr backends
def test_stub_recognizer_satisfies_the_ocr_runner_contract() -> None:
    """``aicls.ocr.StubRecognizer`` really provides ``recognize`` -> ``latex`` / ``elapsed_ms``.

    The runner reads those two attribute names off whatever the recogniser returns; a rename in
    :class:`aicls.ocr.Recognition` would silently produce an empty hypothesis and a CER of 1.
    """
    from aicls.ocr import StubRecognizer

    recognizer = StubRecognizer()
    samples = build_ocr_samples(repeats=1)[:4]
    recognition = recognizer.recognize(samples[0])
    assert isinstance(recognition.latex, str) and recognition.latex
    assert isinstance(recognition.elapsed_ms, float)

    report = run_ocr_eval(recognizer, samples, git=CLEAN, stub=True)
    assert report.n == 4
    assert report.metrics["mean_ms"] >= 0.0


def test_the_ocr_runner_never_lets_the_backend_see_the_label() -> None:
    """The guarantee that replaces the old ``cer == 0.0`` assertion (KNOWN_ISSUES priority 1).

    ``aicls.ocr.StubRecognizer.recognize`` returns ``Ink.label`` whenever it finds one, so a
    runner that passes the sample through unblinded turns the stub into an oracle marking its
    own exam: it scored ``cer = 0.0`` / ``exprate = 1.0`` and those two numbers were promoted
    into ``reports/metrics/latest.json``, next to a contractual target of ``CER < 10 %``.

    The test asserts the stronger property directly — every object handed to the backend has
    ``label is None`` — and then confirms the consequence: the same stub over the same labelled
    corpus can no longer answer perfectly.
    """
    from aicls.ocr import StubRecognizer

    seen: list[Any] = []
    stub = StubRecognizer()

    class Recording:
        """Passes each sample to the real stub and keeps what it was handed."""

        def recognize(self, inp: Any, *, beam: int = 4) -> Any:
            seen.append(inp)
            return stub.recognize(inp, beam=beam)

    samples = build_ocr_samples(repeats=1)[:8]
    assert all(sample.label for sample in samples), "the corpus must carry an answer key"

    report = run_ocr_eval(Recording(), samples, git=CLEAN, stub=True)

    assert len(seen) == len(samples)
    assert all(getattr(handed, "label", None) is None for handed in seen)
    # the geometry survives the blinding, so the backend still has something to recognise
    assert all(
        handed.n_points == original.n_points for handed, original in zip(seen, samples, strict=True)
    )
    # and the answer key itself is untouched: blinding copies, it does not mutate
    assert all(sample.label for sample in samples)
    # the consequence: a stub that cannot read the answer cannot score a perfect exam
    assert report.metrics["cer"] > 0.0
    assert report.metrics["exprate"] < 1.0


def test_stub_transcriber_satisfies_the_asr_runner_contract(tmp_path: Path) -> None:
    """``aicls.asr.StubTranscriber`` provides ``transcribe`` -> ``text`` / ``elapsed_ms``."""
    from aicls.asr import StubTranscriber

    dataset, texts = build_asr_dataset(default_paths()["commands"], n=48, work_dir=tmp_path)
    transcriber = StubTranscriber(sentences=texts)
    transcript = transcriber.transcribe(next(iter(dataset)).audio)
    assert isinstance(transcript.text, str) and transcript.text
    assert isinstance(transcript.elapsed_ms, float)

    report = run_asr_eval(transcriber, dataset, git=CLEAN, stub=True)
    assert report.n == len(dataset)
    assert report.metrics["wer_tr_v1"] >= 0.0
    assert report.breakdown["speaker_disjoint"] is True


def test_the_asr_runner_hands_the_backend_a_waveform_and_nothing_else(tmp_path: Path) -> None:
    """The ASR half of the blinding audit (KNOWN_ISSUES priority 1).

    The OCR runner needed :func:`aicls.eval.runners.blind_sample` because a recogniser is handed
    the sample object. The ASR runner is blind by construction — ``_audio_of`` builds an
    :class:`aicls.asr.types.Audio` and the sample, which carries ``text``, ``speaker`` and
    ``intent``, never leaves the loop. That is a property of the code, not a coincidence, so it
    is pinned here: the day someone "simplifies" the runner into ``transcribe(sample)`` the ASR
    stub becomes an oracle exactly the way the OCR stub was.
    """
    from aicls.asr import StubTranscriber
    from aicls.asr.types import Audio

    dataset, texts = build_asr_dataset(default_paths()["commands"], n=32, work_dir=tmp_path)
    stub = StubTranscriber(sentences=texts)
    seen: list[Any] = []

    class Recording:
        """Passes each waveform to the real stub and keeps what it was handed."""

        def transcribe(self, audio: Any) -> Any:
            seen.append(audio)
            return stub.transcribe(audio)

    run_asr_eval(Recording(), dataset, git=CLEAN, stub=True)

    assert seen and len(seen) == len(dataset)
    for handed in seen:
        assert isinstance(handed, Audio)
        for answer in ("text", "intent", "label", "speaker", "sample_id"):
            assert not hasattr(handed, answer), answer


def test_data_layer_split_files_carry_the_fingerprint_the_asr_runner_requires(
    tmp_path: Path,
) -> None:
    """``aicls.data`` names it ``fingerprint`` on ``Splits`` and ``splits_fingerprint`` on the
    dataset; the ASR runner refuses to measure a WER without it."""
    from aicls.data import AudioCommandDataset

    dataset, _ = build_asr_dataset(default_paths()["commands"], n=24, work_dir=tmp_path)
    assert dataset.splits_fingerprint.startswith("sha256:")
    # the real dataset class exposes the same attribute name
    assert hasattr(AudioCommandDataset, "splits_fingerprint")


# ------------------------------------------------------------------------- eval <-> mathengine
def test_engine_result_srepr_is_what_the_cas_checker_rebuilds() -> None:
    """``Result.srepr`` must round-trip through the metric's own rebuild step.

    This is the join between ``aicls.mathengine`` and the reported CAS accuracy: if the engine
    ever printed something ``sympify`` cannot read, every row would score as an ``error``.
    """
    import sympy

    from aicls.eval.checkers import compare, rebuild_srepr
    from aicls.mathengine import InProcessExecutor, MathEngine

    result = MathEngine(InProcessExecutor()).execute({"op": "diff", "expr": "x^{2}"})
    rebuilt = rebuild_srepr(result.srepr)
    assert rebuilt == 2 * sympy.Symbol("x", real=True)
    # and it compares equal to the same answer written without assumptions
    assert compare("symbolic", rebuilt, 2 * sympy.Symbol("x"), {})[0] is True


def test_curriculum_rows_load_with_the_fields_the_cas_runner_reads() -> None:
    """``aicls.data.Problem`` must keep ``command``/``expected``/``checker``/``topic``."""
    from aicls.data import load_curriculum

    problems = load_curriculum(default_paths()["problems"])
    assert len(problems) > 1000
    header = problems[0]
    assert header.id == "_header"
    assert (header.model_extra or {}).get("dataset_version")
    row = problems[1]
    assert row.command and row.expected.srepr and row.checker
    assert row.input.kind in {"latex", "text_tr"}


# ------------------------------------------------------------------------------ eval <-> book
def test_book_resolver_hits_expose_unit_id_for_the_fetch_metric() -> None:
    """``BookResolver.resolve`` returns objects with ``.unit.id``; the İP-8 number depends on it."""
    paths = default_paths()
    with book_resolver(paths["book_sample"]) as (resolver, coverage):
        assert coverage == 1.0
        hits = resolver.resolve("türev tanımını getir", k=3)
        assert hits and hits[0].unit.id
        report = run_book_eval(resolver, paths["fetch_test"], tagging_coverage=coverage, git=CLEAN)
    assert report.metrics["fetch_top1"] > 0.95
    assert report.metrics["tagging_coverage"] == 1.0


# -------------------------------------------------------------------------------- eval <-> db
def test_eval_report_document_is_accepted_by_the_eval_runs_repository() -> None:
    """``EvalRunRepository.add`` reads the report dict; the field names must line up.

    ``aicls.db`` was written before this package existed, from the same specification. This is
    the test that proves the two specifications were read the same way.
    """
    from aicls.db import EvalKind, EvalRunRepository, make_engine, session_scope, upgrade_head

    report = EvalReport(
        name="parser_valid",
        kind="parser",
        git_sha="a" * 40,
        dirty=False,
        citable=True,
        dataset_fingerprint="sha256:" + "b" * 64,
        dataset_version="latex-2026.09",
        n=3709,
        metrics={"success_rate": 1.0},
        ci95={"success_rate": (0.999, 1.0)},
    )
    engine = make_engine("sqlite+pysqlite:///:memory:")
    try:
        upgrade_head(engine)
        with session_scope(engine) as session:
            row = EvalRunRepository(session).add(report.to_dict(), is_reportable=True)
            assert row.kind == EvalKind.PARSER
            assert row.git_sha == "a" * 40
            assert row.dirty is False
            assert row.n == 3709
            assert row.metrics_json["success_rate"] == 1.0
            assert row.ci95_json["success_rate"] == [0.999, 1.0]
            assert row.normalizer_versions_json == NORMALIZER_VERSIONS
            assert row.is_reportable is True
    finally:
        engine.dispose()


def test_every_eval_kind_of_the_database_has_a_runner() -> None:
    """``aicls.db.EvalKind`` and the harness must name the same seven runs."""
    from aicls.db import EvalKind
    from aicls.eval.harness import KINDS

    assert {kind.value for kind in EvalKind} == set(KINDS)


# ----------------------------------------------------------------------------- eval <-> mkdocs
def test_promoted_latest_json_exposes_the_keys_the_docs_macros_read(tmp_path: Path) -> None:
    """``{{ metrics.<kind>.<metric> }}`` (mkdocs.yml, degerlendirme.md §4) must resolve.

    The macro plugin runs with ``on_undefined: strict``, so a renamed metric key does not
    degrade a page — it breaks the documentation build. The four keys quoted in
    ``docs/ml/degerlendirme.md`` are pinned here.

    A wiring-check entry keeps its keys and sets them to ``null``: the macro still resolves (the
    build survives) but there is no number to quote. That is deliberate — see
    :func:`test_a_stub_backed_run_publishes_no_number_only_a_reason`.

    ``latency_n`` clears :data:`aicls.eval.MIN_VAL_SAMPLES` so that this run really does publish
    a number: both shapes have to appear here or the test only proves half of the contract. Five
    requests used to be enough because the sample floor was not applied to a finished report; a
    p95 over five requests was never a figure anyone should quote.
    """
    output = run_all(
        out_dir=tmp_path, kinds=("ocr", "latency"), git=CLEAN, ocr_repeats=1, latency_n=120
    )
    latest = load_latest(output.latest)
    assert "cer" in latest["ocr"] and latest["ocr"]["cer"] is None
    assert latest["latency"]["mean_ms"] is not None
    for entry in latest.values():
        assert entry["git_sha"] == "d" * 40
        assert entry["normalizer_versions"] == NORMALIZER_VERSIONS
        assert entry["citable"] is True
        assert entry["source_file"].endswith(".json")


def test_a_stub_backed_run_publishes_no_number_only_a_reason(tmp_path: Path) -> None:
    """Promotion is publication, and a stub has nothing to publish (KNOWN_ISSUES priority 1).

    ``latest.json`` is the one file the documentation macros read, so a value in it is a claim.
    Blinding the samples already stops the stub scoring a fake ``0.0``; this is the second half
    of the decision: whatever a stub scores, the headline field stays ``null`` and says why. The
    observed values are not thrown away — they stay under ``wiring_check`` and in the per-run
    file — they are just not published as if they measured a model.

    ``latency`` is the control: it is also marked ``stub`` (its backend models are stubs), but
    the pipeline it timed is real code that really ran, so its number survives promotion.
    """
    output = run_all(
        out_dir=tmp_path, kinds=("ocr", "asr", "latency"), git=CLEAN, ocr_repeats=1, asr_rows=48
    )
    latest = load_latest(output.latest)

    for kind, metric in (("ocr", "cer"), ("ocr", "exprate"), ("asr", "wer_tr_v1")):
        entry = latest[kind]
        assert entry["measurement"] == "wiring_check", kind
        assert entry["stub"] is True, kind
        assert entry[metric] is None, (kind, metric)
        assert entry["ci95"] == {}, kind
        assert entry["wiring_check"]["reason"].startswith("WIRING CHECK"), kind
        # nothing is hidden: what the stub scored is still readable, just not published
        assert isinstance(entry["wiring_check"]["metrics"][metric], float), (kind, metric)

    assert latest["latency"]["measurement"] == "measured"
    assert latest["latency"]["stub"] is True
    assert isinstance(latest["latency"]["mean_ms"], float)
    assert "wiring_check" not in latest["latency"]

    # and the per-run report file still records everything the run observed
    ocr_file = next(path for path in output.files if path.name.startswith("ocr_"))
    document = json.loads(ocr_file.read_text(encoding="utf-8"))
    assert isinstance(document["metrics"]["cer"], float)
    assert document["breakdown"]["blinded"] is True


def test_repository_latest_json_stays_readable_and_checkable() -> None:
    """The committed ``reports/metrics/latest.json`` must parse and pass ``report-check``.

    Before the first promotion it is ``{}``; afterwards every entry must carry its provenance.
    """
    from aicls.eval.provenance import known_shas, read_head
    from aicls.eval.report import check_promoted

    path = default_paths()["out_dir"] / "latest.json"
    latest = load_latest(path)
    assert isinstance(latest, dict)
    head, _ = read_head()
    problems = check_promoted(latest, head_sha=head, ancestors=known_shas())
    citable = [kind for kind, entry in latest.items() if entry.get("citable")]
    # Numbers produced from a dirty tree are allowed to sit in the file, but they must not
    # claim to be citable, and anything that does claim it must survive the gate.
    assert all(not problem.startswith(tuple(citable)) for problem in problems), problems


# ---------------------------------------------------------------------------------- layering
def test_importing_aicls_eval_does_not_import_torch_or_the_training_package() -> None:
    """ADR-0002 ``eval-no-ml``: the judge may not import the trainer, or torch.

    Probed in a clean interpreter (the pattern of ``tests/unit/data/test_layering.py``): reading
    this process' ``sys.modules`` would only describe whichever tests ran first, and clearing it
    would perturb every test that follows.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(default_paths()["root"] / "src")
    probe = (
        "import sys; import aicls.eval; "
        "print('ok', *[m for m in ('torch', 'transformers', 'aicls.ml') if m in sys.modules])"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(default_paths()["root"]),
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["ok"], completed.stdout


# --------------------------------------------------------------------------- the whole harness
@pytest.mark.slow
def test_full_harness_run_over_the_committed_corpora(tmp_path: Path) -> None:
    """The command that produces the progress-report numbers, end to end.

    Every contractual metric must be present, in range, and carry a dataset fingerprint; the
    stub runs must be marked as stubs so no reader can mistake them for evidence.
    """
    output = run_all(out_dir=tmp_path, git=CLEAN, ocr_repeats=1, asr_rows=60, latency_n=20)
    by_kind: dict[str, EvalReport] = {report.kind: report for report in output.reports}
    assert set(by_kind) == {"parser", "cas", "nlu", "ocr", "asr", "book", "latency"}

    assert by_kind["parser"].metrics["success_rate"] > 0.99
    assert by_kind["cas"].metrics["correct_rate"] > 0.99
    assert by_kind["cas"].breakdown["notes"].startswith("PROVENANCE")
    assert by_kind["nlu"].metrics["intent_accuracy"] >= 0.95
    assert by_kind["book"].metrics["fetch_top1"] > 0.95
    assert by_kind["book"].metrics["tagging_coverage"] == 1.0
    assert by_kind["latency"].metrics["mean_ms"] < 3000.0
    assert by_kind["ocr"].breakdown["stub"] is True
    assert by_kind["asr"].breakdown["stub"] is True

    for report in output.reports:
        assert report.dataset_fingerprint.startswith("sha256:"), report.kind
        assert report.normalizer_versions == NORMALIZER_VERSIONS
        assert check_report(report, head_sha="d" * 40, ancestors={"d" * 40}) == []

    latest: dict[str, Any] = json.loads(output.latest.read_text(encoding="utf-8"))
    assert set(latest) == set(by_kind)
    assert "| cas | correct_rate |" in output.summary.read_text(encoding="utf-8")


def test_promote_refuses_to_publish_two_runs_of_one_kind(tmp_path: Path) -> None:
    """The pointer file holds one number per kind; the harness must never guess which."""
    from aicls.core.errors import InputValidationError

    output = run_all(out_dir=tmp_path, kinds=("ocr",), git=CLEAN, ocr_repeats=1, write=False)
    with pytest.raises(InputValidationError):
        promote([*output.reports, *output.reports], out_dir=tmp_path)
