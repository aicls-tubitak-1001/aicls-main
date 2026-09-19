"""The report document, its writer, the Markdown table and ``report-check`` (ADR-0015 §4).

A report file is the unit of evidence in this project: a number without a git SHA, a dirty flag
and a dataset fingerprint cannot be defended in a progress report, so the model refuses to call
such a run citable and ``check_report`` refuses to let it be promoted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aicls.core.errors import InputValidationError
from aicls.eval.messages import tr
from aicls.eval.normalizers import NORMALIZER_VERSIONS
from aicls.eval.report import (
    DEFINITIONS_REF,
    LATEST_FILENAME,
    MIN_REPORTABLE_STEPS,
    MIN_VAL_SAMPLES,
    MODEL_BACKED_KINDS,
    TARGETS,
    WIRING_CHECK,
    WIRING_CHECK_KEY,
    WIRING_CHECK_PREFIX,
    WIRING_CHECK_REASON,
    WIRING_CHECK_SUFFIX,
    WIRING_NOT_CITABLE,
    WIRING_RANDOM_INIT,
    WIRING_STEP_BUDGET_TOO_SMALL,
    WIRING_STUB_BACKEND,
    WIRING_SYNTHETIC_DATA,
    WIRING_UNSPECIFIED,
    WIRING_VAL_SUBSET_TOO_SMALL,
    WIRING_WARMUP_NOT_COMPLETED,
    EvalReport,
    WiringCheckVerdict,
    assess_wiring_check,
    check_promoted,
    check_report,
    load_latest,
    promote,
    render_markdown,
    wiring_check_reason_text,
    write_report,
)

CLEAN_SHA = "a" * 40
OTHER_SHA = "b" * 40


def _report(**overrides: object) -> EvalReport:
    """A citable parser report; ``overrides`` bend one field at a time."""
    payload: dict[str, object] = {
        "name": "parser_valid",
        "kind": "parser",
        "git_sha": CLEAN_SHA,
        "dirty": False,
        "citable": True,
        "dataset_fingerprint": "sha256:" + "c" * 64,
        "dataset_version": "latex-2026.09",
        "n": 3709,
        "metrics": {"success_rate": 1.0},
        "ci95": {"success_rate": (0.999, 1.0)},
    }
    payload.update(overrides)
    return EvalReport(**payload)  # type: ignore[arg-type]  # a deliberately loose test factory


# ------------------------------------------------------------------------------- the document
def test_eval_report_has_fingerprint_git_sha_and_normalizer_versions() -> None:
    """The three provenance fields are mandatory and are serialised with the numbers."""
    report = _report()
    assert report.git_sha == CLEAN_SHA
    assert report.dataset_fingerprint.startswith("sha256:")
    assert report.normalizer_versions == NORMALIZER_VERSIONS
    assert report.definitions_ref == DEFINITIONS_REF
    assert report.timestamp.endswith("Z")

    document = report.to_dict()
    for field in (
        "name",
        "kind",
        "git_sha",
        "dirty",
        "citable",
        "dataset_fingerprint",
        "dataset_version",
        "model_version",
        "model_sha256",
        "config_hash",
        "normalizer_versions",
        "n",
        "metrics",
        "ci95",
        "breakdown",
        "failures_path",
        "timestamp",
        "definitions_ref",
    ):
        assert field in document, field


def test_report_without_a_git_sha_is_never_citable() -> None:
    """A number that cannot be tied to a commit cannot be cited, whatever the caller claims."""
    assert _report(git_sha="", citable=True).citable is False


def test_report_from_a_dirty_tree_is_never_citable() -> None:
    """``dirty=True`` downgrades ``citable`` in the model itself, not only in the checker."""
    assert _report(dirty=True, citable=True).citable is False


def test_report_rejects_an_unknown_kind_and_a_malformed_sha() -> None:
    """``kind`` is the closed set of runners; ``git_sha`` is hex or empty."""
    with pytest.raises(ValueError, match="kind"):
        _report(kind="handwriting")
    with pytest.raises(ValueError, match="git_sha"):
        _report(git_sha="zzzz")


def test_report_short_sha_is_the_first_seven_characters() -> None:
    """The file name and the docs footnote use the same short SHA."""
    assert _report().short_sha == "a" * 7
    assert _report(git_sha="").short_sha == "nogit"


# ------------------------------------------------------------------------------------- writer
def test_write_report_names_the_file_by_kind_and_sha(tmp_path: Path) -> None:
    """``reports/metrics/<name>_<gitsha7>.json`` (docs/ml/degerlendirme.md §4)."""
    path = write_report(_report(), out_dir=tmp_path)
    assert path.name == f"parser_valid_{'a' * 7}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metrics"]["success_rate"] == 1.0
    assert payload["ci95"]["success_rate"] == [0.999, 1.0]
    assert payload["normalizer_versions"] == NORMALIZER_VERSIONS


def test_write_report_is_utf8_lf_and_ends_with_a_newline(tmp_path: Path) -> None:
    """Report files are diffable: UTF-8, LF, trailing newline (CODING_STANDARDS)."""
    path = write_report(_report(dataset_version="müfredat-2026"), out_dir=tmp_path)
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert raw.endswith(b"\n")
    assert "müfredat-2026" in raw.decode("utf-8")


# ---------------------------------------------------------------------------------- promotion
def test_promote_writes_latest_json_keyed_by_kind(tmp_path: Path) -> None:
    """``latest.json`` is what the MkDocs macros read; one entry per evaluation kind."""
    parser = _report()
    latency = _report(
        name="latency_stub",
        kind="latency",
        metrics={"mean_ms": 120.0, "p50_ms": 100.0, "p95_ms": 200.0},
        ci95={},
        n=200,
    )
    path = promote([parser, latency], out_dir=tmp_path)
    assert path.name == LATEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"parser", "latency"}
    assert payload["parser"]["success_rate"] == 1.0
    assert payload["latency"]["mean_ms"] == 120.0
    for entry in payload.values():
        for field in (
            "git_sha",
            "dataset_fingerprint",
            "normalizer_versions",
            "dirty",
            "citable",
            "generated_at",
            "source_file",
            "n",
        ):
            assert field in entry, field


def test_promote_carries_the_stub_flag_into_latest_json(tmp_path: Path) -> None:
    """``latest.json`` says per kind whether the number came from a stub model.

    The MkDocs macros read this file and nothing else, so the caveat has to travel with the
    value; without it a report page could render a stub CER as if it were a measurement.
    """
    payload = json.loads(
        promote(
            [
                _report(),
                _report(kind="ocr", name="ocr_test", ci95={}, breakdown={"stub": True}),
            ],
            out_dir=tmp_path,
        ).read_text(encoding="utf-8")
    )
    assert payload["ocr"]["stub"] is True
    assert payload["parser"]["stub"] is False


def test_promote_publishes_null_for_a_stub_backed_model_run(tmp_path: Path) -> None:
    """A wiring check has nothing to publish (KNOWN_ISSUES priority 1).

    ``latest.json`` is the one file the documentation macros read, so a value in it is a claim.
    The repository published ``ocr.cer = 0.0`` and ``ocr.exprate = 1.0`` from a stub that had
    been handed the answer key — indefensible next to a contractual ``CER < 10 %``. A
    stub-backed OCR/ASR run now publishes ``null`` and the reason; the metric **keys** stay, so
    ``{{ metrics.ocr.cer }}`` still resolves under ``on_undefined: strict`` and the docs build
    does not break on a missing name.
    """
    stub = _report(
        kind="ocr",
        name="ocr_test",
        metrics={"cer": 0.97, "exprate": 0.02},
        ci95={"cer": (0.95, 0.99)},
        breakdown={"stub": True, "notes": "WIRING CHECK, NOT A RESULT: ..."},
    )
    payload = json.loads(promote([stub], out_dir=tmp_path).read_text(encoding="utf-8"))
    entry = payload["ocr"]

    assert entry["cer"] is None and entry["exprate"] is None
    assert "cer" in entry and "exprate" in entry  # the macro must still resolve the name
    assert entry["ci95"] == {}
    assert entry["measurement"] == "wiring_check"
    assert entry["notes"].startswith("WIRING CHECK")
    # nothing is destroyed: what the stub scored is still in the file, just not as a headline
    assert entry["wiring_check"]["metrics"] == {"cer": 0.97, "exprate": 0.02}
    assert entry["wiring_check"]["ci95"] == {"cer": [0.95, 0.99]}
    assert entry["wiring_check"]["reason"].startswith("WIRING CHECK")


def test_promote_still_publishes_a_stub_run_of_a_kind_that_measured_real_code(
    tmp_path: Path,
) -> None:
    """``latency`` is marked ``stub`` too, and its number is real.

    The distinction :data:`aicls.eval.report.MODEL_BACKED_KINDS` draws: for ``ocr``/``asr`` a
    stub backend means the subject of the metric does not exist yet, so there is nothing to
    report. For ``latency`` it means "server side, not the browser figure the contract names" —
    the NLU -> CAS pipeline it timed is real code that really ran, and withholding that number
    would hide a measurement instead of a fake one.
    """
    latency = _report(
        kind="latency",
        name="latency_local",
        metrics={"mean_ms": 4.9},
        ci95={"mean_ms": (3.2, 6.6)},
        breakdown={"stub": True},
    )
    payload = json.loads(promote([latency], out_dir=tmp_path).read_text(encoding="utf-8"))
    entry = payload["latency"]

    assert entry["stub"] is True
    assert entry["measurement"] == "measured"
    assert entry["mean_ms"] == 4.9
    assert entry["ci95"] == {"mean_ms": [3.2, 6.6]}
    assert "wiring_check" not in entry


def test_is_wiring_check_separates_a_missing_model_from_a_caveat() -> None:
    """The property the promotion rule is built on, stated on its own."""
    assert _report(kind="ocr", name="ocr_test", breakdown={"stub": True}).is_wiring_check is True
    assert _report(kind="asr", name="asr_test", breakdown={"stub": True}).is_wiring_check is True
    assert _report(kind="latency", name="lat", breakdown={"stub": True}).is_wiring_check is False
    assert _report(kind="ocr", name="ocr_real").is_wiring_check is False
    assert set(MODEL_BACKED_KINDS) == {"ocr", "asr"}


def test_promote_round_trips_through_load_latest(tmp_path: Path) -> None:
    """``load_latest`` is the reader the docs macros and ``report-check`` share."""
    promote([_report()], out_dir=tmp_path)
    latest = load_latest(tmp_path / LATEST_FILENAME)
    assert latest["parser"]["n"] == 3709
    assert load_latest(tmp_path / "missing.json") == {}


def test_promote_refuses_two_reports_of_the_same_kind(tmp_path: Path) -> None:
    """Silently keeping the last of two ``ocr`` runs would publish an unexplained number."""
    with pytest.raises(InputValidationError):
        promote([_report(), _report(name="parser_other")], out_dir=tmp_path)


# ------------------------------------------------------------------------------- report-check
def test_report_check_accepts_a_clean_ancestor_run() -> None:
    """The happy path: committed SHA, clean tree, fingerprint that matches the corpus."""
    problems = check_report(
        _report(),
        head_sha=OTHER_SHA,
        ancestors={CLEAN_SHA, OTHER_SHA},
        fingerprints={"sha256:" + "c" * 64},
    )
    assert problems == []


def test_report_check_rejects_non_ancestor_sha() -> None:
    """A report from a commit that is not in HEAD's history describes code nobody can check out."""
    problems = check_report(_report(), head_sha=OTHER_SHA, ancestors={OTHER_SHA})
    assert any("git_sha" in problem for problem in problems)


def test_report_check_rejects_dirty_run() -> None:
    """A number produced from an uncommitted tree can never be promoted."""
    problems = check_report(
        _report(dirty=True, citable=False), head_sha=CLEAN_SHA, ancestors={CLEAN_SHA}
    )
    assert any("dirty" in problem for problem in problems)
    assert any("citable" in problem for problem in problems)


def test_report_check_rejects_a_stale_normalizer_version() -> None:
    """A number computed with an old normaliser must not be quoted next to a new one."""
    problems = check_report(
        _report(normalizer_versions={"latex": "v0", "tr": "v1"}),
        head_sha=CLEAN_SHA,
        ancestors={CLEAN_SHA},
    )
    assert any("normalizer" in problem for problem in problems)


def test_report_check_rejects_an_unknown_dataset_fingerprint() -> None:
    """The fingerprint must match a committed corpus, otherwise the corpus is unidentified."""
    problems = check_report(
        _report(),
        head_sha=CLEAN_SHA,
        ancestors={CLEAN_SHA},
        fingerprints={"sha256:" + "d" * 64},
    )
    assert any("fingerprint" in problem for problem in problems)


def test_report_check_rejects_a_missing_definitions_ref() -> None:
    """Every promoted number points at the section that defines it."""
    problems = check_report(_report(definitions_ref=""), head_sha=CLEAN_SHA, ancestors={CLEAN_SHA})
    assert any("definitions_ref" in problem for problem in problems)


def test_check_promoted_reports_every_kind_that_fails(tmp_path: Path) -> None:
    """``poe report-check`` runs over ``latest.json`` and names the offending kind."""
    promote([_report(dirty=True, citable=False)], out_dir=tmp_path)
    latest = load_latest(tmp_path / LATEST_FILENAME)
    problems = check_promoted(latest, head_sha=CLEAN_SHA, ancestors={CLEAN_SHA})
    assert problems
    assert all(problem.startswith("parser:") for problem in problems)


def test_check_promoted_of_an_empty_pointer_file_is_clean() -> None:
    """Before the first promotion ``latest.json`` is ``{}`` and CI must stay green."""
    assert check_promoted({}, head_sha=CLEAN_SHA, ancestors={CLEAN_SHA}) == []


# ----------------------------------------------------------------------------------- markdown
def test_render_markdown_has_one_row_per_metric_with_the_contract_target() -> None:
    """The results table is generated, never hand-edited (docs/ml/degerlendirme.md §4)."""
    text = render_markdown([_report()])
    assert "| parser | success_rate | 1.0000 |" in text
    assert "> 0.99" in text  # the İP-4 contract target
    assert CLEAN_SHA[:7] in text
    assert "sha256:" in text


def test_render_markdown_marks_a_missed_target_and_an_untargeted_metric() -> None:
    """A missed target is visible in the table; a metric without a target says so."""
    missed = _report(kind="ocr", name="ocr_stub", metrics={"cer": 0.5, "ter": 0.4})
    text = render_markdown([missed])
    assert "< 0.10" in text
    assert "0.5000" in text
    lines = [line for line in text.splitlines() if line.startswith("| ocr | ter |")]
    assert lines and "—" in lines[0]


def test_render_markdown_of_no_reports_is_still_a_table() -> None:
    """An empty run produces an empty table, not a crash."""
    text = render_markdown([])
    assert text.strip().startswith("|")


def test_render_markdown_never_gives_a_stub_run_a_pass_or_fail_verdict() -> None:
    """A stub run measured nothing, so its status cell must not answer the contract question.

    The OCR wiring check scores ``cer = 0`` because the stub echoes the label it was handed, and
    the ASR wiring check scores ``wer > 1`` because the stub draws a random sentence. Printing
    "tuttu" and "tutmadı" next to the İP-3 targets for those two rows would be the single most
    damaging kind of mistake this package can make: a reader of the generated table would see a
    contractual metric marked as met (or missed) by a model that does not exist yet.
    """
    stub_ocr = _report(
        kind="ocr", name="ocr_test", metrics={"cer": 0.0}, ci95={}, breakdown={"stub": True}
    )
    stub_asr = _report(
        kind="asr", name="asr_test", metrics={"wer_tr_v1": 1.1}, ci95={}, breakdown={"stub": True}
    )
    text = render_markdown([stub_ocr, stub_asr])
    rows = [line for line in text.splitlines() if line.startswith(("| ocr |", "| asr |"))]
    assert len(rows) == 2
    for row in rows:
        assert "tuttu" not in row and "tutmadı" not in row
        assert "stub" in row
        assert "< 0.10" in row  # the target is still shown, only the verdict is withheld
    # a real run keeps its verdict
    assert "tuttu" in render_markdown([_report()])


def test_targets_cover_every_contractual_metric() -> None:
    """The seven contractual numbers of the proposal all have a frozen target here."""
    assert TARGETS[("ocr", "cer")].threshold == 0.10
    assert TARGETS[("asr", "wer_tr_v1")].threshold == 0.10
    assert TARGETS[("parser", "success_rate")].threshold == 0.99
    assert TARGETS[("cas", "correct_rate")].threshold == 0.99
    assert TARGETS[("latency", "mean_ms")].threshold == 3000.0
    assert TARGETS[("book", "fetch_top1")].threshold == 0.95
    assert TARGETS[("book", "tagging_coverage")].threshold == 1.0
    assert TARGETS[("sus", "sus_score")].threshold == 75.0


def test_load_latest_rejects_a_corrupt_pointer_file(tmp_path: Path) -> None:
    """A broken ``latest.json`` must fail loudly: the docs build reads it with strict macros."""
    path = tmp_path / LATEST_FILENAME
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(InputValidationError):
        load_latest(path)
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(InputValidationError):
        load_latest(path)
    path.write_text("   ", encoding="utf-8")
    assert load_latest(path) == {}


def test_check_promoted_reports_a_malformed_entry() -> None:
    """An entry that is not an object is a problem, not a crash."""
    problems = check_promoted(
        {"parser": "not-an-object"}, head_sha=CLEAN_SHA, ancestors={CLEAN_SHA}
    )
    assert problems == ["parser: entry is not an object"]


def test_check_report_accepts_the_head_commit_itself() -> None:
    """A run made at HEAD is trivially an ancestor of HEAD."""
    assert check_report(_report(git_sha=OTHER_SHA), head_sha=OTHER_SHA, ancestors=set()) == []


# --------------------------------------------------------------------- the wiring-check verdict
def test_a_real_final_run_is_not_a_wiring_check() -> None:
    """The direction that matters most: the flag must not fire on the run that counts.

    The numbers are the shipped OCR schedule (``offline_resnet34_tfm.yaml``): 30 epochs, a
    2000-step warmup and ``eval.max_n_val: 2000``, from a clean tree on the real corpus.
    """
    verdict = assess_wiring_check(
        synthetic_data=False,
        stub_backend=False,
        random_init=False,
        citable=True,
        steps=120_000,
        warmup_steps=2000,
        n_val=2000,
    )
    assert verdict.is_wiring_check is False
    assert verdict.reasons == ()
    assert verdict.summary() == "measurement"
    assert verdict.to_dict() == {"is_wiring_check": False, "codes": [], "reasons": []}


def test_the_run_that_prompted_this_rule_is_a_wiring_check() -> None:
    """The real GPU run of 2026-09-08: 20 steps, 16 validation samples, warmup 2000.

    ``fit_report.json`` said ``wiring_check: false`` because the flag meant only "the dataset was
    synthetic". Three independent reasons apply, and each names the number behind it.
    """
    verdict = assess_wiring_check(
        synthetic_data=False, citable=True, steps=20, warmup_steps=2000, n_val=16
    )
    assert verdict.is_wiring_check is True
    assert verdict.codes == (
        WIRING_WARMUP_NOT_COMPLETED,
        WIRING_STEP_BUDGET_TOO_SMALL,
        WIRING_VAL_SUBSET_TOO_SMALL,
    )
    assert "20 optimizer steps" in verdict.summary()
    assert "16 validation samples" in verdict.summary()


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"synthetic_data": True}, WIRING_SYNTHETIC_DATA),
        ({"stub_backend": True}, WIRING_STUB_BACKEND),
        ({"random_init": True}, WIRING_RANDOM_INIT),
        ({"citable": False}, WIRING_NOT_CITABLE),
        ({"steps": 20, "warmup_steps": 2000}, WIRING_WARMUP_NOT_COMPLETED),
        ({"steps": 20}, WIRING_STEP_BUDGET_TOO_SMALL),
        ({"n_val": 16}, WIRING_VAL_SUBSET_TOO_SMALL),
    ],
)
def test_each_reason_fires_on_its_own(kwargs: dict[str, object], code: str) -> None:
    verdict = assess_wiring_check(**kwargs)  # type: ignore[arg-type]  # parametrised on purpose
    assert code in verdict.codes
    assert verdict.is_wiring_check is True
    assert all(reason["code"] in verdict.codes for reason in verdict.to_dict()["reasons"])


def test_unknown_facts_are_never_held_against_a_run() -> None:
    """Every judgement is opt-in: a caller that cannot observe a fact causes no reason.

    This is what keeps the flag trustworthy. ``aicls profile`` has no validation set at all and
    a pure evaluation run takes no optimizer steps; neither may be called a wiring check for
    something it never claimed to do.
    """
    assert assess_wiring_check().is_wiring_check is False
    assert assess_wiring_check(steps=None, n_val=None, citable=None).is_wiring_check is False
    assert assess_wiring_check(steps=200, warmup_steps=None, n_val=100).is_wiring_check is False
    assert assess_wiring_check(steps=200, warmup_steps=0, n_val=100).is_wiring_check is False


def test_the_thresholds_are_boundaries_not_approximations() -> None:
    assert assess_wiring_check(steps=MIN_REPORTABLE_STEPS).is_wiring_check is False
    assert assess_wiring_check(steps=MIN_REPORTABLE_STEPS - 1).is_wiring_check is True
    assert assess_wiring_check(n_val=MIN_VAL_SAMPLES).is_wiring_check is False
    assert assess_wiring_check(n_val=MIN_VAL_SAMPLES - 1).is_wiring_check is True
    # the warmup rule is "did not get past it", so finishing warmup exactly is enough
    assert assess_wiring_check(steps=2001, warmup_steps=2000).is_wiring_check is False
    assert assess_wiring_check(steps=2000, warmup_steps=2000).is_wiring_check is True


# --------------------------------------------------------------------------------------------
# what a finished report judges about itself
#
# assess_wiring_check has always known how to weigh a dirty tree and a sample count, but until
# 2026-09-09 the only caller was EvalReport.wiring_check_verdict and it passed one fact:
# stub_backend. So a report that was not a stub could not be a wiring check for any other
# reason, and promote() published its numbers. The speech study found the hole from the ASR
# side - the first real Whisper run would file a dev-set WER next to the contractual
# "WER < 10 %" - but nothing about it was specific to ASR; every kind leaked the same way.
# --------------------------------------------------------------------------------------------
def test_a_report_that_cannot_be_tied_to_a_commit_is_a_wiring_check() -> None:
    """A real backend and a real corpus do not make a number quotable on a dirty tree."""
    verdict = _report(citable=False, dirty=True).wiring_check_verdict

    assert verdict.is_wiring_check
    assert WIRING_NOT_CITABLE in verdict.codes


def test_a_report_over_too_few_samples_is_a_wiring_check() -> None:
    """An interval over a handful of samples is arithmetic, not evidence."""
    verdict = _report(n=5).wiring_check_verdict

    assert verdict.is_wiring_check
    assert WIRING_VAL_SUBSET_TOO_SMALL in verdict.codes


def test_a_real_asr_run_on_a_dirty_tree_publishes_no_wer(tmp_path: Path) -> None:
    """The case the speech study named, end to end through promote().

    A genuine Whisper transcription of genuine speech is not a stub, so nothing used to stop
    ``wer_tr_v1`` from being published as a measurement - beside a contractual target it does
    not answer to, from a tree nobody could check it against.
    """
    report = _report(
        name="asr_test",
        kind="asr",
        n=864,
        metrics={"wer_tr_v1": 0.083},
        ci95={},
        citable=False,
        dirty=True,
        breakdown={"stub": False, "speaker_disjoint": True},
    )

    latest = json.loads((promote([report], out_dir=tmp_path)).read_text(encoding="utf-8"))

    assert latest["asr"]["wer_tr_v1"] is None, "a WER nobody can tie to a commit was published"
    assert latest["asr"]["measurement"] == WIRING_CHECK
    assert WIRING_NOT_CITABLE in latest["asr"][WIRING_CHECK_KEY]["codes"]


def test_the_new_facts_do_not_fire_on_a_run_that_should_be_reportable() -> None:
    """The half of the rule that is easy to lose.

    ``assess_wiring_check`` says it plainly: the flag stops being trusted the moment it also
    fires on the final run. Both facts added here are cleared by every run the contract
    actually asks for - a clean tree, and sample counts in the thousands - so this asserts the
    quiet case as deliberately as the loud ones.
    """
    verdict = _report(
        kind="asr", n=864, citable=True, dirty=False, breakdown={"stub": False}
    ).wiring_check_verdict

    assert not verdict.is_wiring_check
    assert verdict.codes == ()


def test_a_wiring_check_that_was_not_a_stub_does_not_claim_a_stub_model() -> None:
    """The warning has to name the reason it actually has.

    A wiring check could only ever mean "stub" for a model-backed kind, so the renderer printed
    the stub sentence for any verdict at all. Once a verdict could also mean a dirty tree, a
    generated corpus or too few samples, that sentence became a false statement about the model:
    it tells a reader a real recogniser was a fake one. That is a worse error than the one the
    warning exists to prevent, because it is wrong in the reader's favour nowhere and against the
    work everywhere.
    """
    report = _report(kind="ocr", n=2000, citable=False, dirty=True, breakdown={"stub": False})

    text = render_markdown([report])

    assert tr("REPORT_STUB_WARNING") not in text
    assert tr("REPORT_WIRING_WARNING") in text
    assert tr("REPORT_NOT_CITABLE") in text
    assert WIRING_NOT_CITABLE in text


def test_a_run_that_really_was_a_stub_still_says_so() -> None:
    """And the other half: the stub sentence must survive for the case it was written for."""
    report = _report(kind="ocr", n=2000, citable=True, dirty=False, breakdown={"stub": True})

    text = render_markdown([report])

    assert tr("REPORT_STUB_WARNING") in text
    assert tr("REPORT_WIRING_WARNING") not in text


def test_assess_wiring_check_rejects_a_negative_threshold() -> None:
    with pytest.raises(InputValidationError):
        assess_wiring_check(min_steps=-1)
    with pytest.raises(InputValidationError):
        assess_wiring_check(min_val_samples=-1)


def test_a_verdict_survives_a_json_round_trip() -> None:
    """The verdict travels between processes as data, so it has to read back."""
    verdict = assess_wiring_check(steps=20, warmup_steps=2000, n_val=16, citable=False)
    restored = WiringCheckVerdict.from_payload(json.loads(json.dumps(verdict.to_dict())))
    assert restored == verdict
    assert restored.codes == verdict.codes


@pytest.mark.parametrize(
    ("payload", "codes"),
    [
        (None, ()),
        (False, ()),
        (True, (WIRING_UNSPECIFIED,)),
        ({"is_wiring_check": False}, ()),
        ({"codes": ["step_budget_too_small"]}, (WIRING_STEP_BUDGET_TOO_SMALL,)),
        ([{"code": "random_init", "detail": "x"}], (WIRING_RANDOM_INIT,)),
        (["random_init", "random_init"], (WIRING_RANDOM_INIT,)),
        ("random_init", (WIRING_RANDOM_INIT,)),
        ({"reason": "WIRING CHECK ..."}, (WIRING_UNSPECIFIED,)),
        (12, ()),
        ([None, 3, {"detail": "no code"}], ()),
    ],
)
def test_from_payload_reads_every_shape_the_record_has_held(
    payload: object, codes: tuple[str, ...]
) -> None:
    """A malformed provenance note must not crash the gate; it just fails to excuse the run."""
    assert WiringCheckVerdict.from_payload(payload).codes == codes


def test_merge_orders_reasons_and_drops_the_placeholder() -> None:
    stub = assess_wiring_check(stub_backend=True)
    recorded = WiringCheckVerdict.from_payload(
        [{"code": WIRING_SYNTHETIC_DATA}, {"code": WIRING_UNSPECIFIED}]
    )
    merged = stub.merge(recorded)

    assert merged.codes == (WIRING_SYNTHETIC_DATA, WIRING_STUB_BACKEND)
    assert WiringCheckVerdict().merge(WiringCheckVerdict.from_payload(True)).codes == (
        WIRING_UNSPECIFIED,
    )


def test_a_report_carries_the_producing_runs_verdict() -> None:
    """A real-data OCR run that trained for 20 steps is a wiring check, stub or not."""
    verdict = assess_wiring_check(steps=20, warmup_steps=2000, n_val=16)
    report = _report(
        kind="ocr",
        name="ocr_excerpt",
        metrics={"cer": 40.4},
        breakdown={"stub": False, WIRING_CHECK_KEY: verdict.to_dict()},
    )
    assert report.is_stub is False
    assert report.is_wiring_check is True
    assert report.wiring_check_verdict.codes == verdict.codes


def test_promote_publishes_the_reasons_next_to_the_nulled_metric(tmp_path: Path) -> None:
    verdict = assess_wiring_check(steps=20, warmup_steps=2000, n_val=16)
    report = _report(
        kind="ocr",
        name="ocr_excerpt",
        metrics={"cer": 40.4},
        n=16,
        breakdown={WIRING_CHECK_KEY: verdict.to_dict()},
    )
    entry = json.loads(promote([report], out_dir=tmp_path).read_text(encoding="utf-8"))["ocr"]

    assert entry["cer"] is None
    assert entry["measurement"] == "wiring_check"
    assert entry["wiring_check"]["codes"] == list(verdict.codes)
    assert entry["wiring_check"]["metrics"] == {"cer": 40.4}
    assert entry["wiring_check"]["reason"].startswith("WIRING CHECK")
    assert "MIN_REPORTABLE_STEPS" in entry["wiring_check"]["reason"]


def test_a_stub_only_verdict_keeps_the_documented_wording() -> None:
    """``reports/README.md`` quotes this string; only a richer verdict changes it."""
    assert wiring_check_reason_text(assess_wiring_check(stub_backend=True)) == WIRING_CHECK_REASON
    assert wiring_check_reason_text(WiringCheckVerdict()) == ""
    text = wiring_check_reason_text(assess_wiring_check(steps=20))
    assert text.startswith(WIRING_CHECK_PREFIX) and text.endswith(WIRING_CHECK_SUFFIX)


def test_report_check_refuses_to_promote_a_wiring_check_run(tmp_path: Path) -> None:
    """``aicls report check`` is the gate between a measurement and a citable number."""
    verdict = assess_wiring_check(steps=20, warmup_steps=2000, n_val=16)
    report = _report(
        kind="ocr",
        name="ocr_excerpt",
        metrics={"cer": 40.4},
        breakdown={WIRING_CHECK_KEY: verdict.to_dict()},
    )
    # per-report provenance is sound, so the plain check stays silent ...
    assert check_report(report, head_sha=CLEAN_SHA, ancestors={CLEAN_SHA}) == []
    # ... but promotion is publication, and that is refused
    refused = check_report(
        report, head_sha=CLEAN_SHA, ancestors={CLEAN_SHA}, reject_wiring_check=True
    )
    assert len(refused) == 1
    assert "wiring check" in refused[0]
    assert WIRING_STEP_BUDGET_TOO_SMALL in refused[0]

    latest = json.loads(promote([report], out_dir=tmp_path).read_text(encoding="utf-8"))
    problems = check_promoted(latest, head_sha=CLEAN_SHA, ancestors={CLEAN_SHA})
    assert [problem for problem in problems if "wiring check" in problem], problems


def test_report_check_still_passes_a_legitimate_final_run(tmp_path: Path) -> None:
    """The other direction, through the whole promote -> check path."""
    report = _report(
        kind="ocr",
        name="ocr_test",
        metrics={"cer": 0.087},
        n=2000,
        breakdown={
            "stub": False,
            WIRING_CHECK_KEY: assess_wiring_check(
                citable=True, steps=120_000, warmup_steps=2000, n_val=2000
            ).to_dict(),
        },
    )
    assert report.is_wiring_check is False
    latest = json.loads(promote([report], out_dir=tmp_path).read_text(encoding="utf-8"))

    assert latest["ocr"]["cer"] == 0.087
    assert latest["ocr"]["measurement"] == "measured"
    assert "wiring_check" not in latest["ocr"]
    assert check_promoted(latest, head_sha=CLEAN_SHA, ancestors={CLEAN_SHA}) == []


def test_the_generated_table_warns_about_a_non_stub_wiring_check() -> None:
    """A real-data 20-step run gets no verdict cell and carries its codes into the table."""
    report = _report(
        kind="ocr",
        name="ocr_excerpt",
        metrics={"cer": 40.4},
        breakdown={WIRING_CHECK_KEY: assess_wiring_check(steps=20).to_dict()},
    )
    text = render_markdown([report])

    assert "| ocr | cer |" in text
    assert tr("REPORT_STATUS_PASS") not in text and tr("REPORT_STATUS_FAIL") not in text
    assert WIRING_STEP_BUDGET_TOO_SMALL in text
