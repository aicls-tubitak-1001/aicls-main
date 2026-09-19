"""The harness: synthetic corpora, the stub runs and the files they produce.

The full run over the committed corpora is an integration test
(``tests/integration/test_eval_contracts.py``); here only the cheap parts are exercised.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aicls.eval.harness import (
    DATASET_VERSIONS,
    KINDS,
    build_asr_dataset,
    build_ocr_samples,
    default_paths,
    main,
    run_all,
    strip_labels,
    synthetic_fingerprint,
)
from aicls.eval.provenance import GitInfo
from aicls.eval.report import LATEST_FILENAME

CLEAN = GitInfo(sha="e" * 40, branch="main", dirty=False, dirty_known=True, source="status")


def test_default_paths_point_at_the_committed_corpora() -> None:
    """The harness reads the sets that are in git, not something under AICLS_DATA_ROOT."""
    paths = default_paths()
    for key in ("valid_inputs", "problems", "commands", "fetch_test"):
        assert paths[key].is_file(), key
    assert paths["book_sample"].is_dir()
    assert paths["out_dir"].name == "metrics"


def test_kinds_cover_every_runner() -> None:
    """One kind per contractual runner (docs/ml/degerlendirme.md §5)."""
    assert KINDS == ("parser", "cas", "nlu", "ocr", "asr", "book", "latency")
    assert set(DATASET_VERSIONS) <= set(KINDS)


def test_build_ocr_samples_is_deterministic_and_labelled() -> None:
    """The synthetic ink is reproducible from its seed and carries its ground truth."""
    first = build_ocr_samples(repeats=1)
    second = build_ocr_samples(repeats=1)
    assert len(first) == len(second) > 0
    assert all(sample.label for sample in first)
    assert [sample.label for sample in first] == [sample.label for sample in second]
    assert first[0].all_points().shape == second[0].all_points().shape


def test_strip_labels_removes_the_ground_truth_only() -> None:
    """The control run must keep the geometry and lose only the answer."""
    labelled = build_ocr_samples(repeats=1)[:3]
    stripped = strip_labels(labelled)
    assert [sample.label for sample in stripped] == [None] * 3
    assert stripped[0].n_strokes == labelled[0].n_strokes


def test_synthetic_fingerprint_is_stable_and_order_independent() -> None:
    """A generated corpus is identified by its generator parameters."""
    left = synthetic_fingerprint({"generator": "g", "seed": 1234})
    right = synthetic_fingerprint({"seed": 1234, "generator": "g"})
    assert left == right
    assert left.startswith("sha256:")
    assert left != synthetic_fingerprint({"generator": "g", "seed": 7})


def test_build_asr_dataset_is_speaker_disjoint_and_carries_a_real_split(tmp_path: Path) -> None:
    """The split comes from ``aicls.data.make_splits`` and is written to a real split file."""
    dataset, texts = build_asr_dataset(default_paths()["commands"], n=60, work_dir=tmp_path)
    assert (tmp_path / "audio_commands_splits.json").is_file()
    assert dataset.splits_fingerprint.startswith("sha256:")
    assert len(dataset) > 0
    assert len(texts) == 60
    test_speakers = {sample.speaker for sample in dataset}
    assert test_speakers  # the test split holds whole speakers
    assert all(sample.audio.n_samples > 0 for sample in dataset)


def test_run_all_writes_reports_latest_and_a_summary(tmp_path: Path) -> None:
    """One command produces the files the documentation macros and reviewers read."""
    output = run_all(out_dir=tmp_path, kinds=("ocr",), git=CLEAN, ocr_repeats=1)
    assert len(output.reports) == 1
    report = output.reports[0]
    assert report.kind == "ocr"
    assert report.breakdown["stub"] is True
    assert report.breakdown["blinded"] is True
    assert "label_leak_control" in report.breakdown
    assert report.dataset_fingerprint.startswith("sha256:")
    assert report.citable is True

    assert output.files[0].is_file()
    latest = json.loads((tmp_path / LATEST_FILENAME).read_text(encoding="utf-8"))
    assert set(latest) == {"ocr"}
    assert latest["ocr"]["source_file"] == output.files[0].name
    summary = output.summary.read_text(encoding="utf-8")
    assert "| ocr | cer |" in summary
    assert "stub" in summary.lower()


def test_run_all_stub_ocr_cannot_echo_the_label_and_says_what_that_is_worth(
    tmp_path: Path,
) -> None:
    """The wiring check and the honesty check, in one run (KNOWN_ISSUES priority 1).

    This test used to assert ``cer == 0.0`` on purpose, because the stub was handed the sample
    with its ``label`` attached and returns that label verbatim — an oracle marking its own
    exam, whose perfect score was promoted into ``reports/metrics/latest.json``. The runner now
    blinds every sample, so the stub has to guess from its fixed curriculum table and scores
    like the lookup it is.

    ``label_leak_control`` re-runs the unblinded version deliberately: it is the receipt showing
    what the leak was worth (a perfect 0.0 against a contractual ``CER < 10 %``) and therefore
    that the blinding is load-bearing rather than decorative.
    """
    report = run_all(out_dir=tmp_path, kinds=("ocr",), git=CLEAN, ocr_repeats=1, write=False)
    ocr = report.reports[0]

    # blinded: the stub is guessing, and a guesser is nowhere near the İP-3 target
    assert ocr.metrics["cer"] > 0.5
    assert ocr.metrics["exprate"] < 0.1
    assert ocr.breakdown["blinded"] is True

    # unblinded: exactly the perfect score the repository used to publish
    control = ocr.breakdown["label_leak_control"]
    assert control["unblinded_cer"] == 0.0
    assert control["unblinded_exprate"] == 1.0
    assert control["blinded_cer"] == ocr.metrics["cer"]
    assert control["n"] == ocr.n

    # and the report says in its own words that this is not evidence
    assert ocr.breakdown["notes"].startswith("WIRING CHECK")


def test_run_all_without_writing_touches_no_file(tmp_path: Path) -> None:
    """``write=False`` is what tests use; it must not create the report directory."""
    output = run_all(
        out_dir=tmp_path / "unused", kinds=("ocr",), git=CLEAN, ocr_repeats=1, write=False
    )
    assert output.files == ()
    assert not (tmp_path / "unused").exists()


def test_main_runs_the_selected_kind_into_the_given_directory(tmp_path: Path) -> None:
    """``python -m aicls.eval --kind ocr --out <dir>`` is the developer entry point."""
    exit_code = main(["--kind", "ocr", "--out", str(tmp_path), "--ocr-repeats", "1"])
    assert exit_code == 0
    assert (tmp_path / LATEST_FILENAME).is_file()
    assert (tmp_path / "summary.md").is_file()


def test_main_rejects_an_unknown_kind(tmp_path: Path) -> None:
    """The kind list is closed; a typo must not silently run nothing."""
    with pytest.raises(SystemExit):
        main(["--kind", "handwriting", "--out", str(tmp_path)])
