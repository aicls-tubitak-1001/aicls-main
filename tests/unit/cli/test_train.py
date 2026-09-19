"""``aicls train`` / ``aicls profile``: the approval gate in both directions (ADR-0012).

Two failure modes matter equally and both are asserted here:

* a gate that **leaks** — a real-data run must exit 3 and leave no run directory behind, and
  ``--i-have-approval`` on its own must not be enough (the lock is triple);
* a gate that is **too strict** — the synthetic smoke configuration must run the real two-step
  loop on CPU with no approval at all, or CI and every developer are blocked.

The synthetic run here is a wiring check, not a measurement: a randomly initialised 60k-parameter
network over generated Bézier scribbles. The test asserts the plumbing (a run directory, a
manifest, a fit report), never a quality number.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.unit.cli.conftest import (
    OCR_REAL_CONFIG,
    OCR_SMOKE_CONFIG,
    REPO_ROOT,
    payload_of,
    problem_of,
    write_yaml,
)

pytestmark = pytest.mark.usefixtures("cli_env")


def _cuda_available() -> bool:
    """Whether this host has a usable CUDA device (the container does, Windows does not)."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a dev dependency here
        return False
    return bool(torch.cuda.is_available())


def test_committed_smoke_config_is_gate_free() -> None:
    """The configuration the CI loop runs must be gate-free by its dataset alone."""
    from aicls.ml.config import load_config

    cfg = load_config(OCR_SMOKE_CONFIG)
    assert cfg.is_gate_free is True
    assert cfg.data.dataset.startswith("synthetic_")
    assert cfg.train.max_steps == 2


def test_cli_train_synthetic_runs_without_gate(
    runner: CliRunner, app_module: object, tmp_path: Path
) -> None:
    """The 2-step synthetic smoke config trains on CPU with no approval of any kind."""
    run_dir = tmp_path / "run"
    result = runner.invoke(
        app_module,
        [
            "train",
            "--config",
            str(OCR_SMOKE_CONFIG),
            "--run-dir",
            str(run_dir),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["plan"]["gate_free"] is True
    assert payload["plan"]["gate_checked"] is False  # the gate was never consulted
    assert payload["report"]["steps"] == 2
    assert payload["report"]["wiring_check"] is True, "a synthetic run is never evidence"
    assert (run_dir / "fit_report.json").is_file()
    assert (run_dir / "run_manifest.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()


def test_cli_train_real_refuses_without_gate(
    runner: CliRunner, app_module: object, data_root: Path
) -> None:
    """A real-data run exits 3 and creates nothing: the refusal happens before any work."""
    before = sorted(p.name for p in (data_root / "runs").iterdir())
    result = runner.invoke(app_module, ["train", "--config", str(OCR_REAL_CONFIG)])
    assert result.exit_code == 3, result.stdout + result.stderr
    assert "GATE_BLOCKED" in result.stderr
    after = sorted(p.name for p in (data_root / "runs").iterdir())
    assert after == before, "a refused run must not leave a run directory behind"


def test_cli_train_real_refuses_with_flag_alone(runner: CliRunner, app_module: object) -> None:
    """``--i-have-approval`` is one of three conditions, never sufficient on its own.

    The assertion is deliberately about the *flag*, not about the ledger. This test used to
    require ``{"env", "ledger"} <= missing``, which quietly encoded "the committed ledger has no
    approved TRAINING row" as a fact; when the gates were opened for the OCR run on 2026-09-08
    the row appeared and the test failed for being right about the wrong thing. The ledger half
    of the triple lock is pinned where it belongs — against fixture ledgers in
    ``tests/unit/core/test_gates.py``, and against the committed file in
    ``test_repo_ledger_grants_only_what_something_actually_needs`` — so this test asserts only
    what it is here to assert: the flag alone does not open the gate, and the flag is credited.
    """
    result = runner.invoke(
        app_module, ["train", "--config", str(OCR_REAL_CONFIG), "--i-have-approval", "--json"]
    )
    assert result.exit_code == 3
    problem = problem_of(result.stderr)
    assert problem["code"] == "GATE_BLOCKED"
    assert problem["gate"] == "TRAINING"
    assert "env" in problem["missing"], "AICLS_GATE_TRAINING is unset in the test environment"
    assert "cli_flag" not in problem["missing"]
    assert problem["missing"], "a blocked gate must name at least one unmet condition"


def test_cli_train_synthetic_flag_rewrites_a_real_config(
    runner: CliRunner, app_module: object
) -> None:
    """``--synthetic`` makes a run gate-free by changing the data, not by skipping the check."""
    result = runner.invoke(
        app_module,
        ["train", "--config", str(OCR_REAL_CONFIG), "--synthetic", "--dry-run", "--json"],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["dataset"] == "synthetic_ink"
    assert payload["gate_free"] is True
    assert payload["gate_checked"] is False
    assert payload["dry_run"] is True


def test_train_synthetic_flag_never_changes_a_real_dataset_silently() -> None:
    """Without ``--synthetic`` the dataset of a real config is untouched (no accidental swap)."""
    from aicls.cli.train import resolve_config

    cfg = resolve_config(OCR_REAL_CONFIG, [], synthetic=False)
    assert cfg.data.dataset == "mathwriting"
    assert cfg.is_gate_free is False


def test_cli_train_override_is_applied(runner: CliRunner, app_module: object) -> None:
    """``--override key=value`` reaches the resolved configuration."""
    result = runner.invoke(
        app_module,
        [
            "train",
            "--config",
            str(OCR_SMOKE_CONFIG),
            "--override",
            "data.batch_size=2",
            "--dry-run",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert payload_of(result.stdout)["batch_size"] == 2


def test_cli_train_accepts_set_as_a_spelling_of_override(
    runner: CliRunner, app_module: object
) -> None:
    """``training/scripts/train.py`` spells this flag ``--set``; the CLI spelled it ``--override``.

    Two doors onto the same trainer that disagree about the name of the same flag is a trap an
    operator only finds mid-run. Both names now work at both doors.
    """
    result = runner.invoke(
        app_module,
        [
            "train",
            "--config",
            str(OCR_SMOKE_CONFIG),
            "--set",
            "data.batch_size=2",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert payload_of(result.stdout)["batch_size"] == 2


def test_cli_train_resumes_from_a_checkpoint(
    runner: CliRunner, app_module: object, tmp_path: Path
) -> None:
    """Continuing an interrupted run must not require leaving the CLI for the script.

    Until 2026-09-09 ``--resume`` existed only on ``training/scripts/train.py``. A nine-hour run
    interrupted at step 18000 is exactly when nobody wants to discover that the command they have
    been using all along cannot pick it up.
    """
    first = tmp_path / "first"
    result = runner.invoke(
        app_module,
        ["train", "--config", str(OCR_SMOKE_CONFIG), "--run-dir", str(first), "--json"],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    checkpoints = sorted((first / "checkpoints").glob("step-*"))
    assert checkpoints, "the smoke run wrote no checkpoint to resume from"

    second = tmp_path / "second"
    result = runner.invoke(
        app_module,
        [
            "train",
            "--config",
            str(OCR_SMOKE_CONFIG),
            "--run-dir",
            str(second),
            "--resume",
            str(checkpoints[-1]),
            # --max-steps cannot raise a budget, only lower it, so the room to continue into has
            # to come from the configuration itself.
            "--set",
            "train.max_steps=4",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["report"]["resumed_from"].endswith(checkpoints[-1].name)
    assert payload["report"]["steps"] == 4, (
        "FitReport.steps is absolute, resumes included: a run that picked up at step 2 and was "
        "given a budget of 4 has performed 4 steps, not 2"
    )


def test_cli_train_takes_the_single_gpu_lock_for_a_gated_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Until 2026-09-09 it did not, and training/scripts/train.py always did.

    One door taking the workstation's single-job lock and the other ignoring it means the
    protection can be walked straight past by using the other command - which is exactly the
    two-trainings-on-one-GPU case the lock exists to prevent.
    """
    from aicls.cli import train as cli_train
    from aicls.ml.config import load_config

    cfg = load_config(OCR_REAL_CONFIG)
    assert not cfg.is_gate_free, "this test needs a gated config to mean anything"
    taken: list[str] = []

    class _Lock:
        def __init__(self, **kwargs: Any) -> None:
            taken.append(str(kwargs.get("job", "")))

        def __enter__(self) -> _Lock:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    import aicls.ml.queue as queue_module

    monkeypatch.setattr(queue_module, "TrainingQueue", _Lock)
    with cli_train._queue_for(cfg, use_queue=True, force=False):
        pass

    assert taken == [cfg.run.name]


def test_cli_train_leaves_the_lock_alone_for_a_gate_free_smoke_run() -> None:
    """A two-step CPU smoke run must not probe nvidia-smi or touch training/queue.lock.

    The script draws the line in the same place (`use_queue`), and a queue taken by every unit
    test is noise that teaches people to pass --no-queue reflexively.
    """
    from contextlib import nullcontext

    from aicls.cli import train as cli_train
    from aicls.ml.config import load_config

    cfg = load_config(OCR_SMOKE_CONFIG)
    assert cfg.is_gate_free

    assert isinstance(cli_train._queue_for(cfg, use_queue=True, force=False), type(nullcontext()))


def test_cli_train_refuses_resume_for_the_whisper_family(
    runner: CliRunner, app_module: object, tmp_path: Path
) -> None:
    """Whisper fine-tunes through Hugging Face, which owns its own resume.

    Accepting the flag and starting from scratch is the one outcome worse than refusing: it looks
    like it worked.
    """
    result = runner.invoke(
        app_module,
        [
            "train",
            "--config",
            str(REPO_ROOT / "training" / "configs" / "asr" / "smoke_synthetic.yaml"),
            "--run-dir",
            str(tmp_path / "run"),
            "--resume",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert "resume" in (result.stdout + result.stderr).lower()


def test_cli_train_refuses_an_unknown_override(runner: CliRunner, app_module: object) -> None:
    """A typo in an override is a validation error (exit 1), not a silently ignored flag."""
    result = runner.invoke(
        app_module,
        ["train", "--config", str(OCR_SMOKE_CONFIG), "--override", "data.nope=1", "--dry-run"],
    )
    assert result.exit_code == 1
    assert "VALIDATION_ERROR" in result.stderr


def test_cli_train_refuses_a_family_without_a_trainer(
    runner: CliRunner, app_module: object, tmp_path: Path
) -> None:
    """wav2vec2 has no fine-tuning wrapper yet; refusing beats pretending it is supported."""
    config = write_yaml(
        tmp_path / "w2v.yaml",
        "run: {name: w2v_smoke}\n"
        "data: {dataset: synthetic_audio, mode: online, n_samples: 4, batch_size: 2,"
        " feature_extractor: log_mel}\n"
        "model: {family: wav2vec2, encoder: null, d_model: 32, layers: 2, heads: 2}\n"
        "train: {max_steps: 1, allow_dirty: true}\n",
    )
    result = runner.invoke(app_module, ["train", "--config", str(config)])
    assert result.exit_code == 1
    assert "family_not_supported_by_cli" in result.stderr


def test_cli_profile_synthetic_writes_a_report(
    runner: CliRunner, app_module: object, tmp_path: Path
) -> None:
    """``aicls profile`` runs gate-free on synthetic data and reports null VRAM on CPU."""
    out = tmp_path / "metrics"
    result = runner.invoke(
        app_module,
        [
            "profile",
            "--config",
            str(OCR_SMOKE_CONFIG),
            "--steps",
            "3",
            "--warmup",
            "1",
            "--run-dir",
            str(tmp_path / "profile-run"),
            "--out",
            str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["measured_steps"] == 2
    # The assertion is "the report tells the truth about the device it ran on", not "this box has
    # no GPU". Hard-coding "cpu" made the test pass on the CPU-only Windows venv and fail inside
    # the CUDA container, where `select_device` legitimately picks cuda for the same config —
    # the identical mistake that `tests/unit/ml/test_profile.py` carried.
    cuda = _cuda_available()
    assert payload["device"] == ("cuda" if cuda else "cpu")
    if cuda:
        assert isinstance(payload["peak_vram_mb"], float), "a CUDA run must report real VRAM"
    else:
        assert payload["peak_vram_mb"] is None, "CPU has no VRAM; a fake 0.0 would be a lie"
    assert payload["wiring_check"] is True
    assert list(out.glob("profile_*.json"))


def test_cli_profile_real_refuses_without_gate(runner: CliRunner, app_module: object) -> None:
    """Profiling real data is a training run and is gated exactly like one."""
    result = runner.invoke(
        app_module, ["profile", "--config", str(OCR_REAL_CONFIG), "--steps", "2"]
    )
    assert result.exit_code == 3
    assert "GATE_BLOCKED" in result.stderr


def test_cli_train_whisper_refuses_a_real_config_without_local_weights(
    runner: CliRunner, app_module: object, tmp_path: Path
) -> None:
    """The ASR route refuses honestly when a checkpoint is named but absent: no hub id, no fetch.

    ``--synthetic`` takes the TRAINING gate out of the picture (the dataset is rewritten to
    ``synthetic_audio``), so what is left is ``model.base_dir`` pointing at a directory the
    ``WEIGHTS_DOWNLOAD`` gate has not populated. A named-but-missing checkpoint is an error; it
    is never quietly replaced by random weights, because a WER from those would be a fabrication.
    """
    config = REPO_ROOT / "training" / "configs" / "asr" / "whisper_small_ft.yaml"
    result = runner.invoke(
        app_module,
        ["train", "--config", str(config), "--synthetic", "--run-dir", str(tmp_path / "run")],
    )
    assert result.exit_code == 1, result.stdout + result.stderr
    assert "WEIGHTS_DOWNLOAD" in result.stderr or "weights_dir_missing" in result.stderr
    assert not (tmp_path / "run" / "checkpoints").exists()


def test_cli_train_whisper_smoke_config_needs_no_weights_at_all(tmp_path: Path) -> None:
    """``asr/smoke_synthetic.yaml`` sets ``model.base_dir: null`` and must stay runnable.

    Wave 5 shipped that configuration and a fine-tuner that demanded a local checkpoint, so the
    one ASR configuration needing no approval could not run through this command. The full run
    is asserted in ``tests/integration/test_cli_contracts.py`` (it needs ``accelerate``); what is
    asserted here is that the CLI's own route resolves without any weights on disk.
    """
    pytest.importorskip("transformers")
    from aicls.ml.config import load_config
    from aicls.ml.whisper_ft import WhisperFineTuner

    cfg = load_config(REPO_ROOT / "training" / "configs" / "asr" / "smoke_synthetic.yaml")
    assert cfg.is_gate_free is True
    assert cfg.model.base_dir is None

    tuner = WhisperFineTuner(cfg, run_dir=tmp_path / "run", allow_dirty=True)
    assert tuner.random_init is True
    assert tuner.weights_dir is None
    assert tuner.processor.tokenizer.vocab_size > 0
