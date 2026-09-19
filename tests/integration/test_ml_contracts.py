"""Cross-module contracts of :mod:`aicls.ml` (wave-5 integration gate).

Five agents wrote :mod:`aicls.ml` in parallel against declared interfaces rather than against
each other's code. These tests cross the seams between them, using the **files that are actually
written** rather than the code that claims to write them:

* the package surface (``import aicls.ml``) and its torch-free guarantee;
* ``training/configs/**.yaml`` against :mod:`aicls.ml.config`;
* ``checkpoint.metric`` against what each loop actually publishes — the contract the wave-5
  configuration owner flagged as unverified, and which was in fact broken (``val_cer`` was never
  produced by :meth:`Trainer.evaluate`, so best-by-CER selection was silently inert);
* :mod:`aicls.ml.trainer` against :mod:`aicls.ml.manifest`, :mod:`aicls.ml.tracking` and
  :mod:`aicls.ml.checkpoint`, driven by one real two-step run on the committed gate-free config;
* :mod:`aicls.ml.trainer` against :mod:`aicls.ml.whisper_ft` — ADR-0006 promises the same
  ``run_manifest.json`` and the same checkpoint layout from both loops.

**Honesty.** The training run here is a WIRING CHECK: a randomly initialised ~53k-parameter model
over deterministic synthetic Bezier scribbles, whose "validation" set is generated from the same
seed as its training set. Nothing asserted below is evidence for the CER < 10 % target; the tests
assert plumbing and provenance, and they assert that the run labels itself as a wiring check.

No test here opens a gate, sets ``AICLS_GATE_*`` or writes to the approval ledger.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "training" / "configs"
OCR_SMOKE_CONFIG = CONFIG_ROOT / "ocr" / "smoke_synthetic.yaml"

#: Names the module spec (``brief/modules/ml.json``) promises on the package itself.
SPEC_SURFACE: tuple[str, ...] = (
    "TrainConfig",
    "load_config",
    "config_hash",
    "register_model",
    "register_dataset",
    "build_model",
    "build_dataset",
    "RunManifest",
    "seed_everything",
    "capture_env",
    "Tracker",
    "JsonlTracker",
    "MlflowTracker",
    "Trainer",
    "FitReport",
    "save_checkpoint",
    "load_checkpoint",
    "Checkpoint",
    "CheckpointMeta",
    "WhisperFineTuner",
    "run_profile",
    "ProfileReport",
    "TrainingQueue",
)

#: Files every run directory must contain after a successful ``fit()`` (ADR-0007 §7).
RUN_DIR_FILES: tuple[str, ...] = (
    "run_manifest.json",
    "env.json",
    "metrics.jsonl",
    "params.json",
    "fit_report.json",
)

#: Files every checkpoint directory of the OCR loop contains (design document §5 + checkpoint.json).
CHECKPOINT_FILES: tuple[str, ...] = (
    "checkpoint.json",
    "config.yaml",
    "metrics.json",
    "model.safetensors",
    "optimizer.pt",
    "rng_state.pt",
    "run_manifest.json",
    "scheduler.pt",
    "vocab.json",
)

#: The three files :func:`aicls.ml.whisper_ft.complete_checkpoint_dir` adds to a Hugging Face
#: checkpoint so that both loops publish the same provenance next to their weights (ADR-0006).
SHARED_CHECKPOINT_FILES: tuple[str, ...] = ("config.yaml", "metrics.json", "run_manifest.json")


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter with ``src`` importable."""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "SYSTEMROOT": "C:\\Windows", "PATH": ""},
    )


# ==============================================================================================
# the package surface
# ==============================================================================================
def test_ml_package_exports_the_documented_surface() -> None:
    """``import aicls.ml`` gives the names the module spec promises, and every ``__all__`` name."""
    from aicls import ml

    missing = [name for name in SPEC_SURFACE if not hasattr(ml, name)]
    assert missing == [], f"aicls.ml does not export: {missing}"
    unresolvable = [name for name in ml.__all__ if not hasattr(ml, name)]
    assert unresolvable == [], f"__all__ lists names that do not resolve: {unresolvable}"


def test_ml_package_does_not_flatten_ambiguous_names() -> None:
    """Three names live in two submodules with different meanings; a flat alias would lie."""
    from aicls import ml
    from aicls.ml import checkpoint, tracking, trainer, whisper_ft

    assert ml.FitReport is trainer.FitReport
    assert ml.WhisperFitReport is whisper_ft.FitReport
    assert ml.FitReport is not ml.WhisperFitReport
    # metrics.jsonl (tracking) and metrics.json (checkpoint) are different files.
    assert tracking.METRICS_FILENAME != checkpoint.METRICS_FILENAME
    assert not hasattr(ml, "METRICS_FILENAME")
    # the two require_transformers() raise different typed errors, so neither may win by default
    assert not hasattr(ml, "require_transformers")


def test_importing_aicls_ml_does_not_import_torch() -> None:
    """``import aicls.ml`` must stay torch-free: the CLI pays this cost on every ``--help``.

    Checked in a subprocess, because any earlier test in this session may already have imported
    torch into this interpreter.
    """
    code = (
        "import sys, json; import aicls.ml; "
        "print(json.dumps(sorted(m for m in sys.modules "
        "if m in {'torch', 'transformers', 'accelerate', 'peft', 'torchvision'})))"
    )
    result = _run_python(code)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip()) == []


# ==============================================================================================
# config tree vs schema
# ==============================================================================================
def test_every_shipped_config_loads_and_hashes_distinctly() -> None:
    """Each ``training/configs/**.yaml`` validates, and no two of them are the same run."""
    from aicls.ml.config import config_hash, load_config

    paths = sorted(CONFIG_ROOT.rglob("*.yaml"))
    assert len(paths) >= 12, "the design document promises twelve configuration files"
    hashes = {path: config_hash(load_config(path)) for path in paths}
    assert len(set(hashes.values())) == len(hashes), f"duplicate configurations: {hashes}"


def test_only_synthetic_configs_are_gate_free() -> None:
    """A configuration is gate-free because of what it trains on, never because it says so."""
    from aicls.ml.config import load_config

    for path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        cfg = load_config(path)
        assert cfg.is_gate_free == cfg.data.dataset.startswith("synthetic_"), path


def test_shipped_checkpoint_metrics_resolve_against_the_loop_that_runs_them() -> None:
    """``checkpoint.metric`` must be a key the loop actually publishes.

    This is the contract the configuration owner flagged as unverified in wave 5, and it was
    broken: the OCR configs ask for ``val_cer`` while ``Trainer.evaluate()`` published only
    ``cer``, so ``CheckpointSaver`` never found its key and best-by-CER selection did nothing.
    The ASR side was worse — Hugging Face raises ``KeyError`` on ``metric_for_best_model``
    it cannot find, so the first evaluation of a Whisper fine-tune would have crashed.
    """
    from aicls.ml.config import load_config
    from aicls.ml.whisper_ft import WER_METRIC_KEYS, hf_metric_name

    # what each loop publishes
    ocr_keys = {"loss", "val_loss", "n_val"} | {
        f"{prefix}{name}"
        for prefix in ("", "val_")
        for name in ("cer", "ter", "wer", "exprate", "symeq")
    }
    hf_keys = {*WER_METRIC_KEYS, "loss"}

    for path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        cfg = load_config(path)
        metric = cfg.checkpoint.metric
        if cfg.model.family in ("whisper", "wav2vec2"):
            assert hf_metric_name(metric) in hf_keys, f"{path}: {metric} unknown to the HF loop"
        else:
            assert metric in ocr_keys, f"{path}: {metric} is never produced by Trainer.evaluate()"


# ==============================================================================================
# one real run: trainer x manifest x tracking x checkpoint
# ==============================================================================================
@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Two real optimizer steps from the committed gate-free configuration.

    WIRING CHECK, not a result: randomly initialised weights over generated scribbles.
    """
    from aicls.ml.config import load_config
    from aicls.ml.registry import DatasetSpec, build_dataset, build_model
    from aicls.ml.trainer import Trainer

    run_dir = tmp_path_factory.mktemp("ml-contract-run") / "run"
    cfg = load_config(OCR_SMOKE_CONFIG)
    assert cfg.is_gate_free, "this fixture must never need the TRAINING gate"
    tokenizer = DatasetSpec.from_config(cfg).tokenizer()
    trainer = Trainer(
        cfg,
        build_model(cfg, tokenizer.vocab_size),
        build_dataset(cfg, "train"),
        build_dataset(cfg, "valid"),
        run_dir=run_dir,
        tokenizer=tokenizer,
        device="cpu",
    )
    report = trainer.fit()
    return {"cfg": cfg, "trainer": trainer, "report": report, "run_dir": run_dir}


def test_run_directory_carries_the_whole_provenance_contract(smoke_run: dict[str, Any]) -> None:
    """ADR-0007: manifest, environment, metrics, params and the fit report, all in the run dir."""
    run_dir: Path = smoke_run["run_dir"]
    missing = [name for name in RUN_DIR_FILES if not (run_dir / name).is_file()]
    assert missing == [], f"run directory is incomplete: {missing}"

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["config_hash"], "the manifest must fingerprint the resolved configuration"
    assert manifest["python"].startswith("3.11")
    assert manifest["torch"], "the torch version belongs in the manifest"
    # "A dirty tree can never be citable" is the property worth protecting, but asserting
    # `citable is False` outright only tested it when the developer happened to have uncommitted
    # work — and inverted into a failure on every clean checkout, which is the case that matters.
    # Assert the biconditional instead: it holds in both states and is strictly stronger.
    assert manifest["citable"] == (bool(manifest["git_sha"]) and not manifest["dirty"])
    if manifest["dirty"]:
        assert manifest["citable"] is False


def test_one_run_directory_carries_exactly_one_run_id(smoke_run: dict[str, Any]) -> None:
    """The manifest, the metric rows, the params record and the report agree on ``run_id``.

    ``RunManifest`` and ``JsonlTracker`` each mint a ULID of their own. Before the wave-5 gate
    they kept both, so one run directory held two different ids and nothing could join
    ``metrics.jsonl`` to ``run_manifest.json`` except by directory name.
    """
    run_dir: Path = smoke_run["run_dir"]
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    params = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "fit_report.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = {manifest["run_id"], params["run_id"], report["run_id"]} | {row["run_id"] for row in rows}
    assert len(ids) == 1, f"one run directory, {len(ids)} run ids: {sorted(ids)}"


def test_validation_metrics_reach_both_the_tracker_and_the_checkpoint_selector(
    smoke_run: dict[str, Any],
) -> None:
    """The configured ``checkpoint.metric`` selects a best checkpoint on a real run.

    ``training/configs/ocr/smoke_synthetic.yaml`` asks for ``val_cer``. If the loop does not
    publish that key, ``best.json`` is never written and ``fit_report.best_value`` stays null —
    which is exactly what the committed tree did before this gate.
    """
    run_dir: Path = smoke_run["run_dir"]
    cfg = smoke_run["cfg"]
    report = smoke_run["report"]

    valid_rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["split"] == "valid"
    ]
    assert valid_rows, "a validation pass must be recorded"
    assert cfg.checkpoint.metric in valid_rows[-1]["metrics"]

    assert report.best_metric == cfg.checkpoint.metric
    assert report.best_value is not None, "best-by-metric selection produced no value"
    assert report.best_step is not None
    assert (run_dir / "checkpoints" / "best.json").is_file()


def test_checkpoint_directory_is_complete_and_reloadable(smoke_run: dict[str, Any]) -> None:
    """Every promised file is there, and the saved weights load back into a fresh model."""
    from aicls.ml.checkpoint import load_checkpoint
    from aicls.ml.registry import DatasetSpec, build_model

    run_dir: Path = smoke_run["run_dir"]
    cfg = smoke_run["cfg"]
    steps = sorted((run_dir / "checkpoints").glob("step-*"))
    assert steps, "the loop wrote no checkpoint"
    latest = steps[-1]
    missing = [name for name in CHECKPOINT_FILES if not (latest / name).is_file()]
    assert missing == [], f"checkpoint {latest.name} is incomplete: {missing}"

    tokenizer = DatasetSpec.from_config(cfg).tokenizer()
    fresh = build_model(cfg, tokenizer.vocab_size)
    checkpoint = load_checkpoint(latest, model=fresh, verify_sha=True)
    assert checkpoint.step == int(latest.name.removeprefix("step-"))
    assert checkpoint.config, "the resolved configuration travels with the weights"
    assert checkpoint.vocab, "the vocabulary travels with the weights (ADR-0006)"


def test_the_report_labels_itself_a_wiring_check(smoke_run: dict[str, Any]) -> None:
    """Synthetic data must never produce a document a reader could mistake for evidence."""
    from aicls.ml.trainer import WIRING_CHECK_NOTE

    report = smoke_run["report"]
    assert report.wiring_check is True
    assert WIRING_CHECK_NOTE in report.notes
    # `citable` deliberately is NOT asserted here. It answers "can this number be traced to a
    # commit", which depends on the checkout and is true on a clean tree; the question this test
    # asks is "could a reader mistake this for evidence", which is `wiring_check`. Conflating the
    # two made this test pass only while someone had uncommitted edits.


# ==============================================================================================
# the two loops publish the same records (ADR-0006)
# ==============================================================================================
def test_both_loops_write_the_same_manifest_schema(
    smoke_run: dict[str, Any], tmp_path: Path
) -> None:
    """The OCR loop and the Whisper wrapper must produce interchangeable manifests.

    Compared as written files: the callback is driven directly (``accelerate`` is not installed,
    so the Hugging Face trainer itself cannot be constructed on this host) and the two documents
    are compared key for key.
    """
    from aicls.ml.config import load_config
    from aicls.ml.whisper_ft import AiclsManifestCallback

    asr_cfg = load_config(CONFIG_ROOT / "asr" / "smoke_synthetic.yaml")
    assert asr_cfg.is_gate_free
    whisper_run = tmp_path / "whisper-run"
    callback = AiclsManifestCallback(asr_cfg, whisper_run, allow_dirty=True)
    callback.on_train_begin()

    ocr_manifest = json.loads(
        (smoke_run["run_dir"] / "run_manifest.json").read_text(encoding="utf-8")
    )
    asr_manifest = json.loads((whisper_run / "run_manifest.json").read_text(encoding="utf-8"))
    assert set(ocr_manifest) == set(asr_manifest), "the two loops disagree on the manifest schema"
    for field in ("git_sha", "dirty", "citable", "python", "torch", "schema_version"):
        assert ocr_manifest[field] == asr_manifest[field], field
    assert (whisper_run / "env.json").is_file()


def test_whisper_checkpoints_gain_the_shared_layout(tmp_path: Path) -> None:
    """``complete_checkpoint_dir`` puts our three provenance files next to the HF weights.

    That is the whole of ADR-0006's "same checkpoint layout" promise that can hold: Hugging Face
    names its directories ``checkpoint-<step>`` and ours ``step-XXXXXXXX``, and only ours carries
    ``checkpoint.json``/``vocab.json``. The *shared* part is the three files a reader (and
    ``aicls models import``) needs, and it is asserted here against real files.
    """
    from aicls.ml.config import load_config
    from aicls.ml.manifest import RunManifest
    from aicls.ml.whisper_ft import complete_checkpoint_dir

    cfg = load_config(CONFIG_ROOT / "asr" / "smoke_synthetic.yaml")
    hf_ckpt = tmp_path / "checkpoints" / "checkpoint-2"
    hf_ckpt.mkdir(parents=True)
    (hf_ckpt / "model.safetensors").write_bytes(b"")

    written = complete_checkpoint_dir(
        hf_ckpt,
        cfg=cfg,
        manifest=RunManifest.capture(cfg, allow_dirty=True),
        metrics={"eval_wer": 1.0},
    )
    assert {path.name for path in written} == set(SHARED_CHECKPOINT_FILES)
    for name in SHARED_CHECKPOINT_FILES:
        assert (hf_ckpt / name).is_file(), name
    assert set(CHECKPOINT_FILES) >= set(SHARED_CHECKPOINT_FILES), (
        "the OCR checkpoint must be a superset of the shared layout"
    )
