"""Unit tests for :mod:`aicls.ml.checkpoint` (docs/ml/egitim-altyapisi.md §5).

Everything runs on CPU with a three-parameter model. The two contractual tests are
``test_checkpoint_roundtrip_contains_manifest_and_vocab`` and ``test_checkpoint_atomic_write``:
a crash between the temporary write and the ``os.replace`` must leave the previous checkpoint
intact and must not leave a directory that :func:`aicls.ml.checkpoint.load_checkpoint` accepts.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import random
import shutil
from typing import Any

import pytest

from aicls.core.errors import AiclsError
from aicls.core.hashing import sha256_file
from aicls.core.paths import long_path
from aicls.ml import checkpoint as c
from aicls.ml.manifest import RunManifest

torch = pytest.importorskip("torch")

SHA = "0123456789abcdef0123456789abcdef01234567"
CFG: dict[str, Any] = {
    "run": {"name": "ckpt_unit", "seed": 1234},
    "model": {"family": "seq2seq_image", "d_model": 32},
}
VOCAB: dict[str, int] = {"<s>": 0, "</s>": 1, "x": 2, "^": 3}


def _manifest() -> RunManifest:
    """A manifest built directly (no git subprocess) — the trainer passes a captured one."""
    return RunManifest(
        run_id="01JMANIFEST0000000000000000",
        run_name="ckpt_unit",
        started_at=datetime.now(UTC),
        git_sha=SHA,
        git_available=True,
        config_hash="c0ffee",
    )


def _model(seed: int = 0) -> Any:
    torch.manual_seed(seed)
    return torch.nn.Linear(4, 3)


def _save(
    path: Path, *, step: int = 10, model: Any = None, metrics: dict[str, float] | None = None
) -> c.CheckpointMeta:
    net = model if model is not None else _model()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    optimizer.step()  # no gradients yet: only silences the scheduler-order warning
    scheduler.step()
    scheduler.step()
    return c.save_checkpoint(
        path,
        model=net,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        cfg=CFG,
        manifest=_manifest(),
        metrics=metrics if metrics is not None else {"cer": 0.30, "loss": 1.5},
        vocab=VOCAB,
    )


# --------------------------------------------------------------------------------- round trip
def test_checkpoint_roundtrip_contains_manifest_and_vocab(tmp_path: Path) -> None:
    """Contractual: the §5 layout is written and every part comes back."""
    ckpt = tmp_path / "ocr-0.1.0"
    original = _model()
    meta = _save(ckpt, step=10, model=original)

    written = {p.name for p in ckpt.iterdir()}
    assert written == {
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "config.yaml",
        "run_manifest.json",
        "vocab.json",
        "metrics.json",
        "rng_state.pt",
        "checkpoint.json",
    }
    assert meta.step == 10
    assert meta.model_sha256 == sha256_file(ckpt / "model.safetensors")

    fresh = _model(seed=99)
    assert not torch.allclose(fresh.weight, original.weight)
    optimizer = torch.optim.SGD(fresh.parameters(), lr=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    loaded = c.load_checkpoint(
        ckpt, model=fresh, optimizer=optimizer, scheduler=scheduler, restore_rng=False
    )

    assert torch.allclose(fresh.weight, original.weight)
    assert torch.allclose(fresh.bias, original.bias)
    assert loaded.step == 10
    assert loaded.vocab == VOCAB
    assert loaded.manifest["git_sha"] == SHA
    assert loaded.config["model"]["d_model"] == 32
    assert loaded.metrics == {"cer": 0.30, "loss": 1.5}
    assert loaded.model_sha256 == meta.model_sha256
    # 0.1 * gamma**2 after the two scheduler steps taken before the save (not the fresh 0.9)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.025)
    assert scheduler.last_epoch == 2


def test_checkpoint_metadata_records_provenance(tmp_path: Path) -> None:
    meta = _save(tmp_path / "ckpt", step=42)
    document = json.loads((tmp_path / "ckpt" / c.META_FILENAME).read_text(encoding="utf-8"))
    assert document["step"] == 42
    assert document["model_sha256"] == meta.model_sha256
    assert document["git_sha"] == SHA
    assert document["run_id"] == "01JMANIFEST0000000000000000"
    assert document["config_hash"] == "c0ffee"
    assert document["format"] in {"safetensors", "torch"}
    assert document["metrics"]["cer"] == 0.30
    assert set(document["files"]) >= {"model.safetensors", "config.yaml"}


def test_checkpoint_restores_rng_state(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    _save(ckpt)
    after_save = torch.randn(4)
    c.load_checkpoint(ckpt, restore_rng=True)
    assert torch.equal(torch.randn(4), after_save)


def test_save_without_optimizer_or_vocab_is_allowed(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    meta = c.save_checkpoint(ckpt, model=_model(), step=1, save_rng=False)
    assert meta.has_optimizer is False
    assert meta.has_rng is False
    names = {p.name for p in ckpt.iterdir()}
    assert names == {"model.safetensors", "checkpoint.json"}
    loaded = c.load_checkpoint(ckpt)
    assert loaded.vocab is None
    assert loaded.config == {}


def test_save_checkpoint_replaces_an_existing_checkpoint(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    _save(ckpt, step=1)
    _save(ckpt, step=2, model=_model(seed=7))
    assert c.load_checkpoint(ckpt).step == 2
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt"]


# ------------------------------------------------------------------------------ atomic writes
def test_checkpoint_atomic_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Contractual: a crash mid-write keeps the old checkpoint and leaves nothing loadable."""
    ckpt = tmp_path / "ckpt"
    first = _save(ckpt, step=10)

    def explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("simulated crash between the temp write and os.replace")

    monkeypatch.setattr(c, "_write_meta", explode)
    with pytest.raises(RuntimeError):
        _save(ckpt, step=20, model=_model(seed=5))

    survivor = c.load_checkpoint(ckpt, restore_rng=False)
    assert survivor.step == 10
    assert survivor.model_sha256 == first.model_sha256
    # No half-written directory is left behind at all, so none can be loaded.
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt"]


def test_checkpoint_survives_a_run_root_past_windows_max_path(tmp_path: Path) -> None:
    """Reproduced 2026-09-08: a 274-character run root broke ``_write_payload`` (``WinError 3``).

    Two things caused it and both are fixed here: the temporary directory name was ~38 characters
    longer than the checkpoint directory it stands in for, and nothing asked Windows to leave the
    260-character ``MAX_PATH`` behind. ``AICLS_DATA_ROOT`` under a deep OneDrive/temp path is
    normal, so this is a real operating condition, not a pathological one.

    On POSIX the same path is simply legal, and the test still proves the writer handles it.
    """
    deep = tmp_path
    while len(str(deep)) < 240:
        deep = deep / ("run-directory-segment" * 2)
    ckpt = deep / "checkpoints" / "step-00002000"
    assert len(str(ckpt)) > 260, len(str(ckpt))

    long_path(ckpt).parent.mkdir(parents=True, exist_ok=True)
    _save(ckpt, step=2000)
    # Saving twice also exercises `_publish`'s rename-away-then-replace, whose backup directory
    # is a second long name in the same parent.
    meta = _save(ckpt, step=2000, model=_model(seed=7))

    assert meta.step == 2000
    assert long_path(ckpt / "model.safetensors").is_file()
    loaded = c.load_checkpoint(ckpt, restore_rng=False)
    assert loaded.step == 2000
    assert loaded.model_sha256 == meta.model_sha256
    assert loaded.metrics == {"cer": 0.30, "loss": 1.5}
    siblings = [p.name for p in long_path(ckpt).parent.iterdir()]
    assert siblings == ["step-00002000"], f"staging directories were left behind: {siblings}"

    shutil.rmtree(long_path(tmp_path / ("run-directory-segment" * 2)), ignore_errors=True)


def test_temporary_directory_name_stays_close_to_the_final_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The staging name is what pushed a legal target over the limit — keep the overhead small."""
    seen: list[Path] = []
    original = c._write_payload

    def record(tmp_dir: Path, **kwargs: Any) -> c.CheckpointMeta:
        seen.append(tmp_dir)
        return original(tmp_dir, **kwargs)

    monkeypatch.setattr(c, "_write_payload", record)
    target = tmp_path / "step-00002000"
    _save(target, step=2000)

    assert len(seen) == 1
    overhead = len(seen[0].name) - len(target.name)
    assert 0 < overhead <= 12, f"{seen[0].name!r} is {overhead} characters longer than the target"


def test_load_checkpoint_refuses_a_directory_without_metadata(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    _save(ckpt)
    (ckpt / c.META_FILENAME).unlink()
    with pytest.raises(AiclsError) as excinfo:
        c.load_checkpoint(ckpt)
    assert excinfo.value.details["reason"] == "checkpoint_incomplete"


def test_load_checkpoint_detects_a_tampered_model_file(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    _save(ckpt)
    (ckpt / "model.safetensors").write_bytes(b"corrupted")
    with pytest.raises(AiclsError) as excinfo:
        c.load_checkpoint(ckpt)
    assert excinfo.value.details["reason"] == "checkpoint_sha_mismatch"


def test_load_checkpoint_refuses_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(AiclsError) as excinfo:
        c.load_checkpoint(tmp_path / "absent")
    assert excinfo.value.details["reason"] == "checkpoint_missing"


def test_load_checkpoint_can_skip_hash_verification(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    _save(ckpt)
    (ckpt / "model.safetensors").write_bytes(b"corrupted")
    loaded = c.load_checkpoint(ckpt, verify_sha=False, restore_rng=False)
    assert loaded.step == 10


# ------------------------------------------------------------------------------------- saver
def test_checkpoint_saver_keeps_last_n_and_best_one(tmp_path: Path) -> None:
    saver = c.CheckpointSaver(tmp_path, keep_last=2, metric="cer", mode="min")
    net = _model()
    values = {10: 0.50, 20: 0.20, 30: 0.35, 40: 0.40}
    for step, cer in values.items():
        saver.save(step, model=net, metrics={"cer": cer}, save_rng=False)

    kept = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert kept == ["step-00000020", "step-00000030", "step-00000040"]
    assert saver.best_step == 20
    assert saver.best_value == pytest.approx(0.20)
    assert saver.last_dir is not None and saver.last_dir.name == "step-00000040"
    best = json.loads((tmp_path / c.BEST_POINTER_FILENAME).read_text(encoding="utf-8"))
    assert best["step"] == 20
    assert best["metric"] == "cer"
    assert best["value"] == pytest.approx(0.20)
    last = json.loads((tmp_path / c.LAST_POINTER_FILENAME).read_text(encoding="utf-8"))
    assert last["step"] == 40
    assert c.load_checkpoint(saver.best_dir, restore_rng=False).step == 20


def test_checkpoint_saver_maximises_when_mode_is_max(tmp_path: Path) -> None:
    saver = c.CheckpointSaver(tmp_path, keep_last=1, metric="exprate", mode="max")
    net = _model()
    for step, value in ((10, 0.10), (20, 0.60), (30, 0.30)):
        saver.save(step, model=net, metrics={"exprate": value}, save_rng=False)
    assert saver.best_step == 20
    assert {p.name for p in tmp_path.iterdir() if p.is_dir()} == {
        "step-00000020",
        "step-00000030",
    }


def test_checkpoint_saver_without_the_metric_still_keeps_last(tmp_path: Path) -> None:
    saver = c.CheckpointSaver(tmp_path, keep_last=1, metric="cer", mode="min")
    net = _model()
    saver.save(1, model=net, metrics={"loss": 2.0}, save_rng=False)
    saver.save(2, model=net, metrics={"loss": 1.0}, save_rng=False)
    assert saver.best_dir is None
    assert {p.name for p in tmp_path.iterdir() if p.is_dir()} == {"step-00000002"}


def test_checkpoint_saver_rejects_an_unknown_mode(tmp_path: Path) -> None:
    with pytest.raises(AiclsError):
        c.CheckpointSaver(tmp_path, mode="sideways")  # type: ignore[arg-type]


def test_checkpoint_saver_rejects_a_negative_keep_last(tmp_path: Path) -> None:
    with pytest.raises(AiclsError) as excinfo:
        c.CheckpointSaver(tmp_path, keep_last=-1)
    assert excinfo.value.details["reason"] == "invalid_keep_last"


def test_checkpoint_saver_rejects_a_retention_that_keeps_nothing(tmp_path: Path) -> None:
    with pytest.raises(AiclsError) as excinfo:
        c.CheckpointSaver(tmp_path, keep_last=0, keep_best=0)
    assert excinfo.value.details["reason"] == "invalid_retention"


def test_checkpoint_saver_keep_last_zero_keeps_only_the_best(tmp_path: Path) -> None:
    """``cfg.checkpoint.keep_last: 0`` means "the best checkpoint and nothing else"."""
    saver = c.CheckpointSaver(tmp_path, keep_last=0, keep_best=1, metric="cer", mode="min")
    net = _model()
    saver.save(1, model=net, metrics={"cer": 0.5}, save_rng=False)
    saver.save(2, model=net, metrics={"cer": 0.9}, save_rng=False)
    assert {p.name for p in tmp_path.iterdir() if p.is_dir()} == {"step-00000001"}


def test_checkpoint_saver_without_keep_best_prunes_the_best_too(tmp_path: Path) -> None:
    saver = c.CheckpointSaver(tmp_path, keep_last=1, keep_best=0, metric="cer", mode="min")
    net = _model()
    saver.save(1, model=net, metrics={"cer": 0.1}, save_rng=False)
    saver.save(2, model=net, metrics={"cer": 0.9}, save_rng=False)
    assert {p.name for p in tmp_path.iterdir() if p.is_dir()} == {"step-00000002"}


def test_checkpoint_saver_defaults_match_the_config_defaults(tmp_path: Path) -> None:
    """Cross-module contract: one retention default, not two (aicls.ml.config.CheckpointCfg)."""
    config_module = pytest.importorskip("aicls.ml.config")
    defaults = config_module.CheckpointCfg()
    saver = c.CheckpointSaver(tmp_path)
    assert (saver.keep_last, saver.keep_best, saver.metric, saver.mode) == (
        defaults.keep_last,
        defaults.keep_best,
        defaults.metric,
        defaults.mode,
    )


# ------------------------------------------------------------------------------- rng helpers
def test_rng_state_roundtrip_is_reproducible() -> None:
    state = c.rng_state()
    first = (torch.randn(3), tuple(random.random() for _ in range(2)))
    c.restore_rng_state(state)
    second = (torch.randn(3), tuple(random.random() for _ in range(2)))
    assert torch.equal(first[0], second[0])
    assert first[1] == second[1]
