"""Unit tests for :mod:`aicls.ml.manifest` (ADR-0007, docs/ml/egitim-altyapisi.md §7).

Git-dependent behaviour is driven through a **fake process runner**, so the assertions do not
depend on the state of this checkout. One test calls the real ``git`` subprocess (read-only) to
prove that the production path works on this host, and one test proves that a missing ``git``
degrades to a recorded "unknown" state instead of crashing.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
from typing import Any

import pytest

from aicls.core.errors import AiclsError
from aicls.ml import manifest as m

SHA = "0123456789abcdef0123456789abcdef01234567"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _config(**over: Any) -> dict[str, Any]:
    """A plain-mapping stand-in for :class:`aicls.ml.config.TrainConfig`."""
    cfg: dict[str, Any] = {
        "run": {"name": "unit_run", "seed": 1234, "deterministic": True},
        "data": {"dataset": "synthetic_ink", "root": "", "batch_size": 4},
        "model": {"family": "seq2seq_image", "d_model": 32},
        "train": {"epochs": 1, "allow_dirty": False},
    }
    cfg.update(over)
    return cfg


def _fake_git(
    monkeypatch: pytest.MonkeyPatch, *, sha: str = SHA, dirty: bool = False, branch: str = "main"
) -> None:
    """Patch :func:`aicls.ml.manifest.run_process` with a scripted ``git``."""

    def fake_run(
        args: list[str] | tuple[str, ...], *, cwd: Path | None = None, timeout_s: float = 10.0
    ) -> m.ProcessResult | None:
        argv = list(args)
        if argv[0] != "git":  # nvidia-smi and friends are "not installed" for this test
            return None
        if "status" in argv:
            out = " M src/aicls/ml/manifest.py\n?? new_file.py\n" if dirty else ""
        elif "diff" in argv:
            out = "diff --git a/x b/x\n+one line\n" if dirty else ""
        elif "--abbrev-ref" in argv:
            out = branch + "\n"
        elif "rev-parse" in argv:
            out = sha + "\n"
        else:  # pragma: no cover - the implementation asks for nothing else
            return m.ProcessResult(tuple(argv), 1, "", "unexpected git call")
        return m.ProcessResult(tuple(argv), 0, out, "")

    monkeypatch.setattr(m, "run_process", fake_run)


def _patch_git_state(monkeypatch: pytest.MonkeyPatch, state: m.GitState) -> None:
    monkeypatch.setattr(m, "capture_git_state", lambda *a, **k: state)


# ------------------------------------------------------------------------------------ seeding
def test_seed_everything_returns_seed_and_configures_determinism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(m.CUBLAS_ENV_VAR, raising=False)
    assert m.seed_everything(4711) == 4711
    assert os.environ[m.CUBLAS_ENV_VAR] == m.CUBLAS_WORKSPACE_CONFIG
    assert os.environ["PYTHONHASHSEED"] == "4711"


def test_seed_everything_reproducible_forward_pass() -> None:
    """Two seeded builds of the same tiny net produce bit-identical outputs (CPU)."""
    torch = pytest.importorskip("torch")

    def forward(seed: int) -> Any:
        m.seed_everything(seed)
        net = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4))
        x = torch.randn(3, 8)
        with torch.no_grad():
            return net(x)

    first, second = forward(1234), forward(1234)
    assert torch.equal(first, second)
    # A no-op seeding would make this pass by accident, so check that the seed matters.
    assert not torch.equal(first, forward(4321))


def test_seed_everything_rejects_out_of_range_seed() -> None:
    with pytest.raises(AiclsError) as excinfo:
        m.seed_everything(-1)
    assert excinfo.value.details["reason"] == "seed_out_of_range"


def test_seed_everything_without_determinism_still_seeds() -> None:
    assert m.seed_everything(7, deterministic=False) == 7


# ------------------------------------------------------------------------------------ git state
def test_git_state_from_real_repository_is_a_real_subprocess_call(repo_root: Path) -> None:
    """The production path shells out to git; on a git-less host it degrades to "unknown"."""
    state = m.capture_git_state(repo_root)
    assert isinstance(state, m.GitState)
    if state.available:
        assert re.fullmatch(r"[0-9a-f]{40}", state.sha), state.sha
        assert isinstance(state.dirty, bool)
    else:  # pragma: no cover - this host has git
        assert state.sha == m.UNKNOWN


def test_capture_git_state_reports_sha_branch_and_clean_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_git(monkeypatch, dirty=False)
    state = m.capture_git_state(tmp_path)
    assert (state.sha, state.branch, state.dirty, state.available) == (SHA, "main", False, True)
    assert state.diff_sha256 is None


def test_capture_git_state_hashes_the_diff_when_dirty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_git(monkeypatch, dirty=True)
    state = m.capture_git_state(tmp_path)
    assert state.dirty is True
    assert state.diff_sha256 is not None
    assert _HEX64.match(state.diff_sha256)


def test_a_run_is_never_citable_when_git_status_did_not_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A timed-out ``git status`` must not produce a citable run.

    ``citable = git_available and not dirty`` and ``dirty`` defaults to ``False``, so reporting
    ``available=True`` on this branch published **citable=True for a run whose working tree was
    never inspected**. Measured 2026-09-08: ``git status --porcelain`` over a ``/mnt`` path in
    WSL takes 10.2 s against a 15 s budget, so a loaded machine really does cross it. The commit
    is still recorded -- it is true and useful -- but the run may not be cited.
    """

    NL = chr(10)

    def only_rev_parse(
        args: list[str] | tuple[str, ...], *, cwd: Path | None = None, timeout_s: float = 10.0
    ) -> m.ProcessResult | None:
        argv = list(args)
        if "status" in argv:
            return m.ProcessResult(tuple(argv), 1, "", "timed out")
        if "--abbrev-ref" in argv:
            return m.ProcessResult(tuple(argv), 0, "main" + NL, "")
        if "rev-parse" in argv:
            return m.ProcessResult(tuple(argv), 0, SHA + NL, "")
        return None

    monkeypatch.setattr(m, "run_process", only_rev_parse)
    state = m.capture_git_state(tmp_path)

    assert state.sha == SHA, "the commit is known and stays recorded"
    assert state.available is False, "an uninspected working tree is not a citable provenance"
    manifest = m.RunManifest(
        run_id="R",
        started_at=datetime.now(UTC),
        git_sha=state.sha,
        dirty=state.dirty,
        git_available=state.available,
        citable=True,
    )
    assert manifest.citable is False


def test_capture_git_state_degrades_to_unknown_without_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(m, "run_process", lambda *a, **k: None)
    state = m.capture_git_state(tmp_path)
    assert (state.available, state.sha, state.dirty) == (False, m.UNKNOWN, False)


def test_run_process_returns_none_for_a_missing_executable(tmp_path: Path) -> None:
    assert m.run_process(["aicls-no-such-binary-xyz", "--version"], cwd=tmp_path) is None


def test_run_process_captures_stdout_of_a_real_command(tmp_path: Path) -> None:
    import sys

    result = m.run_process([sys.executable, "-c", "print('hello')"], cwd=tmp_path)
    assert result is not None
    assert result.ok and result.stdout.strip() == "hello"


# ------------------------------------------------------------------------------------ manifest
def test_run_manifest_captures_git_sha_and_dirty_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_git(monkeypatch, dirty=False)
    manifest = m.RunManifest.capture(_config(), repo_root=Path.cwd())
    assert manifest.git_sha == SHA
    assert manifest.dirty is False
    assert manifest.branch == "main"
    assert manifest.diff_sha256 is None
    assert manifest.citable is True
    assert manifest.run_name == "unit_run"
    assert manifest.seed == 1234
    assert _HEX64.match(manifest.pip_freeze_sha256)
    assert manifest.config_hash
    assert manifest.hostname
    assert isinstance(manifest.started_at, datetime)
    assert manifest.started_at.tzinfo is not None
    assert manifest.python.startswith("3.")
    assert manifest.steps_per_s is None


def test_run_manifest_refuses_dirty_tree_without_allow_dirty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_git(monkeypatch, dirty=True)
    with pytest.raises(m.DirtyWorkingTreeError) as excinfo:
        m.RunManifest.capture(_config(), repo_root=Path.cwd())
    assert isinstance(excinfo.value, AiclsError)
    assert excinfo.value.details["reason"] == "dirty_working_tree"
    assert excinfo.value.details["git_sha"] == SHA


def test_run_manifest_dirty_capture_with_allow_dirty_is_not_citable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_git(monkeypatch, dirty=True)
    manifest = m.RunManifest.capture(_config(), allow_dirty=True, repo_root=Path.cwd())
    assert manifest.dirty is True
    assert manifest.citable is False
    assert manifest.allow_dirty is True
    assert manifest.diff_sha256 is not None


def test_run_manifest_without_git_is_not_citable(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_git_state(monkeypatch, m.GitState())
    manifest = m.RunManifest.capture(_config())
    assert manifest.git_available is False
    assert manifest.git_sha == m.UNKNOWN
    assert manifest.citable is False


def test_citable_cannot_be_forced_on_a_dirty_manifest() -> None:
    manifest = m.RunManifest(
        run_id="R", started_at=datetime.now(UTC), git_available=True, dirty=True, citable=True
    )
    assert manifest.citable is False


def test_run_manifest_exposes_the_documented_contract_fields() -> None:
    documented = {
        "git_sha",
        "dirty",
        "diff_sha256",
        "config_hash",
        "data_fingerprints",
        "pip_freeze_sha256",
        "python",
        "torch",
        "cuda",
        "cudnn",
        "gpu_name",
        "driver",
        "hostname",
        "started_at",
        "steps_per_s",
        "citable",
    }
    assert documented <= set(m.RunManifest.model_fields)


def test_run_manifest_save_load_roundtrip_and_steps_per_s(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_git(monkeypatch, dirty=False)
    manifest = m.RunManifest.capture(_config(), repo_root=Path.cwd())
    path = manifest.save(tmp_path)
    assert path == tmp_path / m.MANIFEST_FILENAME
    assert path.read_bytes().endswith(b"\n")
    assert b"\r\n" not in path.read_bytes()

    manifest.set_steps_per_s(12.5, run_dir=tmp_path)
    loaded = m.RunManifest.load(tmp_path)
    assert loaded.steps_per_s == 12.5
    assert loaded.git_sha == SHA
    assert loaded.run_id == manifest.run_id
    assert loaded.citable is True

    with pytest.raises(AiclsError):
        manifest.set_steps_per_s(0.0)


def test_run_manifest_load_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AiclsError):
        m.RunManifest.load(tmp_path / "nope")


def test_run_manifest_records_dataset_fingerprints_from_the_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from aicls.data.manifest import DataManifest

    root = tmp_path / "tiny_ds"
    root.mkdir()
    (root / "shard_000.npz").write_bytes(b"not really an npz")
    data_manifest = DataManifest.build(root, "tiny_ds", license="internal", source_url="repo://x")
    data_manifest.save(root)

    _fake_git(monkeypatch, dirty=False)
    cfg = _config(data={"dataset": "tiny_ds", "root": str(root)})
    manifest = m.RunManifest.capture(cfg, repo_root=Path.cwd())
    assert manifest.data_fingerprints == {"tiny_ds": data_manifest.fingerprint}


def test_run_manifest_data_fingerprints_survive_a_missing_data_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_git(monkeypatch, dirty=False)
    cfg = _config(data={"dataset": "mathwriting", "root": str(tmp_path / "absent")})
    manifest = m.RunManifest.capture(cfg, repo_root=Path.cwd())
    assert manifest.data_fingerprints == {}


def test_explicit_data_fingerprints_win(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_git(monkeypatch, dirty=False)
    manifest = m.RunManifest.capture(
        _config(), repo_root=Path.cwd(), data_fingerprints={"x": "sha256:deadbeef"}
    )
    assert manifest.data_fingerprints == {"x": "sha256:deadbeef"}


def test_config_train_allow_dirty_permits_a_dirty_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """``train.allow_dirty: true`` in the config has the same effect as the keyword."""
    _fake_git(monkeypatch, dirty=True)
    cfg = _config(train={"epochs": 1, "allow_dirty": True})
    manifest = m.RunManifest.capture(cfg, repo_root=Path.cwd())
    assert manifest.allow_dirty is True
    assert manifest.citable is False


def test_run_manifest_uses_the_config_hash_of_a_real_train_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-module contract with :mod:`aicls.ml.config` (the report cites this hash)."""
    config_module = pytest.importorskip("aicls.ml.config")
    cfg = config_module.TrainConfig(run={"name": "ocr_smoke", "seed": 7})
    _fake_git(monkeypatch, dirty=False)
    manifest = m.RunManifest.capture(cfg, repo_root=Path.cwd())
    assert manifest.config_hash == config_module.config_hash(cfg)
    assert manifest.run_name == "ocr_smoke"
    assert manifest.seed == 7


def test_data_fingerprints_cover_the_audio_manifest_and_splits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "audio.jsonl"
    manifest_path.write_text('{"path": "a.wav", "text": "iki"}\n', encoding="utf-8")
    splits_path = tmp_path / "splits.json"
    splits_path.write_text('{"train": ["spk1"]}', encoding="utf-8")
    _fake_git(monkeypatch, dirty=False)
    cfg = _config(
        data={
            "dataset": "audio_commands",
            "root": "",
            "manifest": str(manifest_path),
            "splits_path": str(splits_path),
        }
    )
    fingerprints = m.RunManifest.capture(cfg, repo_root=Path.cwd()).data_fingerprints
    assert set(fingerprints) == {"audio_commands.manifest", "audio_commands.splits"}
    assert all(value.startswith("sha256:") for value in fingerprints.values())


def test_run_name_falls_back_when_the_config_has_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_git(monkeypatch, dirty=False)
    manifest = m.RunManifest.capture(_config(run={"name": "", "seed": 1}), repo_root=Path.cwd())
    assert manifest.run_name == "run"


# ------------------------------------------------------------------------------- config hashing
def test_config_fingerprint_is_stable_and_sensitive() -> None:
    first = m.config_fingerprint(_config())
    assert first == m.config_fingerprint(_config())
    changed = _config()
    changed["model"]["d_model"] = 64
    assert m.config_fingerprint(changed) != first


def test_config_fingerprint_accepts_a_pydantic_model() -> None:
    from pydantic import BaseModel

    class Tiny(BaseModel):
        lr: float = 1e-4

    assert m.config_fingerprint(Tiny()) == m.config_fingerprint({"lr": 1e-4})


def test_config_fingerprint_rejects_an_unsupported_object() -> None:
    with pytest.raises(AiclsError):
        m.config_fingerprint(object())


# --------------------------------------------------------------------------------- environment
def test_capture_env_writes_env_json(tmp_path: Path) -> None:
    payload = m.capture_env(tmp_path)
    written = json.loads((tmp_path / m.ENV_FILENAME).read_text(encoding="utf-8"))
    assert written == payload
    assert payload["python"]["version"].startswith("3.")
    assert payload["platform"]["hostname"]
    assert isinstance(payload["packages"], list)
    assert any(line.startswith("pydantic==") for line in payload["packages"])
    assert _HEX64.match(payload["pip_freeze_sha256"])
    assert payload["pip_freeze_sha256"] == m.pip_freeze_sha256()
    assert "torch" in payload


def test_capture_env_never_records_secret_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AICLS_JWT_SECRET", "super-secret-value")
    monkeypatch.setenv("AICLS_LLM_API_KEY", "another-secret")
    monkeypatch.setenv("AICLS_DATA_ROOT", str(tmp_path))
    payload = m.capture_env(tmp_path)
    blob = json.dumps(payload, ensure_ascii=False)
    assert "super-secret-value" not in blob
    assert "another-secret" not in blob
    assert payload["env"]["AICLS_DATA_ROOT"] == str(tmp_path)


def test_pip_freeze_lines_are_sorted_and_unique() -> None:
    lines = m.pip_freeze_lines()
    assert lines == sorted(lines)
    assert len(lines) == len(set(lines))


def test_torch_versions_reports_cpu_build_without_crashing() -> None:
    info = m.torch_versions()
    assert set(info) >= {"torch", "cuda", "cudnn", "cuda_available", "device_count"}
    assert isinstance(info["cuda_available"], bool)


def test_gpu_info_is_none_when_nvidia_smi_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m, "run_process", lambda *a, **k: None)
    assert m.nvidia_smi_devices() is None


def test_gpu_info_parses_nvidia_smi_output(monkeypatch: pytest.MonkeyPatch) -> None:
    line = "0, NVIDIA GeForce RTX 5090, 610.88, 32607\n"
    monkeypatch.setattr(
        m, "run_process", lambda *a, **k: m.ProcessResult(("nvidia-smi",), 0, line, "")
    )
    devices = m.nvidia_smi_devices()
    assert devices == [
        {
            "index": 0,
            "name": "NVIDIA GeForce RTX 5090",
            "driver": "610.88",
            "memory_total_mb": 32607.0,
        }
    ]


# ------------------------------------------------------- the shards -> run manifest data link
def _prepare_splits(root: Path, splits: tuple[str, ...], *, seed: int = 1234) -> Path:
    """Prepare ``splits`` of synthetic InkML into ``<root>/shards`` and return the shards root."""
    from aicls.data import prepare_shards, synth_inkml_dataset

    out = root / "shards"
    for index, split in enumerate(splits):
        corpus = synth_inkml_dataset(root / f"ink-{split}", 4, seed + index, prefix=split)
        prepare_shards(corpus, out, split=split, shard_size=2)
    return out


def test_a_run_against_prepared_shards_records_every_split_it_opened(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The defect this closes: ``data_fingerprints: {}`` for a run on real MathWriting shards.

    ``prepare_mathwriting.py`` writes a per-split ``shards_manifest.json``; the manifest reader
    looked only for a root ``manifest.json``, so ``run_manifest.json`` recorded which code ran
    and not which data it ran on.
    """
    from aicls.data.shards import load_shard_manifest, shards_root_fingerprint

    shards = _prepare_splits(tmp_path, ("train", "valid", "test"))
    _fake_git(monkeypatch, dirty=False)
    cfg = _config(data={"dataset": "mathwriting", "root": str(shards)})

    manifest = m.RunManifest.capture(cfg, repo_root=Path.cwd())
    saved = m.RunManifest.load(manifest.save(tmp_path / "run"))
    recorded = saved.data_fingerprints

    assert set(recorded) == {
        "mathwriting",
        "mathwriting.train",
        "mathwriting.valid",
        "mathwriting.test",
    }
    for split in ("train", "valid", "test"):
        assert recorded[f"mathwriting.{split}"] == load_shard_manifest(shards, split).fingerprint
    assert recorded["mathwriting"] == shards_root_fingerprint(
        {split: recorded[f"mathwriting.{split}"] for split in ("train", "valid", "test")}
    )


def test_a_run_records_only_the_splits_it_opens_when_told(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``data_splits`` narrows the record to the splits the loop actually builds."""
    shards = _prepare_splits(tmp_path, ("train", "valid", "test"))
    _fake_git(monkeypatch, dirty=False)
    cfg = _config(data={"dataset": "mathwriting", "root": str(shards)})

    recorded = m.RunManifest.capture(
        cfg, repo_root=Path.cwd(), data_splits=("train", "valid")
    ).data_fingerprints

    assert set(recorded) == {"mathwriting", "mathwriting.train", "mathwriting.valid"}


def test_changing_a_shard_changes_the_recorded_fingerprints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A provenance link that survives a change to the data is not a provenance link."""
    _fake_git(monkeypatch, dirty=False)
    first = _prepare_splits(tmp_path / "a", ("train", "valid"))
    second = _prepare_splits(tmp_path / "b", ("train", "valid"), seed=99)

    before = m.RunManifest.capture(
        _config(data={"dataset": "mathwriting", "root": str(first)}), repo_root=Path.cwd()
    ).data_fingerprints
    after = m.RunManifest.capture(
        _config(data={"dataset": "mathwriting", "root": str(second)}), repo_root=Path.cwd()
    ).data_fingerprints

    assert before["mathwriting.train"] != after["mathwriting.train"]
    assert before["mathwriting.valid"] != after["mathwriting.valid"]
    assert before["mathwriting"] != after["mathwriting"]


def test_a_root_data_manifest_still_wins_the_dataset_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``manifest.json`` (``--manifest`` / ``aicls data verify``) keeps naming the whole root."""
    from aicls.data.manifest import DataManifest

    shards = _prepare_splits(tmp_path, ("train",))
    root_manifest = DataManifest.build(
        shards, "mathwriting", license="CC-BY-NC-SA-4.0", source_url="file://x"
    )
    root_manifest.save(shards)
    _fake_git(monkeypatch, dirty=False)

    recorded = m.RunManifest.capture(
        _config(data={"dataset": "mathwriting", "root": str(shards)}), repo_root=Path.cwd()
    ).data_fingerprints

    assert recorded["mathwriting"] == root_manifest.fingerprint
    assert recorded["mathwriting.train"].startswith("sha256:")


def test_an_unprepared_data_root_is_recorded_as_empty_not_guessed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No manifest anywhere: the run goes ahead and the record says it could name no data."""
    empty = tmp_path / "shards"
    (empty / "train").mkdir(parents=True)
    _fake_git(monkeypatch, dirty=False)

    recorded = m.RunManifest.capture(
        _config(data={"dataset": "mathwriting", "root": str(empty)}), repo_root=Path.cwd()
    ).data_fingerprints

    assert recorded == {}
