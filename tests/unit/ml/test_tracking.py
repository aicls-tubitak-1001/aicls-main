"""Unit tests for :mod:`aicls.ml.tracking` (ADR-0007, docs/ml/egitim-altyapisi.md §6).

``test_jsonl_tracker_schema`` pins the on-disk line format: later reporting code reads
``metrics.jsonl`` and ``params.json``, so a change there is a breaking change.

MLflow is *not* installed in this environment, which makes the "optional dependency missing" path
the real one; the forwarding path is covered with a fake ``mlflow`` module injected into
``sys.modules`` (no MLflow server, no ``mlruns`` directory, no network).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any

import pytest

from aicls.core.errors import AiclsError
from aicls.ml import tracking as t

MLFLOW_INSTALLED = importlib.util.find_spec("mlflow") is not None


def _lines(run_dir: Path) -> list[dict[str, Any]]:
    text = (run_dir / t.METRICS_FILENAME).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ------------------------------------------------------------------------------ JSONL tracker
def test_jsonl_tracker_schema(tmp_path: Path) -> None:
    """The on-disk schema of metrics.jsonl and params.json (contractual, ADR-0007)."""
    tracker = t.JsonlTracker(tmp_path)
    run_id = tracker.start_run("ocr_smoke", {"optim": {"lr": 0.0003}, "run": {"seed": 1234}})
    tracker.log_metrics({"loss": 2.5, "grad_norm": 1.0}, step=100)
    tracker.log_metrics({"cer": 0.42}, step=200, split="valid")
    tracker.end_run()

    rows = _lines(tmp_path)
    assert len(rows) == 2
    assert set(rows[0]) == {
        "schema_version",
        "run_id",
        "step",
        "split",
        "wall_time",
        "ts",
        "metrics",
    }
    assert rows[0]["schema_version"] == t.SCHEMA_VERSION
    assert rows[0]["run_id"] == run_id
    assert rows[0]["step"] == 100
    assert rows[0]["split"] == "train"
    assert isinstance(rows[0]["wall_time"], float)
    assert rows[0]["wall_time"] >= 0.0
    assert rows[0]["ts"].endswith("Z")
    assert rows[0]["metrics"] == {"loss": 2.5, "grad_norm": 1.0}
    assert rows[1]["split"] == "valid"
    assert rows[1]["metrics"] == {"cer": 0.42}

    params = json.loads((tmp_path / t.PARAMS_FILENAME).read_text(encoding="utf-8"))
    assert set(params) == {
        "schema_version",
        "run_id",
        "run_name",
        "started_at",
        "ended_at",
        "status",
        "duration_s",
        "params",
    }
    assert params["run_id"] == run_id
    assert params["run_name"] == "ocr_smoke"
    assert params["params"]["optim"]["lr"] == 0.0003
    assert params["status"] == "finished"
    assert params["ended_at"] is not None
    assert params["duration_s"] >= 0.0


def test_jsonl_tracker_files_are_lf_and_utf8(tmp_path: Path) -> None:
    with t.JsonlTracker(tmp_path) as tracker:
        tracker.start_run("türkçe_koşu", {"note": "eğitim"})
        tracker.log_metrics({"loss": 1.0}, step=1)
    raw = (tmp_path / t.METRICS_FILENAME).read_bytes()
    assert b"\r\n" not in raw
    assert "türkçe_koşu" in (tmp_path / t.PARAMS_FILENAME).read_text(encoding="utf-8")


def test_jsonl_tracker_read_helpers_round_trip(tmp_path: Path) -> None:
    tracker = t.JsonlTracker(tmp_path)
    tracker.start_run("run", {"a": 1})
    for step in (1, 2, 3):
        tracker.log_metrics({"loss": 1.0 / step}, step=step)
    tracker.end_run()
    rows = t.read_metrics(tmp_path)
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert t.read_params(tmp_path)["params"] == {"a": 1}


def test_jsonl_tracker_writes_null_for_non_finite_metrics(tmp_path: Path) -> None:
    """A NaN loss must be recorded, not crash the run, and must stay valid JSON."""
    tracker = t.JsonlTracker(tmp_path)
    tracker.start_run("run")
    tracker.log_metrics({"loss": float("nan"), "lr": float("inf"), "ok": 1.5}, step=7)
    tracker.end_run()
    row = _lines(tmp_path)[0]
    assert row["metrics"] == {"loss": None, "lr": None, "ok": 1.5}


def test_jsonl_tracker_copies_external_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    external = tmp_path / "confusion.json"
    external.write_text('{"a": 1}', encoding="utf-8")
    tracker = t.JsonlTracker(run_dir)
    tracker.start_run("run")
    record = tracker.log_artifact(external)
    tracker.end_run()
    assert (run_dir / t.ARTIFACTS_DIRNAME / "confusion.json").is_file()
    assert record["name"] == "confusion.json"
    assert record["sha256"]
    logged = [
        json.loads(line)
        for line in (run_dir / t.ARTIFACTS_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    assert logged[0]["name"] == "confusion.json"


def test_jsonl_tracker_keeps_artifacts_already_inside_the_run_dir(tmp_path: Path) -> None:
    inside = tmp_path / "run_manifest.json"
    inside.write_text("{}", encoding="utf-8")
    tracker = t.JsonlTracker(tmp_path)
    tracker.start_run("run")
    record = tracker.log_artifact(inside)
    assert record["path"] == "run_manifest.json"
    assert not (tmp_path / t.ARTIFACTS_DIRNAME).exists()


def test_jsonl_tracker_rejects_a_missing_artifact(tmp_path: Path) -> None:
    tracker = t.JsonlTracker(tmp_path)
    tracker.start_run("run")
    with pytest.raises(AiclsError):
        tracker.log_artifact(tmp_path / "nope.json")


def test_jsonl_tracker_refuses_metrics_before_start_run(tmp_path: Path) -> None:
    tracker = t.JsonlTracker(tmp_path)
    with pytest.raises(AiclsError) as excinfo:
        tracker.log_metrics({"loss": 1.0}, step=0)
    assert excinfo.value.details["reason"] == "run_not_started"


def test_jsonl_tracker_rejects_a_non_numeric_metric(tmp_path: Path) -> None:
    tracker = t.JsonlTracker(tmp_path)
    tracker.start_run("run")
    with pytest.raises(AiclsError) as excinfo:
        tracker.log_metrics({"loss": "low"}, step=1)  # type: ignore[dict-item]
    assert excinfo.value.details["reason"] == "metric_not_numeric"


def test_jsonl_tracker_end_run_is_idempotent(tmp_path: Path) -> None:
    tracker = t.JsonlTracker(tmp_path)
    tracker.start_run("run")
    tracker.end_run()
    tracker.end_run(status="failed")
    assert t.read_params(tmp_path)["status"] == "finished"


def test_jsonl_tracker_context_manager_records_a_failure(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError), t.JsonlTracker(tmp_path) as tracker:
        tracker.start_run("run")
        raise RuntimeError("boom")
    assert t.read_params(tmp_path)["status"] == "failed"


def test_read_helpers_on_an_empty_run_dir(tmp_path: Path) -> None:
    assert t.read_metrics(tmp_path) == []
    with pytest.raises(AiclsError):
        t.read_params(tmp_path)


# ---------------------------------------------------------------------------------- protocol
def test_both_trackers_satisfy_the_protocol(tmp_path: Path) -> None:
    assert isinstance(t.JsonlTracker(tmp_path), t.Tracker)
    assert isinstance(t.MlflowTracker(experiment="ocr"), t.Tracker)


def test_flatten_params_produces_dotted_string_pairs() -> None:
    flat = t.flatten_params({"optim": {"lr": 0.0003, "betas": [0.9, 0.98]}, "run": {"seed": 1234}})
    assert flat == {"optim.lr": "0.0003", "optim.betas": "[0.9, 0.98]", "run.seed": "1234"}


# ----------------------------------------------------------------------------- MLflow tracker
@pytest.mark.skipif(MLFLOW_INSTALLED, reason="mlflow is installed; the missing-dep path is moot")
def test_mlflow_tracker_raises_a_clear_error_when_mlflow_is_missing(tmp_path: Path) -> None:
    tracker = t.MlflowTracker(experiment="ocr", tracking_uri=f"file:{tmp_path.as_posix()}")
    with pytest.raises(AiclsError) as excinfo:
        tracker.start_run("run")
    assert excinfo.value.details["reason"] == "mlflow_not_installed"
    assert "mlflow" in excinfo.value.details["hint"]


class _FakeMlflow:
    """A stand-in for the ``mlflow`` module, recording every call made against it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def set_tracking_uri(self, uri: str) -> None:
        self.calls.append(("uri", uri))

    def set_experiment(self, name: str) -> None:
        self.calls.append(("experiment", name))

    def start_run(self, run_name: str | None = None) -> Any:
        self.calls.append(("start", run_name))
        return types.SimpleNamespace(info=types.SimpleNamespace(run_id="fake-run-id"))

    def log_params(self, params: dict[str, str]) -> None:
        self.calls.append(("params", params))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.calls.append(("metrics", (metrics, step)))

    def log_artifact(self, path: str) -> None:
        self.calls.append(("artifact", path))

    def end_run(self, status: str | None = None) -> None:
        self.calls.append(("end", status))


def _install_fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> _FakeMlflow:
    """Put a fake ``mlflow`` in ``sys.modules``; the lazy ``import mlflow`` finds it there."""
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    return fake


def test_mlflow_tracker_forwards_to_mlflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _install_fake_mlflow(monkeypatch)
    artifact = tmp_path / "plot.png"
    artifact.write_bytes(b"png")
    tracker = t.MlflowTracker(experiment="ocr", tracking_uri=f"file:{tmp_path.as_posix()}")
    run_id = tracker.start_run("ocr_smoke", {"optim": {"lr": 0.0003}})
    tracker.log_metrics({"loss": 1.5}, step=10, split="valid")
    tracker.log_artifact(artifact)
    tracker.end_run()

    calls = dict(fake.calls)
    assert run_id == "fake-run-id"
    assert calls["uri"] == f"file:{tmp_path.as_posix()}"
    assert calls["experiment"] == "ocr"
    assert calls["start"] == "ocr_smoke"
    assert calls["params"] == {"optim.lr": "0.0003"}
    assert calls["metrics"] == ({"valid.loss": 1.5}, 10)
    assert calls["artifact"] == str(artifact)
    assert ("end", "FINISHED") in fake.calls


def test_mlflow_tracker_default_uri_points_at_the_data_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _install_fake_mlflow(monkeypatch)
    from aicls.core.settings import settings_override

    with settings_override(data_root=tmp_path):
        tracker = t.MlflowTracker(experiment="asr")
        tracker.start_run("run")
    uri = dict(fake.calls)["uri"]
    assert uri.startswith("file:")
    assert uri.endswith("/mlruns")
