"""Wav2Vec2 CTC edge — the İP-3 ensemble fallback, local checkpoints only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aicls.asr.errors import AsrModelError
from aicls.asr.interfaces import get_transcriber
from aicls.asr.synth import make_speech_like
from aicls.asr.wav2vec2_transcriber import Wav2Vec2Transcriber
from aicls.core.errors import NotFoundError


def test_wav2vec2_builds_from_tiny_random_config_without_download(
    tiny_wav2vec2_dir: Path,
) -> None:
    """A tiny randomly initialised CTC model is saved locally, loaded and decoded offline."""
    transcriber = Wav2Vec2Transcriber.load(tiny_wav2vec2_dir, "cpu")

    assert transcriber.device == "cpu"
    assert transcriber.model_version == tiny_wav2vec2_dir.name

    transcript = transcriber.transcribe(make_speech_like(duration_s=1.0))

    assert isinstance(transcript.text, str)
    assert 0.0 <= transcript.confidence <= 1.0
    assert transcript.elapsed_ms > 0.0
    assert len(transcript.segments) == 1
    assert transcript.segments[0].end_s == pytest.approx(1.0, abs=0.01)


def test_wav2vec2_is_reachable_through_the_factory(tiny_wav2vec2_dir: Path) -> None:
    transcriber = get_transcriber("wav2vec2", model_dir=tiny_wav2vec2_dir)
    assert isinstance(transcriber, Wav2Vec2Transcriber)
    assert transcriber.warmup() is None


def test_wav2vec2_load_reports_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        Wav2Vec2Transcriber.load(tmp_path / "absent")
    assert excinfo.value.details["reason"] == "model_dir_missing"


def test_wav2vec2_load_wraps_a_corrupt_checkpoint(tmp_path: Path) -> None:
    directory = tmp_path / "corrupt"
    directory.mkdir()
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "pytorch_model.bin").write_bytes(b"not a checkpoint")
    with pytest.raises(AsrModelError) as excinfo:
        Wav2Vec2Transcriber.load(directory)
    assert excinfo.value.details["reason"] == "wav2vec2_load_failed"


def test_wav2vec2_wraps_a_forward_failure(tiny_wav2vec2_dir: Path) -> None:
    transcriber = Wav2Vec2Transcriber.load(tiny_wav2vec2_dir)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("kernel exploded")

    transcriber._model = boom
    with pytest.raises(AsrModelError) as excinfo:
        transcriber.transcribe(make_speech_like(duration_s=0.5))
    assert excinfo.value.details["reason"] == "wav2vec2_forward_failed"


def test_backend_helpers_report_missing_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host without the ML stack must get a Turkish "model not ready", not an ImportError."""
    import builtins

    from aicls.asr import _backend

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"torch", "transformers"}:
            raise ImportError(f"no module named {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(AsrModelError) as excinfo:
        _backend.import_torch()
    assert excinfo.value.details["reason"] == "torch_missing"
    with pytest.raises(AsrModelError) as excinfo:
        _backend.import_transformers()
    assert excinfo.value.details["reason"] == "transformers_missing"


def test_select_device_logs_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import SimpleNamespace

    from aicls.asr import _backend

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    assert _backend.select_device("cuda") == "cpu"
    assert _backend.select_device("cpu") == "cpu"
