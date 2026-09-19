"""Tests for the model and dataset registries (docs/ml/egitim-altyapisi.md §3, ADR-0006).

Every model is built at *tiny* size (``d_model 32``, 2 layers) on CPU with randomly initialised
weights. Nothing here is evidence about accuracy: a parameter count is a WIRING CHECK that the
config-to-module path works, not a result (ADR-0012 keeps the real weights behind a gate).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from aicls.core.errors import InputValidationError, NotFoundError
from aicls.ml.config import DATASET_NAMES, MODEL_FAMILIES, DataCfg, ModelCfg, build_config
from aicls.ml.registry import (
    DatasetSpec,
    ModelSpec,
    build_dataset,
    build_model,
    get_dataset_builder,
    get_model_builder,
    register_dataset,
    register_model,
    registered_datasets,
    registered_models,
)

TINY = {"d_model": 32, "layers": 2, "heads": 2, "ff_dim": 64, "dropout": 0.0}
TINY_VOCAB = 40


def _tiny_config(family: str, **extra: object) -> dict[str, object]:
    """A complete TrainConfig payload for one family at tiny size."""
    return {
        "run": {"name": f"tiny_{family}", "seed": 1234},
        "data": {
            "dataset": "synthetic_ink" if family.startswith("seq2seq") else "synthetic_audio",
            "image_size": [32, 64],
            "mode": "offline" if family == "seq2seq_image" else "online",
            "n_samples": 4,
        },
        "model": {"family": family, "max_len": 16, **TINY, **extra},
        "train": {"max_steps": 2},
    }


# ------------------------------------------------------------------------------------------
# registry bookkeeping
# ------------------------------------------------------------------------------------------
def test_registry_covers_every_name_in_the_design_document() -> None:
    assert set(registered_models()) == set(MODEL_FAMILIES)
    assert set(registered_datasets()) == set(DATASET_NAMES)


def test_register_model_rejects_a_duplicate_name() -> None:
    with pytest.raises(InputValidationError):

        @register_model("whisper")
        def _again(spec: ModelSpec, vocab_size: int) -> object:  # pragma: no cover - never called
            return None


def test_register_dataset_rejects_a_duplicate_name() -> None:
    with pytest.raises(InputValidationError):

        @register_dataset("mathwriting")
        def _again(spec: DatasetSpec, split: str) -> object:  # pragma: no cover - never called
            return None


def test_unknown_model_name_lists_the_registered_names() -> None:
    with pytest.raises(NotFoundError) as excinfo:
        get_model_builder("resnet_of_theseus")

    assert excinfo.value.details["name"] == "resnet_of_theseus"
    assert set(excinfo.value.details["registered"]) == set(MODEL_FAMILIES)


def test_unknown_dataset_name_lists_the_registered_names() -> None:
    with pytest.raises(NotFoundError) as excinfo:
        get_dataset_builder("imagenet")

    assert set(excinfo.value.details["registered"]) == set(DATASET_NAMES)


def test_importing_the_registry_does_not_import_torch(repo_root: Path) -> None:
    """``aicls.ml.registry`` must stay importable in an ungated, torch-free process."""
    code = (
        "import json, sys;"
        "import aicls.ml.registry;"
        "print(json.dumps({m: m in sys.modules for m in ('torch', 'transformers')}))"
    )
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}

    result = subprocess.run(  # fixed argv, the venv interpreter
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root),
        timeout=180,
        check=True,
    )

    assert json.loads(result.stdout.strip()) == {"torch": False, "transformers": False}


# ------------------------------------------------------------------------------------------
# models
# ------------------------------------------------------------------------------------------
def test_registry_builds_every_model_at_tiny_size() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    counts: dict[str, int] = {}
    for family in MODEL_FAMILIES:
        cfg = build_config(_tiny_config(family))

        model = build_model(cfg, TINY_VOCAB)

        assert isinstance(model, torch.nn.Module), family
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0, family
        # Tiny really means tiny: a wiring check, never a trained model.
        assert n_params < 3_000_000, (family, n_params)
        counts[family] = n_params

    assert set(counts) == set(MODEL_FAMILIES)


def test_build_model_accepts_a_bare_model_config() -> None:
    torch = pytest.importorskip("torch")
    cfg = ModelCfg(family="seq2seq_stroke", encoder="stroke_bilstm", max_len=16, **TINY)

    model = build_model(cfg, TINY_VOCAB)

    assert isinstance(model, torch.nn.Module)


def test_seq2seq_image_forward_produces_logits_of_the_vocabulary_width() -> None:
    torch = pytest.importorskip("torch")
    cfg = build_config(_tiny_config("seq2seq_image"))
    model = build_model(cfg, TINY_VOCAB)

    images = torch.zeros(2, 1, 32, 64)
    targets = torch.zeros(2, 5, dtype=torch.long)
    with torch.no_grad():
        logits = model(images, targets)

    assert logits.shape == (2, 5, TINY_VOCAB)


def test_whisper_builder_refuses_a_missing_local_directory(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    cfg = build_config(_tiny_config("whisper", base_dir=str(tmp_path / "not-downloaded")))

    with pytest.raises(NotFoundError) as excinfo:
        build_model(cfg, TINY_VOCAB)

    assert excinfo.value.details["reason"] == "model_dir_missing"


def test_build_model_refuses_a_non_positive_vocabulary() -> None:
    pytest.importorskip("torch")
    cfg = build_config(_tiny_config("seq2seq_image"))

    with pytest.raises(InputValidationError):
        build_model(cfg, 0)


def test_a_missing_transformers_install_is_a_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional heavy stack must fail with the project's error, not a bare ImportError."""
    from aicls.ocr.torch_utils import TorchUnavailableError

    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(TorchUnavailableError) as excinfo:
        build_model(build_config(_tiny_config("whisper")), TINY_VOCAB)

    assert excinfo.value.details["missing_package"] == "transformers"


@pytest.mark.parametrize("family", ["whisper", "wav2vec2"])
def test_speech_models_load_from_a_local_directory_without_the_network(
    family: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gated path: a checkpoint comes from a directory, never from a hub id (ADR-0012)."""
    pytest.importorskip("transformers")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    checkpoint = tmp_path / "pretrained" / family
    build_model(build_config(_tiny_config(family)), TINY_VOCAB).save_pretrained(checkpoint)

    cfg = build_config(_tiny_config(family, base_dir=str(checkpoint)))
    model = build_model(cfg, TINY_VOCAB)

    assert model.config.vocab_size == TINY_VOCAB


# ------------------------------------------------------------------------------------------
# datasets
# ------------------------------------------------------------------------------------------
def test_build_dataset_builds_the_synthetic_families() -> None:
    ink_cfg = build_config(_tiny_config("seq2seq_image"))
    audio_cfg = build_config(_tiny_config("whisper"))

    ink = build_dataset(ink_cfg, "train")
    audio = build_dataset(audio_cfg, "train")

    assert len(ink) == 4
    ink_sample = ink[0]
    assert ink_sample.features.ndim == 3  # (1, H, W) offline raster
    assert ink_sample.features.shape[1:] == (32, 64)
    assert ink_sample.label_ids.size > 0

    assert len(audio) == 4
    assert audio[0].features.ndim == 2  # (n_mels, frames) log-Mel
    assert audio[0].text


def test_build_dataset_accepts_a_bare_data_config() -> None:
    data = DataCfg(dataset="synthetic_ink", n_samples=3, mode="online")

    dataset = build_dataset(data, "train")

    assert len(dataset) == 3
    assert dataset[0].features.ndim == 2  # (T, F) stroke features


def test_build_dataset_uses_the_configured_vocabulary(tmp_path: Path) -> None:
    from aicls.ocr import LatexTokenizer

    vocab = tmp_path / "vocab.json"
    LatexTokenizer(["x", "^", "{", "}", "2"]).save(vocab)
    cfg = build_config(
        {
            "data": {"dataset": "synthetic_ink", "n_samples": 2},
            "model": {"vocab_path": str(vocab), **TINY},
        }
    )

    spec = DatasetSpec.from_config(cfg)

    assert spec.vocab_path == vocab
    assert spec.tokenizer().vocab_size == 5 + 4  # five content tokens plus the four specials


def test_build_dataset_requires_a_root_for_mathwriting() -> None:
    cfg = build_config({"data": {"dataset": "mathwriting"}})

    with pytest.raises(InputValidationError) as excinfo:
        build_dataset(cfg, "train")

    assert excinfo.value.details["key"] == "data.root"


def test_synthetic_ratio_reaches_the_training_split_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``data.synthetic_ratio`` is a training curriculum, never a property of the metric.

    The shipped OCR configs mix rendered synthetic ink into training at 0.5 (design document §1).
    Passing that ratio to ``valid`` or ``test`` would mean half of every reported CER was measured
    on machine-generated strokes, and — worse, because it is silent — the ``val_cer`` that selects
    the best checkpoint would be scoring a model on data it is not meant to be judged on. The
    contractual number is a CER on human handwriting; the ratio must stop at the training split.
    """
    import aicls.data as data_module

    seen: list[tuple[str, float]] = []

    class _Spy:
        def __init__(self, _root: Path, split: str, **kwargs: object) -> None:
            seen.append((split, float(kwargs["synthetic_ratio"])))  # type: ignore[arg-type]

        def __len__(self) -> int:
            return 0

    monkeypatch.setattr(data_module, "MathWritingDataset", _Spy)
    cfg = build_config(
        {
            "data": {
                "dataset": "mathwriting",
                "root": str(tmp_path),
                "synthetic_ratio": 0.5,
                "synthetic_warmup_epochs": 0,
            }
        }
    )

    for split in ("train", "valid", "test", "symbols"):
        build_dataset(cfg, split)

    assert seen == [("train", 0.5), ("valid", 0.0), ("test", 0.0), ("symbols", 0.0)]


def test_build_dataset_builds_the_audio_command_corpus(tmp_path: Path) -> None:
    """The real ASR dataset path, on a two-row synthetic manifest (no download, no gate)."""
    from aicls.asr import write_wav_pcm16
    from aicls.asr.synth import make_speech_like

    rows = []
    for index, speaker in enumerate(("spk-a", "spk-b")):
        wav = tmp_path / f"utt{index}.wav"
        wav.write_bytes(write_wav_pcm16(make_speech_like(duration_s=0.5, seed=1234 + index)))
        rows.append(
            json.dumps(
                {
                    "id": f"utt{index}",
                    "path": str(wav),
                    "text": "x karenin türevini al",
                    "speaker_pseudonym": speaker,
                    "split": "train",
                }
            )
        )
    manifest = tmp_path / "audio_commands.jsonl"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    cfg = build_config(
        {
            "data": {"dataset": "audio_commands", "manifest": str(manifest)},
            "model": {"family": "whisper", "num_mel_bins": 40},
        }
    )

    dataset = build_dataset(cfg, "train")

    assert len(dataset) == 2
    sample = dataset[0]
    assert sample.features.shape[0] == 40  # log-Mel bands follow model.num_mel_bins
    assert sample.speaker == "spk-a"


def test_build_dataset_requires_a_manifest_for_audio_commands() -> None:
    cfg = build_config({"data": {"dataset": "audio_commands"}})

    with pytest.raises(InputValidationError) as excinfo:
        build_dataset(cfg, "train")

    assert excinfo.value.details["key"] == "data.manifest"


def test_whisper_feature_extractor_is_read_from_the_local_model_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers = pytest.importorskip("transformers")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    checkpoint = tmp_path / "whisper-tiny-local"
    transformers.WhisperFeatureExtractor(feature_size=40, chunk_length=1).save_pretrained(
        checkpoint
    )
    cfg = build_config(
        {
            "data": {
                "dataset": "synthetic_audio",
                "feature_extractor": "whisper",
                "n_samples": 2,
                "sample_seconds": 0.5,
            },
            "model": {"family": "whisper", "base_dir": str(checkpoint)},
        }
    )

    dataset = build_dataset(cfg, "train")

    assert len(dataset) == 2
    assert dataset[0].features.shape[0] == 40


def test_whisper_feature_extractor_needs_a_local_model_directory() -> None:
    cfg = build_config(
        {
            "data": {"dataset": "synthetic_audio", "feature_extractor": "whisper", "n_samples": 1},
            "model": {"family": "whisper"},
        }
    )

    with pytest.raises(InputValidationError) as excinfo:
        build_dataset(cfg, "train")

    assert excinfo.value.details["key"] == "model.base_dir"
