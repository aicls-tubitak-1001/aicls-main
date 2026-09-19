"""Cross-module contracts of :mod:`aicls.data` (wave-1 lesson: test the boundary, not the fixture).

Three boundaries are exercised here end to end, each with the *real* peer module rather than a
local fake:

1. ``aicls.data`` -> ``aicls.ocr``: synthetic InkML written here is parsed by ``read_inkml``,
   sharded, turned into online features (``normalize_strokes``) and offline rasters
   (``render_ink`` + ``preprocess_image``) and encoded with ``LatexTokenizer``.
2. ``aicls.data`` -> ``aicls.asr``: WAV bytes written by ``write_wav_pcm16`` are read back by
   ``AudioCommandDataset`` and turned into log-Mel features by ``aicls.asr.features.log_mel``.
3. ``aicls.data`` -> ``aicls.ml``/``aicls.eval`` (consumers): the batch dictionaries and the
   dataset item shapes the training loop and the evaluation harness code against.
"""

from __future__ import annotations

from functools import partial
import json
from pathlib import Path

import numpy as np
import pytest

from aicls.asr.audio_io import read_wav_pcm16, write_wav_pcm16
from aicls.asr.features import log_mel
from aicls.data import (
    AudioCommandDataset,
    DataManifest,
    MathWritingDataset,
    collate_audio,
    collate_image,
    collate_ink,
    make_splits,
    prepare_shards,
    synth_audio,
    synth_inkml_dataset,
)
from aicls.data.splits import save_splits
from aicls.ocr import (
    STROKE_FEATURE_DIM,
    Ink,
    PreprocessConfig,
    default_tokenizer,
    normalize_strokes,
    read_inkml,
    render_ink,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def prepared(tmp_path: Path) -> tuple[Path, Path]:
    """A synthetic InkML corpus and its shards (train + synthetic splits)."""
    real = synth_inkml_dataset(tmp_path / "ink" / "train", 10, 1234)
    fake = synth_inkml_dataset(tmp_path / "ink" / "synthetic", 6, 4321)
    shards = tmp_path / "shards"
    prepare_shards(real, shards, split="train", shard_size=4)
    prepare_shards(fake, shards, split="synthetic", shard_size=4)
    return tmp_path / "ink", shards


def test_inkml_written_by_data_is_parsed_by_ocr_without_loss(prepared: tuple[Path, Path]) -> None:
    ink_root, _ = prepared
    for path in sorted((ink_root / "train").glob("*.inkml")):
        ink = read_inkml(path)
        assert isinstance(ink, Ink)
        assert ink.label and ink.writer_id
        assert ink.n_points > 0
        # the ocr feature extractor accepts every sample the data package emits
        features = normalize_strokes(ink)
        assert features.shape[1] == STROKE_FEATURE_DIM
        assert np.isfinite(features).all()
        # ... and so does the renderer used by the offline path
        raster = render_ink(ink, (32, 96))
        assert raster.shape == (32, 96)


def test_ocr_tokenizer_round_trips_every_shard_label(prepared: tuple[Path, Path]) -> None:
    _, shards = prepared
    tokenizer = default_tokenizer()
    ds = MathWritingDataset(shards, "train", mode="online", tokenizer=tokenizer)
    for index in range(len(ds)):
        sample = ds[index]
        assert tokenizer.decode(sample.label_ids.tolist()) == sample.label
        assert sample.label_ids[0] == 1 and sample.label_ids[-1] == 2  # <s> ... </s>


def test_dataset_batches_match_the_trainer_contract(prepared: tuple[Path, Path]) -> None:
    _, shards = prepared
    tokenizer = default_tokenizer()
    online = MathWritingDataset(
        shards, "train", mode="online", tokenizer=tokenizer, synthetic_ratio=0.5
    )
    batch = collate_ink([online[i] for i in range(4)])
    assert set(batch) >= {
        "features",
        "feature_lengths",
        "feature_mask",
        "labels",
        "label_lengths",
        "labels_text",
        "sample_ids",
        "is_synthetic",
    }
    assert batch["features"].ndim == 3
    assert batch["features"].shape[2] == STROKE_FEATURE_DIM
    assert batch["labels"].dtype == np.int64
    assert int(batch["labels"].max()) < tokenizer.vocab_size

    offline = MathWritingDataset(
        shards,
        "train",
        mode="offline",
        tokenizer=tokenizer,
        preprocess=PreprocessConfig(steps=("normalize01", "resize_pad"), size=(32, 96)),
    )
    image_batch = collate_image([offline[i] for i in range(3)])
    assert image_batch["images"].shape == (3, 1, 32, 96)
    assert image_batch["images"].dtype == np.float32


def test_audio_manifest_wav_roundtrip_between_data_and_asr(tmp_path: Path) -> None:
    root = tmp_path / "audio"
    root.mkdir()
    rng = np.random.default_rng(1234)
    rows = []
    for speaker in range(4):
        for k in range(2):
            audio = synth_audio(rng, 0.5)
            path = root / f"P{speaker:05d}-{k}.wav"
            path.write_bytes(write_wav_pcm16(audio))
            # the bytes the data package writes are exactly what the asr reader accepts
            back = read_wav_pcm16(path.read_bytes())
            assert back.sr == audio.sr
            np.testing.assert_allclose(back.samples, audio.samples, atol=1e-4)
            rows.append(
                {
                    "id": f"P{speaker:05d}-{k}",
                    "path": str(path),
                    "text": "türevini al",
                    "speaker_pseudonym": f"P{speaker:05d}",
                    "university": "mus",
                    "gender": "f",
                }
            )
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    splits = make_splits(
        [row["id"] for row in rows],
        group_key=lambda i: i.split("-", 1)[0],
        ratios=(0.5, 0.25, 0.25),
    )
    save_splits(splits, root / "audio_commands_splits.json")

    ds = AudioCommandDataset(manifest, "train", feature_fn=partial(log_mel, n_mels=16))
    batch = collate_audio([ds[i] for i in range(len(ds))])
    assert set(batch) >= {"features", "feature_lengths", "texts", "speakers", "sample_ids"}
    assert batch["features"].shape[1] == 16
    assert batch["features"].dtype == np.float32
    assert len({row.split("-")[0] for row in batch["sample_ids"]}) == len(set(batch["speakers"]))


def test_data_manifest_fingerprint_is_report_grade(tmp_path: Path) -> None:
    """``EvalReport.dataset_fingerprint`` must accept what ``DataManifest`` produces."""
    root = synth_inkml_dataset(tmp_path / "ink", 3, 1234)
    manifest = DataManifest.build(
        root, "synthetic-ink", license="internal", source_url="repo://synthetic"
    )
    assert manifest.fingerprint.startswith("sha256:")
    assert len(manifest.fingerprint) == len("sha256:") + 64
    assert manifest.verify(root).ok


def test_collect_export_manifest_and_split_file_load_in_the_dataset(tmp_path: Path) -> None:
    """``aicls.collect`` writes the manifest this dataset reads — including its ``dev`` split name.

    ``aicls.collect.MANIFEST_FIELDS`` has no ``id`` column (the row id is ``recording_id``) and
    ``aicls.collect.SPLIT_NAMES`` is ``(train, dev, test)``, while the training/eval side says
    ``valid``. Both differences are absorbed here instead of at a rename, so a real export loads.
    """
    from aicls.collect.manifest import MANIFEST_FIELDS, SPLITS_FILENAME, SplitFile, save_splits
    from aicls.collect.models import SPLIT_NAMES

    assert "recording_id" in MANIFEST_FIELDS
    assert "id" not in MANIFEST_FIELDS
    assert SPLIT_NAMES == ("train", "dev", "test")

    root = tmp_path / "audio"
    root.mkdir()
    rng = np.random.default_rng(1234)
    rows = []
    speakers = {"P00001": "train", "P00002": "train", "P00003": "dev", "P00004": "test"}
    for pseudonym, speaker_split in speakers.items():
        for k in range(2):
            audio = synth_audio(rng, 0.4)
            path = root / f"{pseudonym}-{k}.wav"
            path.write_bytes(write_wav_pcm16(audio))
            row = dict.fromkeys(MANIFEST_FIELDS, "")
            row.update(
                {
                    "path": str(path),
                    "text": "türevini al",
                    "speaker_pseudonym": pseudonym,
                    "university": "mus",
                    "gender": "f",
                    "age_band": "18-24",
                    "split": speaker_split,
                    "duration_s": audio.duration_s,
                    "sample_rate": audio.sr,
                    "sha256": "0" * 64,
                    "scenario_tag": "turev",
                    "consent_version": "v1",
                    "source": "contrib",
                    "recording_id": f"{pseudonym}-{k}",
                }
            )
            rows.append(row)
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    save_splits(root / SPLITS_FILENAME, SplitFile(by_speaker=speakers, seed=1234))

    train = AudioCommandDataset(manifest, "train", feature_fn=partial(log_mel, n_mels=16))
    valid = AudioCommandDataset(manifest, "valid", feature_fn=partial(log_mel, n_mels=16))
    dev = AudioCommandDataset(manifest, "dev", feature_fn=partial(log_mel, n_mels=16))

    assert train.speakers == {"P00001", "P00002"}
    assert valid.speakers == {"P00003"}  # collect's "dev" is this side's "valid"
    assert dev.split == "valid"
    assert train.rows[0].id == "P00001-0"  # recording_id, not a synthesised stem
    assert train.splits_fingerprint is not None
    assert train[0].features.shape[0] == 16


def test_collect_split_file_is_readable_by_data_load_splits(tmp_path: Path) -> None:
    from aicls.collect.manifest import SPLITS_FILENAME, SplitFile, save_splits
    from aicls.data.splits import load_splits

    path = tmp_path / SPLITS_FILENAME
    save_splits(path, SplitFile(by_speaker={"P1": "train", "P2": "dev", "P3": "test"}, seed=7))

    splits = load_splits(path)
    assert splits.train == ["P1"]
    assert splits.valid == ["P2"]
    assert splits.test == ["P3"]
    assert splits.seed == 7
    assert splits.source == "collect.speakers"
    assert splits.fingerprint.startswith("sha256:")
