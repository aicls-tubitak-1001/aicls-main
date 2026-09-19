"""AudioCommandDataset over the collect export manifest (docs/ml/veri-yonetimi.md §3)."""

from __future__ import annotations

from functools import partial
import json
from pathlib import Path

import numpy as np
import pytest

from aicls.asr.features import log_mel
from aicls.core.errors import InputValidationError, NotFoundError
from aicls.data import AudioCommandDataset, collate_audio, make_splits
from aicls.data.splits import save_splits

from .conftest import write_audio_manifest

FEATURE_FN = partial(log_mel, n_mels=16)


def test_audio_command_dataset_reads_the_split_and_builds_features(audio_manifest: Path) -> None:
    ds = AudioCommandDataset(audio_manifest, "train", feature_fn=FEATURE_FN)

    assert len(ds) == 6  # speakers 1 and 2 (3 utterances each)
    sample = ds[0]
    assert sample.features.shape[0] == 16
    assert sample.features.dtype == np.float32
    assert sample.text == "x karenin türevini al"
    assert sample.speaker.startswith("P")
    assert sample.sample_rate == 16000
    assert {s.speaker for s in (ds[i] for i in range(len(ds)))} == {"P00001", "P00002"}

    features, text, speaker = sample
    assert text == sample.text and speaker == sample.speaker
    assert features is sample.features


def test_audio_command_dataset_is_speaker_disjoint_across_splits(audio_manifest: Path) -> None:
    speakers = {
        split: {
            AudioCommandDataset(audio_manifest, split, feature_fn=FEATURE_FN)[i].speaker
            for i in range(len(AudioCommandDataset(audio_manifest, split, feature_fn=FEATURE_FN)))
        }
        for split in ("train", "valid", "test")
    }
    assert speakers["train"] & speakers["valid"] == set()
    assert speakers["train"] & speakers["test"] == set()
    assert speakers["valid"] & speakers["test"] == set()


def test_audio_command_dataset_refuses_without_a_speaker_split_file(tmp_path: Path) -> None:
    manifest = write_audio_manifest(tmp_path / "ds", with_split_field=False)
    with pytest.raises(InputValidationError):
        AudioCommandDataset(manifest, "train", feature_fn=FEATURE_FN)


def test_audio_command_dataset_accepts_an_external_split_file(tmp_path: Path) -> None:
    manifest = write_audio_manifest(tmp_path / "ds", with_split_field=False)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    splits = make_splits(
        [row["id"] for row in rows],
        group_key=lambda i: i.split("-", 1)[0],
        ratios=(0.5, 0.25, 0.25),
    )
    path = save_splits(splits, tmp_path / "ds" / "audio_commands_splits.json")

    ds = AudioCommandDataset(manifest, "train", feature_fn=FEATURE_FN, splits=path)
    assert len(ds) == len(splits.train)
    assert ds.splits_fingerprint == splits.fingerprint

    # the sibling file is found automatically as well
    assert len(AudioCommandDataset(manifest, "train", feature_fn=FEATURE_FN)) == len(splits.train)


def test_audio_command_dataset_rejects_missing_manifest_and_unknown_split(
    tmp_path: Path, audio_manifest: Path
) -> None:
    with pytest.raises(NotFoundError):
        AudioCommandDataset(tmp_path / "missing.jsonl", "train", feature_fn=FEATURE_FN)
    with pytest.raises(InputValidationError):
        AudioCommandDataset(audio_manifest, "nope", feature_fn=FEATURE_FN)


def test_audio_command_dataset_rejects_rows_without_path_or_text(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"id": "a", "speaker_pseudonym": "P1", "split": "train"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(InputValidationError):
        AudioCommandDataset(manifest, "train", feature_fn=FEATURE_FN)


def test_audio_command_dataset_reports_a_missing_wav_file(tmp_path: Path) -> None:
    manifest = write_audio_manifest(tmp_path / "ds")
    ds = AudioCommandDataset(manifest, "train", feature_fn=FEATURE_FN)
    Path(ds.rows[0].path).unlink()
    with pytest.raises(NotFoundError):
        _ = ds[0]


def test_collate_audio_pads_features_and_reports_lengths(audio_manifest: Path) -> None:
    ds = AudioCommandDataset(audio_manifest, "train", feature_fn=FEATURE_FN)
    batch = collate_audio([ds[0], ds[1]])

    assert batch["features"].shape[0] == 2
    assert batch["features"].shape[1] == 16
    assert batch["feature_lengths"].tolist() == [
        ds[0].features.shape[-1],
        ds[1].features.shape[-1],
    ]
    assert batch["texts"] == [ds[0].text, ds[1].text]
    assert batch["speakers"] == [ds[0].speaker, ds[1].speaker]
