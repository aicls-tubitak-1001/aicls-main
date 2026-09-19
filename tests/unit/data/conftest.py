"""Shared fixtures for the :mod:`aicls.data` unit tests.

Everything here is synthetic and generated from seed 1234: a handful of InkML files written by
:func:`aicls.data.synth_inkml_dataset`, the shards produced from them and a tiny audio-command
manifest whose WAV files come from :func:`aicls.data.synth_audio`. No fixture touches the network
and none of them is larger than a few hundred kilobytes (docs/gelistirme/test-stratejisi.md §4).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aicls.asr.audio_io import write_wav_pcm16
from aicls.data import prepare_shards, synth_audio, synth_inkml_dataset

SEED = 1234
N_INKML = 12
N_SYNTHETIC_INKML = 8


@pytest.fixture
def inkml_root(tmp_path: Path) -> Path:
    """A directory of 12 synthetic InkML files (flat layout)."""
    return synth_inkml_dataset(tmp_path / "inkml", N_INKML, SEED)


@pytest.fixture
def shards_root(tmp_path: Path, inkml_root: Path) -> Path:
    """Shards for a ``train`` split built from :fixture:`inkml_root`."""
    out = tmp_path / "shards"
    prepare_shards(inkml_root, out, split="train", shard_size=5)
    return out


@pytest.fixture
def mixed_shards_root(tmp_path: Path) -> Path:
    """Shards holding both a ``train`` and a ``synthetic`` split (for the mixing sampler)."""
    real = synth_inkml_dataset(tmp_path / "ink-real", N_INKML, SEED)
    fake = synth_inkml_dataset(tmp_path / "ink-synth", N_SYNTHETIC_INKML, SEED + 1)
    out = tmp_path / "shards-mixed"
    prepare_shards(real, out, split="train", shard_size=5)
    prepare_shards(fake, out, split="synthetic", shard_size=5)
    return out


def write_audio_manifest(
    root: Path,
    *,
    n_speakers: int = 4,
    per_speaker: int = 3,
    with_split_field: bool = True,
) -> Path:
    """Write a tiny ``manifest.jsonl`` plus its WAV files and return the manifest path."""
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(SEED)
    universities = ("mus", "mersin", "amasya")
    for s in range(n_speakers):
        pseudonym = f"P{s + 1:05d}"
        split = ("train", "train", "valid", "test")[s % 4]
        for k in range(per_speaker):
            audio = synth_audio(rng, 0.6)
            path = audio_dir / f"{pseudonym}-{k:03d}.wav"
            path.write_bytes(write_wav_pcm16(audio))
            row = {
                "id": f"{pseudonym}-{k:03d}",
                "path": str(path),
                "text": "x karenin türevini al",
                "speaker_pseudonym": pseudonym,
                "university": universities[s % len(universities)],
                "gender": ("f", "m")[s % 2],
                "age_band": "18-24",
                "duration_s": audio.duration_s,
                "source": "collected",
            }
            if with_split_field:
                row["split"] = split
            rows.append(row)
    manifest = root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return manifest


@pytest.fixture
def audio_manifest(tmp_path: Path) -> Path:
    """A 4-speaker audio-command manifest whose rows carry their split."""
    return write_audio_manifest(tmp_path / "audio-ds")
