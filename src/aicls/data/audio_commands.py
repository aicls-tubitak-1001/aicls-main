"""Audio-command dataset over the collect export manifest (docs/ml/veri-yonetimi.md §3).

The corpus this reads is produced by ``aicls contrib export-manifest``: one JSONL row per accepted
recording, carrying **only** the pseudonymised fields (path, text, speaker pseudonym, university,
gender band, age band, source) — never an identity column (KVKK, ADR-0008).

A speaker-disjoint split assignment is **mandatory**: either every row carries a ``split`` field,
or a split file (``audio_commands_splits.json``, written by :func:`aicls.data.make_splits`) sits
next to the manifest or is passed explicitly. Without one the dataset refuses to load, because a
WER measured on a speaker the model has heard during training is not a WER
(``test_asr_eval_refuses_without_speaker_split_file``).

Audio decoding goes through :mod:`aicls.asr.audio_io` (stdlib ``wave`` + numpy, ADR-0017), so no
``torchaudio``/``librosa`` is needed and the whole path runs offline on CPU.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from aicls.asr.audio_io import read_wav_pcm16, resample_to_16k
from aicls.asr.interfaces import TARGET_SAMPLE_RATE, Audio
from aicls.core.logging import get_logger
from aicls.data.errors import invalid_data, missing_path
from aicls.data.splits import (
    CORE_SPLITS,
    SPLITS_FILENAME,
    Splits,
    load_splits,
    normalize_split_name,
)

_log = get_logger(__name__)

#: Feature extractor contract: an :class:`~aicls.asr.interfaces.Audio` in, an array out.
FeatureFn = Callable[[Audio], np.ndarray]

#: Manifest fields that must never appear in an export (KVKK, docs/ml/veri-yonetimi.md §4).
FORBIDDEN_FIELDS: frozenset[str] = frozenset({"user_id", "email", "name", "birth_date", "ip"})


@dataclass(frozen=True, slots=True)
class AudioRow:
    """One manifest row (already pseudonymised)."""

    id: str
    path: str
    text: str
    speaker: str
    split: str
    university: str = ""
    gender: str = ""
    age_band: str = ""
    duration_s: float = 0.0
    source: str = "collected"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AudioSample:
    """One audio training item.

    Attributes:
        features: Whatever ``feature_fn`` produced (log-Mel ``(n_mels, T)`` by default).
        text: Reference transcript.
        speaker: Speaker pseudonym.
        sample_id: Manifest row id.
        sample_rate: Sample rate of the decoded waveform (always 16 kHz).
        duration_s: Duration of the decoded waveform.
    """

    features: np.ndarray
    text: str
    speaker: str
    sample_id: str
    sample_rate: int = TARGET_SAMPLE_RATE
    duration_s: float = 0.0

    def __iter__(self) -> Iterator[Any]:
        """Yield ``features, text, speaker`` (the documented item tuple)."""
        yield self.features
        yield self.text
        yield self.speaker

    def __getitem__(self, index: int) -> Any:
        """``sample[0]`` features, ``sample[1]`` text, ``sample[2]`` speaker."""
        return (self.features, self.text, self.speaker)[index]


class AudioCommandDataset:
    """Recordings of one split of the Turkish math-command corpus.

    Args:
        manifest: Path to the JSONL export manifest.
        split: ``train``, ``valid`` or ``test``.
        feature_fn: Turns an :class:`~aicls.asr.interfaces.Audio` into features
            (``aicls.asr.features.log_mel``, or a Whisper feature extractor wrapper).
        splits: Split assignment — a :class:`~aicls.data.splits.Splits`, a path to a split file,
            or ``None`` to use the rows' own ``split`` field / the sibling split file.
        max_seconds: Reject recordings longer than this when decoding.

    Raises:
        NotFoundError: If the manifest does not exist.
        InputValidationError: On an unknown split, a malformed row, or a missing split
            assignment (a speaker-disjoint split file is mandatory).
    """

    def __init__(
        self,
        manifest: Path,
        split: str,
        *,
        feature_fn: FeatureFn,
        splits: Splits | Path | None = None,
        max_seconds: float = 30.0,
    ) -> None:
        self._manifest = Path(manifest)
        if not self._manifest.is_file():
            raise missing_path(self._manifest, "audio_manifest_missing")
        canonical = normalize_split_name(split)
        if canonical not in CORE_SPLITS:
            raise invalid_data("unknown_split", split=split, available=list(CORE_SPLITS))

        self._split = canonical
        self._feature_fn = feature_fn
        self._max_seconds = float(max_seconds)
        self._splits = self._resolve_splits(splits)
        self._rows = self._read_rows()
        _log.info(
            "data.audio_commands.opened",
            manifest=str(self._manifest),
            split=split,
            n_rows=len(self._rows),
            speakers=len({row.speaker for row in self._rows}),
        )

    # ------------------------------------------------------------------ dataset protocol
    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> AudioSample:
        """Decode row ``index`` and build its features.

        Raises:
            IndexError: If ``index`` is out of range.
            NotFoundError: If the WAV file named by the row is missing.
            AsrInputError: If the WAV file is not a valid 16-bit PCM container.
        """
        if index < 0:
            index += len(self._rows)
        if not 0 <= index < len(self._rows):
            raise IndexError(f"dataset index out of range: {index}")
        row = self._rows[index]
        audio = self._read_audio(row)
        features = np.asarray(self._feature_fn(audio), dtype=np.float32)
        return AudioSample(
            features=features,
            text=row.text,
            speaker=row.speaker,
            sample_id=row.id,
            sample_rate=audio.sr,
            duration_s=audio.duration_s,
        )

    def __iter__(self) -> Iterator[AudioSample]:
        for index in range(len(self)):
            yield self[index]

    # ------------------------------------------------------------------ introspection
    @property
    def rows(self) -> list[AudioRow]:
        """The manifest rows of this split, in file order."""
        return self._rows

    @property
    def split(self) -> str:
        """Served split name."""
        return self._split

    @property
    def speakers(self) -> set[str]:
        """Speaker pseudonyms present in this split."""
        return {row.speaker for row in self._rows}

    @property
    def splits_fingerprint(self) -> str | None:
        """Fingerprint of the split file (cited in every ASR report), if one was used."""
        return self._splits.fingerprint if self._splits is not None else None

    # ------------------------------------------------------------------ internals
    def _resolve_splits(self, splits: Splits | Path | None) -> Splits | None:
        if isinstance(splits, Splits):
            return splits
        if splits is not None:
            return load_splits(Path(splits))
        sibling = self._manifest.parent / SPLITS_FILENAME
        if sibling.is_file():
            return load_splits(sibling)
        return None

    def _read_audio(self, row: AudioRow) -> Audio:
        path = Path(row.path)
        if not path.is_file():
            raise missing_path(path, "audio_file_missing", sample_id=row.id)
        audio = read_wav_pcm16(path.read_bytes(), max_seconds=self._max_seconds)
        return audio if audio.sr == TARGET_SAMPLE_RATE else resample_to_16k(audio)

    def _read_rows(self) -> list[AudioRow]:
        index = self._splits.index() if self._splits is not None else {}
        rows: list[AudioRow] = []
        assigned = 0
        for line_no, line in enumerate(
            self._manifest.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise invalid_data(
                    "manifest_row_not_json", path=str(self._manifest), line=line_no
                ) from exc
            forbidden = sorted(FORBIDDEN_FIELDS & set(payload))
            if forbidden:
                raise invalid_data(
                    "manifest_row_has_identity_fields", line=line_no, fields=forbidden
                )
            sample_id = str(
                payload.get("id")
                or payload.get("recording_id")
                or Path(str(payload.get("path", ""))).stem
            )
            speaker = str(payload.get("speaker_pseudonym") or payload.get("speaker") or "")
            raw_split = payload.get("split") or index.get(sample_id) or index.get(speaker) or ""
            split = normalize_split_name(str(raw_split)) if raw_split else ""
            if split:
                assigned += 1
            if split != self._split:
                continue
            path = payload.get("path")
            text = payload.get("text")
            if not path or not isinstance(text, str) or not text.strip():
                raise invalid_data(
                    "manifest_row_incomplete",
                    line=line_no,
                    sample_id=sample_id,
                    missing=[k for k in ("path", "text") if not payload.get(k)],
                )
            rows.append(
                AudioRow(
                    id=sample_id,
                    path=str(path),
                    text=text,
                    speaker=speaker,
                    split=split,
                    university=str(payload.get("university") or ""),
                    gender=str(payload.get("gender") or ""),
                    age_band=str(payload.get("age_band") or ""),
                    duration_s=float(payload.get("duration_s") or 0.0),
                    source=str(payload.get("source") or "collected"),
                )
            )
        if assigned == 0:
            raise invalid_data(
                "speaker_split_missing",
                manifest=str(self._manifest),
                expected_file=SPLITS_FILENAME,
            )
        return rows


class SyntheticAudioDataset:
    """Deterministic synthetic utterances — the CPU smoke dataset for the ASR path.

    Args:
        n: Number of samples.
        seed: Base seed; sample ``i`` uses ``seed + i``.
        feature_fn: Feature extractor; ``None`` returns the raw waveform as features.
        seconds: Duration of each utterance.
        texts: Transcript pool; defaults to a few Turkish command sentences from
            :mod:`aicls.asr.stub`.
        snr_db: Optional noise level.

    Raises:
        InputValidationError: If ``n`` is not positive.
    """

    def __init__(
        self,
        n: int,
        *,
        seed: int = 1234,
        feature_fn: FeatureFn | None = None,
        seconds: float = 1.0,
        texts: tuple[str, ...] | None = None,
        snr_db: float | None = None,
    ) -> None:
        if int(n) <= 0:
            raise invalid_data("n_not_positive", n=int(n))
        from aicls.asr.stub import STUB_SENTENCES

        self._n = int(n)
        self._seed = int(seed)
        self._feature_fn = feature_fn
        self._seconds = float(seconds)
        self._texts = tuple(texts) if texts else tuple(STUB_SENTENCES)
        self._snr_db = snr_db

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> AudioSample:
        """Generate sample ``index`` from ``seed + index``.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        from aicls.data.synthetic import synth_audio

        if index < 0:
            index += self._n
        if not 0 <= index < self._n:
            raise IndexError(f"dataset index out of range: {index}")
        audio = synth_audio(
            np.random.default_rng(self._seed + index), self._seconds, snr_db=self._snr_db
        )
        features = (
            np.asarray(self._feature_fn(audio), dtype=np.float32)
            if self._feature_fn is not None
            else audio.samples
        )
        return AudioSample(
            features=features,
            text=self._texts[index % len(self._texts)],
            speaker=f"S{index % 4:05d}",
            sample_id=f"synthetic-{index:06d}",
            sample_rate=audio.sr,
            duration_s=audio.duration_s,
        )

    def __iter__(self) -> Iterator[AudioSample]:
        for index in range(len(self)):
            yield self[index]


__all__ = [
    "FORBIDDEN_FIELDS",
    "AudioCommandDataset",
    "AudioRow",
    "AudioSample",
    "FeatureFn",
    "SyntheticAudioDataset",
]
