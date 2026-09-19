"""Training-manifest export: no identity, no withdrawn or rejected rows, frozen test speakers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aicls.asr import make_speech_like, write_wav_pcm16
from aicls.collect import (
    MANIFEST_FIELDS,
    SPLIT_NAMES,
    Demographics,
    ManifestReport,
    accept_consent,
    accept_recording,
    current_consent,
    export_manifest,
    load_splits,
    withdraw,
)
from aicls.core.errors import InputValidationError
from aicls.core.hashing import sha256_bytes
from aicls.core.ids import ulid
from aicls.db import PromptSentence, RecordingRepository, Speaker
from tests.unit.collect.conftest import good_wav

#: Every key a manifest row may ever carry (checked field by field below).
EXPECTED_FIELDS = (
    "path",
    "text",
    "speaker_pseudonym",
    "university",
    "gender",
    "age_band",
    "split",
    "duration_s",
    "sample_rate",
    "sha256",
    "scenario_tag",
    "consent_version",
    "source",
    "recording_id",
)

#: Column names that would identify a participant; none may appear in an export.
FORBIDDEN_FIELDS = frozenset(
    {
        "user_id",
        "email",
        "name",
        "display_name",
        "speaker_id",
        "ip_hash",
        "user_agent_hash",
        "token_hash",
        "birth_date",
        "phone",
        "consented_at",
    }
)

_WAV = write_wav_pcm16(make_speech_like(duration_s=1.5))


def _contribute(
    repo: RecordingRepository,
    speaker: Speaker,
    prompt: PromptSentence,
    data_root: Path,
    *,
    qc_passed: bool = True,
    duration_s: float = 1.5,
) -> Path:
    """Store one recording (file plus row) without re-running quality control."""
    rel = f"audio/raw/{speaker.pseudonym}/{ulid()}.wav"
    path = data_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_WAV)
    qc = {"passed": qc_passed, "checks": [], "reasons_tr": [] if qc_passed else ["Gürültülü."]}
    repo.add(
        speaker.id,
        prompt.id,
        rel,
        qc,
        duration_s,
        sha256_bytes(_WAV),
        size_bytes=len(_WAV),
    )
    return path


def _consent(repo: RecordingRepository, *, university: str = "mus", gender: str = "f") -> Speaker:
    """Register one more pseudonymous speaker through the real consent flow."""
    return accept_consent(
        repo,
        user_id=None,
        version=current_consent().version,
        demographics=Demographics(age_band="18-24", gender=gender, university=university),
        ip_hash=None,
    )


def _rows(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class TestManifestContent:
    """The manifest carries training data and nothing that could identify a participant."""

    def test_export_manifest_has_no_identity_fields(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
        data_root: Path,
        tmp_path: Path,
    ) -> None:
        from aicls.db import UserRepository

        user = UserRepository(repo.session).create("ada@example.edu.tr", "argon2id$h", "Ada")
        linked = accept_consent(
            repo,
            user_id=user.id,
            version=current_consent().version,
            demographics=Demographics(age_band="25-34", gender="m", university="mersin"),
            ip_hash="c" * 64,
        )
        _contribute(repo, linked, prompts[0], data_root)
        accept_recording(repo, speaker_id=speaker.id, prompt_id=prompts[1].id, wav=good_wav())

        out = tmp_path / "manifest.jsonl"
        report = export_manifest(repo, out=out, data_root=data_root)

        rows = _rows(out)
        assert len(rows) == 2
        assert report.rows == 2
        for row in rows:
            assert set(row) == set(EXPECTED_FIELDS)
            assert set(row) == set(MANIFEST_FIELDS)
            assert not set(row) & FORBIDDEN_FIELDS
            assert row["speaker_pseudonym"] in ("P00001", "P00002")
            assert row["split"] in SPLIT_NAMES
        blob = out.read_text(encoding="utf-8")
        for secret in (user.id, user.email, linked.id, speaker.id, "c" * 64, "Ada"):
            assert secret not in blob

    def test_rows_carry_the_training_payload(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
        data_root: Path,
        tmp_path: Path,
    ) -> None:
        decision = accept_recording(
            repo, speaker_id=speaker.id, prompt_id=prompts[0].id, wav=good_wav()
        )
        out = tmp_path / "manifest.jsonl"
        export_manifest(repo, out=out, data_root=data_root)
        (row,) = _rows(out)
        assert row["path"] == decision.rel_path
        assert row["text"] == prompts[0].text_tr
        assert row["scenario_tag"] == prompts[0].scenario_tag
        assert row["sample_rate"] == 16000
        assert row["sha256"] == decision.sha256
        assert row["consent_version"] == current_consent().version
        assert row["source"] == "contrib"
        assert row["university"] == "mus"
        assert row["gender"] == "f"
        assert row["age_band"] == "18-24"

    def test_source_label_can_be_overridden_for_internal_runs(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
        data_root: Path,
        tmp_path: Path,
    ) -> None:
        _contribute(repo, speaker, prompts[0], data_root)
        out = tmp_path / "manifest.jsonl"
        report = export_manifest(repo, out=out, data_root=data_root, source="internal")
        assert report.source == "internal"
        assert _rows(out)[0]["source"] == "internal"

    def test_report_summarises_the_export(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
        data_root: Path,
        tmp_path: Path,
    ) -> None:
        for prompt in prompts[:3]:
            _contribute(repo, speaker, prompt, data_root, duration_s=2.0)
        out = tmp_path / "manifest.jsonl"
        report = export_manifest(repo, out=out, data_root=data_root)
        assert isinstance(report, ManifestReport)
        assert report.path == out
        assert report.rows == 3
        assert report.speakers == 1
        assert report.seconds == pytest.approx(6.0)
        assert report.hours == pytest.approx(6.0 / 3600.0)
        assert sum(report.by_split.values()) == 3
        assert report.consent_versions == (current_consent().version,)
        assert report.fingerprint.startswith("sha256:")
        assert report.splits_path.is_file()

    def test_export_is_byte_reproducible(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
        data_root: Path,
        tmp_path: Path,
    ) -> None:
        _contribute(repo, speaker, prompts[0], data_root)
        first = export_manifest(repo, out=tmp_path / "a.jsonl", data_root=data_root)
        second = export_manifest(repo, out=tmp_path / "b.jsonl", data_root=data_root)
        assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
        assert first.fingerprint == second.fingerprint

    def test_rejects_a_non_jsonl_destination(
        self, repo: RecordingRepository, data_root: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(InputValidationError):
            export_manifest(repo, out=tmp_path / "manifest.csv", data_root=data_root)


class TestManifestExclusions:
    """Only ``qc_passed AND deleted_at IS NULL AND withdrawn_at IS NULL`` may be trained on."""

    def test_export_manifest_excludes_withdrawn_and_qc_failed(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
        data_root: Path,
        tmp_path: Path,
    ) -> None:
        keeper = _consent(repo, university="mersin")
        kept = _contribute(repo, keeper, prompts[0], data_root)
        rejected = _contribute(repo, keeper, prompts[1], data_root, qc_passed=False)
        gone = _contribute(repo, speaker, prompts[2], data_root)

        withdraw(repo, speaker.id, data_root=data_root)
        assert not gone.exists()

        out = tmp_path / "manifest.jsonl"
        report = export_manifest(repo, out=out, data_root=data_root)

        paths = [row["path"] for row in _rows(out)]
        assert paths == [kept.relative_to(data_root).as_posix()]
        assert rejected.relative_to(data_root).as_posix() not in paths
        assert report.rows == 1
        assert report.speakers == 1
        assert report.excluded_qc_failed == 1
        assert report.excluded_withdrawn == 1
        assert "P00001" not in out.read_text(encoding="utf-8")

    def test_rows_whose_audio_is_missing_are_reported_and_skipped(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
        data_root: Path,
        tmp_path: Path,
    ) -> None:
        kept = _contribute(repo, speaker, prompts[0], data_root)
        lost = _contribute(repo, speaker, prompts[1], data_root)
        lost.unlink()

        out = tmp_path / "manifest.jsonl"
        report = export_manifest(repo, out=out, data_root=data_root)
        assert report.missing_files == 1
        assert [row["path"] for row in _rows(out)] == [kept.relative_to(data_root).as_posix()]

        loose = export_manifest(
            repo, out=tmp_path / "loose.jsonl", data_root=data_root, verify_files=False
        )
        assert loose.rows == 2
        assert loose.missing_files == 0


class TestSplits:
    """Speaker-disjoint, stratified, frozen across report periods (ADR-0008)."""

    def _population(
        self,
        repo: RecordingRepository,
        prompts: list[PromptSentence],
        data_root: Path,
        n: int,
        *,
        university: str = "mus",
    ) -> list[Speaker]:
        speakers = []
        for index in range(n):
            person = _consent(repo, university=university)
            _contribute(repo, person, prompts[index % len(prompts)], data_root)
            speakers.append(person)
        return speakers

    def test_split_is_speaker_disjoint(
        self,
        repo: RecordingRepository,
        prompts: list[PromptSentence],
        data_root: Path,
        tmp_path: Path,
    ) -> None:
        people = self._population(repo, prompts, data_root, 10)
        for person in people[:3]:  # a few speakers contribute twice
            _contribute(repo, person, prompts[4], data_root)

        out = tmp_path / "manifest.jsonl"
        export_manifest(repo, out=out, data_root=data_root)

        by_speaker: dict[str, set[str]] = {}
        for row in _rows(out):
            by_speaker.setdefault(str(row["speaker_pseudonym"]), set()).add(str(row["split"]))
        assert all(len(splits) == 1 for splits in by_speaker.values())

        groups: dict[str, set[str]] = {}
        for pseudonym, splits in by_speaker.items():
            groups.setdefault(next(iter(splits)), set()).add(pseudonym)
        assert not groups["train"] & groups["test"]
        assert set(groups) <= set(SPLIT_NAMES)

    def test_split_ratios_are_respected(
        self,
        repo: RecordingRepository,
        prompts: list[PromptSentence],
        data_root: Path,
        tmp_path: Path,
    ) -> None:
        self._population(repo, prompts, data_root, 10)
        report = export_manifest(repo, out=tmp_path / "manifest.jsonl", data_root=data_root)
        assert report.by_split_speakers == {"train": 8, "dev": 1, "test": 1}

    def test_split_is_stratified_over_universities(
        self,
        repo: RecordingRepository,
        prompts: list[PromptSentence],
        data_root: Path,
        tmp_path: Path,
    ) -> None:
        self._population(repo, prompts, data_root, 10, university="mus")
        self._population(repo, prompts, data_root, 10, university="mersin")
        out = tmp_path / "manifest.jsonl"
        export_manifest(repo, out=out, data_root=data_root)
        universities = {str(row["university"]) for row in _rows(out) if row["split"] == "test"}
        assert universities == {"mus", "mersin"}

    def test_split_freezes_previous_test_speakers(
        self,
        repo: RecordingRepository,
        prompts: list[PromptSentence],
        data_root: Path,
        tmp_path: Path,
    ) -> None:
        self._population(repo, prompts, data_root, 10)
        out = tmp_path / "manifest.jsonl"
        first = export_manifest(repo, out=out, data_root=data_root)
        before = {str(row["speaker_pseudonym"]): str(row["split"]) for row in _rows(out)}
        frozen_test = {p for p, split in before.items() if split == "test"}
        assert frozen_test

        # A second collection campaign doubles the population.
        self._population(repo, prompts, data_root, 10)
        second = export_manifest(repo, out=out, data_root=data_root)
        after = {str(row["speaker_pseudonym"]): str(row["split"]) for row in _rows(out)}

        assert {p for p, split in after.items() if split == "test"} >= frozen_test
        for pseudonym, split in before.items():
            assert after[pseudonym] == split, pseudonym
        assert second.rows > first.rows
        assert second.frozen_speakers == 10

        stored = load_splits(first.splits_path)
        assert stored.seed == 1234
        assert len(stored.by_speaker) == 20
        assert {p for p, s in stored.by_speaker.items() if s == "test"} >= frozen_test

    def test_split_seed_changes_the_assignment_for_new_speakers(
        self,
        repo: RecordingRepository,
        prompts: list[PromptSentence],
        data_root: Path,
        tmp_path: Path,
    ) -> None:
        self._population(repo, prompts, data_root, 10)
        a = export_manifest(repo, out=tmp_path / "a.jsonl", splits_path=tmp_path / "a_splits.json")
        b = export_manifest(
            repo,
            out=tmp_path / "b.jsonl",
            splits_path=tmp_path / "b_splits.json",
            split_seed=99,
        )
        assert a.by_split_speakers == b.by_split_speakers
        first = load_splits(a.splits_path).by_speaker
        second = load_splits(b.splits_path).by_speaker
        assert first != second

    def test_corrupt_splits_file_is_rejected(
        self,
        repo: RecordingRepository,
        prompts: list[PromptSentence],
        data_root: Path,
        tmp_path: Path,
    ) -> None:
        splits_path = tmp_path / "audio_commands_splits.json"
        splits_path.write_text("{not json", encoding="utf-8")
        with pytest.raises(InputValidationError):
            export_manifest(repo, out=tmp_path / "manifest.jsonl", splits_path=splits_path)
