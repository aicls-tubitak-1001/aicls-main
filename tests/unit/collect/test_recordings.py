"""Recording acceptance policy: consent first, quality control second, only then a file."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from sqlalchemy import select

from aicls.asr import QcThresholds
from aicls.collect import (
    ConsentRequiredError,
    ContributionGateError,
    RecordingDecision,
    accept_recording,
    speaker_audio_dir,
)
from aicls.core.errors import AiclsError, ErrorCode, NotFoundError
from aicls.core.hashing import sha256_file
from aicls.core.ids import is_ulid
from aicls.core.settings import Settings, settings_override
from aicls.db import Asset, PromptSentence, Recording, RecordingRepository, Speaker
from tests.unit.collect.conftest import bad_wav, good_wav, short_wav, silent_wav


def _raw_root(data_root: Path) -> Path:
    return data_root / "audio" / "raw"


def _all_files(data_root: Path) -> list[Path]:
    root = _raw_root(data_root)
    return sorted(p for p in root.rglob("*") if p.is_file())


class TestConsentGate:
    """No consent, no recording (``GET /contribute/consent/current`` then 403)."""

    def test_consent_required_before_recording(
        self,
        repo: RecordingRepository,
        prompts: list[PromptSentence],
        data_root: Path,
    ) -> None:
        with pytest.raises(ConsentRequiredError) as excinfo:
            accept_recording(
                repo,
                speaker_id="P00001",
                prompt_id=prompts[0].id,
                wav=good_wav(),
                thresholds=QcThresholds(),
            )
        assert excinfo.value.code is ErrorCode.CONSENT_REQUIRED
        assert excinfo.value.http_status == 403
        assert _all_files(data_root) == []
        assert repo.session.scalars(select(Recording)).all() == []

    def test_withdrawn_speaker_may_not_record_again(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
    ) -> None:
        from aicls.collect import withdraw

        withdraw(repo, speaker.id)
        with pytest.raises(ConsentRequiredError):
            accept_recording(repo, speaker_id=speaker.id, prompt_id=prompts[0].id, wav=good_wav())

    def test_stale_consent_version_must_be_re_accepted(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
    ) -> None:
        older = repo.upsert_consent_version("v0-archived", "Eski onam metni")
        speaker.consent_version_id = older.id
        repo.flush()
        with pytest.raises(ConsentRequiredError) as excinfo:
            accept_recording(repo, speaker_id=speaker.id, prompt_id=prompts[0].id, wav=good_wav())
        assert excinfo.value.details["accepted_version"] == "v0-archived"

    def test_gate_blocks_recording_without_an_ethics_reference(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
    ) -> None:
        with (
            settings_override(ethics_approval_ref=""),
            pytest.raises(ContributionGateError),
        ):
            accept_recording(repo, speaker_id=speaker.id, prompt_id=prompts[0].id, wav=good_wav())


class TestQualityGate:
    """A rejected upload never reaches the disk and never becomes a row."""

    def test_recording_rejected_by_qc_writes_no_file(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
        data_root: Path,
    ) -> None:
        decision = accept_recording(
            repo,
            speaker_id=speaker.id,
            prompt_id=prompts[0].id,
            wav=bad_wav(),
            thresholds=QcThresholds(),
        )
        assert isinstance(decision, RecordingDecision)
        assert decision.accepted is False
        assert decision.recording_id is None
        assert decision.path is None
        assert decision.rel_path is None
        assert decision.qc.passed is False
        assert decision.reasons_tr
        assert "snr_db" in {check.name for check in decision.qc.failed_checks}
        assert _all_files(data_root) == []
        assert repo.session.scalars(select(Recording)).all() == []

    @pytest.mark.parametrize("payload", [short_wav, silent_wav])
    def test_other_quality_failures_also_write_nothing(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
        data_root: Path,
        payload: Callable[[], bytes],
    ) -> None:
        decision = accept_recording(
            repo, speaker_id=speaker.id, prompt_id=prompts[0].id, wav=payload()
        )
        assert decision.accepted is False
        assert _all_files(data_root) == []

    def test_thresholds_may_be_tightened_per_campaign(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
        data_root: Path,
    ) -> None:
        strict = QcThresholds(min_snr_db=99.0)
        assert (
            accept_recording(
                repo,
                speaker_id=speaker.id,
                prompt_id=prompts[0].id,
                wav=good_wav(),
                thresholds=strict,
            ).accepted
            is False
        )
        assert _all_files(data_root) == []


class TestAcceptedRecording:
    """An accepted upload lands under ``AICLS_DATA_ROOT/audio/raw/<pseudonym>/<ulid>.wav``."""

    def test_writes_file_row_and_digest(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
        data_root: Path,
    ) -> None:
        decision = accept_recording(
            repo,
            speaker_id=speaker.id,
            prompt_id=prompts[0].id,
            wav=good_wav(),
            thresholds=QcThresholds(),
        )
        assert decision.accepted is True
        assert decision.qc.passed is True
        assert decision.reasons_tr == []
        assert decision.recording_id is not None
        assert is_ulid(decision.recording_id)

        path = decision.path
        assert path is not None
        assert path.is_file()
        assert path.parent == speaker_audio_dir(speaker.pseudonym, data_root=data_root)
        assert path.parent.name == "P00001"
        assert path.suffix == ".wav"
        assert is_ulid(path.stem)
        assert decision.rel_path == f"audio/raw/P00001/{path.name}"
        assert list(path.parent.iterdir()) == [path]  # no leftover temporary file

        row = repo.session.get(Recording, decision.recording_id)
        assert row is not None
        assert row.speaker_id == speaker.id
        assert row.prompt_id == prompts[0].id
        assert row.qc_passed is True
        assert row.sample_rate == 16000
        assert row.deleted_at is None
        assert row.sha256 == sha256_file(path)
        assert row.qc_json["passed"] is True
        assert [c["name"] for c in row.qc_json["checks"]] == [
            check.name for check in decision.qc.checks
        ]
        assert 1.9 < row.duration_s < 2.1

        asset = repo.session.get(Asset, row.asset_id)
        assert asset is not None
        assert asset.path == decision.rel_path
        assert asset.mime == "audio/wav"
        assert asset.kind.value == "audio_wav"
        assert asset.bytes == path.stat().st_size

    def test_stored_audio_is_canonical_16k_mono(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
    ) -> None:
        from aicls.asr import read_wav_pcm16

        decision = accept_recording(
            repo, speaker_id=speaker.id, prompt_id=prompts[0].id, wav=good_wav()
        )
        assert decision.path is not None
        audio = read_wav_pcm16(decision.path.read_bytes())
        assert audio.sr == 16000
        assert audio.samples.ndim == 1

    def test_two_recordings_get_distinct_files(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
        data_root: Path,
    ) -> None:
        first = accept_recording(
            repo, speaker_id=speaker.id, prompt_id=prompts[0].id, wav=good_wav()
        )
        second = accept_recording(
            repo, speaker_id=speaker.id, prompt_id=prompts[1].id, wav=good_wav()
        )
        assert first.path != second.path
        assert len(_all_files(data_root)) == 2

    def test_as_dict_matches_the_api_payload(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
    ) -> None:
        payload = accept_recording(
            repo, speaker_id=speaker.id, prompt_id=prompts[0].id, wav=good_wav()
        ).as_dict()
        assert set(payload) == {"id", "accepted", "qc", "duration_s"}
        assert set(payload["qc"]) == {"passed", "checks", "reasons_tr"}
        assert "path" not in payload  # a filesystem path never reaches a client


class TestInputValidation:
    """Everything that arrives from the browser is validated before anything is written."""

    def test_unknown_prompt_is_not_found(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
        data_root: Path,
    ) -> None:
        with pytest.raises(NotFoundError):
            accept_recording(
                repo, speaker_id=speaker.id, prompt_id="01JQZZZZZZZZZZZZZZZZZZZZZZ", wav=good_wav()
            )
        assert _all_files(data_root) == []

    def test_inactive_prompt_is_refused(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
    ) -> None:
        prompts[0].is_active = False
        repo.flush()
        with pytest.raises(NotFoundError):
            accept_recording(repo, speaker_id=speaker.id, prompt_id=prompts[0].id, wav=good_wav())

    def test_non_wav_payload_is_refused_without_writing(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
        data_root: Path,
    ) -> None:
        with pytest.raises(AiclsError) as excinfo:
            accept_recording(
                repo,
                speaker_id=speaker.id,
                prompt_id=prompts[0].id,
                wav=b"\x89PNG\r\n\x1a\n" + b"0" * 200,
            )
        assert excinfo.value.code is ErrorCode.ASR_INPUT_INVALID
        assert _all_files(data_root) == []
        assert repo.session.scalars(select(Recording)).all() == []

    def test_settings_data_root_is_used_when_none_is_given(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
        collect_settings: Settings,
    ) -> None:
        decision = accept_recording(
            repo, speaker_id=speaker.id, prompt_id=prompts[0].id, wav=good_wav()
        )
        assert decision.path is not None
        assert collect_settings.data_root in decision.path.parents
