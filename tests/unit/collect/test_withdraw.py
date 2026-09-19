"""KVKK withdrawal: the audio disappears, the proof of consent stays (veri-yonetimi §4)."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from aicls.collect import (
    Demographics,
    WithdrawReport,
    accept_consent,
    accept_recording,
    current_consent,
    speaker_audio_dir,
    withdraw,
)
from aicls.core.errors import NotFoundError
from aicls.db import ConsentVersion, PromptSentence, Recording, RecordingRepository, Speaker
from tests.unit.collect.conftest import good_wav


class TestWithdraw:
    """``POST /contribute/withdraw`` / ``aicls contrib withdraw --speaker P00012``."""

    def test_withdraw_deletes_files_keeps_consent_audit(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
        data_root: Path,
    ) -> None:
        paths = []
        for prompt in prompts[:2]:
            decision = accept_recording(
                repo, speaker_id=speaker.id, prompt_id=prompt.id, wav=good_wav()
            )
            assert decision.path is not None
            paths.append(decision.path)
        # A derived feature file next to the audio must go as well.
        features = data_root / "audio" / "features" / speaker.pseudonym
        features.mkdir(parents=True, exist_ok=True)
        (features / "mel.npy").write_bytes(b"derived")

        consent_id = speaker.consent_version_id
        report = withdraw(repo, speaker.id, data_root=data_root)

        assert isinstance(report, WithdrawReport)
        assert report.speaker_pseudonym == "P00001"
        assert report.recordings_deleted == 2
        assert report.files_deleted == 3
        assert report.bytes_deleted > 0
        assert report.consent_kept is True
        assert report.withdrawn_at is not None

        # 1. every audio file and derived feature is gone, and so is the speaker directory
        assert all(not path.exists() for path in paths)
        assert not speaker_audio_dir(speaker.pseudonym, data_root=data_root).exists()
        assert not features.exists()

        # 2. the speaker row survives as an audit record, stripped of the network identifiers
        stored = repo.get_speaker(speaker.id)
        assert stored is not None
        assert stored.withdrawn_at is not None
        assert stored.ip_hash is None
        assert stored.user_agent_hash is None
        assert stored.pseudonym == "P00001"
        assert stored.consent_version_id == consent_id

        # 3. the consent text version itself is untouched (KVKK proof of consent)
        assert repo.session.get(ConsentVersion, consent_id) is not None

        # 4. the recordings are soft-deleted and no longer exportable
        rows = list(repo.session.scalars(select(Recording)))
        assert len(rows) == 2
        assert all(row.deleted_at is not None for row in rows)
        assert repo.exportable() == []

    def test_withdraw_without_recordings_is_a_no_op_report(
        self, repo: RecordingRepository, speaker: Speaker, data_root: Path
    ) -> None:
        report = withdraw(repo, speaker.pseudonym, data_root=data_root)
        assert report.recordings_deleted == 0
        assert report.files_deleted == 0
        assert report.bytes_deleted == 0
        assert report.consent_kept is True

    def test_withdraw_is_idempotent(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
        data_root: Path,
    ) -> None:
        accept_recording(repo, speaker_id=speaker.id, prompt_id=prompts[0].id, wav=good_wav())
        first = withdraw(repo, speaker.id, data_root=data_root)
        second = withdraw(repo, speaker.id, data_root=data_root)
        assert first.files_deleted == 1
        assert second.files_deleted == 0
        assert second.recordings_deleted == 0

    def test_withdraw_unknown_speaker_is_not_found(
        self, repo: RecordingRepository, data_root: Path
    ) -> None:
        with pytest.raises(NotFoundError):
            withdraw(repo, "P99999", data_root=data_root)

    def test_withdraw_leaves_other_speakers_alone(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
        data_root: Path,
        demographics: Demographics,
    ) -> None:
        other = accept_consent(
            repo,
            user_id=None,
            version=current_consent().version,
            demographics=demographics,
            ip_hash=None,
        )
        kept = accept_recording(repo, speaker_id=other.id, prompt_id=prompts[0].id, wav=good_wav())
        accept_recording(repo, speaker_id=speaker.id, prompt_id=prompts[1].id, wav=good_wav())

        withdraw(repo, speaker.id, data_root=data_root)

        assert kept.path is not None
        assert kept.path.is_file()
        assert len(repo.exportable()) == 1
