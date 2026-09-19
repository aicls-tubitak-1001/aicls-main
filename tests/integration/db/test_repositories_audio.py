"""Behaviour of the voice data-collection repository (KVKK rules, İP-3 balancing)."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from aicls.core.errors import NotFoundError
from aicls.db import (
    AssetKind,
    AssetRepository,
    ConsentVersion,
    PromptSentence,
    RecognitionRepository,
    RecordingRepository,
    ReviewStatus,
    Speaker,
    Split,
    User,
)

pytestmark = pytest.mark.integration

PASSING_QC = {"passed": True, "snr_db": 22.0, "clipping_ratio": 0.0, "silence_ratio": 0.1}
FAILING_QC = {"passed": False, "snr_db": 4.0, "reasons_tr": ["Ortam gürültüsü yüksek."]}


def _record(
    repo: RecordingRepository,
    speaker: Speaker,
    prompt: PromptSentence,
    qc: dict[str, object] = PASSING_QC,
    duration_s: float = 3.0,
) -> object:
    return repo.add(speaker.id, prompt.id, f"audio/raw/{prompt.id}.wav", qc, duration_s, "a" * 64)


class TestConsentAndSpeakers:
    """Consent versions are immutable and speakers carry only pseudonymous data."""

    def test_upsert_consent_version_is_idempotent(self, session: Session) -> None:
        repo = RecordingRepository(session)
        first = repo.upsert_consent_version("v1", "Onam metni", sha256="a" * 64)
        second = repo.upsert_consent_version("v1", "DEĞİŞTİRİLMİŞ metin")
        assert second is first
        assert first.text_tr == "Onam metni"
        assert repo.get_consent_version("v1") is first
        assert repo.get_consent_version("v9") is None

    def test_pseudonyms_increment_and_hold_no_identity(
        self, session: Session, consent: ConsentVersion
    ) -> None:
        repo = RecordingRepository(session)
        first = repo.create_speaker(consent.id, university="mus", gender="f", age_band="18-24")
        second = repo.create_speaker(consent.id, university="mersin", gender="m", age_band="45+")
        assert (first.pseudonym, second.pseudonym) == ("P00001", "P00002")
        assert first.user_id is None
        assert repo.get_speaker("P00002") is second
        assert repo.get_speaker(second.id) is second
        assert repo.get_speaker("P99999") is None

    def test_speaker_can_be_linked_to_a_user_for_progress_tracking(
        self, session: Session, consent: ConsentVersion, user: User
    ) -> None:
        repo = RecordingRepository(session)
        speaker = repo.create_speaker(consent.id, user_id=user.id)
        assert repo.speaker_for_user(user.id) is speaker
        assert repo.speaker_for_user("0" * 26) is None


class TestPromptBalancing:
    """``next_prompts`` implements the least-recorded balancing of İP-3."""

    def test_add_prompts_is_idempotent(self, session: Session) -> None:
        repo = RecordingRepository(session)
        rows = [{"text_tr": "türev al"}, {"text_tr": "limit hesapla"}, {"text_tr": ""}]
        assert repo.add_prompts(rows) == 2
        assert repo.add_prompts(rows) == 0

    def test_next_prompts_prefers_least_recorded_and_skips_own_recordings(
        self, session: Session, speaker: Speaker, prompts: list[PromptSentence]
    ) -> None:
        repo = RecordingRepository(session)
        busy, quiet = prompts[0], prompts[1]
        other = repo.create_speaker(speaker.consent_version_id)
        _record(repo, other, busy)
        _record(repo, other, busy)
        ordered = repo.next_prompts(speaker.id, 3)
        assert ordered[0].id in {quiet.id, prompts[2].id}
        assert ordered[-1].id == busy.id

        _record(repo, speaker, quiet)
        remaining = [p.id for p in repo.next_prompts(speaker.id, 5)]
        assert quiet.id not in remaining

    def test_inactive_prompts_are_never_offered(
        self, session: Session, speaker: Speaker, prompts: list[PromptSentence]
    ) -> None:
        repo = RecordingRepository(session)
        prompts[0].is_active = False
        session.flush()
        assert prompts[0].id not in {p.id for p in repo.next_prompts(speaker.id, 10)}


class TestRecordingsAndWithdrawal:
    """Recording rows, the QC queue, manifest eligibility and KVKK withdrawal."""

    def test_add_creates_the_audio_asset_and_lifts_qc_fields(
        self, session: Session, speaker: Speaker, prompts: list[PromptSentence]
    ) -> None:
        repo = RecordingRepository(session)
        recording = _record(repo, speaker, prompts[0])
        assert recording.qc_passed is True
        assert recording.snr_db == 22.0
        assert recording.qc_json == PASSING_QC
        assert recording.review_status is ReviewStatus.PENDING
        assert recording.split is Split.UNASSIGNED
        asset = AssetRepository(session).get(recording.asset_id)
        assert asset is not None
        assert asset.kind is AssetKind.AUDIO_WAV
        assert asset.path.startswith("audio/raw/")
        assert asset.mime == "audio/wav"

    def test_set_review_and_set_split(
        self, session: Session, speaker: Speaker, prompts: list[PromptSentence], user: User
    ) -> None:
        repo = RecordingRepository(session)
        recording = _record(repo, speaker, prompts[0])
        repo.set_review(recording.id, "fixed", reviewer_id=user.id, transcript_fix="türev al")
        assert recording.review_status is ReviewStatus.FIXED
        assert recording.transcript_final == "türev al"
        assert recording.reviewed_by == user.id
        repo.set_split(recording.id, "test")
        assert recording.split is Split.TEST
        with pytest.raises(NotFoundError):
            repo.set_review("0" * 26, "accepted")
        with pytest.raises(NotFoundError):
            repo.set_split("0" * 26, "train")

    def test_qc_queue_lists_pending_recordings_only(
        self, session: Session, speaker: Speaker, prompts: list[PromptSentence]
    ) -> None:
        repo = RecordingRepository(session)
        pending = _record(repo, speaker, prompts[0])
        reviewed = _record(repo, speaker, prompts[1])
        repo.set_review(reviewed.id, ReviewStatus.ACCEPTED)
        assert [r.id for r in repo.qc_queue()] == [pending.id]
        assert [r.id for r in repo.qc_queue("accepted")] == [reviewed.id]

    def test_exportable_excludes_withdrawn_and_qc_failed(
        self, session: Session, speaker: Speaker, prompts: list[PromptSentence]
    ) -> None:
        repo = RecordingRepository(session)
        good = _record(repo, speaker, prompts[0])
        _record(repo, speaker, prompts[1], FAILING_QC)
        withdrawn_speaker = repo.create_speaker(speaker.consent_version_id)
        _record(repo, withdrawn_speaker, prompts[2])
        repo.mark_withdrawn(withdrawn_speaker.pseudonym)

        exported = repo.exportable()
        assert [row[0].id for row in exported] == [good.id]
        assert exported[0][1].pseudonym == speaker.pseudonym
        assert exported[0][2].id == prompts[0].id

    def test_mark_withdrawn_soft_deletes_and_clears_hashes(
        self, session: Session, speaker: Speaker, prompts: list[PromptSentence]
    ) -> None:
        repo = RecordingRepository(session)
        speaker.ip_hash = "h" * 64
        speaker.user_agent_hash = "u" * 64
        _record(repo, speaker, prompts[0])
        _record(repo, speaker, prompts[1])
        assert repo.mark_withdrawn(speaker.id) == 2
        assert speaker.withdrawn_at is not None
        assert speaker.ip_hash is None and speaker.user_agent_hash is None
        # The consent audit row survives the withdrawal.
        assert repo.get_consent_version("v1") is not None
        assert repo.mark_withdrawn(speaker.id) == 0
        with pytest.raises(NotFoundError):
            repo.mark_withdrawn("P00404")

    def test_stats_aggregates_hours_speakers_and_distributions(
        self, session: Session, speaker: Speaker, prompts: list[PromptSentence]
    ) -> None:
        repo = RecordingRepository(session)
        second = repo.create_speaker(
            speaker.consent_version_id, university="amasya", gender="m", age_band="25-34"
        )
        _record(repo, speaker, prompts[0], PASSING_QC, 1800.0)
        _record(repo, speaker, prompts[1], PASSING_QC, 1800.0)
        _record(repo, second, prompts[2], PASSING_QC, 3600.0)
        _record(repo, second, prompts[0], FAILING_QC, 5.0)

        stats = repo.stats()
        assert stats.utterances == 3
        assert stats.speakers == 2
        assert stats.hours_accepted == pytest.approx(2.0)
        assert stats.by_university == {"amasya": 1, "mus": 2}
        assert stats.by_gender == {"f": 2, "m": 1}
        assert stats.by_age_band == {"18-24": 2, "25-34": 1}
        assert stats.qc_pass_rate == pytest.approx(0.75)

    def test_stats_on_an_empty_database(self, session: Session) -> None:
        stats = RecordingRepository(session).stats()
        assert (stats.utterances, stats.speakers, stats.qc_pass_rate) == (0, 0, 0.0)
        assert stats.hours_accepted == 0.0


class TestRecognitionCorrections:
    """OCR/ASR outputs and the corrections that feed retraining."""

    def test_recognition_correction_sets_timestamp(self, session: Session, user: User) -> None:
        repo = RecognitionRepository(session)
        recognition = repo.add(
            "x^{2}", user_id=user.id, confidence=0.8, model_version="ocr-0.1.0", elapsed_ms=120
        )
        assert repo.get(recognition.id) is recognition
        assert recognition.corrected_at is None
        repo.set_correction(recognition.id, "x^2")
        assert recognition.corrected_latex == "x^2"
        assert recognition.corrected_at is not None
        assert [r.id for r in repo.corrections()] == [recognition.id]
        assert repo.corrections(limit=1)[0].id == recognition.id
        with pytest.raises(NotFoundError):
            repo.set_correction("0" * 26, "x")

    def test_transcript_correction(self, session: Session, user: User) -> None:
        repo = RecognitionRepository(session)
        transcript = repo.add_transcript(
            "x karenin türevini al", user_id=user.id, confidence=0.9, model_version="asr-0.1.0"
        )
        repo.set_transcript_correction(transcript.id, "x karenin türevini al.")
        assert transcript.corrected_text.endswith(".")
        with pytest.raises(NotFoundError):
            repo.set_transcript_correction("0" * 26, "x")
