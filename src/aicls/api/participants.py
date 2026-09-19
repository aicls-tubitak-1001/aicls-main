"""Participant lookup for the voice-collection tab (``speakers.token_hash``).

A contributor may be anonymous: the consent endpoint hands back a pseudonym and an opaque
participant token, and every later call presents that token in ``X-Participant-Token``. Only its
SHA-256 digest is stored, so the database never holds a credential that could be replayed if it
leaked (docs/guvenlik-gizlilik.md §2).

:mod:`aicls.db` owns no query for this — ``RecordingRepository`` addresses a speaker by id or by
pseudonym — so the three extra reads live here, still behind a
:class:`aicls.db.repositories.base.Repository` subclass (ADR-0003 rule 8).
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select

from aicls.db.base import utcnow
from aicls.db.models import Recording, Speaker
from aicls.db.repositories.base import Repository

#: Header carrying the opaque participant token of an anonymous contributor.
PARTICIPANT_TOKEN_HEADER = "X-Participant-Token"


@dataclass(frozen=True, slots=True)
class Progress:
    """``GET /contribute/me/progress`` figures for one speaker."""

    n_accepted: int = 0
    seconds_accepted: float = 0.0


class ParticipantRepository(Repository):
    """Reads and writes the API needs on top of ``RecordingRepository``."""

    def by_token_hash(self, digest: str) -> Speaker | None:
        """Return the speaker whose participant token hashes to ``digest``."""
        if not digest:
            return None
        stmt = select(Speaker).where(Speaker.token_hash == digest)
        return self.session.scalars(stmt).one_or_none()

    def set_token(self, speaker: Speaker, digest: str) -> Speaker:
        """Store the digest of a freshly minted participant token."""
        speaker.token_hash = digest
        self.session.flush()
        return speaker

    def progress(self, speaker_id: str) -> Progress:
        """Accepted utterance count and total seconds of one speaker."""
        row = self.session.execute(
            select(
                func.count(Recording.id),
                func.coalesce(func.sum(Recording.duration_s), 0.0),
            ).where(
                Recording.speaker_id == speaker_id,
                Recording.qc_passed.is_(True),
                Recording.deleted_at.is_(None),
            )
        ).one()
        return Progress(n_accepted=int(row[0] or 0), seconds_accepted=float(row[1] or 0.0))

    def owned_recording(self, recording_id: str, speaker_id: str) -> Recording | None:
        """Return a live recording only when ``speaker_id`` owns it (ownership check)."""
        stmt = select(Recording).where(
            Recording.id == recording_id,
            Recording.speaker_id == speaker_id,
            Recording.deleted_at.is_(None),
        )
        return self.session.scalars(stmt).one_or_none()

    def soft_delete(self, recording_id: str, speaker_id: str) -> bool:
        """Soft-delete one of the speaker's recordings; ``False`` when there is nothing to do."""
        recording = self.owned_recording(recording_id, speaker_id)
        if recording is None:
            return False
        recording.deleted_at = utcnow()
        self.session.flush()
        return True


__all__ = ["PARTICIPANT_TOKEN_HEADER", "ParticipantRepository", "Progress"]
