"""Voice data-collection repository (docs/mimari/veri-modeli.md §3.3, KVKK).

``RecordingRepository`` is the single query surface :mod:`aicls.collect` uses: it registers
consent versions and pseudonymous speakers, balances the prompt pool towards the least-recorded
sentences, stores an accepted recording together with its audio asset, aggregates the
contribution dashboard and performs a KVKK withdrawal.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

from sqlalchemy import func, select

from aicls.core.errors import NotFoundError
from aicls.core.messages_tr import msg
from aicls.db.base import utcnow
from aicls.db.enums import (
    AgeBand,
    AssetKind,
    Gender,
    ReviewStatus,
    Split,
    University,
    coerce_enum,
)
from aicls.db.models import Asset, ConsentVersion, PromptSentence, Recording, Speaker
from aicls.db.repositories.base import Repository, clamp_limit, field_of
from aicls.db.results import ContribStats

#: Prefix and width of a speaker pseudonym (``P00001``).
PSEUDONYM_PREFIX = "P"
PSEUDONYM_DIGITS = 5
_SECONDS_PER_HOUR = 3600.0


class RecordingRepository(Repository):
    """Queries over ``consent_versions``, ``speakers``, ``prompt_sentences`` and ``recordings``."""

    # -- consent ------------------------------------------------------------------------------
    def upsert_consent_version(
        self,
        version: str,
        text_tr: str,
        *,
        sha256: str = "",
        effective_from: date | None = None,
        ethics_approval_ref: str = "",
    ) -> ConsentVersion:
        """Insert the consent text version if it is new, else return the stored row.

        Consent texts are immutable: an existing row is never rewritten, so an acceptance always
        points at exactly the words the participant saw.
        """
        stmt = select(ConsentVersion).where(ConsentVersion.version == version)
        existing = self.session.scalars(stmt).one_or_none()
        if existing is not None:
            return existing
        row = ConsentVersion(
            version=version,
            text_tr=text_tr,
            sha256=sha256,
            effective_from=effective_from or utcnow().date(),
            ethics_approval_ref=ethics_approval_ref,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def get_consent_version(self, version: str) -> ConsentVersion | None:
        """Return a stored consent version or ``None``."""
        stmt = select(ConsentVersion).where(ConsentVersion.version == version)
        return self.session.scalars(stmt).one_or_none()

    # -- speakers -----------------------------------------------------------------------------
    def next_pseudonym(self) -> str:
        """Return the next free ``P00001``-style pseudonym."""
        highest = self.session.scalars(
            select(Speaker.pseudonym).order_by(Speaker.pseudonym.desc()).limit(1)
        ).one_or_none()
        number = int(highest[1:]) + 1 if highest else 1
        return f"{PSEUDONYM_PREFIX}{number:0{PSEUDONYM_DIGITS}d}"

    def create_speaker(
        self,
        consent_version_id: str,
        *,
        user_id: str | None = None,
        age_band: AgeBand | str = AgeBand.B25_34,
        gender: Gender | str = Gender.NA,
        university: University | str = University.OTHER,
        accent_region: str | None = None,
        ip_hash: str | None = None,
        user_agent_hash: str | None = None,
        token_hash: str | None = None,
        consented_at: datetime | None = None,
    ) -> Speaker:
        """Register a pseudonymous voice contributor who has accepted a consent version."""
        speaker = Speaker(
            user_id=user_id,
            pseudonym=self.next_pseudonym(),
            consent_version_id=consent_version_id,
            consented_at=consented_at or utcnow(),
            ip_hash=ip_hash,
            user_agent_hash=user_agent_hash,
            age_band=coerce_enum(AgeBand, age_band),
            gender=coerce_enum(Gender, gender),
            university=coerce_enum(University, university),
            accent_region=accent_region,
            token_hash=token_hash,
        )
        self.session.add(speaker)
        self.session.flush()
        return speaker

    def get_speaker(self, speaker_id: str) -> Speaker | None:
        """Return a speaker by id or by pseudonym (``P00012``), else ``None``."""
        if speaker_id.startswith(PSEUDONYM_PREFIX) and len(speaker_id) == PSEUDONYM_DIGITS + 1:
            stmt = select(Speaker).where(Speaker.pseudonym == speaker_id)
            return self.session.scalars(stmt).one_or_none()
        return self.session.get(Speaker, speaker_id)

    def speaker_for_user(self, user_id: str) -> Speaker | None:
        """Return the speaker linked to a signed-in user, if any."""
        stmt = select(Speaker).where(Speaker.user_id == user_id)
        return self.session.scalars(stmt).one_or_none()

    # -- prompts ------------------------------------------------------------------------------
    def add_prompts(self, prompts: Sequence[Any]) -> int:
        """Insert prompt sentences from mappings or objects; return how many were added.

        Recognised fields: ``text_tr``, ``text_normalized``, ``scenario_tag``, ``intent``,
        ``difficulty``, ``is_active``, ``target_count``. Sentences whose ``text_tr`` already
        exists are skipped, so seeding is idempotent.
        """
        existing = set(self.session.scalars(select(PromptSentence.text_tr)))
        added = 0
        for prompt in prompts:
            text_tr = str(field_of(prompt, "text_tr", "text", default=""))
            if not text_tr or text_tr in existing:
                continue
            self.session.add(
                PromptSentence(
                    text_tr=text_tr,
                    text_normalized=str(field_of(prompt, "text_normalized", default=text_tr)),
                    scenario_tag=str(field_of(prompt, "scenario_tag", default="genel")),
                    intent=field_of(prompt, "intent"),
                    difficulty=int(field_of(prompt, "difficulty", default=1)),
                    is_active=bool(field_of(prompt, "is_active", default=True)),
                    target_count=int(field_of(prompt, "target_count", default=0)),
                )
            )
            existing.add(text_tr)
            added += 1
        self.session.flush()
        return added

    def next_prompts(self, speaker_id: str, n: int = 5) -> list[PromptSentence]:
        """Return the ``n`` least-recorded active prompts this speaker has not read yet.

        Balancing keeps the collected corpus even across reading scenarios: prompts are ordered
        by how many recordings exist for them, then by id so the result is deterministic.
        """
        already = select(Recording.prompt_id).where(Recording.speaker_id == speaker_id)
        counts = (
            select(Recording.prompt_id, func.count(Recording.id).label("n"))
            .group_by(Recording.prompt_id)
            .subquery()
        )
        stmt = (
            select(PromptSentence)
            .outerjoin(counts, counts.c.prompt_id == PromptSentence.id)
            .where(PromptSentence.is_active.is_(True), PromptSentence.id.not_in(already))
            .order_by(func.coalesce(counts.c.n, 0), PromptSentence.id)
            .limit(clamp_limit(n))
        )
        return list(self.session.scalars(stmt))

    # -- recordings ---------------------------------------------------------------------------
    def add(
        self,
        speaker_id: str,
        prompt_id: str,
        path: str,
        qc: Mapping[str, Any],
        duration_s: float,
        sha256: str,
        *,
        sample_rate: int = 16000,
        size_bytes: int = 0,
        transcript_final: str = "",
    ) -> Recording:
        """Store an accepted recording and its audio asset.

        Args:
            speaker_id: Owning speaker (ULID).
            prompt_id: Prompt sentence that was read.
            path: WAV path relative to ``AICLS_DATA_ROOT`` (``audio/raw/...``).
            qc: Quality-control verdict; ``passed``, ``snr_db``, ``clipping_ratio`` and
                ``silence_ratio`` are lifted into columns, the whole document is kept in
                ``qc_json``.
            duration_s: Utterance length in seconds.
            sha256: Digest of the WAV bytes.
            sample_rate: Sampling rate of the stored file.
            size_bytes: File size, stored on the asset row.
            transcript_final: Reference transcript (defaults to the prompt text at export time).

        Returns:
            The inserted :class:`~aicls.db.models.Recording`.
        """
        asset = Asset(
            kind=AssetKind.AUDIO_WAV,
            path=path,
            sha256=sha256,
            bytes=size_bytes,
            mime="audio/wav",
        )
        self.session.add(asset)
        self.session.flush()
        recording = Recording(
            speaker_id=speaker_id,
            prompt_id=prompt_id,
            asset_id=asset.id,
            duration_s=float(duration_s),
            sample_rate=sample_rate,
            sha256=sha256,
            qc_passed=bool(qc.get("passed", qc.get("qc_passed", False))),
            qc_json=dict(qc),
            snr_db=float(qc.get("snr_db", 0.0) or 0.0),
            clipping_ratio=float(qc.get("clipping_ratio", 0.0) or 0.0),
            silence_ratio=float(qc.get("silence_ratio", 0.0) or 0.0),
            transcript_final=transcript_final,
        )
        self.session.add(recording)
        self.session.flush()
        return recording

    def set_review(
        self,
        recording_id: str,
        verdict: ReviewStatus | str,
        *,
        reviewer_id: str | None = None,
        transcript_fix: str | None = None,
    ) -> Recording:
        """Apply a human QC decision (``POST /admin/qc/{recording_id}/review``).

        Raises:
            NotFoundError: If no such recording exists.
        """
        recording = self.session.get(Recording, recording_id)
        if recording is None:
            raise NotFoundError(
                msg("NOT_FOUND"), details={"table": "recordings", "id": recording_id}
            )
        recording.review_status = coerce_enum(ReviewStatus, verdict)
        recording.reviewed_by = reviewer_id
        if transcript_fix is not None:
            recording.transcript_final = transcript_fix
        self.session.flush()
        return recording

    def set_split(self, recording_id: str, split: Split | str) -> None:
        """Assign a recording to a dataset split (speaker-disjoint splits, ADR-0008).

        Raises:
            NotFoundError: If no such recording exists.
        """
        recording = self.session.get(Recording, recording_id)
        if recording is None:
            raise NotFoundError(
                msg("NOT_FOUND"), details={"table": "recordings", "id": recording_id}
            )
        recording.split = coerce_enum(Split, split)
        self.session.flush()

    def qc_queue(
        self, status: ReviewStatus | str = ReviewStatus.PENDING, *, limit: int | None = None
    ) -> list[Recording]:
        """Return recordings awaiting the given review state, oldest first."""
        stmt = (
            select(Recording)
            .where(
                Recording.review_status == coerce_enum(ReviewStatus, status),
                Recording.deleted_at.is_(None),
            )
            .order_by(Recording.id)
            .limit(clamp_limit(limit))
        )
        return list(self.session.scalars(stmt))

    def exportable(self) -> list[tuple[Recording, Speaker, PromptSentence]]:
        """Return ``(recording, speaker, prompt)`` triples eligible for a training manifest.

        Eligible means: QC passed, not soft-deleted and the speaker has not withdrawn consent —
        the rule behind ``test_export_manifest_excludes_withdrawn_and_qc_failed``.
        """
        stmt = (
            select(Recording, Speaker, PromptSentence)
            .join(Speaker, Speaker.id == Recording.speaker_id)
            .join(PromptSentence, PromptSentence.id == Recording.prompt_id)
            .where(
                Recording.qc_passed.is_(True),
                Recording.deleted_at.is_(None),
                Speaker.withdrawn_at.is_(None),
            )
            .order_by(Recording.id)
        )
        return [(row[0], row[1], row[2]) for row in self.session.execute(stmt)]

    def mark_withdrawn(self, speaker_id: str) -> int:
        """Withdraw a speaker's consent (KVKK): soft-delete every recording, keep the audit row.

        The ``speakers`` row itself is kept (with ``withdrawn_at`` set and the identifying hashes
        cleared) because the consent acceptance is an audit record; the audio files are removed
        from disk by :func:`aicls.collect.withdraw`.

        Args:
            speaker_id: Speaker ULID or pseudonym.

        Returns:
            Number of recordings marked as deleted.

        Raises:
            NotFoundError: If no such speaker exists.
        """
        speaker = self.get_speaker(speaker_id)
        if speaker is None:
            raise NotFoundError(msg("NOT_FOUND"), details={"table": "speakers", "id": speaker_id})
        now = utcnow()
        speaker.withdrawn_at = now
        speaker.ip_hash = None
        speaker.user_agent_hash = None
        stmt = select(Recording).where(
            Recording.speaker_id == speaker.id, Recording.deleted_at.is_(None)
        )
        recordings = list(self.session.scalars(stmt))
        for recording in recordings:
            recording.deleted_at = now
        self.session.flush()
        return len(recordings)

    def stats(self) -> ContribStats:
        """Aggregate the contribution dashboard (``GET /contribute/stats``)."""
        live = (Recording.deleted_at.is_(None),)
        accepted = (*live, Recording.qc_passed.is_(True), Speaker.withdrawn_at.is_(None))
        totals = self.session.execute(
            select(func.count(Recording.id), func.coalesce(func.sum(Recording.duration_s), 0.0))
            .select_from(Recording)
            .join(Speaker, Speaker.id == Recording.speaker_id)
            .where(*accepted)
        ).one()
        utterances = int(totals[0] or 0)
        hours = float(totals[1] or 0.0) / _SECONDS_PER_HOUR
        speakers = int(
            self.session.scalar(
                select(func.count(func.distinct(Recording.speaker_id)))
                .select_from(Recording)
                .join(Speaker, Speaker.id == Recording.speaker_id)
                .where(*accepted)
            )
            or 0
        )
        total_live = int(self.session.scalar(select(func.count(Recording.id)).where(*live)) or 0)
        passed_live = int(
            self.session.scalar(
                select(func.count(Recording.id)).where(*live, Recording.qc_passed.is_(True))
            )
            or 0
        )
        return ContribStats(
            hours_accepted=hours,
            utterances=utterances,
            speakers=speakers,
            by_university=self._group_counts(Speaker.university, accepted),
            by_gender=self._group_counts(Speaker.gender, accepted),
            by_age_band=self._group_counts(Speaker.age_band, accepted),
            qc_pass_rate=(passed_live / total_live) if total_live else 0.0,
        )

    def _group_counts(self, column: Any, conditions: tuple[Any, ...]) -> dict[str, int]:
        """Count accepted recordings grouped by a speaker attribute."""
        stmt = (
            select(column, func.count(Recording.id))
            .select_from(Recording)
            .join(Speaker, Speaker.id == Recording.speaker_id)
            .where(*conditions)
            .group_by(column)
        )
        return {
            str(getattr(key, "value", key)): int(count) for key, count in self.session.execute(stmt)
        }


__all__ = ["PSEUDONYM_DIGITS", "PSEUDONYM_PREFIX", "RecordingRepository"]
