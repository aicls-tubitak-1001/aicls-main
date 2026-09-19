"""Voice data-collection tables (docs/mimari/veri-modeli.md §3.3, KVKK).

"Model Gelişimine Katkıda Bulun" stores a versioned consent acceptance per speaker, a balanced
pool of prompt sentences, and one row per uploaded recording with its automatic quality-control
verdict. Speakers are identified by a pseudonym (``P00001``); the optional ``user_id`` link exists
only for progress tracking and never leaves the database in a training manifest.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    CHAR,
    Boolean,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from aicls.db.base import (
    MYSQL_TABLE_KWARGS,
    Base,
    Digest,
    TimestampMixin,
    UlidPk,
    enum_column,
    ulid_fk,
)
from aicls.db.enums import AgeBand, Gender, ReviewStatus, Split, University


class ConsentVersion(Base, TimestampMixin):
    """An immutable, hashed version of the Turkish consent text (KVKK audit trail)."""

    __tablename__ = "consent_versions"
    __table_args__ = (MYSQL_TABLE_KWARGS,)

    id: Mapped[UlidPk]
    version: Mapped[str] = mapped_column(String(16), unique=True, nullable=False)
    text_tr: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[Digest] = mapped_column(default="")
    effective_from: Mapped[date] = mapped_column(Date, nullable=False)
    ethics_approval_ref: Mapped[str] = mapped_column(String(80), default="")


class Speaker(Base, TimestampMixin):
    """A voice contributor: pseudonym, accepted consent version and demographic bands only."""

    __tablename__ = "speakers"
    __table_args__ = (
        Index("ix_speakers_university_gender", "university", "gender"),
        MYSQL_TABLE_KWARGS,
    )

    id: Mapped[UlidPk]
    user_id: Mapped[str | None] = ulid_fk(
        "users.id", ondelete="SET NULL", nullable=True, unique=True
    )
    pseudonym: Mapped[str] = mapped_column(CHAR(6), unique=True, nullable=False)
    consent_version_id: Mapped[str] = ulid_fk("consent_versions.id", ondelete="RESTRICT")
    consented_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ip_hash: Mapped[str | None] = mapped_column(String(80), nullable=True, default=None)
    user_agent_hash: Mapped[str | None] = mapped_column(String(80), nullable=True, default=None)
    age_band: Mapped[AgeBand] = mapped_column(enum_column(AgeBand, "age_band"))
    gender: Mapped[Gender] = mapped_column(enum_column(Gender, "gender"))
    university: Mapped[University] = mapped_column(enum_column(University, "university"))
    accent_region: Mapped[str | None] = mapped_column(String(40), nullable=True, default=None)
    token_hash: Mapped[str | None] = mapped_column(String(80), nullable=True, default=None)
    withdrawn_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)


class PromptSentence(Base, TimestampMixin):
    """A Turkish sentence the contributor is asked to read (scenario-tagged, İP-3)."""

    __tablename__ = "prompt_sentences"
    __table_args__ = (
        Index("ix_prompt_sentences_scenario_tag_is_active", "scenario_tag", "is_active"),
        MYSQL_TABLE_KWARGS,
    )

    id: Mapped[UlidPk]
    text_tr: Mapped[str] = mapped_column(String(500), nullable=False)
    text_normalized: Mapped[str] = mapped_column(String(500), default="")
    scenario_tag: Mapped[str] = mapped_column(String(40), default="genel")
    intent: Mapped[str | None] = mapped_column(String(40), nullable=True, default=None)
    difficulty: Mapped[int] = mapped_column(SmallInteger, default=1, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    target_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class Recording(Base, TimestampMixin):
    """One uploaded utterance with its QC verdict, review state and dataset split."""

    __tablename__ = "recordings"
    __table_args__ = (
        Index("ix_recordings_prompt_id_qc_passed", "prompt_id", "qc_passed"),
        MYSQL_TABLE_KWARGS,
    )

    id: Mapped[UlidPk]
    speaker_id: Mapped[str] = ulid_fk("speakers.id", ondelete="CASCADE", index=True)
    prompt_id: Mapped[str] = ulid_fk("prompt_sentences.id", ondelete="RESTRICT")
    asset_id: Mapped[str] = ulid_fk("assets.id", ondelete="CASCADE")
    duration_s: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    sample_rate: Mapped[int] = mapped_column(Integer, default=16000, nullable=False)
    sha256: Mapped[Digest] = mapped_column(default="")
    qc_passed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    qc_json: Mapped[dict[str, Any]] = mapped_column(default=dict)
    snr_db: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    clipping_ratio: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    silence_ratio: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    review_status: Mapped[ReviewStatus] = mapped_column(
        enum_column(ReviewStatus, "review_status"), default=ReviewStatus.PENDING, index=True
    )
    reviewed_by: Mapped[str | None] = ulid_fk("users.id", ondelete="SET NULL", nullable=True)
    transcript_final: Mapped[str] = mapped_column(String(500), default="")
    split: Mapped[Split] = mapped_column(
        enum_column(Split, "split"), default=Split.UNASSIGNED, index=True
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)


__all__ = ["ConsentVersion", "PromptSentence", "Recording", "Speaker"]
