"""Operation, asset and model-output tables (docs/mimari/veri-modeli.md §3.1–§3.2).

``operations`` is the audit trail of the contractual pipeline (input → command → SymPy result);
``assets`` addresses every binary artefact by a path relative to ``AICLS_DATA_ROOT``;
``recognitions`` / ``transcripts`` keep OCR and ASR outputs together with the user corrections
that feed İP-3 model improvement.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text
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
from aicls.db.enums import AssetKind, InputKind, NluSource, ResultKind


class Asset(Base, TimestampMixin):
    """A stored file (plot, ink, audio, book figure) addressed relative to ``AICLS_DATA_ROOT``."""

    __tablename__ = "assets"
    __table_args__ = (MYSQL_TABLE_KWARGS,)

    id: Mapped[UlidPk]
    kind: Mapped[AssetKind] = mapped_column(enum_column(AssetKind, "kind"), index=True)
    path: Mapped[str] = mapped_column(String(400), nullable=False)
    sha256: Mapped[Digest] = mapped_column(default="")
    bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    mime: Mapped[str] = mapped_column(String(64), default="application/octet-stream")
    owner_id: Mapped[str | None] = ulid_fk("users.id", ondelete="SET NULL", nullable=True)
    retention_until: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, default=None, index=True
    )


class Operation(Base, TimestampMixin):
    """One end-to-end request: input, recognised command, SymPy result and stage timings."""

    __tablename__ = "operations"
    __table_args__ = (
        Index("ix_operations_workspace_id_created_at", "workspace_id", "created_at"),
        MYSQL_TABLE_KWARGS,
    )

    id: Mapped[UlidPk]
    workspace_id: Mapped[str] = ulid_fk("workspaces.id", ondelete="CASCADE")
    user_id: Mapped[str] = ulid_fk("users.id", ondelete="CASCADE")
    input_kind: Mapped[InputKind] = mapped_column(enum_column(InputKind, "input_kind"))
    source_text: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    input_latex: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    command_json: Mapped[dict[str, Any]] = mapped_column(default=dict)
    result_kind: Mapped[ResultKind | None] = mapped_column(
        enum_column(ResultKind, "result_kind"), nullable=True, default=None
    )
    result_latex: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    result_srepr: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    steps_json: Mapped[dict[str, Any] | None] = mapped_column(nullable=True, default=None)
    plot_asset_id: Mapped[str | None] = ulid_fk("assets.id", ondelete="SET NULL", nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(32), nullable=True, default=None)
    timings_json: Mapped[dict[str, Any]] = mapped_column(default=dict)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    model_versions_json: Mapped[dict[str, Any]] = mapped_column(default=dict)
    nlu_source: Mapped[NluSource | None] = mapped_column(
        enum_column(NluSource, "nlu_source"), nullable=True, default=None
    )


class Recognition(Base, TimestampMixin):
    """A handwriting-recognition output plus the user's correction, if any (İP-3 feedback)."""

    __tablename__ = "recognitions"
    __table_args__ = (MYSQL_TABLE_KWARGS,)

    id: Mapped[UlidPk]
    user_id: Mapped[str | None] = ulid_fk("users.id", ondelete="SET NULL", nullable=True)
    operation_id: Mapped[str | None] = ulid_fk("operations.id", ondelete="SET NULL", nullable=True)
    ink_asset_id: Mapped[str | None] = ulid_fk("assets.id", ondelete="SET NULL", nullable=True)
    image_asset_id: Mapped[str | None] = ulid_fk("assets.id", ondelete="SET NULL", nullable=True)
    latex: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    alternatives_json: Mapped[dict[str, Any]] = mapped_column(default=dict)
    model_version: Mapped[str] = mapped_column(String(40), index=True, default="")
    elapsed_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    corrected_latex: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    corrected_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, default=None, index=True
    )


class Transcript(Base, TimestampMixin):
    """A speech-recognition output plus the user's correction, if any (İP-3 feedback)."""

    __tablename__ = "transcripts"
    __table_args__ = (MYSQL_TABLE_KWARGS,)

    id: Mapped[UlidPk]
    user_id: Mapped[str | None] = ulid_fk("users.id", ondelete="SET NULL", nullable=True)
    operation_id: Mapped[str | None] = ulid_fk("operations.id", ondelete="SET NULL", nullable=True)
    audio_asset_id: Mapped[str | None] = ulid_fk("assets.id", ondelete="SET NULL", nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    model_version: Mapped[str] = mapped_column(String(40), index=True, default="")
    elapsed_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    corrected_text: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)


__all__ = ["Asset", "Operation", "Recognition", "Transcript"]
