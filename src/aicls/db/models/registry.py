"""Model registry, evaluation, telemetry, feedback and audit tables (§3.5).

``model_versions`` is the single source of truth for which OCR/ASR checkpoint is live
(``models/registry.yaml`` is only a seed); ``eval_runs`` stores the same JSON that
``reports/metrics/*.json`` carries so the admin pages and the MkDocs macros read one source;
``latency_samples`` is how the contractual "< 3 s to rendered result" metric is measured;
``feedback`` / ``sus_responses`` collect the İP-9 pilot data; ``audit_log`` records every
sensitive action (role change, consent withdrawal, book load, model activation, export, QC).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from aicls.db.base import (
    MYSQL_TABLE_KWARGS,
    Base,
    Digest,
    GitSha,
    TimestampMixin,
    UlidPk,
    UlidStr,
    enum_column,
    ulid_fk,
)
from aicls.db.enums import (
    EvalKind,
    FeedbackCategory,
    FeedbackStatus,
    Modality,
    ModelComponent,
)


class ModelVersion(Base, TimestampMixin):
    """A registered OCR/ASR checkpoint; exactly one row per component may be active."""

    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("component", "version", name="uq_model_versions_component_version"),
        Index("ix_model_versions_component_is_active", "component", "is_active"),
        MYSQL_TABLE_KWARGS,
    )

    id: Mapped[UlidPk]
    component: Mapped[ModelComponent] = mapped_column(enum_column(ModelComponent, "component"))
    version: Mapped[str] = mapped_column(String(40), nullable=False)
    path: Mapped[str] = mapped_column(String(400), default="")
    sha256: Mapped[Digest] = mapped_column(default="")
    config_hash: Mapped[Digest] = mapped_column(default="")
    git_sha: Mapped[GitSha] = mapped_column(default="")
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True, default=None)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)
    promoted_by: Mapped[str | None] = ulid_fk("users.id", ondelete="SET NULL", nullable=True)


class EvalRun(Base, TimestampMixin):
    """A persisted :class:`aicls.eval.EvalReport`; ``is_reportable`` marks citable numbers."""

    __tablename__ = "eval_runs"
    __table_args__ = (
        Index("ix_eval_runs_kind_created_at", "kind", "created_at"),
        MYSQL_TABLE_KWARGS,
    )

    id: Mapped[UlidPk]
    kind: Mapped[EvalKind] = mapped_column(enum_column(EvalKind, "kind"))
    name: Mapped[str] = mapped_column(String(120), default="")
    git_sha: Mapped[GitSha] = mapped_column(default="")
    dirty: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    config_hash: Mapped[str | None] = mapped_column(String(80), nullable=True, default=None)
    dataset_fingerprint: Mapped[Digest] = mapped_column(default="")
    dataset_version: Mapped[str] = mapped_column(String(40), default="")
    model_version: Mapped[str | None] = mapped_column(String(40), nullable=True, default=None)
    normalizer_versions_json: Mapped[dict[str, Any]] = mapped_column(default=dict)
    n: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(default=dict)
    ci95_json: Mapped[dict[str, Any]] = mapped_column(default=dict)
    report_path: Mapped[str] = mapped_column(String(400), default="")
    is_reportable: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    created_by: Mapped[str | None] = ulid_fk("users.id", ondelete="SET NULL", nullable=True)


class LatencySample(Base, TimestampMixin):
    """One measured request: server stage timings plus the browser's render time (< 3 s metric)."""

    __tablename__ = "latency_samples"
    __table_args__ = (
        Index("ix_latency_samples_modality_created_at", "modality", "created_at"),
        MYSQL_TABLE_KWARGS,
    )

    id: Mapped[UlidPk]
    operation_id: Mapped[str | None] = ulid_fk("operations.id", ondelete="SET NULL", nullable=True)
    request_id: Mapped[UlidStr]
    modality: Mapped[Modality] = mapped_column(enum_column(Modality, "modality"))
    endpoint: Mapped[str] = mapped_column(String(80), default="")
    server_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    stage_json: Mapped[dict[str, Any]] = mapped_column(default=dict)
    client_total_ms: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    render_ms: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    model_versions_json: Mapped[dict[str, Any]] = mapped_column(default=dict)


class Feedback(Base, TimestampMixin):
    """A free-text pilot-study feedback entry with its triage state (İP-9)."""

    __tablename__ = "feedback"
    __table_args__ = (
        Index("ix_feedback_status_category", "status", "category"),
        MYSQL_TABLE_KWARGS,
    )

    id: Mapped[UlidPk]
    user_id: Mapped[str | None] = ulid_fk("users.id", ondelete="SET NULL", nullable=True)
    category: Mapped[FeedbackCategory] = mapped_column(enum_column(FeedbackCategory, "category"))
    text: Mapped[str] = mapped_column(Text, nullable=False)
    page: Mapped[str] = mapped_column(String(120), default="")
    screenshot_asset_id: Mapped[str | None] = ulid_fk(
        "assets.id", ondelete="SET NULL", nullable=True
    )
    status: Mapped[FeedbackStatus] = mapped_column(
        enum_column(FeedbackStatus, "status"), default=FeedbackStatus.OPEN
    )
    collected_week: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)


class SusResponse(Base, TimestampMixin):
    """A System Usability Scale questionnaire (10 answers 1–5) and its computed score."""

    __tablename__ = "sus_responses"
    __table_args__ = (MYSQL_TABLE_KWARGS,)

    id: Mapped[UlidPk]
    user_id: Mapped[str | None] = ulid_fk("users.id", ondelete="SET NULL", nullable=True)
    cohort: Mapped[str] = mapped_column(String(40), default="", index=True)
    answers_json: Mapped[dict[str, Any]] = mapped_column(default=dict)
    score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    collected_week: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)


class AuditLog(Base, TimestampMixin):
    """An append-only record of a sensitive action (docs/guvenlik-gizlilik.md)."""

    __tablename__ = "audit_log"
    __table_args__ = (
        Index("ix_audit_log_action_created_at", "action", "created_at"),
        MYSQL_TABLE_KWARGS,
    )

    id: Mapped[UlidPk]
    actor_id: Mapped[str | None] = ulid_fk("users.id", ondelete="SET NULL", nullable=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    target_table: Mapped[str] = mapped_column(String(40), default="")
    target_id: Mapped[str] = mapped_column(String(64), default="")
    detail_json: Mapped[dict[str, Any]] = mapped_column(default=dict)
    ip_hash: Mapped[str] = mapped_column(String(80), default="")


__all__ = [
    "AuditLog",
    "EvalRun",
    "Feedback",
    "LatencySample",
    "ModelVersion",
    "SusResponse",
]
