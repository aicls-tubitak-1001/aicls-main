"""Administration and health payloads (api.md §3.1 and §3.11)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from aicls.api.schemas.common import ApiModel, RequestModel, RoleName


class LiveOut(ApiModel):
    """``GET /health/live`` response."""

    status: str = "ok"


class SandboxOut(ApiModel):
    """Sandbox pool health (ADR-0016)."""

    workers: int = 0
    restarts: int = 0
    started: bool = False


class ReadyOut(ApiModel):
    """``GET /health/ready`` response; 503 when anything is not ready."""

    status: str
    db: str = "ok"
    models: dict[str, str] = Field(default_factory=dict)
    sandbox: SandboxOut = SandboxOut()


class VersionOut(ApiModel):
    """``GET /health/version`` response."""

    version: str
    git_sha: str = ""
    profile: str = "full"
    models: dict[str, str | None] = Field(default_factory=dict)


class ModelVersionOut(ApiModel):
    """One row of ``model_versions`` (``GET /admin/models``)."""

    component: str
    version: str
    path: str = ""
    sha256: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    is_active: bool = False
    promoted_at: str | None = None


class ActivateIn(RequestModel):
    """``POST /admin/models/activate`` body."""

    component: Literal["ocr", "asr"]
    version: str = Field(min_length=1, max_length=40)


class EvalRunOut(ApiModel):
    """One persisted evaluation report (``GET /admin/eval-runs``)."""

    id: str
    kind: str
    name: str = ""
    n: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)
    dataset_fingerprint: str = ""
    model_version: str | None = None
    is_reportable: bool = False
    created_at: str = ""


class LatencyStatsOut(ApiModel):
    """``GET /admin/metrics/latency`` response (the contractual "< 3 s" figure)."""

    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    n: int = 0
    by_modality: dict[str, float] = Field(default_factory=dict)


class QcQueueItemOut(ApiModel):
    """One recording awaiting human review (``GET /admin/qc/queue``)."""

    id: str
    speaker_id: str
    prompt_id: str
    duration_s: float = 0.0
    qc_passed: bool = False
    review_status: str = "pending"
    transcript_final: str = ""


class QcReviewIn(RequestModel):
    """``POST /admin/qc/{recording_id}/review`` body."""

    verdict: Literal["accepted", "rejected", "fixed"]
    transcript_fix: str | None = Field(default=None, max_length=500)


class RoleIn(RequestModel):
    """``PATCH /admin/users/{id}/role`` body."""

    role: RoleName


class AuditOut(ApiModel):
    """One audit-log row."""

    id: str
    actor_id: str | None = None
    action: str
    target_table: str = ""
    target_id: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""


class ExportJobOut(ApiModel):
    """``POST /admin/contributions/export`` response."""

    job_id: str
    rows: int = 0
    speakers: int = 0
    hours: float = 0.0
    path: str = ""
    fingerprint: str = ""


__all__ = [
    "ActivateIn",
    "AuditOut",
    "EvalRunOut",
    "ExportJobOut",
    "LatencyStatsOut",
    "LiveOut",
    "ModelVersionOut",
    "QcQueueItemOut",
    "QcReviewIn",
    "ReadyOut",
    "RoleIn",
    "SandboxOut",
    "VersionOut",
]
