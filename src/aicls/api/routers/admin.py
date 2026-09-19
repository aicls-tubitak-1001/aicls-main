"""Administration endpoints (api.md §3.11).

Two routers are exported because ADR-0018 mounts different subsets per profile:

* :data:`router` — the full administration surface (``full`` profile);
* :data:`qc_router` — the quality-control subset the collection VM needs (``collect`` profile):
  the review queue, the contribution dashboard, feedback triage and the audit log. The VM never
  gets the model registry or the evaluation history, because it has neither.

Every state-changing action writes an ``audit_log`` row (docs/guvenlik-gizlilik.md §1).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query, Request, status

from aicls.api.deps import AdminUser, AppSettings, DbSession, Registry, StaffUser
from aicls.api.schemas import (
    ActivateIn,
    AuditOut,
    ContribStatsOut,
    EvalRunOut,
    ExportJobOut,
    FeedbackOut,
    FeedbackPatch,
    LatencyStatsOut,
    LoadReportOut,
    ModelVersionOut,
    Page,
    QcQueueItemOut,
    QcReviewIn,
    RoleIn,
    UserOut,
)
from aicls.core import messages_tr
from aicls.core.errors import NotFoundError
from aicls.core.ids import ulid
from aicls.core.logging import get_logger
from aicls.db.models import EvalRun, ModelVersion, Recording, User
from aicls.db.repositories.audio import RecordingRepository
from aicls.db.repositories.identity import UserRepository
from aicls.db.repositories.registry import (
    AuditRepository,
    EvalRunRepository,
    FeedbackRepository,
    LatencyRepository,
    ModelVersionRepository,
)

router = APIRouter(prefix="/admin", tags=["admin"])
qc_router = APIRouter(prefix="/admin", tags=["admin"])

#: File name of the manifest ``POST /admin/contributions/export`` writes.
MANIFEST_NAME = "audio_commands.jsonl"

_logger = get_logger(__name__)


def _model_out(row: ModelVersion) -> ModelVersionOut:
    """Convert a ``model_versions`` row into its API shape."""
    return ModelVersionOut(
        component=row.component.value,
        version=row.version,
        path=row.path,
        sha256=row.sha256,
        metrics=dict(row.metrics_json or {}),
        run_id=row.run_id,
        is_active=row.is_active,
        promoted_at=row.promoted_at.isoformat() if row.promoted_at else None,
    )


def _eval_out(row: EvalRun) -> EvalRunOut:
    """Convert an ``eval_runs`` row into its API shape."""
    return EvalRunOut(
        id=row.id,
        kind=row.kind.value,
        name=row.name,
        n=row.n,
        metrics=dict(row.metrics_json or {}),
        dataset_fingerprint=row.dataset_fingerprint,
        model_version=row.model_version,
        is_reportable=row.is_reportable,
        created_at=row.created_at.isoformat(),
    )


def _queue_out(row: Recording) -> QcQueueItemOut:
    """Convert a ``recordings`` row into a review-queue item."""
    return QcQueueItemOut(
        id=row.id,
        speaker_id=row.speaker_id,
        prompt_id=row.prompt_id,
        duration_s=round(row.duration_s, 3),
        qc_passed=row.qc_passed,
        review_status=row.review_status.value,
        transcript_final=row.transcript_final,
    )


# ---------------------------------------------------------------------------------------------
# model registry and evaluation (full profile only)
# ---------------------------------------------------------------------------------------------
@router.get("/models", response_model=list[ModelVersionOut])
def list_models(_user: AdminUser, session: DbSession) -> list[ModelVersionOut]:
    """Every registered checkpoint, newest first."""
    return [_model_out(row) for row in ModelVersionRepository(session).list_versions()]


@router.post("/models/activate", status_code=status.HTTP_204_NO_CONTENT)
def activate_model(
    body: ActivateIn, user: AdminUser, session: DbSession, registry: Registry
) -> None:
    """Promote one checkpoint and reload the registry (audited).

    Raises:
        NotFoundError: If the version is not registered.
    """
    ModelVersionRepository(session).activate(body.component, body.version, promoted_by=user.id)
    AuditRepository(session).log(
        user.id,
        "model_activate",
        ("model_versions", body.version),
        {"component": body.component},
    )
    registry.reload()
    _logger.info("api.admin.model_activated", component=body.component, version=body.version)


@router.get("/eval-runs", response_model=list[EvalRunOut])
def list_eval_runs(
    _user: AdminUser,
    session: DbSession,
    kind: Annotated[str | None, Query(max_length=16)] = None,
    since: Annotated[datetime | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[EvalRunOut]:
    """Persisted evaluation reports, newest first."""
    return [_eval_out(row) for row in EvalRunRepository(session).list(kind, since, limit=limit)]


@router.get("/eval-runs/{run_id}", response_model=EvalRunOut)
def get_eval_run(run_id: str, _user: AdminUser, session: DbSession) -> EvalRunOut:
    """One evaluation report."""
    row = session.get(EvalRun, run_id)
    if row is None:
        raise NotFoundError(messages_tr.MSG_NOT_FOUND, details={"table": "eval_runs", "id": run_id})
    return _eval_out(row)


@router.get("/metrics/latency", response_model=LatencyStatsOut)
def latency_metrics(
    _user: StaffUser,
    session: DbSession,
    since: Annotated[datetime | None, Query()] = None,
    modality: Annotated[str | None, Query(max_length=16)] = None,
) -> LatencyStatsOut:
    """Mean/p50/p95 of the measured end-to-end time (the contractual "< 3 s" figure)."""
    stats = LatencyRepository(session).stats(since, modality)
    return LatencyStatsOut(
        mean_ms=round(stats.mean_ms, 3),
        p50_ms=round(stats.p50_ms, 3),
        p95_ms=round(stats.p95_ms, 3),
        n=stats.n,
        by_modality={key: round(value, 3) for key, value in stats.by_modality.items()},
    )


@router.post("/book/load", response_model=LoadReportOut)
async def load_book(request: Request, user: AdminUser, session: DbSession) -> LoadReportOut:
    """Load a zipped Markdown book directory (422 when tagging is not 100 %, İP-8)."""
    from aicls.api.bookload import load_book_archive

    report = await load_book_archive(request, session)
    AuditRepository(session).log(user.id, "book_load", ("book_units", ""), {"units": report.units})
    return LoadReportOut(
        units=report.units,
        triples=report.triples,
        aliases=report.aliases,
        tagged_ratio=report.tagged_ratio,
    )


@router.post("/contributions/export", response_model=ExportJobOut)
def export_contributions(
    user: AdminUser, session: DbSession, settings: AppSettings
) -> ExportJobOut:
    """Write the speaker-disjoint ASR training manifest and return its summary."""
    from aicls.collect import export_manifest

    out = settings.data_root / "manifests" / MANIFEST_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    report = export_manifest(RecordingRepository(session), out=out, data_root=settings.data_root)
    job_id = ulid()
    AuditRepository(session).log(
        user.id, "contributions_export", ("recordings", job_id), {"rows": report.rows}
    )
    return ExportJobOut(
        job_id=job_id,
        rows=report.rows,
        speakers=report.speakers,
        hours=round(report.hours, 4),
        path=report.path.as_posix(),
        fingerprint=report.fingerprint,
    )


@router.get("/users", response_model=Page[UserOut])
def list_users(
    _user: AdminUser,
    session: DbSession,
    cursor: Annotated[str | None, Query(max_length=26)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> Page[UserOut]:
    """One page of accounts, newest first."""
    page = UserRepository(session).page(cursor=cursor, limit=limit)
    return Page[UserOut](
        items=[UserOut.model_validate(row, from_attributes=True) for row in page.items],
        next_cursor=page.next_cursor,
    )


@router.patch("/users/{user_id}/role", status_code=status.HTTP_204_NO_CONTENT)
def set_user_role(user_id: str, body: RoleIn, user: AdminUser, session: DbSession) -> None:
    """Change a user's role (audited)."""
    UserRepository(session).set_role(user_id, body.role)
    AuditRepository(session).log(user.id, "role_change", ("users", user_id), {"role": body.role})


# ---------------------------------------------------------------------------------------------
# quality-control subset (mounted on both profiles)
# ---------------------------------------------------------------------------------------------
@qc_router.get("/qc/queue", response_model=list[QcQueueItemOut])
def qc_queue(
    _user: StaffUser,
    session: DbSession,
    review_status: Annotated[str, Query(alias="status", max_length=16)] = "pending",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[QcQueueItemOut]:
    """Recordings awaiting a human decision, oldest first."""
    rows = RecordingRepository(session).qc_queue(review_status, limit=limit)
    return [_queue_out(row) for row in rows]


@qc_router.post("/qc/{recording_id}/review", status_code=status.HTTP_204_NO_CONTENT)
def qc_review(recording_id: str, body: QcReviewIn, user: StaffUser, session: DbSession) -> None:
    """Apply a human quality decision to one recording (audited)."""
    RecordingRepository(session).set_review(
        recording_id, body.verdict, reviewer_id=user.id, transcript_fix=body.transcript_fix
    )
    AuditRepository(session).log(
        user.id, "qc_review", ("recordings", recording_id), {"verdict": body.verdict}
    )


@qc_router.get("/contributions/stats", response_model=ContribStatsOut)
def contributions_stats(_user: StaffUser, session: DbSession) -> ContribStatsOut:
    """The İP-3 collection dashboard, from the administration side."""
    stats = RecordingRepository(session).stats()
    return ContribStatsOut(
        hours_accepted=round(stats.hours_accepted, 4),
        utterances=stats.utterances,
        speakers=stats.speakers,
        by_university=dict(stats.by_university),
        by_gender=dict(stats.by_gender),
        by_age_band=dict(stats.by_age_band),
        qc_pass_rate=round(stats.qc_pass_rate, 4),
    )


@qc_router.patch("/feedback/{feedback_id}", response_model=FeedbackOut)
def triage_feedback(
    feedback_id: str, body: FeedbackPatch, user: StaffUser, session: DbSession
) -> FeedbackOut:
    """Move one feedback entry through triage (audited)."""
    from aicls.api.routers.feedback import feedback_out

    row = FeedbackRepository(session).set_status(feedback_id, body.status)
    AuditRepository(session).log(
        user.id, "feedback_triage", ("feedback", feedback_id), {"status": body.status}
    )
    return feedback_out(row)


@qc_router.get("/audit", response_model=Page[AuditOut])
def audit_log(
    _user: AdminUser,
    session: DbSession,
    action: Annotated[str | None, Query(max_length=64)] = None,
    cursor: Annotated[str | None, Query(max_length=26)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> Page[AuditOut]:
    """One page of the audit log, newest first."""
    page = AuditRepository(session).page(action=action, cursor=cursor, limit=limit)
    return Page[AuditOut](
        items=[
            AuditOut(
                id=row.id,
                actor_id=row.actor_id,
                action=row.action,
                target_table=row.target_table,
                target_id=row.target_id,
                detail=dict(row.detail_json or {}),
                created_at=row.created_at.isoformat(),
            )
            for row in page.items
        ],
        next_cursor=page.next_cursor,
    )


def user_out(row: User) -> UserOut:
    """Convert a ``users`` row into its API shape (shared with the users router)."""
    return UserOut(
        id=row.id,
        email=row.email,
        display_name=row.display_name,
        role=row.role.value,
        created_at=row.created_at,
    )


__all__ = ["MANIFEST_NAME", "qc_router", "router", "user_out"]
