"""Model registry, evaluation, telemetry, feedback and audit repositories (§3.5)."""

from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from datetime import datetime
import math
from typing import Any

from sqlalchemy import select

from aicls.core.errors import InputValidationError, NotFoundError
from aicls.core.messages_tr import msg
from aicls.db.base import utcnow
from aicls.db.enums import (
    EvalKind,
    FeedbackCategory,
    FeedbackStatus,
    Modality,
    ModelComponent,
    coerce_enum,
)
from aicls.db.models import AuditLog, EvalRun, Feedback, LatencySample, ModelVersion, SusResponse
from aicls.db.repositories.base import Repository, clamp_limit, field_of
from aicls.db.results import LatencyStats, Page

#: Number of answers in a System Usability Scale questionnaire.
SUS_ITEMS = 10
#: Rows scanned at most by :meth:`LatencyRepository.stats`.
LATENCY_SCAN_LIMIT = 100_000


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return the ``fraction`` percentile of ``values`` (nearest-rank, ADR-0015).

    Args:
        values: Sample values; need not be sorted.
        fraction: Percentile in ``[0, 1]`` (``0.5`` = median, ``0.95`` = p95).

    Returns:
        The nearest-rank percentile, or ``0.0`` for an empty sample.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def sus_score(answers: Sequence[int]) -> float:
    """Compute the SUS score of ten 1–5 answers (odd items ``x-1``, even items ``5-x``, ×2.5).

    :func:`aicls.eval.sus_score` is the authoritative definition (ADR-0015); this copy exists so
    that :mod:`aicls.db` — which sits below ``eval`` in the layering — can store a score when the
    caller did not compute one.

    Raises:
        InputValidationError: If the answer count or a value is out of range.
    """
    if len(answers) != SUS_ITEMS or any(not 1 <= int(a) <= 5 for a in answers):
        raise InputValidationError(
            msg("VALIDATION_ERROR"), details={"field": "answers", "expected": "10 values in 1..5"}
        )
    total = 0
    for index, answer in enumerate(answers):
        total += (int(answer) - 1) if index % 2 == 0 else (5 - int(answer))
    return total * 2.5


class ModelVersionRepository(Repository):
    """Queries over ``model_versions`` — the single registry of trained checkpoints."""

    def register(
        self,
        component: ModelComponent | str,
        version: str,
        path: str,
        sha256: str,
        metrics: Mapping[str, Any] | None = None,
        run_id: str | None = None,
        *,
        config_hash: str = "",
        git_sha: str = "",
    ) -> ModelVersion:
        """Insert or update a checkpoint entry (idempotent on ``(component, version)``)."""
        kind = coerce_enum(ModelComponent, component)
        stmt = select(ModelVersion).where(
            ModelVersion.component == kind, ModelVersion.version == version
        )
        row = self.session.scalars(stmt).one_or_none()
        if row is None:
            row = ModelVersion(component=kind, version=version)
            self.session.add(row)
        row.path = path
        row.sha256 = sha256
        row.config_hash = config_hash
        row.git_sha = git_sha
        row.run_id = run_id
        row.metrics_json = dict(metrics or {})
        self.session.flush()
        return row

    def active(self, component: ModelComponent | str) -> ModelVersion | None:
        """Return the active checkpoint of a component, or ``None`` when none is promoted."""
        stmt = select(ModelVersion).where(
            ModelVersion.component == coerce_enum(ModelComponent, component),
            ModelVersion.is_active.is_(True),
        )
        return self.session.scalars(stmt).first()

    def activate(
        self, component: ModelComponent | str, version: str, *, promoted_by: str | None = None
    ) -> ModelVersion:
        """Promote one version and deactivate every other version of the component.

        Raises:
            NotFoundError: If the version is not registered.
        """
        kind = coerce_enum(ModelComponent, component)
        rows = list(
            self.session.scalars(select(ModelVersion).where(ModelVersion.component == kind))
        )
        target = next((row for row in rows if row.version == version), None)
        if target is None:
            raise NotFoundError(
                msg("NOT_FOUND"),
                details={"table": "model_versions", "component": kind.value, "version": version},
            )
        for row in rows:
            row.is_active = row is target
        target.promoted_at = utcnow()
        target.promoted_by = promoted_by
        self.session.flush()
        return target

    def list_versions(self, component: ModelComponent | str | None = None) -> list[ModelVersion]:
        """Return registered versions, newest first (``GET /admin/models``)."""
        stmt = select(ModelVersion)
        if component is not None:
            stmt = stmt.where(ModelVersion.component == coerce_enum(ModelComponent, component))
        return list(self.session.scalars(stmt.order_by(ModelVersion.id.desc())))


class EvalRunRepository(Repository):
    """Queries over ``eval_runs`` — persisted :class:`aicls.eval.EvalReport` documents."""

    def add(self, report: Mapping[str, Any], *, is_reportable: bool = False) -> EvalRun:
        """Persist an evaluation report.

        Args:
            report: The report document (``aicls.eval.EvalReport`` dumped to a dict). Keys are
                read leniently so a report gains fields without breaking this writer.
            is_reportable: ``True`` marks the numbers as citable in a TÜBİTAK report
                (``aicls report promote``).

        Returns:
            The inserted row.
        """
        row = EvalRun(
            kind=coerce_enum(EvalKind, field_of(report, "kind", default="parser")),
            name=str(field_of(report, "name", default="")),
            git_sha=str(field_of(report, "git_sha", default="")),
            dirty=bool(field_of(report, "dirty", default=False)),
            config_hash=field_of(report, "config_hash"),
            dataset_fingerprint=str(field_of(report, "dataset_fingerprint", default="")),
            dataset_version=str(field_of(report, "dataset_version", default="")),
            model_version=field_of(report, "model_version"),
            normalizer_versions_json=dict(
                field_of(report, "normalizer_versions", "normalizer_versions_json", default={})
            ),
            n=int(field_of(report, "n", default=0)),
            metrics_json=dict(field_of(report, "metrics", "metrics_json", default={})),
            ci95_json=dict(field_of(report, "ci95", "ci95_json", default={})),
            report_path=str(field_of(report, "report_path", "failures_path", default="")),
            is_reportable=is_reportable,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def list(
        self,
        kind: EvalKind | str | None = None,
        since: datetime | None = None,
        *,
        limit: int | None = None,
    ) -> builtins.list[EvalRun]:
        """Return evaluation runs, newest first (``GET /admin/eval-runs``).

        ``builtins.list`` is spelled out because this class defines a method named ``list``
        (the name is part of the public contract in modules/db.json).
        """
        stmt = select(EvalRun)
        if kind is not None:
            stmt = stmt.where(EvalRun.kind == coerce_enum(EvalKind, kind))
        if since is not None:
            stmt = stmt.where(EvalRun.created_at >= since)
        stmt = stmt.order_by(EvalRun.id.desc()).limit(clamp_limit(limit))
        return list(self.session.scalars(stmt))

    def reportable(self, kind: EvalKind | str | None = None) -> builtins.list[EvalRun]:
        """Return only citable runs (``is_reportable``), newest first."""
        stmt = select(EvalRun).where(EvalRun.is_reportable.is_(True))
        if kind is not None:
            stmt = stmt.where(EvalRun.kind == coerce_enum(EvalKind, kind))
        return list(self.session.scalars(stmt.order_by(EvalRun.id.desc())))


class LatencyRepository(Repository):
    """Queries over ``latency_samples`` — how the "< 3 s" contract metric is measured."""

    def add(self, sample: Mapping[str, Any]) -> None:
        """Store one latency sample (``POST /telemetry/latency``).

        Recognised keys: ``operation_id``, ``request_id``, ``modality``, ``endpoint``,
        ``server_ms``, ``stage``/``stage_json``, ``client_total_ms``, ``render_ms``,
        ``model_versions``/``model_versions_json``.
        """
        row = LatencySample(
            operation_id=field_of(sample, "operation_id"),
            request_id=str(field_of(sample, "request_id", default="")),
            modality=coerce_enum(Modality, field_of(sample, "modality", default="text")),
            endpoint=str(field_of(sample, "endpoint", default="")),
            server_ms=float(field_of(sample, "server_ms", default=0.0)),
            stage_json=dict(field_of(sample, "stage", "stage_json", "timings", default={})),
            client_total_ms=field_of(sample, "client_total_ms"),
            render_ms=field_of(sample, "render_ms"),
            model_versions_json=dict(
                field_of(sample, "model_versions", "model_versions_json", default={})
            ),
        )
        self.session.add(row)
        self.session.flush()

    def stats(
        self, since: datetime | None = None, modality: Modality | str | None = None
    ) -> LatencyStats:
        """Aggregate mean/p50/p95 of the end-to-end time (``GET /admin/metrics/latency``).

        The end-to-end value is ``client_total_ms`` when the browser reported it and
        ``server_ms`` otherwise, so the number is the time to the *rendered* result.
        """
        stmt = select(
            LatencySample.modality, LatencySample.server_ms, LatencySample.client_total_ms
        )
        if since is not None:
            stmt = stmt.where(LatencySample.created_at >= since)
        if modality is not None:
            stmt = stmt.where(LatencySample.modality == coerce_enum(Modality, modality))
        rows = list(self.session.execute(stmt.limit(LATENCY_SCAN_LIMIT)))
        totals: list[float] = []
        grouped: dict[str, list[float]] = {}
        for row_modality, server_ms, client_total_ms in rows:
            value = float(client_total_ms if client_total_ms is not None else server_ms)
            totals.append(value)
            grouped.setdefault(str(getattr(row_modality, "value", row_modality)), []).append(value)
        mean = sum(totals) / len(totals) if totals else 0.0
        return LatencyStats(
            mean_ms=mean,
            p50_ms=percentile(totals, 0.5),
            p95_ms=percentile(totals, 0.95),
            n=len(totals),
            by_modality={key: sum(vals) / len(vals) for key, vals in sorted(grouped.items())},
        )


class FeedbackRepository(Repository):
    """Queries over ``feedback`` and ``sus_responses`` (İP-9 pilot study)."""

    def add_feedback(
        self,
        user_id: str | None,
        category: FeedbackCategory | str,
        text: str,
        page: str = "",
        *,
        screenshot_asset_id: str | None = None,
        collected_week: int | None = None,
    ) -> Feedback:
        """Store a feedback entry in state ``open``."""
        row = Feedback(
            user_id=user_id,
            category=coerce_enum(FeedbackCategory, category),
            text=text,
            page=page,
            screenshot_asset_id=screenshot_asset_id,
            collected_week=collected_week,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def add_sus(
        self,
        user_id: str | None,
        cohort: str,
        answers: Sequence[int],
        *,
        score: float | None = None,
        collected_week: int | None = None,
    ) -> SusResponse:
        """Store a SUS questionnaire and its score (computed when not supplied).

        Raises:
            InputValidationError: If ``answers`` is not ten values in 1–5 and no ``score`` was
                supplied.
        """
        computed = score if score is not None else sus_score(answers)
        row = SusResponse(
            user_id=user_id,
            cohort=cohort,
            answers_json={"answers": [int(a) for a in answers]},
            score=float(computed),
            collected_week=collected_week,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def page(
        self,
        status: FeedbackStatus | str | None = None,
        category: FeedbackCategory | str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> Page[Feedback]:
        """Return one page of feedback entries, newest first (``GET /feedback``)."""
        stmt = select(Feedback)
        if status is not None:
            stmt = stmt.where(Feedback.status == coerce_enum(FeedbackStatus, status))
        if category is not None:
            stmt = stmt.where(Feedback.category == coerce_enum(FeedbackCategory, category))
        return self.cursor_page(stmt, Feedback.id, cursor=cursor, limit=limit)

    def set_status(self, feedback_id: str, status: FeedbackStatus | str) -> Feedback:
        """Update the triage state of a feedback entry.

        Raises:
            NotFoundError: If no such entry exists.
        """
        row = self.session.get(Feedback, feedback_id)
        if row is None:
            raise NotFoundError(msg("NOT_FOUND"), details={"table": "feedback", "id": feedback_id})
        row.status = coerce_enum(FeedbackStatus, status)
        self.session.flush()
        return row

    def sus_average(self, cohort: str | None = None) -> float:
        """Mean SUS score of a cohort (contract target > 75); ``0.0`` when there are none."""
        stmt = select(SusResponse.score)
        if cohort is not None:
            stmt = stmt.where(SusResponse.cohort == cohort)
        scores = [float(value) for value in self.session.scalars(stmt)]
        return sum(scores) / len(scores) if scores else 0.0


class AuditRepository(Repository):
    """Append-only writer for ``audit_log`` (docs/guvenlik-gizlilik.md)."""

    def log(
        self,
        actor_id: str | None,
        action: str,
        target: str | tuple[str, str] | None = None,
        detail: Mapping[str, Any] | None = None,
        *,
        ip_hash: str = "",
    ) -> AuditLog:
        """Record one sensitive action.

        Args:
            actor_id: User who performed the action, or ``None`` for a system action.
            action: Stable action name (``"role_change"``, ``"consent_withdraw"``, ...).
            target: ``(table, id)`` pair, or ``"table:id"``, or just a table name.
            detail: Structured context; stored as JSON.
            ip_hash: Hashed client address, when the action came from a request.

        Returns:
            The inserted row (also returned so callers can assert on it in tests).
        """
        if isinstance(target, tuple):
            table, identifier = target[0], target[1]
        elif isinstance(target, str):
            table, _, identifier = target.partition(":")
        else:
            table, identifier = "", ""
        row = AuditLog(
            actor_id=actor_id,
            action=action,
            target_table=table,
            target_id=identifier,
            detail_json=dict(detail or {}),
            ip_hash=ip_hash,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def page(
        self,
        action: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> Page[AuditLog]:
        """Return one page of audit entries, newest first (``GET /admin/audit``)."""
        stmt = select(AuditLog)
        if action is not None:
            stmt = stmt.where(AuditLog.action == action)
        return self.cursor_page(stmt, AuditLog.id, cursor=cursor, limit=limit)


__all__ = [
    "LATENCY_SCAN_LIMIT",
    "SUS_ITEMS",
    "AuditRepository",
    "EvalRunRepository",
    "FeedbackRepository",
    "LatencyRepository",
    "ModelVersionRepository",
    "percentile",
    "sus_score",
]
