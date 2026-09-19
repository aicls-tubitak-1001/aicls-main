"""Repositories — the only layer that builds queries (ADR-0003, dialect rule 8).

Each repository is constructed with a :class:`~sqlalchemy.orm.Session` and never commits; the
caller owns the transaction through :func:`aicls.db.engine.session_scope` or the FastAPI
dependency :func:`aicls.db.engine.get_session`.
"""

from __future__ import annotations

from aicls.db.repositories.audio import RecordingRepository
from aicls.db.repositories.base import Repository, clamp_limit, field_of
from aicls.db.repositories.book import BookRepository
from aicls.db.repositories.compute import (
    AssetRepository,
    OperationRepository,
    RecognitionRepository,
)
from aicls.db.repositories.identity import UserRepository, WorkspaceRepository
from aicls.db.repositories.registry import (
    AuditRepository,
    EvalRunRepository,
    FeedbackRepository,
    LatencyRepository,
    ModelVersionRepository,
)

__all__ = [
    "AssetRepository",
    "AuditRepository",
    "BookRepository",
    "EvalRunRepository",
    "FeedbackRepository",
    "LatencyRepository",
    "ModelVersionRepository",
    "OperationRepository",
    "RecognitionRepository",
    "RecordingRepository",
    "Repository",
    "UserRepository",
    "WorkspaceRepository",
    "clamp_limit",
    "field_of",
]
