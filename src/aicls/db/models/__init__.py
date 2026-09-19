"""All ORM models, grouped by the sections of docs/mimari/veri-modeli.md §3.

Importing this package populates :attr:`aicls.db.base.Base.metadata` with every table, which is
what ``tools/gen_erd.py``, the Alembic ``env.py`` and the schema-rule tests rely on.
"""

from __future__ import annotations

from aicls.db.models.audio import ConsentVersion, PromptSentence, Recording, Speaker
from aicls.db.models.book import BookAlias, BookAsset, BookTriple, BookUnit
from aicls.db.models.compute import Asset, Operation, Recognition, Transcript
from aicls.db.models.identity import RefreshToken, User, Workspace, WorkspaceMember
from aicls.db.models.registry import (
    AuditLog,
    EvalRun,
    Feedback,
    LatencySample,
    ModelVersion,
    SusResponse,
)

#: Every table name the initial migration creates, in dependency order.
TABLE_NAMES: tuple[str, ...] = (
    "users",
    "refresh_tokens",
    "workspaces",
    "workspace_members",
    "assets",
    "operations",
    "recognitions",
    "transcripts",
    "consent_versions",
    "speakers",
    "prompt_sentences",
    "recordings",
    "book_units",
    "book_aliases",
    "book_triples",
    "book_assets",
    "model_versions",
    "eval_runs",
    "latency_samples",
    "feedback",
    "sus_responses",
    "audit_log",
)

__all__ = [
    "TABLE_NAMES",
    "Asset",
    "AuditLog",
    "BookAlias",
    "BookAsset",
    "BookTriple",
    "BookUnit",
    "ConsentVersion",
    "EvalRun",
    "Feedback",
    "LatencySample",
    "ModelVersion",
    "Operation",
    "PromptSentence",
    "Recognition",
    "Recording",
    "RefreshToken",
    "Speaker",
    "SusResponse",
    "Transcript",
    "User",
    "Workspace",
    "WorkspaceMember",
]
