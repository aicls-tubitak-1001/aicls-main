"""FastAPI application: routers, services, WebSocket streams, security middleware.

Public contract (``modules/api.json``, docs/mimari/api.md)::

    create_app(settings=None) -> FastAPI
    get_db(); get_current_user(token) -> User; require_role(*roles)
    PipelineService.run_text / run_latex / run_ink / run_audio -> PipelineResult
    ModelRegistry.get_recognizer() / get_transcriber() / reload()
    WsEnvelope(type, seq, payload)

The package sits above every domain package and below nothing: it imports ``aicls.core``,
``aicls.db``, ``aicls.latex2sympy``, ``aicls.mathengine``, ``aicls.nlu``, ``aicls.ocr``,
``aicls.asr``, ``aicls.book`` and ``aicls.collect``, and never ``aicls.ml``, ``aicls.eval``,
``aicls.data`` or ``torch`` (the ``api-no-training`` and ``no-torch-in-core`` contracts of
ADR-0002).
"""

from __future__ import annotations

from aicls.api.app import API_PREFIX, WS_PREFIX, create_app
from aicls.api.deps import (
    get_current_user,
    get_db,
    get_optional_user,
    get_pipeline,
    get_registry,
    require_role,
)
from aicls.api.errors import AuthError, ForbiddenError, RateLimitedError, problem_response
from aicls.api.model_registry import ModelRegistry
from aicls.api.pipeline import (
    ClarificationCandidate,
    ClarificationPayload,
    NluTrace,
    PipelineResult,
    PipelineService,
)
from aicls.api.ratelimit import RateLimiter, RateRule
from aicls.api.security import WsTicketStore
from aicls.api.ws.protocol import WsEnvelope

__all__ = [
    "API_PREFIX",
    "WS_PREFIX",
    "AuthError",
    "ClarificationCandidate",
    "ClarificationPayload",
    "ForbiddenError",
    "ModelRegistry",
    "NluTrace",
    "PipelineResult",
    "PipelineService",
    "RateLimitedError",
    "RateLimiter",
    "RateRule",
    "WsEnvelope",
    "WsTicketStore",
    "create_app",
    "get_current_user",
    "get_db",
    "get_optional_user",
    "get_pipeline",
    "get_registry",
    "problem_response",
    "require_role",
]
