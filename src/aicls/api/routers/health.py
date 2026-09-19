"""Liveness, readiness and version endpoints (api.md §3.1; open, no authentication).

``/health/ready`` is what a load balancer and the deployment runbook poll: it answers 503 while
the database is unreachable, while the sandbox pool is down, or while a component reports
``training`` — the state ADR-0018 requires during a GPU run on the workstation.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status
from sqlalchemy import text

from aicls.api.deps import AppSettings, DbSession, Registry
from aicls.api.schemas import LiveOut, ReadyOut, SandboxOut, VersionOut
from aicls.core.logging import get_logger

router = APIRouter(prefix="/health", tags=["health"])

#: States that make ``GET /health/ready`` answer 503.
NOT_READY_STATES: frozenset[str] = frozenset({"training"})

_logger = get_logger(__name__)


@router.get("/live", response_model=LiveOut)
def live() -> LiveOut:
    """The process is up (no dependency is touched)."""
    return LiveOut(status="ok")


@router.get("/ready", response_model=ReadyOut)
def ready(request: Request, response: Response, session: DbSession, registry: Registry) -> ReadyOut:
    """Database, models and sandbox pool; 503 when any of them is not ready."""
    db_state = "ok"
    try:
        session.execute(text("SELECT 1"))  # raw-sql-ok: liveness probe, no user input
    except Exception as exc:
        db_state = "error"
        _logger.warning("api.health.db_error", error=str(exc)[:200])
    executor = getattr(request.app.state, "executor", None)
    sandbox = SandboxOut(
        workers=int(getattr(executor, "workers", 1) or 0),
        restarts=int(getattr(executor, "restarts", 0) or 0),
        started=bool(getattr(executor, "started", executor is not None)),
    )
    models: dict[str, str] = {name: str(state) for name, state in registry.states().items()}
    ok = db_state == "ok" and sandbox.started and not (NOT_READY_STATES & set(models.values()))
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyOut(status="ok" if ok else "not_ready", db=db_state, models=models, sandbox=sandbox)


@router.get("/version", response_model=VersionOut)
def version(settings: AppSettings, registry: Registry) -> VersionOut:
    """Package version, git revision, profile and the active model versions."""
    from aicls import __version__

    return VersionOut(
        version=__version__,
        git_sha="",
        profile=settings.profile,
        models=registry.versions(),
    )


__all__ = ["NOT_READY_STATES", "router"]
