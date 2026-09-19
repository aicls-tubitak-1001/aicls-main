"""``create_app`` — the FastAPI application factory (ADR-0001, ADR-0014, ADR-0018).

The factory is the whole configuration surface of the service. It decides:

* **which routers exist** — ``AICLS_PROFILE`` selects the ``full`` workstation surface or the
  ``collect`` VM subset (ADR-0018);
* **what runs at startup** — the lifespan migrates the database, starts the CAS executor
  (a spawned sandbox pool in ``dev``/``prod``, an in-process executor under ``AICLS_ENV=test``
  so a test does not pay a process spawn per application) and warms the model registry;
* **what wraps every request** — CORS, security headers, rate limiting and the request-context
  middleware that installs the ``StageTimer`` behind ``Server-Timing``;
* **what a failure looks like** — the RFC 9457 handlers of :mod:`aicls.api.errors`.

Nothing is a module-level singleton: every collaborator lives on ``app.state``, so two
applications with two in-memory databases can exist in one process (which is exactly what the
integration tests do).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aicls import __version__
from aicls.api.errors import install_exception_handlers
from aicls.api.middleware import (
    RateLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from aicls.api.model_registry import ModelRegistry
from aicls.api.pipeline import PipelineService
from aicls.api.ratelimit import RateLimiter
from aicls.api.routers import routers_for
from aicls.api.security import WsTicketStore, jwt_secret
from aicls.api.ws.audio import audio_stream
from aicls.api.ws.common import WsTimings
from aicls.api.ws.ink import ink_stream
from aicls.core.logging import configure_logging, get_logger
from aicls.core.settings import Settings, get_settings
from aicls.db.engine import make_engine, session_scope
from aicls.db.migrate import upgrade_head
from aicls.mathengine import LruCache, MathEngine

#: Base path of the REST API (URI versioning, api.md §1).
API_PREFIX = "/api/v1"

#: Base path of the WebSocket streams.
WS_PREFIX = "/ws/v1"

#: Environment variable holding the instructor invite code (empty disables the role).
INVITE_CODE_ENV = "AICLS_INVITE_CODE"

#: OpenAPI description shown at ``/api/v1/docs``.
DESCRIPTION = (
    "AI-CLS — TÜBİTAK 1001 #125E748. El yazısı ve ses girdisini SymPy ile hesaplanan "
    "sonuçlara dönüştüren Temel Analiz platformunun REST ve WebSocket arayüzü."
)

_logger = get_logger(__name__)


def build_executor(settings: Settings) -> object:
    """Return the CAS executor for this environment.

    ``dev``/``prod`` get the spawned sandbox pool of ADR-0016 (a killed process really stops);
    ``test`` gets the in-process executor, because a spawn costs seconds per application and the
    tests need dozens of them.

    The pool is built from **this** application's settings. ``pool_from_settings()`` reads the
    process-wide cached ``get_settings()``, so calling it bare made ``create_app(settings)``
    silently start a pool sized by the environment rather than by the settings object it was
    handed -- observed 2026-09-08: ``build_executor(Settings(sandbox_workers=1))`` produced a
    two-worker pool.

    Args:
        settings: The application settings this executor belongs to.

    Returns:
        An **unstarted** executor; the lifespan calls ``start()``.
    """
    if settings.env == "test":
        from aicls.mathengine.executor import InProcessExecutor

        return InProcessExecutor()
    from aicls.mathengine.sandbox import pool_from_settings

    return pool_from_settings(
        workers=settings.sandbox_workers, timeout_s=settings.compute_timeout_s
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Migrate the database, start the executor and warm the model registry."""
    settings: Settings = app.state.settings
    engine = make_engine(settings.db_url)
    upgrade_head(engine)
    app.state.engine = engine

    executor = build_executor(settings)
    start = getattr(executor, "start", None)
    if callable(start):
        start()
    app.state.executor = executor

    registry = ModelRegistry(settings, lambda: session_scope(engine))
    app.state.registry = registry
    app.state.pipeline = PipelineService(
        MathEngine(executor, LruCache(), settings.compute_timeout_s),  # type: ignore[arg-type]
        registry,
        settings,
    )
    registry.warmup()
    _logger.info("api.started", profile=settings.profile, env=settings.env, version=__version__)
    try:
        yield
    finally:
        stop = getattr(executor, "stop", None)
        if callable(stop):
            stop()
        engine.dispose()
        _logger.info("api.stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Runtime settings; :func:`aicls.core.settings.get_settings` when omitted.

    Returns:
        The application, with the routers of ``settings.profile`` mounted under ``/api/v1``,
        the two WebSocket streams under ``/ws/v1`` and every middleware and error handler in
        place. The database, executor and models are created by the lifespan, so building an
        application is cheap and side-effect free.
    """
    resolved = settings if settings is not None else get_settings()
    configure_logging(resolved.log_level)

    app = FastAPI(
        title="AI-CLS API",
        version=__version__,
        description=DESCRIPTION,
        openapi_url=f"{API_PREFIX}/openapi.json",
        docs_url=f"{API_PREFIX}/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    secret = jwt_secret(resolved)
    app.state.settings = resolved
    app.state.jwt_secret = secret
    app.state.tickets = WsTicketStore()
    app.state.limiter = RateLimiter()
    app.state.ws_timings = WsTimings()
    app.state.invite_code = os.environ.get(INVITE_CODE_ENV, "")
    app.state.engine = None
    app.state.executor = None
    app.state.registry = None
    app.state.pipeline = None

    # Added last runs first: CORS -> security headers -> rate limit -> request context.
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        limiter=app.state.limiter,
        secret=secret,
        prefix=API_PREFIX,
        enabled=resolved.rate_limit_enabled,
    )
    app.add_middleware(SecurityHeadersMiddleware)
    if resolved.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved.cors_origins),
            allow_credentials=True,
            allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Request-Id", "X-Participant-Token"],
            expose_headers=["X-Request-Id", "Server-Timing", "Retry-After"],
        )

    install_exception_handlers(app)
    for router in routers_for(resolved.profile):
        app.include_router(router, prefix=API_PREFIX)
    app.add_api_websocket_route(f"{WS_PREFIX}/ink", ink_stream, name="ws_ink")
    app.add_api_websocket_route(f"{WS_PREFIX}/audio", audio_stream, name="ws_audio")
    return app


__all__ = [
    "API_PREFIX",
    "DESCRIPTION",
    "INVITE_CODE_ENV",
    "WS_PREFIX",
    "build_executor",
    "create_app",
    "lifespan",
]
