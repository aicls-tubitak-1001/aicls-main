"""``aicls serve`` — run the FastAPI application with uvicorn.

The command owns exactly two decisions and delegates everything else to :mod:`aicls.api`:

* which **profile** the process runs (``full`` mounts every router, ``collect`` only the voice
  data-collection ones) — it is exported as ``AICLS_PROFILE`` *before* the settings are built, so
  the ``--reload`` worker subprocess sees the same value as the parent;
* where the server binds. The default host is ``127.0.0.1``: a development server must not be
  reachable from the network unless someone asks for it.

:func:`run_server` is the seam. It is the only function that imports uvicorn and the only one
that blocks, so a test can drive the whole command and assert the resolved
:class:`ServeSpec` without starting a socket.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Annotated

import typer

from aicls.cli.common import cli_logger, emit, use_json

HELP = "Run the AI-CLS API server (uvicorn)."

#: Import string uvicorn is given; ``factory=True`` because ``create_app`` builds the app.
APP_FACTORY = "aicls.api.app:create_app"

#: Loopback by default — never bind a development server to every interface implicitly.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


@dataclass(frozen=True, slots=True)
class ServeSpec:
    """The resolved server configuration (what :func:`run_server` is handed).

    Attributes:
        host: Interface to bind.
        port: TCP port.
        profile: ``full`` or ``collect`` (:class:`aicls.core.settings.ProfileName`).
        reload: Whether uvicorn watches the sources and restarts.
        log_level: uvicorn log level, derived from ``AICLS_LOG_LEVEL`` when not given.
        app: Import string of the application factory.
    """

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    profile: str = "full"
    reload: bool = False
    log_level: str = "info"
    app: str = APP_FACTORY


def run_server(spec: ServeSpec) -> None:  # pragma: no cover - blocks on a real socket
    """Start uvicorn with ``spec`` (the blocking part of ``aicls serve``)."""
    import uvicorn

    uvicorn.run(
        spec.app,
        factory=True,
        host=spec.host,
        port=spec.port,
        reload=spec.reload,
        log_level=spec.log_level,
    )


def serve(
    host: Annotated[str, typer.Option("--host", help="Interface to bind.")] = DEFAULT_HOST,
    port: Annotated[int, typer.Option("--port", help="TCP port to listen on.")] = DEFAULT_PORT,
    profile: Annotated[
        str,
        typer.Option(
            "--profile",
            help="Router profile: 'full' (whole platform) or 'collect' (voice contribution only).",
        ),
    ] = "full",
    reload: Annotated[
        bool, typer.Option("--reload/--no-reload", help="Restart on source changes (development).")
    ] = False,
    log_level: Annotated[
        str, typer.Option("--log-level", help="uvicorn log level; defaults to AICLS_LOG_LEVEL.")
    ] = "",
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the resolved server configuration as JSON.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Resolve and print the configuration, do not serve.")
    ] = False,
) -> None:
    """Run the API server.

    Raises:
        aicls.core.errors.InputValidationError: If ``--profile`` is not ``full`` or ``collect``.
    """
    from aicls.core.errors import InputValidationError
    from aicls.core.messages_tr import MSG_VALIDATION_ERROR
    from aicls.core.settings import get_settings

    use_json(json_output)
    if profile not in ("full", "collect"):
        raise InputValidationError(
            MSG_VALIDATION_ERROR,
            details={"reason": "unknown_profile", "profile": profile, "expected": "full|collect"},
        )
    # The reload worker is a fresh process; the environment is the only channel that reaches it.
    os.environ["AICLS_PROFILE"] = profile
    get_settings.cache_clear()
    settings = get_settings()

    spec = ServeSpec(
        host=host,
        port=int(port),
        profile=profile,
        reload=bool(reload),
        log_level=(log_level or settings.log_level).lower(),
    )
    payload = {
        "host": spec.host,
        "port": spec.port,
        "profile": spec.profile,
        "reload": spec.reload,
        "log_level": spec.log_level,
        "app": spec.app,
        "env": settings.env,
        "db_url": settings.db_url,
        "started": not dry_run,
    }
    emit(
        payload,
        f"profile={spec.profile} env={settings.env} log_level={spec.log_level}",
        f"listening on http://{spec.host}:{spec.port} (reload={spec.reload})",
    )
    if dry_run:
        return
    cli_logger(__name__).info(
        "cli.serve.starting", host=spec.host, port=spec.port, profile=spec.profile
    )
    run_server(spec)


__all__ = [
    "APP_FACTORY",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "HELP",
    "ServeSpec",
    "run_server",
    "serve",
]
