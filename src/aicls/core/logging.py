"""Structured logging (structlog) with JSON output and request-id context.

Library code obtains a logger with :func:`get_logger` and never uses ``print``. The API layer
binds the request id with :func:`request_context` (or :func:`bind_request_id`) so every line of a
request carries ``request_id``; the problem-details ``instance`` field uses the same id.

Configuration happens once via :func:`configure_logging` (the CLI and app factory call it); if a
logger is requested before that, defaults are read from ``AICLS_LOG_LEVEL``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import logging as _stdlib_logging
import os
import sys
from typing import Any, TextIO, cast

import structlog
from structlog.typing import FilteringBoundLogger, Processor

from aicls.core import messages_tr
from aicls.core.errors import AiclsError, InputValidationError
from aicls.core.ids import ulid

_LEVELS: dict[str, int] = {
    "DEBUG": _stdlib_logging.DEBUG,
    "INFO": _stdlib_logging.INFO,
    "WARNING": _stdlib_logging.WARNING,
    "ERROR": _stdlib_logging.ERROR,
    "CRITICAL": _stdlib_logging.CRITICAL,
}
_REQUEST_ID_KEY = "request_id"


@dataclass
class _State:
    configured: bool = False
    level: str = "INFO"


_state = _State()


class _LazyStdout:
    """A ``sys.stdout`` proxy that resolves the real stream on every write.

    ``structlog.PrintLoggerFactory`` stores the file object it is given, so binding
    ``sys.stdout`` at configuration time freezes the destination for the life of the process.
    Anything that replaces ``sys.stdout`` afterwards — pytest's ``capsys``, ``contextlib
    .redirect_stdout``, a CLI that re-points logging at stderr — then writes past the
    replacement. Writing through this proxy keeps the destination late-bound, which is what every
    caller means by "log to stdout".
    """

    def write(self, message: str) -> int:
        """Write ``message`` to whatever ``sys.stdout`` is at this moment."""
        return sys.stdout.write(message)

    def flush(self) -> None:
        """Flush the current ``sys.stdout``."""
        sys.stdout.flush()

    def __repr__(self) -> str:
        return "<aicls stdout proxy>"


def _resolve_level(level: str | None) -> str:
    name = (level or os.environ.get("AICLS_LOG_LEVEL") or "INFO").strip().upper()
    if name not in _LEVELS:
        raise InputValidationError(
            messages_tr.msg("INVALID_LOG_LEVEL", level=name), details={"level": name}
        )
    return name


def configure_logging(
    level: str | None = None,
    *,
    json_output: bool = True,
    stream: TextIO | None = None,
) -> None:
    """Configure structlog for the process (safe to call again to reconfigure).

    Args:
        level: ``DEBUG``/``INFO``/``WARNING``/``ERROR``/``CRITICAL``; defaults to
            ``AICLS_LOG_LEVEL`` or ``INFO``.
        json_output: Emit one JSON object per line (production/CI) or a plain console line (dev).
        stream: Destination; the default writes to whatever ``sys.stdout`` is **at write time**
            (see :class:`_LazyStdout`), tests pass a ``StringIO``.

    Raises:
        InputValidationError: On an unknown level name.
    """
    name = _resolve_level(level)
    renderer: Processor = (
        structlog.processors.JSONRenderer(ensure_ascii=False, sort_keys=False)
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        renderer,
    ]
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(_LEVELS[name]),
        logger_factory=structlog.PrintLoggerFactory(
            file=stream if stream is not None else cast("TextIO", _LazyStdout())
        ),
        cache_logger_on_first_use=False,
    )
    _state.configured = True
    _state.level = name


def reset_logging() -> None:
    """Restore structlog defaults and forget the configured level (tests)."""
    structlog.reset_defaults()
    _state.configured = False
    _state.level = "INFO"


def is_configured() -> bool:
    """True once :func:`configure_logging` has run (and not been reset)."""
    return _state.configured


def current_level() -> str:
    """Name of the configured level (``INFO`` before configuration)."""
    return _state.level


def get_logger(name: str = "aicls") -> FilteringBoundLogger:
    """Return a bound logger carrying ``logger=name``; configures defaults on first use."""
    if not _state.configured:
        configure_logging()
    return cast(FilteringBoundLogger, structlog.get_logger().bind(logger=name))


def bind_request_id(request_id: str) -> None:
    """Attach ``request_id`` to every log line of the current context (contextvars)."""
    structlog.contextvars.bind_contextvars(**{_REQUEST_ID_KEY: request_id})


def clear_request_context() -> None:
    """Remove all context-bound fields (call at the end of a request)."""
    structlog.contextvars.clear_contextvars()


@contextmanager
def request_context(request_id: str | None = None) -> Iterator[str]:
    """Bind a request id (generated ULID when omitted) for the duration of the block.

    Yields:
        The request id in effect.
    """
    rid = request_id or ulid()
    tokens = structlog.contextvars.bind_contextvars(**{_REQUEST_ID_KEY: rid})
    try:
        yield rid
    finally:
        structlog.contextvars.reset_contextvars(**tokens)


def log_error(
    logger: FilteringBoundLogger,
    err: AiclsError,
    *,
    request_id: str | None = None,
    **extra: Any,
) -> None:
    """Log an :class:`AiclsError` with its structured fields.

    4xx-class errors are logged at ``warning``; 5xx-class at ``error`` (with the traceback).
    """
    fields: dict[str, Any] = {
        "code": err.code.value,
        "http_status": err.http_status,
        "message_tr": err.message_tr,
        "details": err.details,
        **extra,
    }
    if request_id is not None:
        fields[_REQUEST_ID_KEY] = request_id
    if err.http_status >= 500:
        logger.error("aicls_error", exc_info=err, **fields)
    else:
        logger.warning("aicls_error", **fields)


__all__ = [
    "bind_request_id",
    "clear_request_context",
    "configure_logging",
    "current_level",
    "get_logger",
    "is_configured",
    "log_error",
    "request_context",
    "reset_logging",
]
