"""API error types and the RFC 9457 ``application/problem+json`` handlers (api.md §2).

Every failure that leaves the application is rendered by :func:`problem_response` from an
:class:`~aicls.core.errors.AiclsError`, so the wire format is identical no matter which layer
raised: ``{type, title, status, detail_tr, code, instance, ...}`` plus whatever structured
fields the error carried in ``details`` (``position`` for a parse error, ``construct`` for an
unsupported LaTeX construct, ``errors`` for a validation failure).

Three error types are introduced here because no lower package owns them: :class:`AuthError`
(missing/expired token), :class:`ForbiddenError` (role or ownership) and
:class:`RateLimitedError` (token bucket exhausted, carries ``Retry-After``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse, Response

from aicls.core import messages_tr
from aicls.core.errors import AiclsError, ErrorCode
from aicls.core.logging import get_logger

#: Media type of every error response (RFC 9457).
PROBLEM_MEDIA_TYPE = "application/problem+json"

#: Header carrying the correlation id of a request.
REQUEST_ID_HEADER = "X-Request-Id"

#: HTTP status -> :class:`~aicls.core.errors.ErrorCode` for framework-raised exceptions.
STATUS_TO_CODE: dict[int, ErrorCode] = {
    400: ErrorCode.VALIDATION_ERROR,
    401: ErrorCode.AUTH_INVALID,
    403: ErrorCode.FORBIDDEN,
    404: ErrorCode.NOT_FOUND,
    405: ErrorCode.VALIDATION_ERROR,
    413: ErrorCode.UPLOAD_INVALID,
    415: ErrorCode.UPLOAD_INVALID,
    422: ErrorCode.VALIDATION_ERROR,
    429: ErrorCode.RATE_LIMITED,
    500: ErrorCode.INTERNAL,
    503: ErrorCode.GATE_BLOCKED,
    504: ErrorCode.CAS_TIMEOUT,
}

#: Number of pydantic error entries copied into the problem document.
MAX_VALIDATION_ERRORS = 8

_logger = get_logger(__name__)


class AuthError(AiclsError):
    """No usable credentials were presented (``AUTH_INVALID``, HTTP 401)."""

    default_code = ErrorCode.AUTH_INVALID

    def __init__(
        self, message_tr: str | None = None, *, details: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(self.default_code, message_tr, details, None)


class ForbiddenError(AiclsError):
    """The caller is authenticated but not allowed (``FORBIDDEN``, HTTP 403)."""

    default_code = ErrorCode.FORBIDDEN

    def __init__(
        self, message_tr: str | None = None, *, details: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(self.default_code, message_tr, details, None)


class RateLimitedError(AiclsError):
    """A rate-limit bucket is empty (``RATE_LIMITED``, HTTP 429).

    Args:
        retry_after: Whole seconds the client should wait; echoed in ``Retry-After`` and in the
            Turkish message.
        details: Extra structured fields (``scope``, ``limit``, ``window_s``).
    """

    default_code = ErrorCode.RATE_LIMITED

    retry_after: int

    def __init__(self, retry_after: int = 1, *, details: Mapping[str, Any] | None = None) -> None:
        self.retry_after = max(1, int(retry_after))
        merged: dict[str, Any] = {"retry_after": self.retry_after}
        if details:
            merged.update(details)
        super().__init__(
            self.default_code,
            messages_tr.msg("RATE_LIMITED_RETRY", retry_after=self.retry_after),
            merged,
            None,
        )


def problem_response(error: AiclsError, *, request_id: str | None = None) -> JSONResponse:
    """Render ``error`` as an ``application/problem+json`` response.

    Args:
        error: The failure to report.
        request_id: Correlation id placed in ``instance`` and in ``X-Request-Id``.

    Returns:
        A :class:`~starlette.responses.JSONResponse` with the problem media type and, for a
        rate-limit error, a ``Retry-After`` header.
    """
    document = error.to_problem(instance=request_id)
    headers: dict[str, str] = {}
    if request_id:
        headers[REQUEST_ID_HEADER] = request_id
    if isinstance(error, RateLimitedError):
        headers["Retry-After"] = str(error.retry_after)
    if error.code is ErrorCode.AUTH_INVALID:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(
        status_code=error.http_status,
        content=document,
        media_type=PROBLEM_MEDIA_TYPE,
        headers=headers,
    )


def _request_id(request: Request) -> str | None:
    """Correlation id recorded by :class:`aicls.api.middleware.RequestContextMiddleware`."""
    value = getattr(request.state, "request_id", None)
    return str(value) if value else None


async def aicls_error_handler(request: Request, exc: Exception) -> Response:
    """Render any :class:`~aicls.core.errors.AiclsError` (the common case)."""
    if not isinstance(exc, AiclsError):  # pragma: no cover - registered for this type only
        return await unhandled_error_handler(request, exc)
    if exc.http_status >= 500:
        _logger.error("api.error", code=exc.code.value, path=request.url.path)
    else:
        _logger.info("api.error", code=exc.code.value, path=request.url.path)
    return problem_response(exc, request_id=_request_id(request))


async def validation_error_handler(request: Request, exc: Exception) -> Response:
    """Render a pydantic request-validation failure as ``VALIDATION_ERROR`` (422)."""
    if not isinstance(exc, RequestValidationError):  # pragma: no cover - type-guarded
        return await unhandled_error_handler(request, exc)
    entries = [
        {"loc": [str(part) for part in item.get("loc", ())], "msg": str(item.get("msg", ""))}
        for item in exc.errors()[:MAX_VALIDATION_ERRORS]
    ]
    error = AiclsError(
        ErrorCode.VALIDATION_ERROR, messages_tr.MSG_VALIDATION_ERROR, {"errors": entries}
    )
    return problem_response(error, request_id=_request_id(request))


async def http_exception_handler(request: Request, exc: Exception) -> Response:
    """Render a framework :class:`~starlette.exceptions.HTTPException` as a problem document."""
    if not isinstance(exc, StarletteHTTPException):  # pragma: no cover - type-guarded
        return await unhandled_error_handler(request, exc)
    code = STATUS_TO_CODE.get(exc.status_code)
    if code is None:
        code = ErrorCode.INTERNAL if exc.status_code >= 500 else ErrorCode.VALIDATION_ERROR
    detail = exc.detail if isinstance(exc.detail, str) and exc.detail else None
    error = AiclsError(
        code,
        messages_tr.msg(code),
        {"reason": detail} if detail else None,
        exc.status_code,
    )
    return problem_response(error, request_id=_request_id(request))


async def unhandled_error_handler(request: Request, exc: Exception) -> Response:
    """Last resort: never leak a traceback, always answer ``INTERNAL`` (500)."""
    _logger.error(
        "api.unhandled", path=request.url.path, kind=type(exc).__name__, error=str(exc)[:200]
    )
    error = AiclsError(ErrorCode.INTERNAL, messages_tr.MSG_INTERNAL, {"kind": type(exc).__name__})
    return problem_response(error, request_id=_request_id(request))


def install_exception_handlers(app: FastAPI) -> None:
    """Register every problem-details handler on ``app``."""
    app.add_exception_handler(AiclsError, aicls_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)


__all__ = [
    "MAX_VALIDATION_ERRORS",
    "PROBLEM_MEDIA_TYPE",
    "REQUEST_ID_HEADER",
    "STATUS_TO_CODE",
    "AuthError",
    "ForbiddenError",
    "RateLimitedError",
    "aicls_error_handler",
    "http_exception_handler",
    "install_exception_handlers",
    "problem_response",
    "unhandled_error_handler",
    "validation_error_handler",
]
