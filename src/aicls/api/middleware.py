"""Request-scoped middleware: correlation id, ``Server-Timing``, security headers, rate limit.

Order matters and is fixed in :func:`aicls.api.app.create_app`; from the outside in a request
passes CORS, then the security headers, then the rate limiter, then the request context. The
context middleware is innermost so that the ``StageTimer`` it installs is already there when a
router adds ``nlu``/``parse``/``cas``/``render`` stages, and so that every response — including
the problem documents produced by the exception handlers — leaves with ``X-Request-Id`` and
``Server-Timing`` (ADR-0014).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import re

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from aicls.api.errors import REQUEST_ID_HEADER, RateLimitedError, problem_response
from aicls.api.ratelimit import RateLimiter, RateRule, rules_for
from aicls.core.ids import ulid
from aicls.core.logging import request_context
from aicls.core.timing import StageTimer

#: Header the ``Server-Timing`` metrics are written to (RFC 9211).
SERVER_TIMING_HEADER = "Server-Timing"

#: Security headers of docs/guvenlik-gizlilik.md §4. ``style-src 'unsafe-inline'`` is required by
#: KaTeX, which writes inline styles for every rendered formula.
SECURITY_HEADERS: dict[str, str] = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; font-src 'self'; "
        "style-src 'self' 'unsafe-inline'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "microphone=(self)",
}

#: Shape a client-supplied ``X-Request-Id`` must have to be echoed back (header injection guard).
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{8,64}$")

Dispatch = Callable[[Request], Awaitable[Response]]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id and a :class:`~aicls.core.timing.StageTimer` to every request."""

    async def dispatch(self, request: Request, call_next: Dispatch) -> Response:
        """Install ``request.state.request_id`` / ``request.state.timer`` and stamp the response."""
        supplied = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = supplied if _REQUEST_ID_RE.match(supplied) else ulid()
        timer = StageTimer()
        request.state.request_id = request_id
        request.state.timer = timer
        with request_context(request_id):
            response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[SERVER_TIMING_HEADER] = timer.server_timing_header()
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add the fixed security headers to every response (docs/guvenlik-gizlilik.md §4)."""

    async def dispatch(self, request: Request, call_next: Dispatch) -> Response:
        """Merge :data:`SECURITY_HEADERS` into the response, never overriding a router's value."""
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply the api.md §1 rate limits before a router or a database session is involved.

    Args:
        app: The wrapped ASGI application.
        limiter: Shared :class:`~aicls.api.ratelimit.RateLimiter`.
        secret: JWT secret, used to read ``sub`` out of a bearer token for ``user``-scoped rules.
        prefix: API path prefix stripped before the rules are matched (``"/api/v1"``).
        enabled: ``False`` disables every rule (``AICLS_RATE_LIMIT_ENABLED=false``).
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        limiter: RateLimiter,
        secret: str,
        prefix: str = "",
        enabled: bool = True,
    ) -> None:
        super().__init__(app)
        self._limiter = limiter
        self._secret = secret
        self._prefix = prefix
        self._enabled = enabled

    async def dispatch(self, request: Request, call_next: Dispatch) -> Response:
        """Refuse with 429 + ``Retry-After`` when a bucket is empty."""
        if not self._enabled:
            return await call_next(request)
        path = request.url.path
        if self._prefix and path.startswith(self._prefix):
            path = path[len(self._prefix) :] or "/"
        rules = rules_for(path)
        if rules:
            error = self._exceeded(request, rules)
            if error is not None:
                return problem_response(
                    error, request_id=getattr(request.state, "request_id", None)
                )
        return await call_next(request)

    def _exceeded(self, request: Request, rules: tuple[RateRule, ...]) -> RateLimitedError | None:
        """Return the error for the first exhausted rule, or ``None``."""
        for rule in rules:
            identity = self._identity(request, rule)
            retry_after = self._limiter.check(rule, identity)
            if retry_after is not None:
                return RateLimitedError(
                    max(1, int(retry_after) + 1),
                    details={"rule": rule.name, "limit": rule.limit, "window_s": rule.window_s},
                )
        return None

    def _identity(self, request: Request, rule: RateRule) -> str:
        """Bucket identity: the client address, or the token subject for ``user`` rules."""
        client = request.client.host if request.client else "unknown"
        if rule.scope == "ip":
            return client
        header = request.headers.get("Authorization", "")
        if header.lower().startswith("bearer "):
            from aicls.api.security import peek_subject

            subject = peek_subject(header[7:].strip(), self._secret)
            if subject:
                return subject
        return client


__all__ = [
    "SECURITY_HEADERS",
    "SERVER_TIMING_HEADER",
    "RateLimitMiddleware",
    "RequestContextMiddleware",
    "SecurityHeadersMiddleware",
]
