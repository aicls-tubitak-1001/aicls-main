"""FastAPI dependencies: database session, current user, role guard, shared services.

Everything a router needs comes from here, and everything here reads its collaborators off
``request.app.state`` rather than a module-level global. That is what lets a test build two
applications with two in-memory databases in the same process, and what makes
``test_api_rbac_matrix_every_router_every_role`` a matter of overriding one dependency.

Authorisation is two-layered, exactly as docs/guvenlik-gizlilik.md §1 requires: the *role* is
checked by :func:`require_role`, and *ownership* is checked in the repository layer with
``(resource_id, user_id)`` — never against an id the client sent.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from aicls.api.errors import AuthError, ForbiddenError
from aicls.api.model_registry import ModelRegistry
from aicls.api.pipeline import PipelineService
from aicls.api.security import WsTicketStore, decode_access_token
from aicls.core import messages_tr
from aicls.core.settings import Settings
from aicls.core.timing import StageTimer
from aicls.db.engine import session_scope
from aicls.db.enums import Role
from aicls.db.models import User
from aicls.db.repositories.identity import UserRepository

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aicls.api.ratelimit import RateLimiter

#: Prefix of the ``Authorization`` header value the API accepts.
BEARER_PREFIX = "bearer "


def get_settings_dep(request: Request) -> Settings:
    """The :class:`~aicls.core.settings.Settings` this application was built with."""
    settings: Settings = request.app.state.settings
    return settings


def get_db(request: Request) -> Iterator[Session]:
    """Yield a request-scoped session on this application's engine (commit on success)."""
    with session_scope(request.app.state.engine) as session:
        yield session


def get_pipeline(request: Request) -> PipelineService:
    """The shared :class:`~aicls.api.pipeline.PipelineService`."""
    service: PipelineService = request.app.state.pipeline
    return service


def get_registry(request: Request) -> ModelRegistry:
    """The shared :class:`~aicls.api.model_registry.ModelRegistry`."""
    registry: ModelRegistry = request.app.state.registry
    return registry


def get_tickets(request: Request) -> WsTicketStore:
    """The shared WebSocket ticket store."""
    store: WsTicketStore = request.app.state.tickets
    return store


def get_limiter(request: Request) -> RateLimiter:
    """The shared rate limiter (exposed so admin tooling can inspect it)."""
    limiter: RateLimiter = request.app.state.limiter
    return limiter


def get_timer(request: Request) -> StageTimer:
    """The request's :class:`~aicls.core.timing.StageTimer` (``Server-Timing`` source)."""
    timer = getattr(request.state, "timer", None)
    if isinstance(timer, StageTimer):
        return timer
    fresh = StageTimer()
    request.state.timer = fresh
    return fresh


def bearer_token(request: Request) -> str | None:
    """Extract the bearer token from the ``Authorization`` header, or ``None``."""
    header = request.headers.get("Authorization", "")
    if not header.lower().startswith(BEARER_PREFIX):
        return None
    value = header[len(BEARER_PREFIX) :].strip()
    return value or None


def user_from_token(token: str, session: Session, secret: str) -> User:
    """Resolve an access token to its (active) user.

    Raises:
        AuthError: If the token is invalid or the account is missing or disabled.
    """
    claims = decode_access_token(token, secret)
    user = UserRepository(session).get(str(claims["sub"]))
    if user is None or not user.is_active:
        raise AuthError(details={"reason": "user_inactive"})
    return user


def get_current_user(
    request: Request,
    token: Annotated[str | None, Depends(bearer_token)] = None,
    session: Annotated[Session, Depends(get_db)] = None,  # type: ignore[assignment]  # filled by FastAPI
) -> User:
    """The signed-in user.

    Raises:
        AuthError: If no bearer token was sent, or it is invalid or expired.
    """
    if not token:
        raise AuthError(messages_tr.MSG_AUTH_INVALID, details={"reason": "missing_token"})
    return user_from_token(token, session, request.app.state.jwt_secret)


def get_optional_user(
    request: Request,
    token: Annotated[str | None, Depends(bearer_token)] = None,
    session: Annotated[Session, Depends(get_db)] = None,  # type: ignore[assignment]  # filled by FastAPI
) -> User | None:
    """The signed-in user, or ``None`` for an anonymous caller (open endpoints)."""
    if not token:
        return None
    try:
        return user_from_token(token, session, request.app.state.jwt_secret)
    except AuthError:
        return None


def require_role(*roles: Role | str) -> Callable[[User], User]:
    """Build a dependency that admits only the listed roles (``admin`` always passes).

    Args:
        *roles: Role names or :class:`~aicls.db.enums.Role` members.

    Returns:
        A dependency returning the current user.

    Example:
        ``@router.get("/stats", dependencies=[Depends(require_role("instructor"))])``
    """
    allowed = {Role(str(getattr(role, "value", role))) for role in roles} | {Role.ADMIN}

    def dependency(user: Annotated[User, Depends(get_current_user)]) -> User:
        """Return ``user`` when their role is allowed."""
        if user.role not in allowed:
            raise ForbiddenError(
                messages_tr.MSG_FORBIDDEN,
                details={
                    "required": sorted(role.value for role in allowed),
                    "role": user.role.value,
                },
            )
        return user

    return dependency


#: Common annotated dependencies, so routers read as one line per parameter.
CurrentUser = Annotated[User, Depends(get_current_user)]
OptionalUser = Annotated[User | None, Depends(get_optional_user)]
AdminUser = Annotated[User, Depends(require_role(Role.ADMIN))]
StaffUser = Annotated[User, Depends(require_role(Role.INSTRUCTOR))]
DbSession = Annotated[Session, Depends(get_db)]
Pipeline = Annotated[PipelineService, Depends(get_pipeline)]
Registry = Annotated[ModelRegistry, Depends(get_registry)]
Tickets = Annotated[WsTicketStore, Depends(get_tickets)]
AppSettings = Annotated[Settings, Depends(get_settings_dep)]
Timer = Annotated[StageTimer, Depends(get_timer)]


__all__ = [
    "BEARER_PREFIX",
    "AdminUser",
    "AppSettings",
    "CurrentUser",
    "DbSession",
    "OptionalUser",
    "Pipeline",
    "Registry",
    "StaffUser",
    "Tickets",
    "Timer",
    "bearer_token",
    "get_current_user",
    "get_db",
    "get_limiter",
    "get_optional_user",
    "get_pipeline",
    "get_registry",
    "get_settings_dep",
    "get_tickets",
    "get_timer",
    "require_role",
    "user_from_token",
]
