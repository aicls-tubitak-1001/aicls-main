"""Registration, login, refresh rotation, logout and WebSocket tickets (api.md §3.2).

The refresh flow is the security-sensitive part and follows docs/guvenlik-gizlilik.md §1 exactly:
the cookie holds an opaque secret, the database holds only its SHA-256 digest, every use mints a
replacement in the same *family*, and presenting an already-rotated token revokes the whole
family — the standard answer to a stolen refresh cookie, because the thief and the victim cannot
both keep using the chain without one of them replaying.
"""

from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Request, Response, status

from aicls.api.deps import CurrentUser, DbSession, Tickets, get_settings_dep
from aicls.api.errors import AuthError
from aicls.api.schemas import LoginIn, RegisterIn, TokenOut, UserOut, WsTicketOut
from aicls.api.security import (
    ACCESS_TTL_S,
    LOCKOUT_MINUTES,
    MAX_FAILED_LOGINS,
    REFRESH_COOKIE,
    REFRESH_TTL_S,
    RefreshTokenRepository,
    create_access_token,
    hash_password,
    hash_token,
    new_refresh_token,
    verify_password,
)
from aicls.collect.service import hash_ip
from aicls.core import messages_tr
from aicls.core.errors import InputValidationError
from aicls.core.logging import get_logger
from aicls.db.base import utcnow
from aicls.db.enums import Role
from aicls.db.models import User
from aicls.db.repositories.identity import UserRepository, normalise_email

router = APIRouter(prefix="/auth", tags=["auth"])

_logger = get_logger(__name__)


def _client_ip_hash(request: Request) -> str:
    """Salted hash of the client address (a raw address never reaches the database)."""
    host = request.client.host if request.client else ""
    if not host:
        return ""
    return hash_ip(host) or ""


def _set_refresh_cookie(request: Request, response: Response, raw: str) -> None:
    """Write the ``aicls_refresh`` cookie (httpOnly, SameSite=Strict, Secure in prod)."""
    settings = get_settings_dep(request)
    response.set_cookie(
        REFRESH_COOKIE,
        raw,
        max_age=REFRESH_TTL_S,
        httponly=True,
        secure=settings.is_prod,
        samesite="strict",
        path="/api/v1/auth",
    )


def _issue(request: Request, response: Response, session: DbSession, user: User) -> TokenOut:
    """Mint an access token and a fresh refresh-token family for ``user``."""
    raw, digest = new_refresh_token()
    RefreshTokenRepository(session).create(user.id, digest, ip_hash=_client_ip_hash(request))
    _set_refresh_cookie(request, response, raw)
    token = create_access_token(
        user_id=user.id, role=user.role.value, secret=request.app.state.jwt_secret
    )
    return TokenOut(access_token=token, token_type="bearer", expires_in=ACCESS_TTL_S)


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(request: Request, body: RegisterIn, session: DbSession) -> User:
    """Create an account (``student`` by default, ``instructor`` with a valid invite code).

    Raises:
        InputValidationError: If the address is already registered.
    """
    repo = UserRepository(session)
    email = normalise_email(body.email)
    if repo.get_by_email(email) is not None:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR, details={"field": "email", "reason": "taken"}
        )
    role = Role.STUDENT
    expected = str(request.app.state.invite_code or "")
    if body.invite_code and expected and body.invite_code == expected:
        role = Role.INSTRUCTOR
    user = repo.create(email, hash_password(body.password), body.display_name, role)
    _logger.info("api.auth.registered", user_id=user.id, role=role.value)
    return user


@router.post("/login", response_model=TokenOut)
def login(request: Request, response: Response, body: LoginIn, session: DbSession) -> TokenOut:
    """Exchange credentials for an access token plus a refresh cookie.

    Ten consecutive failures lock the account for fifteen minutes
    (``test_api_login_lockout_after_10_failures``).

    Raises:
        AuthError: On unknown address, wrong password, locked or disabled account. The Turkish
            text is identical in every case so an attacker cannot enumerate accounts.
    """
    repo = UserRepository(session)
    user = repo.get_by_email(body.email)
    now = utcnow()
    if user is None or not user.is_active:
        raise AuthError(messages_tr.MSG_AUTH_INVALID, details={"reason": "unknown_or_disabled"})
    if user.locked_until is not None and user.locked_until > now:
        raise AuthError(messages_tr.MSG_AUTH_INVALID, details={"reason": "locked"})
    if not verify_password(user.password_hash, body.password):
        user.failed_logins += 1
        if user.failed_logins >= MAX_FAILED_LOGINS:
            user.locked_until = now + timedelta(minutes=LOCKOUT_MINUTES)
            _logger.warning("api.auth.locked", user_id=user.id, minutes=LOCKOUT_MINUTES)
        # Committed explicitly: the request is about to fail, and `session_scope` rolls back on
        # an exception — a lockout counter that a wrong password erases is not a lockout.
        session.commit()
        raise AuthError(messages_tr.MSG_AUTH_INVALID, details={"reason": "bad_password"})
    user.failed_logins = 0
    user.locked_until = None
    user.last_login_at = now
    session.flush()
    return _issue(request, response, session, user)


@router.post("/refresh", response_model=TokenOut)
def refresh(request: Request, response: Response, session: DbSession) -> TokenOut:
    """Rotate the refresh cookie and mint a new access token.

    Raises:
        AuthError: If the cookie is missing, unknown, expired or already rotated. A replayed
            token additionally revokes every live token of its family.
    """
    raw = request.cookies.get(REFRESH_COOKIE, "")
    if not raw:
        raise AuthError(messages_tr.MSG_AUTH_INVALID, details={"reason": "missing_cookie"})
    repo = RefreshTokenRepository(session)
    row = repo.by_hash(hash_token(raw))
    if row is None:
        raise AuthError(messages_tr.MSG_AUTH_INVALID, details={"reason": "unknown_refresh"})
    if row.revoked_at is not None:
        revoked = repo.revoke_family(row.family_id)
        # Same reason as the lockout counter: the family revocation must survive the 401.
        session.commit()
        _logger.warning("api.auth.refresh_reuse", user_id=row.user_id, revoked=revoked)
        raise AuthError(messages_tr.MSG_AUTH_INVALID, details={"reason": "reused_refresh"})
    if row.expires_at <= utcnow():
        raise AuthError(messages_tr.MSG_AUTH_INVALID, details={"reason": "expired_refresh"})
    user = UserRepository(session).get(row.user_id)
    if user is None or not user.is_active:
        raise AuthError(messages_tr.MSG_AUTH_INVALID, details={"reason": "user_inactive"})
    replacement, _row = repo.rotate(row, ip_hash=_client_ip_hash(request))
    _set_refresh_cookie(request, response, replacement)
    token = create_access_token(
        user_id=user.id, role=user.role.value, secret=request.app.state.jwt_secret
    )
    return TokenOut(access_token=token, token_type="bearer", expires_in=ACCESS_TTL_S)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request, response: Response, session: DbSession) -> None:
    """Revoke the refresh family and clear the cookie; always answers 204."""
    raw = request.cookies.get(REFRESH_COOKIE, "")
    if raw:
        repo = RefreshTokenRepository(session)
        row = repo.by_hash(hash_token(raw))
        if row is not None:
            repo.revoke_family(row.family_id)
    response.delete_cookie(REFRESH_COOKIE, path="/api/v1/auth")


@router.post("/ws-ticket", response_model=WsTicketOut)
def ws_ticket(user: CurrentUser, tickets: Tickets) -> WsTicketOut:
    """Mint a single-use 60 s ticket for ``/ws/v1/ink`` and ``/ws/v1/audio`` (api.md §1)."""
    value, expires_in = tickets.issue(user.id, user.role.value)
    return WsTicketOut(ticket=value, expires_in=expires_in)


__all__ = ["router"]
