"""Passwords, JWT access tokens, rotating refresh tokens and WebSocket tickets.

The parameters are the ones docs/guvenlik-gizlilik.md §1 fixes:

* **passwords** — argon2id, ``time_cost=3``, ``memory_cost=64 MiB``, ``parallelism=2``, at least
  10 characters, 15-minute lockout after 10 failed attempts;
* **access token** — PyJWT HS256, 15 minutes, claims ``sub`` (user id), ``role``, ``jti``;
* **refresh token** — an opaque 256-bit secret stored only as a SHA-256 digest, 7 days, rotated
  on every use; replaying a rotated token revokes the whole family;
* **WebSocket ticket** — single-use, 60 seconds, kept in memory, so a JWT never appears in a URL
  and therefore never in a server or proxy log.

Nothing here touches the request object: the module is pure so the security rules can be unit
tested without an application (``tests/unit/api/test_security.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import secrets
import time
from typing import Any, Final

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
import jwt
from sqlalchemy import select

from aicls.api.errors import AuthError
from aicls.core import messages_tr
from aicls.core.errors import InputValidationError
from aicls.core.ids import ulid
from aicls.core.settings import Settings
from aicls.db.base import utcnow
from aicls.db.models import RefreshToken
from aicls.db.repositories.base import Repository

# --------------------------------------------------------------------------------------------
# constants (docs/guvenlik-gizlilik.md §1)
# --------------------------------------------------------------------------------------------
#: argon2id time cost (iterations).
ARGON2_TIME_COST: Final = 3
#: argon2id memory cost in KiB (64 MiB).
ARGON2_MEMORY_COST: Final = 65536
#: argon2id lanes.
ARGON2_PARALLELISM: Final = 2
#: argon2id output length in bytes.
ARGON2_HASH_LEN: Final = 32
#: argon2id salt length in bytes.
ARGON2_SALT_LEN: Final = 16

#: Reduced argon2id cost used **only** under ``AICLS_ENV=test``. The production parameters above
#: cost ~40 ms per hash, and the RBAC matrix alone needs well over a thousand of them; a suite
#: that takes a minute to hash passwords stops being run. Verification reads its parameters from
#: the stored hash, so nothing else changes, and ``test_password_hasher_uses_production_cost``
#: pins the production numbers.
ARGON2_TEST_TIME_COST: Final = 1
ARGON2_TEST_MEMORY_COST: Final = 8192
ARGON2_TEST_PARALLELISM: Final = 1

#: Shortest accepted password.
MIN_PASSWORD_LENGTH: Final = 10
#: Longest accepted password (argon2 is happy with more; this bounds the request body).
MAX_PASSWORD_LENGTH: Final = 128
#: Failed logins that trigger a lockout.
MAX_FAILED_LOGINS: Final = 10
#: Lockout duration in minutes.
LOCKOUT_MINUTES: Final = 15

#: Clock skew tolerated when validating ``iat``/``nbf``/``exp`` (seconds).
#:
#: A host clock that steps **backwards** makes a token minted a moment ago look as if it were
#: issued in the future, and PyJWT >= 2.10 rejects that with ``ImmatureSignatureError``. Measured
#: inside the WSL2 container on 2026-09-08: ``api.auth.registered`` at ``05:04:58.600Z`` followed
#: by the very next request at ``05:04:57.698Z`` -- a 0.9 s step back -- produced a spurious 401
#: on a freshly issued token. NTP does this on virtual machines, so a school pilot would see real
#: users signed out at random. The cost is stated plainly: an expired token also stays acceptable
#: for this long, which is 3.3 % of the 900 s access-token lifetime.
CLOCK_SKEW_LEEWAY_S: Final = 30

#: Access-token lifetime in seconds (``TokenOut.expires_in``).
ACCESS_TTL_S: Final = 900
#: Refresh-token lifetime in seconds.
REFRESH_TTL_S: Final = 7 * 24 * 3600
#: WebSocket ticket lifetime in seconds.
WS_TICKET_TTL_S: Final = 60

#: Name of the refresh cookie.
REFRESH_COOKIE = "aicls_refresh"
#: JWT signing algorithm.
JWT_ALGORITHM = "HS256"
#: ``typ`` claim of an access token.
ACCESS_TOKEN_TYPE = "access"

_hashers: dict[str, PasswordHasher] = {}


def argon2_parameters(env: str) -> tuple[int, int, int]:
    """Return ``(time_cost, memory_cost, parallelism)`` for ``env``."""
    if env == "test":
        return ARGON2_TEST_TIME_COST, ARGON2_TEST_MEMORY_COST, ARGON2_TEST_PARALLELISM
    return ARGON2_TIME_COST, ARGON2_MEMORY_COST, ARGON2_PARALLELISM


def password_hasher(env: str | None = None) -> PasswordHasher:
    """Return the cached argon2id hasher for ``env`` (``settings.env`` when omitted)."""
    from aicls.core.settings import get_settings

    resolved = env if env is not None else get_settings().env
    hasher = _hashers.get(resolved)
    if hasher is None:
        time_cost, memory_cost, parallelism = argon2_parameters(resolved)
        hasher = PasswordHasher(
            time_cost=time_cost,
            memory_cost=memory_cost,
            parallelism=parallelism,
            hash_len=ARGON2_HASH_LEN,
            salt_len=ARGON2_SALT_LEN,
        )
        _hashers[resolved] = hasher
    return hasher


def check_password_policy(password: str) -> str:
    """Return ``password`` if it satisfies the policy.

    Raises:
        InputValidationError: If it is shorter than :data:`MIN_PASSWORD_LENGTH` or longer than
            :data:`MAX_PASSWORD_LENGTH`.
    """
    if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={
                "field": "password",
                "min_length": MIN_PASSWORD_LENGTH,
                "max_length": MAX_PASSWORD_LENGTH,
            },
        )
    return password


def hash_password(password: str) -> str:
    """Hash a password with argon2id (policy-checked first)."""
    return password_hasher().hash(check_password_policy(password))


def verify_password(stored_hash: str, password: str) -> bool:
    """Whether ``password`` matches ``stored_hash``; never raises for a wrong password."""
    try:
        return bool(password_hasher().verify(stored_hash, password))
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(stored_hash: str) -> bool:
    """Whether ``stored_hash`` was produced with weaker parameters than the current policy."""
    try:
        return bool(password_hasher().check_needs_rehash(stored_hash))
    except InvalidHashError:  # pragma: no cover - only for a corrupted column
        return True


def jwt_secret(settings: Settings) -> str:
    """Return the HMAC signing secret.

    ``AICLS_JWT_SECRET`` may be empty in ``dev``/``test`` (the settings model only enforces the
    32-character minimum in ``prod``), so a per-process random secret is generated in that case:
    tokens then stop working on restart, which is the correct behaviour for an unconfigured
    development host and impossible to mistake for a production configuration.
    """
    configured = settings.jwt_secret.get_secret_value()
    if configured:
        return configured
    return _fallback_secret()


_fallback: str | None = None


def _fallback_secret() -> str:
    global _fallback  # noqa: PLW0603 - one ephemeral secret per process
    if _fallback is None:
        _fallback = secrets.token_hex(32)
    return _fallback


def create_access_token(
    *, user_id: str, role: str, secret: str, ttl_s: int = ACCESS_TTL_S, now: float | None = None
) -> str:
    """Mint an HS256 access token carrying ``sub``, ``role`` and ``jti`` (api.md §1)."""
    issued = int(now if now is not None else time.time())
    payload = {
        "sub": user_id,
        "role": role,
        "jti": ulid(),
        "typ": ACCESS_TOKEN_TYPE,
        "iat": issued,
        "exp": issued + int(ttl_s),
    }
    return str(jwt.encode(payload, secret, algorithm=JWT_ALGORITHM))


def decode_access_token(token: str, secret: str) -> dict[str, Any]:
    """Verify and decode an access token.

    ``leeway`` is :data:`CLOCK_SKEW_LEEWAY_S`, so a host clock that steps backwards does not
    invalidate a token that was just issued. Signature verification is unaffected.

    Raises:
        AuthError: If the token is malformed, expired or not an access token.
    """
    try:
        claims: dict[str, Any] = jwt.decode(
            token, secret, algorithms=[JWT_ALGORITHM], leeway=CLOCK_SKEW_LEEWAY_S
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError(details={"reason": "expired"}) from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError(details={"reason": "invalid_token"}) from exc
    if claims.get("typ") != ACCESS_TOKEN_TYPE or not claims.get("sub"):
        raise AuthError(details={"reason": "wrong_token_type"})
    return claims


def peek_subject(token: str, secret: str) -> str | None:
    """Return the ``sub`` claim of a *valid* token, or ``None``.

    Used by the rate-limit middleware, which must not raise before the router runs.
    """
    try:
        return str(decode_access_token(token, secret)["sub"])
    except AuthError:
        return None


def new_refresh_token() -> tuple[str, str]:
    """Return ``(raw token, digest)``; only the digest is ever stored."""
    raw = secrets.token_urlsafe(32)
    return raw, hash_token(raw)


def hash_token(raw: str) -> str:
    """SHA-256 hex digest of an opaque token (``refresh_tokens.token_hash``)."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class RefreshTokenRepository(Repository):
    """Queries over ``refresh_tokens``.

    :mod:`aicls.db` ships no repository for this table (it has none in ``modules/db.json``), so
    the API owns one — still a :class:`aicls.db.repositories.base.Repository` subclass, so the
    "repositories are the only place that builds queries" rule of ADR-0003 holds.
    """

    def create(
        self,
        user_id: str,
        token_hash: str,
        *,
        family_id: str | None = None,
        ttl_s: int = REFRESH_TTL_S,
        ip_hash: str = "",
    ) -> RefreshToken:
        """Insert one refresh token and return it."""
        row = RefreshToken(
            user_id=user_id,
            token_hash=token_hash,
            family_id=family_id or ulid(),
            expires_at=utcnow() + timedelta(seconds=ttl_s),
            ip_hash=ip_hash,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def by_hash(self, token_hash: str) -> RefreshToken | None:
        """Return the row with this digest, or ``None``."""
        stmt = select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        return self.session.scalars(stmt).one_or_none()

    def family(self, family_id: str) -> list[RefreshToken]:
        """Every token of one login family, oldest first."""
        stmt = (
            select(RefreshToken)
            .where(RefreshToken.family_id == family_id)
            .order_by(RefreshToken.id)
        )
        return list(self.session.scalars(stmt))

    def revoke_family(self, family_id: str, *, when: datetime | None = None) -> int:
        """Revoke every live token of a family; return how many were revoked."""
        moment = when or utcnow()
        revoked = 0
        for row in self.family(family_id):
            if row.revoked_at is None:
                row.revoked_at = moment
                revoked += 1
        self.session.flush()
        return revoked

    def revoke_for_user(self, user_id: str) -> int:
        """Revoke every live token of a user (logout everywhere)."""
        stmt = select(RefreshToken).where(
            RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None)
        )
        moment = utcnow()
        revoked = 0
        for row in self.session.scalars(stmt):
            row.revoked_at = moment
            revoked += 1
        self.session.flush()
        return revoked

    def rotate(self, current: RefreshToken, *, ip_hash: str = "") -> tuple[str, RefreshToken]:
        """Replace ``current`` with a fresh token of the same family.

        Returns:
            ``(raw token, stored row)`` — the raw value goes into the cookie and is never stored.
        """
        raw, digest = new_refresh_token()
        replacement = self.create(
            current.user_id, digest, family_id=current.family_id, ip_hash=ip_hash
        )
        current.revoked_at = utcnow()
        current.replaced_by = replacement.id
        self.session.flush()
        return raw, replacement


@dataclass(frozen=True, slots=True)
class Ticket:
    """A redeemed WebSocket ticket."""

    user_id: str
    role: str
    expires_at: float


class WsTicketStore:
    """Single-use, short-lived WebSocket tickets held in memory (api.md §1).

    Args:
        ttl_s: Ticket lifetime in seconds.
        clock: Monotonic seconds source; injectable for tests.
    """

    def __init__(self, *, ttl_s: int = WS_TICKET_TTL_S, clock: Any = None) -> None:
        self._ttl_s = int(ttl_s)
        self._clock = clock if clock is not None else time.monotonic
        self._tickets: dict[str, Ticket] = {}

    @property
    def ttl_s(self) -> int:
        """Ticket lifetime in seconds."""
        return self._ttl_s

    def issue(self, user_id: str, role: str) -> tuple[str, int]:
        """Mint a ticket for ``user_id`` and return ``(ticket, expires_in)``."""
        self._purge()
        value = secrets.token_urlsafe(24)
        self._tickets[value] = Ticket(user_id, role, self._clock() + self._ttl_s)
        return value, self._ttl_s

    def consume(self, value: str) -> Ticket:
        """Redeem a ticket exactly once.

        Raises:
            AuthError: If the ticket is unknown, already used or expired.
        """
        self._purge()
        ticket = self._tickets.pop(value, None)
        if ticket is None:
            raise AuthError(details={"reason": "ticket_invalid"})
        if ticket.expires_at < self._clock():
            raise AuthError(details={"reason": "ticket_expired"})
        return ticket

    def _purge(self) -> None:
        now = self._clock()
        for value, ticket in list(self._tickets.items()):
            if ticket.expires_at < now:
                del self._tickets[value]

    @property
    def size(self) -> int:
        """Number of live tickets."""
        return len(self._tickets)


__all__ = [
    "ACCESS_TOKEN_TYPE",
    "ACCESS_TTL_S",
    "ARGON2_HASH_LEN",
    "ARGON2_MEMORY_COST",
    "ARGON2_PARALLELISM",
    "ARGON2_SALT_LEN",
    "ARGON2_TEST_MEMORY_COST",
    "ARGON2_TEST_PARALLELISM",
    "ARGON2_TEST_TIME_COST",
    "ARGON2_TIME_COST",
    "JWT_ALGORITHM",
    "LOCKOUT_MINUTES",
    "MAX_FAILED_LOGINS",
    "MAX_PASSWORD_LENGTH",
    "MIN_PASSWORD_LENGTH",
    "REFRESH_COOKIE",
    "REFRESH_TTL_S",
    "WS_TICKET_TTL_S",
    "RefreshTokenRepository",
    "Ticket",
    "WsTicketStore",
    "argon2_parameters",
    "check_password_policy",
    "create_access_token",
    "decode_access_token",
    "hash_password",
    "hash_token",
    "jwt_secret",
    "needs_rehash",
    "new_refresh_token",
    "password_hasher",
    "peek_subject",
    "verify_password",
]
