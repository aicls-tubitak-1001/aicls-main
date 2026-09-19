"""Unit tests for :mod:`aicls.api.security` (docs/guvenlik-gizlilik.md §1)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
import time

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from aicls.api.errors import AuthError
from aicls.api.security import (
    ACCESS_TOKEN_TYPE,
    ACCESS_TTL_S,
    ARGON2_MEMORY_COST,
    ARGON2_PARALLELISM,
    ARGON2_TIME_COST,
    CLOCK_SKEW_LEEWAY_S,
    MAX_PASSWORD_LENGTH,
    MIN_PASSWORD_LENGTH,
    REFRESH_TTL_S,
    RefreshTokenRepository,
    WsTicketStore,
    argon2_parameters,
    check_password_policy,
    create_access_token,
    decode_access_token,
    hash_password,
    hash_token,
    jwt_secret,
    new_refresh_token,
    password_hasher,
    peek_subject,
    verify_password,
)
from aicls.core.errors import InputValidationError
from aicls.core.settings import Settings
from aicls.db import UserRepository, make_engine, session_scope, upgrade_head
from aicls.db.base import utcnow

SECRET = "0123456789abcdef" * 4


@pytest.fixture
def engine() -> Iterator[Engine]:
    """A migrated in-memory database."""
    created = make_engine("sqlite+pysqlite:///:memory:")
    upgrade_head(created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """An open session on the migrated database."""
    with session_scope(engine) as opened:
        yield opened


# ---------------------------------------------------------------------------------------------
# passwords
# ---------------------------------------------------------------------------------------------
def test_password_hasher_uses_the_documented_production_cost() -> None:
    """The production parameters are the ones docs/guvenlik-gizlilik.md §1 fixes."""
    assert argon2_parameters("prod") == (ARGON2_TIME_COST, ARGON2_MEMORY_COST, ARGON2_PARALLELISM)
    assert argon2_parameters("dev") == (ARGON2_TIME_COST, ARGON2_MEMORY_COST, ARGON2_PARALLELISM)
    assert (ARGON2_TIME_COST, ARGON2_MEMORY_COST, ARGON2_PARALLELISM) == (3, 65536, 2)
    hasher = password_hasher("prod")
    assert hasher.time_cost == 3
    assert hasher.memory_cost == 65536
    assert hasher.parallelism == 2


def test_password_hasher_uses_a_cheaper_cost_only_in_tests() -> None:
    """``AICLS_ENV=test`` gets a cheaper cost; every other environment does not."""
    test_params = argon2_parameters("test")
    assert test_params != argon2_parameters("prod")
    assert test_params[1] < ARGON2_MEMORY_COST


def test_hash_and_verify_round_trip() -> None:
    """A hash verifies its own password and refuses any other."""
    stored = hash_password("Kalkulus2026!")
    assert stored.startswith("$argon2id$")
    assert verify_password(stored, "Kalkulus2026!") is True
    assert verify_password(stored, "BaskaParola1") is False
    assert verify_password("bozuk-hash", "Kalkulus2026!") is False


def test_hashes_are_salted() -> None:
    """Two hashes of the same password differ (a per-password salt, not a global one)."""
    assert hash_password("Kalkulus2026!") != hash_password("Kalkulus2026!")


@pytest.mark.parametrize("password", ["", "kisa", "a" * (MIN_PASSWORD_LENGTH - 1)])
def test_password_policy_rejects_short_passwords(password: str) -> None:
    """The 10-character minimum is enforced in one place."""
    with pytest.raises(InputValidationError) as failure:
        check_password_policy(password)
    assert failure.value.details["field"] == "password"


def test_password_policy_rejects_an_absurdly_long_password() -> None:
    """A very long password is refused before argon2 is asked to hash it."""
    with pytest.raises(InputValidationError):
        check_password_policy("a" * (MAX_PASSWORD_LENGTH + 1))


# ---------------------------------------------------------------------------------------------
# access tokens
# ---------------------------------------------------------------------------------------------
def test_access_token_carries_sub_role_and_jti() -> None:
    """The claims are the ones api.md §1 names."""
    token = create_access_token(user_id="01J", role="instructor", secret=SECRET)
    claims = decode_access_token(token, SECRET)
    assert claims["sub"] == "01J"
    assert claims["role"] == "instructor"
    assert claims["typ"] == ACCESS_TOKEN_TYPE
    assert claims["jti"]
    assert claims["exp"] > claims["iat"]


def test_access_token_rejects_another_secret_and_an_expired_token() -> None:
    """A forged or expired token never decodes."""
    token = create_access_token(user_id="01J", role="student", secret=SECRET)
    with pytest.raises(AuthError):
        decode_access_token(token, "baska-sir")

    stale = create_access_token(
        user_id="01J", role="student", secret=SECRET, ttl_s=1, now=1_600_000_000
    )
    with pytest.raises(AuthError) as expired:
        decode_access_token(stale, SECRET)
    assert expired.value.details["reason"] == "expired"


def test_a_backwards_clock_step_does_not_invalidate_a_fresh_token() -> None:
    """A host clock that jumps back must not sign a user out.

    Measured in the WSL2 container on 2026-09-08: ``api.auth.registered`` at ``05:04:58.600Z``
    followed by the very next request at ``05:04:57.698Z``. PyJWT >= 2.10 rejects an ``iat`` in
    the future, so a token minted 0.9 s earlier answered 401 with ``reason=invalid_token``. NTP
    steps virtual-machine clocks like this, so it is a production hazard and not a test flake.
    """
    now = time.time()
    fresh = create_access_token(user_id="01J", role="student", secret=SECRET, now=now + 5)
    claims = decode_access_token(fresh, SECRET)
    assert claims["sub"] == "01J"

    beyond = create_access_token(
        user_id="01J", role="student", secret=SECRET, now=now + CLOCK_SKEW_LEEWAY_S + 120
    )
    # the tolerance is bounded, not unlimited
    with pytest.raises(AuthError):
        decode_access_token(beyond, SECRET)


def test_the_leeway_is_small_against_the_token_lifetime() -> None:
    """The skew tolerance also extends expiry, so it must stay a rounding error of the TTL."""
    assert 0 < CLOCK_SKEW_LEEWAY_S <= ACCESS_TTL_S // 10


def test_peek_subject_never_raises() -> None:
    """The rate limiter reads the subject without being able to fail the request."""
    token = create_access_token(user_id="01J", role="student", secret=SECRET)
    assert peek_subject(token, SECRET) == "01J"
    assert peek_subject("cop", SECRET) is None
    assert peek_subject(token, "baska") is None


def test_jwt_secret_falls_back_to_an_ephemeral_value() -> None:
    """An unconfigured development host still works, with a per-process secret."""
    configured = jwt_secret(Settings(jwt_secret=SECRET))
    assert configured == SECRET
    generated = jwt_secret(Settings(env="dev"))
    assert len(generated) >= 32
    assert jwt_secret(Settings(env="dev")) == generated


# ---------------------------------------------------------------------------------------------
# refresh tokens
# ---------------------------------------------------------------------------------------------
def test_refresh_tokens_are_stored_only_as_digests(session: Session) -> None:
    """The raw token never reaches the database."""
    user = UserRepository(session).create("a@ornek.edu.tr", "hash", "Ada")
    raw, digest = new_refresh_token()
    assert digest == hash_token(raw)
    repo = RefreshTokenRepository(session)
    row = repo.create(user.id, digest)
    assert row.token_hash == digest
    assert raw not in row.token_hash
    assert repo.by_hash(digest) is row
    assert repo.by_hash(hash_token("baska")) is None
    assert row.expires_at > utcnow() + timedelta(seconds=REFRESH_TTL_S - 60)


def test_rotation_keeps_the_family_and_revokes_the_predecessor(session: Session) -> None:
    """Rotation is what makes a replay detectable (docs/guvenlik-gizlilik.md §1)."""
    user = UserRepository(session).create("b@ornek.edu.tr", "hash", "Ada")
    repo = RefreshTokenRepository(session)
    first = repo.create(user.id, hash_token("ilk"))
    raw, second = repo.rotate(first)
    assert second.family_id == first.family_id
    assert first.revoked_at is not None
    assert first.replaced_by == second.id
    assert repo.by_hash(hash_token(raw)) is second

    revoked = repo.revoke_family(first.family_id)
    assert revoked == 1
    assert second.revoked_at is not None
    assert len(repo.family(first.family_id)) == 2


def test_revoke_for_user_kills_every_live_token(session: Session) -> None:
    """Changing a password logs every device out."""
    user = UserRepository(session).create("c@ornek.edu.tr", "hash", "Ada")
    repo = RefreshTokenRepository(session)
    repo.create(user.id, hash_token("bir"))
    repo.create(user.id, hash_token("iki"))
    assert repo.revoke_for_user(user.id) == 2
    assert repo.revoke_for_user(user.id) == 0


# ---------------------------------------------------------------------------------------------
# WebSocket tickets
# ---------------------------------------------------------------------------------------------
def test_ws_ticket_is_single_use_and_short_lived() -> None:
    """A ticket works exactly once and expires on its own."""
    clock = {"now": 0.0}
    store = WsTicketStore(ttl_s=60, clock=lambda: clock["now"])
    value, expires_in = store.issue("01J", "student")
    assert expires_in == 60
    assert store.size == 1

    ticket = store.consume(value)
    assert ticket.user_id == "01J"
    assert ticket.role == "student"
    assert store.size == 0
    with pytest.raises(AuthError) as reused:
        store.consume(value)
    assert reused.value.details["reason"] == "ticket_invalid"

    other, _ = store.issue("01K", "admin")
    clock["now"] = 61.0
    with pytest.raises(AuthError):
        store.consume(other)
    assert store.size == 0
