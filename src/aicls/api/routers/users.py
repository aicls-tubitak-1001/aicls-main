"""The signed-in user's own profile (api.md §3.2)."""

from __future__ import annotations

from fastapi import APIRouter

from aicls.api.deps import CurrentUser, DbSession
from aicls.api.schemas import UserOut, UserPatch
from aicls.api.security import RefreshTokenRepository, hash_password
from aicls.db.models import User

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> User:
    """Return the caller's own profile."""
    return user


@router.patch("/me", response_model=UserOut)
def patch_me(body: UserPatch, user: CurrentUser, session: DbSession) -> User:
    """Update the display name and/or the password.

    Changing the password revokes every live refresh token of the account, so a session opened
    with the old credentials cannot survive the change (docs/guvenlik-gizlilik.md §1).
    """
    if body.display_name is not None:
        user.display_name = body.display_name
    if body.password is not None:
        user.password_hash = hash_password(body.password)
        RefreshTokenRepository(session).revoke_for_user(user.id)
    session.flush()
    return user


__all__ = ["router"]
