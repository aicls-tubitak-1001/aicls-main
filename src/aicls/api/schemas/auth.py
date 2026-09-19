"""Identity, user and workspace payloads (api.md §3.2 and §3.3)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from aicls.api.schemas.common import (
    ApiModel,
    EmailAddress,
    InputKindName,
    RequestModel,
    ResultKindName,
    RoleName,
    TimingsOut,
)
from aicls.api.security import MAX_PASSWORD_LENGTH, MIN_PASSWORD_LENGTH

#: Longest display name (``users.display_name`` is ``VARCHAR(120)``).
MAX_DISPLAY_NAME = 120
#: Longest workspace title (``workspaces.title`` is ``VARCHAR(200)``).
MAX_TITLE = 200


class RegisterIn(RequestModel):
    """``POST /auth/register`` body."""

    email: EmailAddress
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)
    display_name: str = Field(min_length=1, max_length=MAX_DISPLAY_NAME)
    invite_code: str | None = Field(default=None, max_length=64)


class LoginIn(RequestModel):
    """``POST /auth/login`` body."""

    email: EmailAddress
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)


class TokenOut(ApiModel):
    """``TokenOut`` — the bearer access token (the refresh token travels as a cookie)."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


class WsTicketOut(ApiModel):
    """``POST /auth/ws-ticket`` response: a single-use 60 s WebSocket ticket."""

    ticket: str
    expires_in: int


class UserOut(ApiModel):
    """A user as the API exposes it (never the password hash)."""

    id: str
    email: str
    display_name: str
    role: RoleName
    created_at: datetime


class UserPatch(RequestModel):
    """``PATCH /users/me`` body; both fields optional."""

    display_name: str | None = Field(default=None, min_length=1, max_length=MAX_DISPLAY_NAME)
    password: str | None = Field(
        default=None, min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH
    )


class WorkspaceIn(RequestModel):
    """``POST /workspaces`` body."""

    title: str = Field(min_length=1, max_length=MAX_TITLE)


class WorkspacePatch(RequestModel):
    """``PATCH /workspaces/{id}`` body."""

    title: str | None = Field(default=None, min_length=1, max_length=MAX_TITLE)


class WorkspaceOut(ApiModel):
    """A calculus workspace of one user."""

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    n_operations: int = 0


class OperationOut(ApiModel):
    """One row of the history panel (api.md §3.3)."""

    id: str
    input_kind: InputKindName
    input_summary: str = ""
    input_latex: str | None = None
    command: dict[str, Any] | None = None
    result_latex: str | None = None
    result_kind: ResultKindName | None = None
    plot_url: str | None = None
    error_code: str | None = None
    timings: TimingsOut = TimingsOut()
    created_at: datetime


__all__ = [
    "MAX_DISPLAY_NAME",
    "MAX_TITLE",
    "LoginIn",
    "OperationOut",
    "RegisterIn",
    "TokenOut",
    "UserOut",
    "UserPatch",
    "WorkspaceIn",
    "WorkspaceOut",
    "WorkspacePatch",
    "WsTicketOut",
]
