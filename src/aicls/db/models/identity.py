"""Identity and workspace tables (docs/mimari/veri-modeli.md §3.1).

``users`` / ``refresh_tokens`` back the JWT flow with refresh-token rotation families;
``workspaces`` / ``workspace_members`` scope a user's calculus session history.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import CHAR, Boolean, DateTime, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from aicls.db.base import (
    MYSQL_TABLE_KWARGS,
    Base,
    Digest,
    TimestampMixin,
    UlidPk,
    UlidStr,
    enum_column,
    ulid_fk,
)
from aicls.db.enums import Role, University, WorkspaceRole


class User(Base, TimestampMixin):
    """A registered platform user (student, instructor, contributor or admin)."""

    __tablename__ = "users"
    __table_args__ = (MYSQL_TABLE_KWARGS,)

    id: Mapped[UlidPk]
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    role: Mapped[Role] = mapped_column(enum_column(Role, "role"), default=Role.STUDENT)
    university: Mapped[University | None] = mapped_column(
        enum_column(University, "university"), nullable=True, default=None
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    failed_logins: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)


class RefreshToken(Base, TimestampMixin):
    """A hashed refresh token; rotation keeps one ``family_id`` per login (docs/guvenlik...)."""

    __tablename__ = "refresh_tokens"
    __table_args__ = (MYSQL_TABLE_KWARGS,)

    id: Mapped[UlidPk]
    user_id: Mapped[str] = ulid_fk("users.id", ondelete="CASCADE", index=True)
    token_hash: Mapped[Digest] = mapped_column(unique=True)
    family_id: Mapped[UlidStr] = mapped_column(index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)
    replaced_by: Mapped[str | None] = mapped_column(CHAR(26), nullable=True, default=None)
    ip_hash: Mapped[Digest] = mapped_column(default="")


class Workspace(Base, TimestampMixin):
    """A named container for a user's operations (calculus session)."""

    __tablename__ = "workspaces"
    __table_args__ = (MYSQL_TABLE_KWARGS,)

    id: Mapped[UlidPk]
    owner_id: Mapped[str] = ulid_fk("users.id", ondelete="CASCADE", index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    settings_json: Mapped[dict[str, Any]] = mapped_column(default=dict)
    is_shared: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class WorkspaceMember(Base, TimestampMixin):
    """Membership of a user in a shared workspace."""

    __tablename__ = "workspace_members"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id", "user_id", name="uq_workspace_members_workspace_id_user_id"
        ),
        MYSQL_TABLE_KWARGS,
    )

    id: Mapped[UlidPk]
    workspace_id: Mapped[str] = ulid_fk("workspaces.id", ondelete="CASCADE")
    user_id: Mapped[str] = ulid_fk("users.id", ondelete="CASCADE")
    role_in_ws: Mapped[WorkspaceRole] = mapped_column(
        enum_column(WorkspaceRole, "role_in_ws"), default=WorkspaceRole.VIEWER
    )


__all__ = ["RefreshToken", "User", "Workspace", "WorkspaceMember"]
