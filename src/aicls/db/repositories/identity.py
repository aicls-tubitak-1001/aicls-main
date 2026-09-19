"""User and workspace repositories (docs/mimari/veri-modeli.md §3.1)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import or_, select

from aicls.core.errors import NotFoundError
from aicls.core.messages_tr import msg
from aicls.db.enums import Role, University, WorkspaceRole, coerce_enum
from aicls.db.models import User, Workspace, WorkspaceMember
from aicls.db.repositories.base import Repository
from aicls.db.results import Page


def normalise_email(email: str) -> str:
    """Return the canonical storage form of an e-mail address (trimmed, lower-case)."""
    return email.strip().lower()


class UserRepository(Repository):
    """Queries over ``users`` (registration, login lookup, role administration)."""

    def get(self, user_id: str) -> User | None:
        """Return the user with ``user_id`` or ``None``."""
        return self.session.get(User, user_id)

    def get_by_email(self, email: str) -> User | None:
        """Return the user registered with ``email`` (case-insensitive) or ``None``."""
        stmt = select(User).where(User.email == normalise_email(email))
        return self.session.scalars(stmt).one_or_none()

    def create(
        self,
        email: str,
        password_hash: str,
        display_name: str,
        role: Role | str = Role.STUDENT,
        *,
        university: University | str | None = None,
    ) -> User:
        """Insert a new user and return it (the caller commits).

        Args:
            email: Login address; stored lower-case.
            password_hash: argon2id hash produced by the API layer — never a plain password.
            display_name: Name shown in the UI.
            role: Platform role; defaults to ``student``.
            university: Optional partner-university band.

        Raises:
            InputValidationError: If ``role`` or ``university`` is not a known value.
        """
        user = User(
            email=normalise_email(email),
            password_hash=password_hash,
            display_name=display_name,
            role=coerce_enum(Role, role),
            university=coerce_enum(University, university) if university is not None else None,
        )
        self.session.add(user)
        self.session.flush()
        return user

    def set_role(self, user_id: str, role: Role | str) -> None:
        """Change a user's role.

        Raises:
            NotFoundError: If no such user exists.
            InputValidationError: If ``role`` is unknown.
        """
        user = self.session.get(User, user_id)
        if user is None:
            raise NotFoundError(msg("NOT_FOUND"), details={"table": "users", "id": user_id})
        user.role = coerce_enum(Role, role)
        self.session.flush()

    def set_active(self, user_id: str, *, is_active: bool) -> None:
        """Enable or disable a user account.

        Raises:
            NotFoundError: If no such user exists.
        """
        user = self.session.get(User, user_id)
        if user is None:
            raise NotFoundError(msg("NOT_FOUND"), details={"table": "users", "id": user_id})
        user.is_active = is_active
        self.session.flush()

    def page(self, cursor: str | None = None, limit: int | None = None) -> Page[User]:
        """Return one page of users, newest first (``GET /admin/users``)."""
        return self.cursor_page(select(User), User.id, cursor=cursor, limit=limit)


class WorkspaceRepository(Repository):
    """Queries over ``workspaces`` and ``workspace_members``."""

    def create(
        self,
        owner_id: str,
        title: str,
        *,
        settings: dict[str, Any] | None = None,
        is_shared: bool = False,
    ) -> Workspace:
        """Insert a workspace owned by ``owner_id`` and return it."""
        workspace = Workspace(
            owner_id=owner_id,
            title=title,
            settings_json=dict(settings or {}),
            is_shared=is_shared,
        )
        self.session.add(workspace)
        self.session.flush()
        return workspace

    def list_for_user(self, user_id: str) -> list[Workspace]:
        """Return every workspace the user owns or is a member of, newest first."""
        member_ws = select(WorkspaceMember.workspace_id).where(WorkspaceMember.user_id == user_id)
        stmt = (
            select(Workspace)
            .where(or_(Workspace.owner_id == user_id, Workspace.id.in_(member_ws)))
            .order_by(Workspace.id.desc())
        )
        return list(self.session.scalars(stmt))

    def get_owned(self, ws_id: str, user_id: str) -> Workspace | None:
        """Return the workspace only if ``user_id`` owns it (authorisation helper)."""
        stmt = select(Workspace).where(Workspace.id == ws_id, Workspace.owner_id == user_id)
        return self.session.scalars(stmt).one_or_none()

    def add_member(
        self, ws_id: str, user_id: str, role_in_ws: WorkspaceRole | str = WorkspaceRole.VIEWER
    ) -> WorkspaceMember:
        """Add (or re-role) a member of a shared workspace."""
        stmt = select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == ws_id, WorkspaceMember.user_id == user_id
        )
        member = self.session.scalars(stmt).one_or_none()
        if member is None:
            member = WorkspaceMember(
                workspace_id=ws_id,
                user_id=user_id,
                role_in_ws=coerce_enum(WorkspaceRole, role_in_ws),
            )
            self.session.add(member)
        else:
            member.role_in_ws = coerce_enum(WorkspaceRole, role_in_ws)
        self.session.flush()
        return member

    def delete(self, ws_id: str, user_id: str) -> bool:
        """Delete a workspace the user owns; return ``False`` when it does not exist."""
        workspace = self.get_owned(ws_id, user_id)
        if workspace is None:
            return False
        self.session.delete(workspace)
        self.session.flush()
        return True


__all__ = ["UserRepository", "WorkspaceRepository", "normalise_email"]
