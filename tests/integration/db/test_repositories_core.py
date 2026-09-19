"""Behaviour of the identity, workspace and operation repositories (on migrated SQLite)."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from aicls.core.errors import InputValidationError, NotFoundError
from aicls.db import (
    AssetKind,
    AssetRepository,
    AuditRepository,
    InputKind,
    OperationCreate,
    OperationRepository,
    ResultKind,
    Role,
    User,
    UserRepository,
    Workspace,
    WorkspaceRepository,
)
from aicls.db.repositories.base import MAX_PAGE_SIZE, clamp_limit, field_of

pytestmark = pytest.mark.integration


class TestUserRepository:
    """``users``: registration, lookup and role administration."""

    def test_create_normalises_email_and_defaults_to_student(self, session: Session) -> None:
        repo = UserRepository(session)
        user = repo.create("  Ada@Example.EDU.tr ", "argon2id$hash", "Ada")
        assert user.email == "ada@example.edu.tr"
        assert user.role is Role.STUDENT
        assert user.is_active is True
        assert len(user.id) == 26

    def test_get_by_email_is_case_insensitive_and_returns_none_when_absent(
        self, session: Session, user: User
    ) -> None:
        repo = UserRepository(session)
        assert repo.get_by_email("ADA@example.edu.tr") is user
        assert repo.get_by_email("nobody@example.edu.tr") is None

    def test_set_role_and_set_active(self, session: Session, user: User) -> None:
        repo = UserRepository(session)
        repo.set_role(user.id, "instructor")
        assert user.role is Role.INSTRUCTOR
        repo.set_role(user.id, Role.ADMIN)
        assert user.role is Role.ADMIN
        repo.set_active(user.id, is_active=False)
        assert user.is_active is False

    def test_set_role_rejects_unknown_user_and_unknown_role(
        self, session: Session, user: User
    ) -> None:
        repo = UserRepository(session)
        with pytest.raises(NotFoundError):
            repo.set_role("0" * 26, Role.ADMIN)
        with pytest.raises(NotFoundError):
            repo.set_active("0" * 26, is_active=True)
        with pytest.raises(InputValidationError):
            repo.set_role(user.id, "root")

    def test_page_returns_users_newest_first_with_a_cursor(self, session: Session) -> None:
        repo = UserRepository(session)
        for index in range(5):
            repo.create(f"u{index}@example.edu.tr", "h", f"U{index}")
        first = repo.page(limit=2)
        assert len(first.items) == 2
        assert first.next_cursor is not None
        second = repo.page(cursor=first.next_cursor, limit=2)
        assert {u.id for u in first.items} & {u.id for u in second.items} == set()
        last = repo.page(cursor=second.next_cursor, limit=2)
        assert last.next_cursor is None
        assert len(last) == 1


class TestWorkspaceRepository:
    """``workspaces`` and ``workspace_members``."""

    def test_create_and_list_for_owner(self, session: Session, user: User) -> None:
        repo = WorkspaceRepository(session)
        first = repo.create(user.id, "Limitler", settings={"theme": "dark"})
        second = repo.create(user.id, "Türev")
        listed = repo.list_for_user(user.id)
        assert [ws.id for ws in listed] == [second.id, first.id]
        assert first.settings_json == {"theme": "dark"}
        assert first.is_shared is False

    def test_list_for_user_includes_shared_memberships(self, session: Session, user: User) -> None:
        repo = WorkspaceRepository(session)
        other = UserRepository(session).create("bob@example.edu.tr", "h", "Bob")
        shared = repo.create(other.id, "Ortak", is_shared=True)
        assert repo.list_for_user(user.id) == []
        repo.add_member(shared.id, user.id, "editor")
        assert [ws.id for ws in repo.list_for_user(user.id)] == [shared.id]
        # Re-adding only changes the role.
        member = repo.add_member(shared.id, user.id, "viewer")
        assert member.role_in_ws.value == "viewer"

    def test_get_owned_enforces_ownership(self, session: Session, user: User) -> None:
        repo = WorkspaceRepository(session)
        other = UserRepository(session).create("eve@example.edu.tr", "h", "Eve")
        workspace = repo.create(user.id, "Özel")
        assert repo.get_owned(workspace.id, user.id) is workspace
        assert repo.get_owned(workspace.id, other.id) is None

    def test_delete_returns_false_for_a_foreign_workspace(
        self, session: Session, user: User, workspace: Workspace
    ) -> None:
        repo = WorkspaceRepository(session)
        other = UserRepository(session).create("mallory@example.edu.tr", "h", "M")
        assert repo.delete(workspace.id, other.id) is False
        assert repo.delete(workspace.id, user.id) is True
        assert repo.get_owned(workspace.id, user.id) is None


class TestOperationRepository:
    """``operations``: the workspace history of the contractual pipeline."""

    def test_add_accepts_a_dataclass_and_stores_json_columns(
        self, session: Session, user: User, workspace: Workspace
    ) -> None:
        repo = OperationRepository(session)
        operation = repo.add(
            workspace.id,
            user.id,
            OperationCreate(
                input_kind=InputKind.TEXT,
                source_text="x karenin türevini al",
                input_latex="x^2",
                command={"op": "diff", "expr": "x**2"},
                result_kind=ResultKind.EXPR,
                result_latex="2 x",
                timings={"nlu": 3.0, "cas": 12.5},
                model_versions={"asr": "asr-0.1.0"},
                nlu_source="rules",
            ),
        )
        assert operation.command_json["op"] == "diff"
        assert operation.timings_json["cas"] == 12.5
        assert operation.nlu_source is not None and operation.nlu_source.value == "rules"
        assert repo.get(operation.id) is operation

    def test_add_accepts_a_plain_mapping_with_column_aliases(
        self, session: Session, user: User, workspace: Workspace
    ) -> None:
        repo = OperationRepository(session)
        operation = repo.add(
            workspace.id,
            user.id,
            {
                "input_kind": "latex",
                "input_latex": "\\frac{d}{dx} x^2",
                "command_json": {"op": "diff"},
                "timings_json": {"total": 42.0},
                "result_kind": "expr",
            },
        )
        assert operation.input_kind is InputKind.LATEX
        assert operation.command_json == {"op": "diff"}
        assert operation.timings_json == {"total": 42.0}

    def test_add_rejects_an_unknown_enum_value(
        self, session: Session, user: User, workspace: Workspace
    ) -> None:
        with pytest.raises(InputValidationError):
            OperationRepository(session).add(workspace.id, user.id, {"input_kind": "telepathy"})

    def test_page_is_scoped_to_the_workspace_and_pages_backwards(
        self, session: Session, user: User, workspace: Workspace
    ) -> None:
        repo = OperationRepository(session)
        other_ws = WorkspaceRepository(session).create(user.id, "Diğer")
        ids = [repo.add(workspace.id, user.id, {"input_kind": "text"}).id for _ in range(3)]
        repo.add(other_ws.id, user.id, {"input_kind": "text"})
        first = repo.page(workspace.id, limit=2)
        assert [op.id for op in first.items] == list(reversed(ids))[:2]
        second = repo.page(workspace.id, cursor=first.next_cursor, limit=2)
        assert [op.id for op in second.items] == [ids[0]]
        assert second.next_cursor is None

    def test_delete_is_scoped_to_the_workspace(
        self, session: Session, user: User, workspace: Workspace
    ) -> None:
        repo = OperationRepository(session)
        other_ws = WorkspaceRepository(session).create(user.id, "Diğer")
        operation = repo.add(workspace.id, user.id, {"input_kind": "text"})
        assert repo.delete(other_ws.id, operation.id) is False
        assert repo.delete(workspace.id, operation.id) is True
        assert repo.get(operation.id) is None


class TestAssetAndAuditRepositories:
    """``assets`` retention and the append-only ``audit_log``."""

    def test_asset_expiry_lists_only_passed_retention_dates(self, session: Session) -> None:
        from datetime import datetime, timedelta

        from aicls.db.base import utcnow

        repo = AssetRepository(session)
        now = utcnow()
        old = repo.add(
            AssetKind.INK_JSON,
            "ink/old.json",
            retention_until=now - timedelta(days=1),
            size_bytes=10,
        )
        repo.add(AssetKind.PLOT_PNG, "plots/new.png", retention_until=now + timedelta(days=1))
        repo.add(AssetKind.PLOT_PNG, "plots/keep.png")
        expired = repo.expired(now=now)
        assert [asset.id for asset in expired] == [old.id]
        assert repo.get(old.id) is old
        assert isinstance(old.created_at, datetime)

    def test_audit_log_accepts_tuple_string_and_missing_targets(
        self, session: Session, user: User
    ) -> None:
        repo = AuditRepository(session)
        first = repo.log(user.id, "role_change", ("users", user.id), {"to": "admin"})
        second = repo.log(user.id, "book_load", "book_units:tanim-2.4", ip_hash="h" * 64)
        third = repo.log(None, "retention_sweep")
        assert (first.target_table, first.target_id) == ("users", user.id)
        assert (second.target_table, second.target_id) == ("book_units", "tanim-2.4")
        assert (third.target_table, third.target_id, third.actor_id) == ("", "", None)
        page = repo.page(action="role_change")
        assert [row.id for row in page.items] == [first.id]
        assert len(repo.page().items) == 3


def test_clamp_limit_and_field_of_helpers() -> None:
    """Pagination clamping and the mapping/attribute reader used across repositories."""
    assert clamp_limit(None) == 50
    assert clamp_limit(0) == 1
    assert clamp_limit(10_000) == MAX_PAGE_SIZE

    class Obj:
        alias = "b"
        primary = None

    assert field_of({"primary": "a"}, "primary", "alias") == "a"
    assert field_of({"primary": None, "alias": "b"}, "primary", "alias") == "b"
    assert field_of(Obj(), "primary", "alias") == "b"
    assert field_of(Obj(), "missing", default=7) == 7
