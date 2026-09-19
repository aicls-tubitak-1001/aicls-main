"""Workspaces and the operation history, including TeX/PDF export (api.md §3.3).

Ownership is enforced by the repository (``get_owned(ws_id, user_id)``), never by trusting an id
the client sent, and a workspace that belongs to somebody else answers 404 rather than 403 so an
identifier cannot be probed (the IDOR test of docs/guvenlik-gizlilik.md §9).
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Query, Response, status
from sqlalchemy import func, select

from aicls.api.deps import CurrentUser, DbSession
from aicls.api.export import operation_to_pdf, operation_to_tex
from aicls.api.schemas import (
    OperationOut,
    Page,
    TimingsOut,
    WorkspaceIn,
    WorkspaceOut,
    WorkspacePatch,
)
from aicls.core import messages_tr
from aicls.core.errors import NotFoundError
from aicls.db.models import Operation, Workspace
from aicls.db.repositories.compute import OperationRepository
from aicls.db.repositories.identity import WorkspaceRepository

router = APIRouter(prefix="/workspaces", tags=["workspaces"])

#: Characters of the input echoed in ``OperationOut.input_summary``.
SUMMARY_CHARS = 120


def _not_found(ws_id: str) -> NotFoundError:
    """The single answer for "missing" and "not yours" (no existence oracle)."""
    return NotFoundError(messages_tr.MSG_NOT_FOUND, details={"table": "workspaces", "id": ws_id})


def _owned(session: DbSession, ws_id: str, user: CurrentUser) -> Workspace:
    """Return the workspace ``user`` owns, or raise :func:`_not_found`."""
    workspace = WorkspaceRepository(session).get_owned(ws_id, user.id)
    if workspace is None:
        raise _not_found(ws_id)
    return workspace


def _workspace_out(session: DbSession, workspace: Workspace) -> WorkspaceOut:
    """Build the response model, counting the workspace's operations."""
    count = session.scalar(
        select(func.count(Operation.id)).where(Operation.workspace_id == workspace.id)
    )
    return WorkspaceOut(
        id=workspace.id,
        title=workspace.title,
        created_at=workspace.created_at,
        updated_at=workspace.updated_at,
        n_operations=int(count or 0),
    )


def operation_out(row: Operation) -> OperationOut:
    """Convert an ``operations`` row into its API shape."""
    summary = row.source_text or row.input_latex or ""
    return OperationOut(
        id=row.id,
        input_kind=row.input_kind.value,
        input_summary=summary[:SUMMARY_CHARS],
        input_latex=row.input_latex,
        command=dict(row.command_json or {}) or None,
        result_latex=row.result_latex,
        result_kind=row.result_kind.value if row.result_kind is not None else None,
        plot_url=(f"/api/v1/compute/plots/{row.plot_asset_id}.png" if row.plot_asset_id else None),
        error_code=row.error_code,
        timings=TimingsOut.model_validate(dict(row.timings_json or {})),
        created_at=row.created_at,
    )


@router.get("", response_model=Page[WorkspaceOut])
def list_workspaces(user: CurrentUser, session: DbSession) -> Page[WorkspaceOut]:
    """Every workspace the caller owns or is a member of, newest first.

    Returned as a page for shape consistency with the rest of api.md §1, but always in one page:
    ``WorkspaceRepository.list_for_user`` has no cursor, and a user's workspace list is small by
    construction (one per calculus session).
    """
    rows = WorkspaceRepository(session).list_for_user(user.id)
    return Page[WorkspaceOut](items=[_workspace_out(session, row) for row in rows])


@router.post("", response_model=WorkspaceOut, status_code=status.HTTP_201_CREATED)
def create_workspace(body: WorkspaceIn, user: CurrentUser, session: DbSession) -> WorkspaceOut:
    """Create a workspace owned by the caller."""
    workspace = WorkspaceRepository(session).create(user.id, body.title)
    return _workspace_out(session, workspace)


@router.get("/{ws_id}", response_model=WorkspaceOut)
def get_workspace(ws_id: str, user: CurrentUser, session: DbSession) -> WorkspaceOut:
    """One workspace of the caller."""
    return _workspace_out(session, _owned(session, ws_id, user))


@router.patch("/{ws_id}", response_model=WorkspaceOut)
def patch_workspace(
    ws_id: str, body: WorkspacePatch, user: CurrentUser, session: DbSession
) -> WorkspaceOut:
    """Rename a workspace."""
    workspace = _owned(session, ws_id, user)
    if body.title is not None:
        workspace.title = body.title
    session.flush()
    return _workspace_out(session, workspace)


@router.delete("/{ws_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_workspace(ws_id: str, user: CurrentUser, session: DbSession) -> None:
    """Delete a workspace and its operations."""
    if not WorkspaceRepository(session).delete(ws_id, user.id):
        raise _not_found(ws_id)


@router.get("/{ws_id}/operations", response_model=Page[OperationOut])
def list_operations(
    ws_id: str,
    user: CurrentUser,
    session: DbSession,
    cursor: Annotated[str | None, Query(max_length=26)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> Page[OperationOut]:
    """One page of the workspace history, newest first."""
    _owned(session, ws_id, user)
    page = OperationRepository(session).page(ws_id, cursor=cursor, limit=limit)
    return Page[OperationOut](
        items=[operation_out(row) for row in page.items], next_cursor=page.next_cursor
    )


@router.delete("/{ws_id}/operations/{op_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_operation(ws_id: str, op_id: str, user: CurrentUser, session: DbSession) -> None:
    """Remove one operation from the history."""
    _owned(session, ws_id, user)
    if not OperationRepository(session).delete(ws_id, op_id):
        raise NotFoundError(messages_tr.MSG_NOT_FOUND, details={"table": "operations", "id": op_id})


@router.post("/{ws_id}/operations/{op_id}/export")
def export_operation(
    ws_id: str,
    op_id: str,
    user: CurrentUser,
    session: DbSession,
    export_format: Annotated[Literal["tex", "pdf"], Query(alias="format")] = "tex",
) -> Response:
    """Export one operation as LaTeX source or a one-page PDF (A.2.2)."""
    _owned(session, ws_id, user)
    row = OperationRepository(session).get(op_id)
    if row is None or row.workspace_id != ws_id:
        raise NotFoundError(messages_tr.MSG_NOT_FOUND, details={"table": "operations", "id": op_id})
    if export_format == "tex":
        return Response(
            content=operation_to_tex(row),
            media_type="text/x-tex; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="aicls-{op_id}.tex"'},
        )
    return Response(
        content=operation_to_pdf(row),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="aicls-{op_id}.pdf"'},
    )


__all__ = ["SUMMARY_CHARS", "operation_out", "router"]
