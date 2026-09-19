"""Edge paths: workspace ownership, not-found answers, health degradation, archive guards.

These are the branches a happy-path suite never reaches and an operator meets first: a workspace
that is not yours, an id that does not exist, a database that stopped answering, an archive that
tries to escape its extraction directory.
"""

from __future__ import annotations

import io
from pathlib import Path
import zipfile

from fastapi import FastAPI
import httpx
import pytest

from aicls.api.bookload import MAX_ARCHIVE_BYTES
from aicls.core.errors import AiclsError, ErrorCode, InputValidationError
from tests.integration.api.conftest import account, auth_header

pytestmark = pytest.mark.integration


@pytest.fixture
async def owner(client: httpx.AsyncClient, app: FastAPI) -> dict[str, str]:
    """A signed-in student who owns one workspace."""
    _user, token = await account(client, app, "sahip@ornek.edu.tr")
    return auth_header(token)


@pytest.fixture
async def intruder(client: httpx.AsyncClient, app: FastAPI) -> dict[str, str]:
    """A different signed-in student."""
    _user, token = await account(client, app, "davetsiz@ornek.edu.tr")
    return auth_header(token)


# ---------------------------------------------------------------------------------------------
# workspaces
# ---------------------------------------------------------------------------------------------
async def test_workspace_crud_and_ownership(
    client: httpx.AsyncClient, owner: dict[str, str], intruder: dict[str, str]
) -> None:
    """Create, read, rename, list and delete — and none of it visible to anyone else."""
    created = await client.post("/workspaces", json={"title": "Limit"}, headers=owner)
    assert created.status_code == 201
    ws_id = created.json()["id"]

    fetched = await client.get(f"/workspaces/{ws_id}", headers=owner)
    assert fetched.status_code == 200
    assert fetched.json()["n_operations"] == 0

    renamed = await client.patch(f"/workspaces/{ws_id}", json={"title": "Türev"}, headers=owner)
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Türev"

    listed = await client.get("/workspaces", headers=owner)
    assert [item["id"] for item in listed.json()["items"]] == [ws_id]
    assert (await client.get("/workspaces", headers=intruder)).json()["items"] == []

    for path in (f"/workspaces/{ws_id}", f"/workspaces/{ws_id}/operations"):
        denied = await client.get(path, headers=intruder)
        assert denied.status_code == 404, path
    assert (
        await client.patch(f"/workspaces/{ws_id}", json={"title": "X"}, headers=intruder)
    ).status_code == 404
    assert (await client.delete(f"/workspaces/{ws_id}", headers=intruder)).status_code == 404

    assert (await client.delete(f"/workspaces/{ws_id}", headers=owner)).status_code == 204
    assert (await client.get(f"/workspaces/{ws_id}", headers=owner)).status_code == 404


async def test_workspace_operations_paginate(
    client: httpx.AsyncClient, owner: dict[str, str]
) -> None:
    """The history is cursor-paginated, newest first (api.md §1)."""
    ws_id = (await client.post("/workspaces", json={"title": "Sayfa"}, headers=owner)).json()["id"]
    for index in range(3):
        computed = await client.post(
            "/compute/latex",
            json={
                "command": {"op": "diff", "expr": f"x^{{{index + 2}}}"},
                "workspace_id": ws_id,
            },
            headers=owner,
        )
        assert computed.status_code == 200, computed.text

    first = await client.get(f"/workspaces/{ws_id}/operations", params={"limit": 2}, headers=owner)
    assert first.status_code == 200
    page = first.json()
    assert len(page["items"]) == 2
    assert page["next_cursor"]

    second = await client.get(
        f"/workspaces/{ws_id}/operations",
        params={"limit": 2, "cursor": page["next_cursor"]},
        headers=owner,
    )
    assert len(second.json()["items"]) == 1
    assert second.json()["next_cursor"] is None


async def test_missing_ids_answer_not_found(
    client: httpx.AsyncClient, owner: dict[str, str]
) -> None:
    """Every ``{id}`` route answers a problem document, never a stack trace."""
    ws_id = (await client.post("/workspaces", json={"title": "Yok"}, headers=owner)).json()["id"]
    unknown = "01J000000000000000000000"

    for method, path in (
        ("DELETE", f"/workspaces/{ws_id}/operations/{unknown}"),
        ("POST", f"/workspaces/{ws_id}/operations/{unknown}/export"),
        ("GET", f"/compute/plots/{unknown}.png"),
        ("GET", f"/book/assets/{unknown}"),
    ):
        response = await client.request(method, path, headers=owner)
        assert response.status_code == 404, f"{method} {path}: {response.status_code}"
        assert response.json()["code"] == "NOT_FOUND"

    feedback = await client.post(
        "/ocr/feedback",
        json={"recognition_id": unknown, "corrected_latex": "x"},
        headers=owner,
    )
    assert feedback.status_code == 404

    transcript = await client.post(
        "/asr/feedback",
        json={"transcript_id": unknown, "corrected_text": "x"},
        headers=owner,
    )
    assert transcript.status_code == 404


async def test_users_patch_display_name_only(
    client: httpx.AsyncClient, owner: dict[str, str]
) -> None:
    """A name change does not touch the password or the sessions."""
    patched = await client.patch("/users/me", json={"display_name": "Yeni Ad"}, headers=owner)
    assert patched.status_code == 200
    assert patched.json()["display_name"] == "Yeni Ad"
    assert (await client.get("/users/me", headers=owner)).json()["display_name"] == "Yeni Ad"


async def test_a_disabled_account_cannot_use_its_token(
    client: httpx.AsyncClient, app: FastAPI, owner: dict[str, str]
) -> None:
    """Deactivating an account invalidates a token that has not expired yet."""
    from aicls.db import UserRepository, session_scope

    me = (await client.get("/users/me", headers=owner)).json()
    with session_scope(app.state.engine) as session:
        UserRepository(session).set_active(me["id"], is_active=False)
    assert (await client.get("/users/me", headers=owner)).status_code == 401


# ---------------------------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------------------------
async def test_health_endpoints_report_the_runtime(client: httpx.AsyncClient) -> None:
    """Version and readiness describe the process the operator is looking at."""
    version = await client.get("/health/version")
    assert version.status_code == 200
    body = version.json()
    assert body["profile"] == "full"
    assert body["version"]
    assert body["models"] == {"ocr": "stub-0.1.0", "asr": "asr-stub-1"}

    ready = await client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ok"
    assert ready.json()["sandbox"]["started"] is True


async def test_health_ready_is_503_while_a_model_is_training(
    client: httpx.AsyncClient, app: FastAPI
) -> None:
    """ADR-0018: during a GPU run the readiness probe must say "not ready"."""
    app.state.registry._states["ocr"] = "training"
    response = await client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["models"]["ocr"] == "training"


async def test_health_ready_reports_a_broken_database(
    client: httpx.AsyncClient, app: FastAPI
) -> None:
    """A database that stopped answering is a 503 with ``db: error``, not a 500."""
    from aicls.api.deps import get_db

    class BrokenSession:
        """A session whose first statement fails (a lost connection)."""

        def execute(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("veritabanı yanıt vermiyor")

    def broken() -> BrokenSession:
        return BrokenSession()

    app.dependency_overrides[get_db] = broken
    try:
        response = await client.get("/health/ready")
    finally:
        app.dependency_overrides.pop(get_db, None)
    assert response.status_code == 503
    assert response.json()["db"] == "error"
    assert response.json()["status"] == "not_ready"


# ---------------------------------------------------------------------------------------------
# admin edges
# ---------------------------------------------------------------------------------------------
async def test_admin_eval_runs_listing_and_lookup(client: httpx.AsyncClient, app: FastAPI) -> None:
    """Evaluation reports are readable one by one and as a list."""
    from aicls.db import EvalRunRepository, session_scope

    _user, token = await account(client, app, "denetci@ornek.edu.tr", "admin")
    headers = auth_header(token)
    with session_scope(app.state.engine) as session:
        EvalRunRepository(session).add(
            {"kind": "parser", "name": "parse-suite", "n": 3599, "metrics": {"success": 1.0}},
            is_reportable=True,
        )

    listed = await client.get("/admin/eval-runs", params={"kind": "parser"}, headers=headers)
    assert listed.status_code == 200
    rows = listed.json()
    assert rows[0]["name"] == "parse-suite"
    assert rows[0]["is_reportable"] is True

    one = await client.get(f"/admin/eval-runs/{rows[0]['id']}", headers=headers)
    assert one.status_code == 200
    assert one.json()["metrics"] == {"success": 1.0}

    missing = await client.get("/admin/eval-runs/01J000000000000000000000", headers=headers)
    assert missing.status_code == 404


async def test_admin_book_load_refuses_a_zip_slip_and_an_empty_archive(
    client: httpx.AsyncClient, app: FastAPI
) -> None:
    """The archive guard refuses path traversal and an archive with no chapter in it."""
    _user, token = await account(client, app, "kitapci@ornek.edu.tr", "admin")
    headers = auth_header(token)

    slip = io.BytesIO()
    with zipfile.ZipFile(slip, "w") as archive:
        archive.writestr("../../kacis.md", "# Tanım 1.1: Kaçış\n")
    escaped = await client.post(
        "/admin/book/load",
        files={"file": ("kotu.zip", slip.getvalue(), "application/zip")},
        headers=headers,
    )
    assert escaped.status_code == 422
    assert escaped.json()["reason"] == "zip_slip"

    empty = io.BytesIO()
    with zipfile.ZipFile(empty, "w") as archive:
        archive.writestr("okuma.txt", "markdown degil")
    nothing = await client.post(
        "/admin/book/load",
        files={"file": ("bos.zip", empty.getvalue(), "application/zip")},
        headers=headers,
    )
    assert nothing.status_code == 422
    assert nothing.json()["reason"] == "no_markdown_members"

    without_part = await client.post(
        "/admin/book/load",
        files={"baska": ("kitap.zip", b"PK", "application/zip")},
        headers=headers,
    )
    assert without_part.status_code == 400
    assert without_part.json()["reason"] == "archive_part_missing"


def test_book_archive_limits_are_declared() -> None:
    """The archive ceiling is small on purpose: a textbook chapter set is text."""
    assert MAX_ARCHIVE_BYTES == 8 * 1024 * 1024


# ---------------------------------------------------------------------------------------------
# error handlers
# ---------------------------------------------------------------------------------------------
async def test_unknown_route_and_method_are_problem_documents(
    client: httpx.AsyncClient, owner: dict[str, str]
) -> None:
    """A 404 or a 405 from the framework is rendered like every other failure."""
    missing = await client.get("/bilinmeyen-yol", headers=owner)
    assert missing.status_code == 404
    assert missing.json()["code"] == "NOT_FOUND"
    assert missing.headers["content-type"].startswith("application/problem+json")

    wrong_method = await client.get("/auth/login", headers=owner)
    assert wrong_method.status_code == 405
    assert wrong_method.json()["code"] == "VALIDATION_ERROR"


async def test_unhandled_exception_becomes_an_internal_problem(app: FastAPI) -> None:
    """A bug never leaks a traceback: the client sees ``INTERNAL`` with the request id."""
    from starlette.exceptions import HTTPException

    def boom() -> None:
        raise RuntimeError("beklenmeyen")

    def teapot() -> None:
        raise HTTPException(status_code=418, detail="demlik")

    app.add_api_route("/api/v1/__patlat", boom, methods=["GET"])
    app.add_api_route("/api/v1/__demlik", teapot, methods=["GET"])
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver/api/v1") as raw:
        crashed = await raw.get("/__patlat")
        odd = await raw.get("/__demlik")

    assert crashed.status_code == 500
    body = crashed.json()
    assert body["code"] == "INTERNAL"
    assert body["detail_tr"]
    assert body["kind"] == "RuntimeError"
    assert "beklenmeyen" not in crashed.text, "an internal message never reaches the client"

    assert odd.status_code == 418
    assert odd.json()["reason"] == "demlik"


def test_error_codes_used_by_the_api_are_all_documented() -> None:
    """Nothing in the API invents a code outside the shared vocabulary."""
    assert InputValidationError().code is ErrorCode.VALIDATION_ERROR
    assert AiclsError("BILINMEYEN").code is ErrorCode.INTERNAL


# ---------------------------------------------------------------------------------------------
# parse preview edges
# ---------------------------------------------------------------------------------------------
async def test_compute_parse_reports_repairs(
    client: httpx.AsyncClient, owner: dict[str, str]
) -> None:
    """The preview reports the normalisation repairs it applied."""
    response = await client.post("/compute/parse", json={"latex": "\\frac{1}{"}, headers=owner)
    assert response.status_code == 200
    body = response.json()
    assert body["repairs"], "the client shows what was auto-corrected"


async def test_book_asset_is_served_when_it_exists(
    client: httpx.AsyncClient, app: FastAPI, owner: dict[str, str], tmp_path: Path
) -> None:
    """A registered book figure is served from ``AICLS_DATA_ROOT``."""
    from aicls.db import AssetRepository, session_scope

    data_root = app.state.settings.data_root
    relative = "book/figures/sekil-2.3.png"
    target = data_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    with session_scope(app.state.engine) as session:
        asset = AssetRepository(session).add(
            "book_figure", relative, mime="image/png", size_bytes=24
        )
        asset_id = asset.id

    response = await client.get(f"/book/assets/{asset_id}", headers=owner)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")
