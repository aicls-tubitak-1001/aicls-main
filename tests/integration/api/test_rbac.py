"""The role matrix over every router and every role (docs/guvenlik-gizlilik.md §1, §9.2).

The table below is the machine-readable form of api.md's "rol" column. For each entry the test
asserts the *gate*, not the outcome: a role that may call the endpoint must get anything except
401/403, and a role that may not must get exactly 403. That distinction matters because a
missing row would otherwise be hidden by a 404 or a 422 from the body.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
import httpx
import pytest

from tests.integration.api.conftest import ROLES, account, auth_header

pytestmark = pytest.mark.integration

#: ``(method, path, allowed roles, json body)``. ``None`` in the roles column means "any signed-in
#: user"; an empty tuple would mean "nobody", which no endpoint uses.
MATRIX: tuple[tuple[str, str, tuple[str, ...] | None, dict[str, Any] | None], ...] = (
    # identity and workspaces — every signed-in role
    ("GET", "/users/me", None, None),
    ("GET", "/workspaces", None, None),
    ("POST", "/workspaces", None, {"title": "Deneme"}),
    # compute (İP-4)
    ("POST", "/compute/text", None, {"text": "x karenin türevini al"}),
    ("POST", "/compute/latex", None, {"command": {"op": "diff", "expr": "x^{2}"}}),
    ("POST", "/compute/parse", None, {"latex": "x^{2}"}),
    ("GET", "/compute/plots/01J000000000000000000000.png", None, None),
    # handwriting and speech (İP-3)
    ("POST", "/ocr/recognize", None, {"ink": {"strokes": []}}),
    ("POST", "/ocr/feedback", None, {"recognition_id": "01J0", "corrected_latex": "x"}),
    ("POST", "/asr/feedback", None, {"transcript_id": "01J0", "corrected_text": "x"}),
    # book (İP-8)
    ("GET", "/book/chapters", None, None),
    ("GET", "/book/search?q=t%C3%BCrev", None, None),
    ("POST", "/book/resolve", None, {"text": "türev tanımını getir"}),
    ("GET", "/book/units/tanim-2.4", None, None),
    ("GET", "/book/export.jsonld", ("instructor", "admin"), None),
    ("GET", "/book/export.ttl", ("instructor", "admin"), None),
    # feedback and telemetry (İP-9, ADR-0014)
    ("POST", "/feedback", None, {"category": "usability", "text": "not", "page": "/"}),
    ("POST", "/feedback/sus", None, {"answers": [3] * 10, "cohort": "pilot"}),
    ("GET", "/feedback", ("instructor", "admin"), None),
    (
        "POST",
        "/telemetry/latency",
        None,
        {"modality": "text", "client_submit_ts": 1.0, "client_render_ts": 2.0, "render_ms": 5.0},
    ),
    # contribution dashboard
    ("GET", "/contribute/stats", ("instructor", "admin"), None),
    # administration
    ("GET", "/admin/models", ("admin",), None),
    ("POST", "/admin/models/activate", ("admin",), {"component": "ocr", "version": "yok"}),
    ("GET", "/admin/eval-runs", ("admin",), None),
    ("GET", "/admin/metrics/latency", ("instructor", "admin"), None),
    ("POST", "/admin/contributions/export", ("admin",), None),
    ("GET", "/admin/users", ("admin",), None),
    ("GET", "/admin/audit", ("admin",), None),
    ("GET", "/admin/qc/queue", ("instructor", "admin"), None),
    (
        "POST",
        "/admin/qc/01J000000000000000000000/review",
        ("instructor", "admin"),
        {"verdict": "accepted"},
    ),
    ("GET", "/admin/contributions/stats", ("instructor", "admin"), None),
    (
        "PATCH",
        "/admin/feedback/01J000000000000000000000",
        ("instructor", "admin"),
        {"status": "triaged"},
    ),
    ("PATCH", "/admin/users/01J000000000000000000000/role", ("admin",), {"role": "student"}),
)

#: Endpoints that must work without any credentials at all.
OPEN_ENDPOINTS: tuple[str, ...] = (
    "/health/live",
    "/health/ready",
    "/health/version",
    "/contribute/consent/current",
)


@pytest.fixture
async def tokens(client: httpx.AsyncClient, app: FastAPI) -> dict[str, str]:
    """One signed-in account per role."""
    issued: dict[str, str] = {}
    for role in ROLES:
        _user, token = await account(client, app, f"{role}@ornek.edu.tr", role)
        issued[role] = token
    return issued


async def _call(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    body: dict[str, Any] | None,
    token: str | None,
) -> httpx.Response:
    """Issue one matrix call."""
    headers = auth_header(token) if token else {}
    if method == "GET":
        return await client.get(path, headers=headers)
    if method == "DELETE":
        return await client.delete(path, headers=headers)
    if method == "PATCH":
        return await client.patch(path, json=body, headers=headers)
    return await client.post(path, json=body, headers=headers)


@pytest.mark.parametrize(("method", "path", "allowed", "body"), MATRIX)
@pytest.mark.parametrize("role", ROLES)
async def test_api_rbac_matrix_every_router_every_role(
    client: httpx.AsyncClient,
    tokens: dict[str, str],
    role: str,
    method: str,
    path: str,
    allowed: tuple[str, ...] | None,
    body: dict[str, Any] | None,
) -> None:
    """Every (router, role) pair answers the role gate the documentation promises."""
    response = await _call(client, method, path, body, tokens[role])
    permitted = allowed is None or role in allowed
    if permitted:
        assert response.status_code not in (401, 403), (
            f"{role} must be allowed on {method} {path}: {response.status_code} {response.text[:200]}"
        )
    else:
        assert response.status_code == 403, (
            f"{role} must be refused on {method} {path}: {response.status_code}"
        )
        assert response.json()["code"] == "FORBIDDEN"


@pytest.mark.parametrize(("method", "path", "allowed", "body"), MATRIX)
async def test_api_every_matrix_endpoint_requires_a_token(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    allowed: tuple[str, ...] | None,
    body: dict[str, Any] | None,
) -> None:
    """No endpoint of the matrix is reachable anonymously."""
    del allowed
    response = await _call(client, method, path, body, None)
    assert response.status_code == 401, f"{method} {path} answered {response.status_code}"
    assert response.json()["code"] == "AUTH_INVALID"


@pytest.mark.parametrize("path", OPEN_ENDPOINTS)
async def test_api_open_endpoints_need_no_token(client: httpx.AsyncClient, path: str) -> None:
    """Health and the consent text stay reachable without credentials."""
    response = await client.get(path)
    assert response.status_code in (200, 503)


async def test_api_student_cannot_activate_model_403(
    client: httpx.AsyncClient, tokens: dict[str, str]
) -> None:
    """The single most sensitive administrative action, spelled out."""
    response = await client.post(
        "/admin/models/activate",
        json={"component": "ocr", "version": "ocr-0.3.0"},
        headers=auth_header(tokens["student"]),
    )
    assert response.status_code == 403
    body = response.json()
    assert body["code"] == "FORBIDDEN"
    assert body["role"] == "student"
    assert "admin" in body["required"]


async def test_api_expired_or_forged_token_is_401(client: httpx.AsyncClient) -> None:
    """A token signed with another secret never authenticates."""
    import jwt

    forged = jwt.encode(
        {"sub": "01J000000000000000000000", "role": "admin", "typ": "access", "exp": 4102444800},
        "baska-bir-sir",
        algorithm="HS256",
    )
    response = await client.get("/users/me", headers=auth_header(forged))
    assert response.status_code == 401
    assert response.json()["code"] == "AUTH_INVALID"
