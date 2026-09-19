"""Profile gating (ADR-0018) and the OpenAPI contract (api.md §5).

``docs/api/openapi.json`` is the machine-readable half of the İP-2 "API Dokümantasyonu"
deliverable and the source ``web/src/api/generated/schema.d.ts`` is produced from. It is checked
in, so any change to a route or a schema has to be regenerated deliberately —
``test_openapi_schema_matches_the_checked_in_document`` is the drift gate CI reruns.
"""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path

from fastapi import FastAPI
import httpx
import pytest

from aicls.api import create_app
from aicls.api.app import API_PREFIX
from aicls.api.routers import COLLECT_ROUTERS, FULL_ROUTERS, routers_for
from aicls.core.settings import Settings, get_settings

pytestmark = pytest.mark.integration

#: Path prefixes that must NOT exist on the collection VM (ADR-0018).
FORBIDDEN_ON_COLLECT: tuple[str, ...] = (
    "/compute",
    "/ocr",
    "/asr",
    "/book",
    "/workspaces",
    "/telemetry",
    "/admin/models",
    "/admin/eval-runs",
    "/admin/metrics",
)

#: Path prefixes the collection VM must serve.
REQUIRED_ON_COLLECT: tuple[str, ...] = (
    "/health",
    "/auth",
    "/users",
    "/contribute",
    "/feedback",
    "/admin/qc",
)


@pytest.fixture
def collect_settings(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> Iterator[Settings]:
    """Settings on the ``collect`` profile."""
    monkeypatch.setenv("AICLS_PROFILE", "collect")
    get_settings.cache_clear()
    try:
        yield get_settings()
    finally:
        get_settings.cache_clear()


def paths_of(app: FastAPI) -> set[str]:
    """Every mounted REST path, without the ``/api/v1`` prefix.

    Read from the generated OpenAPI document rather than ``app.routes``: Starlette 1.6 keeps
    included routers as opaque wrappers, and the document is anyway the surface a client sees.
    """
    return {
        path.removeprefix(API_PREFIX)
        for path in app.openapi()["paths"]
        if path.startswith(API_PREFIX)
    }


def test_api_collect_profile_mounts_only_allowed_routers(collect_settings: Settings) -> None:
    """The collection VM exposes the tab it hosts and nothing that needs the GPU host."""
    app = create_app(collect_settings)
    paths = paths_of(app)
    for prefix in REQUIRED_ON_COLLECT:
        assert any(path.startswith(prefix) for path in paths), f"{prefix} missing on collect"
    for prefix in FORBIDDEN_ON_COLLECT:
        assert not any(path.startswith(prefix) for path in paths), f"{prefix} leaked onto collect"
    assert routers_for("collect") == COLLECT_ROUTERS
    assert routers_for("full") == FULL_ROUTERS
    assert routers_for("bilinmeyen") == FULL_ROUTERS


async def test_api_collect_profile_serves_the_contribution_flow(
    collect_settings: Settings,
) -> None:
    """The routers that *are* mounted on ``collect`` actually work there."""
    app = create_app(collect_settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver/api/v1"
        ) as client:
            assert (await client.get("/health/live")).status_code == 200
            consent = await client.get("/contribute/consent/current")
            assert consent.status_code == 200
            assert consent.json()["version"]
            assert (await client.post("/compute/text", json={"text": "x"})).status_code == 404


def test_api_full_profile_mounts_every_documented_router(app: FastAPI) -> None:
    """Every router named in api.md §3 exists on the workstation profile."""
    paths = paths_of(app)
    for prefix in (
        "/health",
        "/auth",
        "/users",
        "/workspaces",
        "/compute",
        "/ocr",
        "/asr",
        "/book",
        "/contribute",
        "/feedback",
        "/telemetry",
        "/admin",
    ):
        assert any(path.startswith(prefix) for path in paths), f"{prefix} missing on full"


def test_openapi_document_is_well_formed(app: FastAPI) -> None:
    """The generated document names the version, the base path and the two error media types."""
    schema = app.openapi()
    assert schema["info"]["title"] == "AI-CLS API"
    assert schema["openapi"].startswith("3.")
    assert f"{API_PREFIX}/compute/text" in schema["paths"]
    assert f"{API_PREFIX}/auth/login" in schema["paths"]
    recognize = schema["paths"][f"{API_PREFIX}/ocr/recognize"]["post"]
    assert set(recognize["requestBody"]["content"]) == {
        "application/json",
        "multipart/form-data",
    }


def test_openapi_schema_matches_the_checked_in_document(app: FastAPI, repo_root: Path) -> None:
    """``docs/api/openapi.json`` is not stale (the CI ``openapi-drift`` gate, api.md §5)."""
    import sys

    sys.path.insert(0, str(repo_root / "scripts"))
    from export_openapi import dump

    target = repo_root / "docs" / "api" / "openapi.json"
    assert target.is_file(), "run `python scripts/export_openapi.py` and commit the result"
    current = json.loads(target.read_text(encoding="utf-8"))
    generated = json.loads(dump(app.openapi()))
    assert current == generated, (
        "docs/api/openapi.json is stale — run `python scripts/export_openapi.py`"
    )


def test_openapi_is_stable_across_two_builds(api_settings: Settings) -> None:
    """Two applications built from the same settings produce byte-identical schemas."""
    first = json.dumps(create_app(api_settings).openapi(), sort_keys=True)
    second = json.dumps(create_app(api_settings).openapi(), sort_keys=True)
    assert first == second
