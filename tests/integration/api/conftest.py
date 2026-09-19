"""Fixtures for the API integration tests.

The application is driven through ``httpx.ASGITransport`` — the ASGI app is called in-process, so
no socket is opened and no uvicorn worker exists (docs/gelistirme/test-stratejisi.md §4). The
lifespan is entered explicitly with ``app.router.lifespan_context`` because ``ASGITransport``
does not run it; that is what starts the executor, the model registry and the migrated
in-memory database.

WebSocket flows cannot go through ``ASGITransport`` (httpx speaks HTTP only), so the two
``tests/integration/api/test_ws_*.py`` modules use ``starlette.testclient.TestClient``, which
also runs the app in-process (a portal thread, still no server socket).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI
import httpx
import pytest
import pytest_asyncio

from aicls.api import create_app
from aicls.core.settings import Settings, get_settings

#: Password used by every seeded account (>= 10 characters, api.md §3.2).
TEST_PASSWORD = "Kalkulus2026!"

#: Roles the RBAC matrix is built from.
ROLES: tuple[str, ...] = ("student", "instructor", "contributor", "admin")


@pytest.fixture
def api_settings(test_env: dict[str, str]) -> Iterator[Settings]:
    """Process-wide :class:`Settings` built from the hermetic ``test_env`` mapping."""
    get_settings.cache_clear()
    try:
        yield get_settings()
    finally:
        get_settings.cache_clear()


@pytest.fixture
def app(api_settings: Settings) -> FastAPI:
    """A FastAPI application on the ``full`` profile with stub models."""
    return create_app(api_settings)


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """An ``httpx.AsyncClient`` bound to ``app`` with the lifespan running."""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver/api/v1"
        ) as http_client:
            yield http_client


async def register(
    client: httpx.AsyncClient,
    email: str,
    *,
    display_name: str = "Test Kullanıcı",
    password: str = TEST_PASSWORD,
) -> dict[str, Any]:
    """Register one account and return the ``UserOut`` payload."""
    response = await client.post(
        "/auth/register",
        json={"email": email, "password": password, "display_name": display_name},
    )
    assert response.status_code == 201, response.text
    payload: dict[str, Any] = response.json()
    return payload


async def login(
    client: httpx.AsyncClient, email: str, *, password: str = TEST_PASSWORD
) -> dict[str, Any]:
    """Log in and return the ``TokenOut`` payload (the refresh cookie stays on ``client``)."""
    response = await client.post("/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    payload: dict[str, Any] = response.json()
    return payload


def auth_header(token: str) -> dict[str, str]:
    """``Authorization: Bearer <token>``."""
    return {"Authorization": f"Bearer {token}"}


async def account(
    client: httpx.AsyncClient, app: FastAPI, email: str, role: str = "student"
) -> tuple[dict[str, Any], str]:
    """Register an account, force ``role`` in the database and return ``(user, access token)``."""
    user = await register(client, email)
    if role != "student":
        set_role(app, user["id"], role)
    token = (await login(client, email))["access_token"]
    return user, token


def set_role(app: FastAPI, user_id: str, role: str) -> None:
    """Promote a user directly through the repository (no admin endpoint needed to bootstrap)."""
    from aicls.db import UserRepository, session_scope

    with session_scope(app.state.engine) as session:
        UserRepository(session).set_role(user_id, role)


@pytest.fixture
def book_datasets(repo_root: Path) -> Path:
    """Path of ``datasets/book`` (the İP-8 sample chapters)."""
    return repo_root / "datasets" / "book"


def load_book(app: FastAPI, book_datasets: Path) -> None:
    """Load the two sample Markdown chapters into the application's database."""
    from aicls.book import load_tag_result, tag_markdown
    from aicls.db import BookRepository, session_scope

    with session_scope(app.state.engine) as session:
        repo = BookRepository(session)
        for name, chapter in (
            ("sample/bolum-01-limit.md", "bolum-1"),
            ("sample/bolum-02-turev.md", "bolum-2"),
        ):
            path = book_datasets / name
            result = tag_markdown(
                path.read_text(encoding="utf-8"), chapter=chapter, source=path.name
            )
            load_tag_result(result, repo)
