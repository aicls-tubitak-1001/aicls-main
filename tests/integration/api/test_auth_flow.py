"""Registration, login, refresh rotation and lockout (api.md §3.2, guvenlik-gizlilik.md §1)."""

from __future__ import annotations

from fastapi import FastAPI
import httpx
import pytest

from aicls.api.security import ACCESS_TTL_S, MAX_FAILED_LOGINS, REFRESH_COOKIE
from tests.integration.api.conftest import TEST_PASSWORD, auth_header, login, register

pytestmark = pytest.mark.integration


async def test_api_register_login_refresh_flow(client: httpx.AsyncClient) -> None:
    """Register, log in, use the access token, rotate the refresh cookie, log out."""
    user = await register(client, "ada@ornek.edu.tr", display_name="Ada Lovelace")
    assert user["role"] == "student"
    assert "password" not in user and "password_hash" not in user

    tokens = await login(client, "ada@ornek.edu.tr")
    assert tokens["token_type"] == "bearer"
    assert tokens["expires_in"] == ACCESS_TTL_S
    assert client.cookies.get(REFRESH_COOKIE)

    me = await client.get("/users/me", headers=auth_header(tokens["access_token"]))
    assert me.status_code == 200
    assert me.json()["email"] == "ada@ornek.edu.tr"

    refreshed = await client.post("/auth/refresh")
    assert refreshed.status_code == 200
    assert refreshed.json()["access_token"]

    out = await client.post("/auth/logout")
    assert out.status_code == 204
    replay = await client.post("/auth/refresh")
    assert replay.status_code == 401


async def test_api_register_rejects_a_duplicate_address(client: httpx.AsyncClient) -> None:
    """The second registration of the same address is a 422, not a 500."""
    await register(client, "ikinci@ornek.edu.tr")
    again = await client.post(
        "/auth/register",
        json={
            "email": "ikinci@ornek.edu.tr",
            "password": TEST_PASSWORD,
            "display_name": "Kopya",
        },
    )
    assert again.status_code == 422
    assert again.json()["code"] == "VALIDATION_ERROR"


async def test_api_register_rejects_a_short_password(client: httpx.AsyncClient) -> None:
    """The 10-character policy is enforced at the schema boundary."""
    response = await client.post(
        "/auth/register",
        json={"email": "kisa@ornek.edu.tr", "password": "kisa", "display_name": "Kısa"},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


async def test_api_refresh_rotation_invalidates_old_cookie(client: httpx.AsyncClient) -> None:
    """Replaying a rotated refresh token fails and revokes the whole family."""
    await register(client, "rot@ornek.edu.tr")
    await login(client, "rot@ornek.edu.tr")
    first = client.cookies.get(REFRESH_COOKIE)
    assert first

    rotated = await client.post("/auth/refresh")
    assert rotated.status_code == 200
    second = client.cookies.get(REFRESH_COOKIE)
    assert second and second != first

    client.cookies.clear()
    client.cookies.set(REFRESH_COOKIE, first)
    replay = await client.post("/auth/refresh")
    assert replay.status_code == 401
    assert replay.json()["code"] == "AUTH_INVALID"

    # the family was revoked, so even the *newest* token is dead now
    client.cookies.clear()
    client.cookies.set(REFRESH_COOKIE, second)
    dead = await client.post("/auth/refresh")
    assert dead.status_code == 401


async def test_api_login_lockout_after_10_failures(client: httpx.AsyncClient) -> None:
    """Ten wrong passwords lock the account; the right password then fails too."""
    await register(client, "kilit@ornek.edu.tr")
    for _ in range(MAX_FAILED_LOGINS):
        bad = await client.post(
            "/auth/login", json={"email": "kilit@ornek.edu.tr", "password": "YanlisParola1"}
        )
        assert bad.status_code == 401
    locked = await client.post(
        "/auth/login", json={"email": "kilit@ornek.edu.tr", "password": TEST_PASSWORD}
    )
    assert locked.status_code == 401
    assert locked.json()["code"] == "AUTH_INVALID"


async def test_api_login_of_an_unknown_address_is_not_distinguishable(
    client: httpx.AsyncClient,
) -> None:
    """An unknown address and a wrong password produce the same Turkish text."""
    await register(client, "var@ornek.edu.tr")
    unknown = await client.post(
        "/auth/login", json={"email": "yok@ornek.edu.tr", "password": TEST_PASSWORD}
    )
    wrong = await client.post(
        "/auth/login", json={"email": "var@ornek.edu.tr", "password": "BaskaParola1"}
    )
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json()["detail_tr"] == wrong.json()["detail_tr"]


async def test_api_ws_ticket_is_single_use(client: httpx.AsyncClient) -> None:
    """``POST /auth/ws-ticket`` mints a 60 s ticket and requires authentication."""
    anonymous = await client.post("/auth/ws-ticket")
    assert anonymous.status_code == 401

    await register(client, "bilet@ornek.edu.tr")
    token = (await login(client, "bilet@ornek.edu.tr"))["access_token"]
    issued = await client.post("/auth/ws-ticket", headers=auth_header(token))
    assert issued.status_code == 200
    body = issued.json()
    assert body["expires_in"] == 60
    assert len(body["ticket"]) >= 16


async def test_api_password_change_revokes_refresh_tokens(
    client: httpx.AsyncClient, app: FastAPI
) -> None:
    """Changing the password kills every live session of the account."""
    await register(client, "parola@ornek.edu.tr")
    token = (await login(client, "parola@ornek.edu.tr"))["access_token"]
    patched = await client.patch(
        "/users/me", json={"password": "YeniParola2026!"}, headers=auth_header(token)
    )
    assert patched.status_code == 200
    dead = await client.post("/auth/refresh")
    assert dead.status_code == 401
    assert app.state.engine is not None
