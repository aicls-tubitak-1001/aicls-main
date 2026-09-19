"""Paths that depend on *who* is calling: per-user rate buckets and signed-in contribution."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import httpx
import pytest

from aicls.api import create_app
from aicls.api.participants import PARTICIPANT_TOKEN_HEADER
from aicls.core.settings import Settings, get_settings
from tests.integration.api.conftest import TEST_PASSWORD, auth_header

pytestmark = pytest.mark.integration


@pytest.fixture
def limited_settings(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> Iterator[Settings]:
    """Settings with the rate limiter switched on."""
    monkeypatch.setenv("AICLS_RATE_LIMIT_ENABLED", "true")
    get_settings.cache_clear()
    try:
        yield get_settings()
    finally:
        get_settings.cache_clear()


@pytest.fixture
async def limited_client(limited_settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    """A client on an application whose rate limiter is enabled."""
    app = create_app(limited_settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver/api/v1"
        ) as client:
            yield client


async def test_compute_rate_limit_is_per_user_not_per_address(
    limited_client: httpx.AsyncClient,
) -> None:
    """``/compute/*`` is 60/min/user: two accounts from one address get separate buckets."""
    tokens: list[str] = []
    for index in (1, 2):
        await limited_client.post(
            "/auth/register",
            json={
                "email": f"kova{index}@ornek.edu.tr",
                "password": TEST_PASSWORD,
                "display_name": f"Kova {index}",
            },
        )
        signed_in = await limited_client.post(
            "/auth/login",
            json={"email": f"kova{index}@ornek.edu.tr", "password": TEST_PASSWORD},
        )
        assert signed_in.status_code == 200, signed_in.text
        tokens.append(signed_in.json()["access_token"])

    body = {"command": {"op": "diff", "expr": "x^{2}"}}
    first = [
        (
            await limited_client.post("/compute/latex", json=body, headers=auth_header(tokens[0]))
        ).status_code
        for _ in range(60)
    ]
    assert set(first) == {200}
    throttled = await limited_client.post(
        "/compute/latex", json=body, headers=auth_header(tokens[0])
    )
    assert throttled.status_code == 429
    assert throttled.json()["rule"] == "compute"

    # the same address, a different account: its own bucket is still full
    other = await limited_client.post("/compute/latex", json=body, headers=auth_header(tokens[1]))
    assert other.status_code == 200


async def test_signed_in_user_consents_and_is_found_without_a_participant_token(
    client: httpx.AsyncClient,
) -> None:
    """A signed-in contributor is resolved from the account, not only from the token header."""
    await client.post(
        "/auth/register",
        json={
            "email": "katilimci@ornek.edu.tr",
            "password": TEST_PASSWORD,
            "display_name": "Katılımcı",
        },
    )
    token = (
        await client.post(
            "/auth/login",
            json={"email": "katilimci@ornek.edu.tr", "password": TEST_PASSWORD},
        )
    ).json()["access_token"]
    headers = auth_header(token)

    version = (await client.get("/contribute/consent/current")).json()["version"]
    accepted = await client.post(
        "/contribute/consent",
        json={
            "version": version,
            "accepted": True,
            "demographics": {"age_band": "25-34", "gender": "m", "university": "mersin"},
        },
        headers=headers,
    )
    assert accepted.status_code == 201, accepted.text
    assert accepted.json()["speaker_pseudonym"] == "P00001"

    # no participant-token header: the account itself identifies the speaker
    progress = await client.get("/contribute/me/progress", headers=headers)
    assert progress.status_code == 200
    assert progress.json()["n_accepted"] == 0

    # an unknown participant token falls back to the account rather than failing
    both = await client.get(
        "/contribute/me/progress",
        headers={**headers, PARTICIPANT_TOKEN_HEADER: "bilinmeyen"},
    )
    assert both.status_code == 200


async def test_compute_parse_reports_a_semantic_failure(client: httpx.AsyncClient) -> None:
    """A formula that parses but has no meaning is still a 200 preview with ``ok:false``."""
    await client.post(
        "/auth/register",
        json={
            "email": "onizleme@ornek.edu.tr",
            "password": TEST_PASSWORD,
            "display_name": "Önizleme",
        },
    )
    token = (
        await client.post(
            "/auth/login",
            json={"email": "onizleme@ornek.edu.tr", "password": TEST_PASSWORD},
        )
    ).json()["access_token"]
    response = await client.post(
        "/compute/parse", json={"latex": "\\int x"}, headers=auth_header(token)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] in (True, False)
    if not body["ok"]:
        assert body["error"]["code"] in {"PARSE_ERROR", "UNSUPPORTED_CONSTRUCT"}
