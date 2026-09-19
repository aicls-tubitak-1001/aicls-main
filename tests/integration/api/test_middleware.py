"""Request id, ``Server-Timing``, security headers, CORS and rate limiting (ADR-0014, §4, §7)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

from fastapi import FastAPI
import httpx
import pytest

from aicls.api.middleware import SECURITY_HEADERS, SERVER_TIMING_HEADER
from aicls.core.settings import Settings, get_settings
from tests.integration.api.conftest import TEST_PASSWORD, account, auth_header

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
    from aicls.api import create_app

    app = create_app(limited_settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver/api/v1"
        ) as client:
            yield client


async def test_api_server_timing_header_present(client: httpx.AsyncClient, app: FastAPI) -> None:
    """Every response carries ``X-Request-Id`` and a ``Server-Timing`` line (ADR-0014)."""
    live = await client.get("/health/live")
    assert live.status_code == 200
    assert live.headers["X-Request-Id"]
    assert "app;dur=" in live.headers[SERVER_TIMING_HEADER]

    _user, token = await account(client, app, "zaman@ornek.edu.tr")
    computed = await client.post(
        "/compute/text", json={"text": "x karenin türevini al"}, headers=auth_header(token)
    )
    assert computed.status_code == 200
    timing = computed.headers[SERVER_TIMING_HEADER]
    assert "nlu;dur=" in timing
    assert "cas;dur=" in timing
    assert "render;dur=" in timing
    assert "app;dur=" in timing


async def test_api_request_id_is_echoed_when_the_client_supplies_one(
    client: httpx.AsyncClient,
) -> None:
    """A well-formed client correlation id is reused; a malformed one is replaced."""
    supplied = await client.get("/health/live", headers={"X-Request-Id": "web-01J-abc-123"})
    assert supplied.headers["X-Request-Id"] == "web-01J-abc-123"

    injected = await client.get("/health/live", headers={"X-Request-Id": "bad value\r\nSet-Cookie"})
    assert injected.headers["X-Request-Id"] != "bad value\r\nSet-Cookie"


async def test_api_error_responses_also_carry_the_headers(client: httpx.AsyncClient) -> None:
    """A problem document is a response like any other: it is stamped too."""
    response = await client.get("/users/me")
    assert response.status_code == 401
    assert response.headers["X-Request-Id"]
    assert SERVER_TIMING_HEADER in response.headers
    assert response.json()["instance"] == response.headers["X-Request-Id"]


@pytest.mark.parametrize("header", sorted(SECURITY_HEADERS))
async def test_api_security_headers_are_set(client: httpx.AsyncClient, header: str) -> None:
    """The docs/guvenlik-gizlilik.md §4 headers are on every response."""
    response = await client.get("/health/live")
    assert response.headers[header] == SECURITY_HEADERS[header]


async def test_api_cors_allows_only_the_configured_origin(client: httpx.AsyncClient) -> None:
    """``AICLS_CORS_ORIGINS`` is an allow-list, not a wildcard."""
    allowed = await client.get("/health/live", headers={"Origin": "http://testserver"})
    assert allowed.headers.get("access-control-allow-origin") == "http://testserver"

    refused = await client.get("/health/live", headers={"Origin": "http://kotu.example"})
    assert "access-control-allow-origin" not in refused.headers


async def test_api_rate_limit_returns_429_with_retry_after(
    limited_client: httpx.AsyncClient,
) -> None:
    """``/auth/*`` is 5 requests per minute per address; the sixth is a 429 with ``Retry-After``."""
    body = {"email": "hiz@ornek.edu.tr", "password": TEST_PASSWORD}
    statuses = [(await limited_client.post("/auth/login", json=body)).status_code for _ in range(5)]
    assert all(code == 401 for code in statuses), statuses

    throttled = await limited_client.post("/auth/login", json=body)
    assert throttled.status_code == 429
    assert int(throttled.headers["Retry-After"]) >= 1
    document = throttled.json()
    assert document["code"] == "RATE_LIMITED"
    assert document["rule"] == "auth_ip"
    assert document["limit"] == 5
    assert response_is_problem(throttled)


def response_is_problem(response: httpx.Response) -> bool:
    """Whether ``response`` is an ``application/problem+json`` document."""
    return response.headers["content-type"].startswith("application/problem+json")


async def test_api_rate_limit_does_not_touch_unlimited_paths(
    limited_client: httpx.AsyncClient,
) -> None:
    """Health is never rate limited: a probe must not be able to lock out monitoring."""
    for _ in range(40):
        assert (await limited_client.get("/health/live")).status_code == 200
