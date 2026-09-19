"""Stage budgets of the latency architecture (ADR-0014 §Karar).

The contract metric is "command → rendered result, average < 3 s". The browser half of it is
measured in production through ``POST /telemetry/latency``; what CI can measure is the *server*
half with the deterministic stub models, and ADR-0014 fixes those budgets:

===============================  ========
stage                            budget
===============================  ========
text path (p50, warm)            300 ms
CAS command                      150 ms
book resolution                  20 ms
===============================  ========

**Corpus provenance.** These numbers are measured on synthetic, in-repository inputs — four
Turkish command sentences from the İP-4 command inventory and the two sample chapters of
``datasets/book`` — with the stub recogniser and stub transcriber, in one process, on the
development host. They are a *regression fence*, not the contractual figure: the contractual
figure is the browser-measured mean over the pilot study (İP-9), and it cannot exist before the
pilot. Nothing here may be quoted as the "< 3 s" evidence.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
import statistics
import time

from fastapi import FastAPI
import httpx
import pytest

from aicls.api import create_app
from aicls.core.settings import Settings, get_settings

pytestmark = [pytest.mark.perf, pytest.mark.integration]

#: Budgets of ADR-0014 (stub models, server side, milliseconds).
TEXT_PATH_P50_MS = 300.0
CAS_BUDGET_MS = 150.0
BOOK_BUDGET_MS = 20.0

#: Measured samples per budget (plus a warm-up that is discarded).
SAMPLES = 20

#: Utterances measured, drawn from the İP-4 command inventory examples.
UTTERANCES: tuple[str, ...] = (
    "x karenin türevini al",
    "sıfırdan bire x kare integralini hesapla",
    "x sonsuza giderken bir bölü x limitini al",
    "x kare eksi dört eşittir sıfır denklemini çöz",
)


@pytest.fixture
def perf_settings(test_env: dict[str, str]) -> Iterator[Settings]:
    """Hermetic settings for the measurement run."""
    get_settings.cache_clear()
    try:
        yield get_settings()
    finally:
        get_settings.cache_clear()


@pytest.fixture
def perf_app(perf_settings: Settings) -> FastAPI:
    """A fresh application for the measurement run."""
    return create_app(perf_settings)


@pytest.fixture
async def perf_client(perf_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """A client with the lifespan running (executor started, models warmed)."""
    async with perf_app.router.lifespan_context(perf_app):
        transport = httpx.ASGITransport(app=perf_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver/api/v1"
        ) as client:
            yield client


async def _signed_in(client: httpx.AsyncClient) -> dict[str, str]:
    """Register and log in one student."""
    await client.post(
        "/auth/register",
        json={
            "email": "olcum@ornek.edu.tr",
            "password": "Kalkulus2026!",
            "display_name": "Ölçüm",
        },
    )
    token = (
        await client.post(
            "/auth/login", json={"email": "olcum@ornek.edu.tr", "password": "Kalkulus2026!"}
        )
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def test_latency_budget_text_path_p50_under_300ms(perf_client: httpx.AsyncClient) -> None:
    """The whole text path (NLU + parse + CAS + render) stays under the ADR-0014 p50 budget."""
    headers = await _signed_in(perf_client)
    # warm-up: the first call pays for the inventory, the grammar and the SymPy import
    warm = await perf_client.post("/compute/text", json={"text": UTTERANCES[0]}, headers=headers)
    assert warm.status_code == 200, warm.text

    durations: list[float] = []
    for index in range(SAMPLES):
        text = UTTERANCES[index % len(UTTERANCES)]
        started = time.perf_counter()
        response = await perf_client.post("/compute/text", json={"text": text}, headers=headers)
        durations.append((time.perf_counter() - started) * 1000.0)
        assert response.status_code == 200, response.text
        assert response.json()["kind"] == "result"

    p50 = statistics.median(durations)
    samples = ", ".join(f"{value:.1f}" for value in sorted(durations))
    assert p50 < TEXT_PATH_P50_MS, (
        f"text path p50 {p50:.1f} ms exceeds the {TEXT_PATH_P50_MS:.0f} ms budget "
        f"(samples: {samples})"
    )


async def test_latency_budget_cas_stage_under_150ms(perf_client: httpx.AsyncClient) -> None:
    """The CAS stage alone, as reported in ``ComputeOut.timings``, stays inside its budget."""
    headers = await _signed_in(perf_client)
    commands = (
        {"op": "diff", "expr": "x^{3} + 2 x"},
        {"op": "integrate", "expr": "x^{2}", "var": "x"},
        {"op": "factor", "expr": "x^{2} - 4"},
        {"op": "simplify", "expr": "x^{2} + 2 x"},
    )
    await perf_client.post("/compute/latex", json={"command": commands[0]}, headers=headers)

    stages: list[float] = []
    for index in range(SAMPLES):
        command = dict(commands[index % len(commands)])
        command["expr"] = f"{command['expr']} + {index}"
        response = await perf_client.post(
            "/compute/latex", json={"command": command}, headers=headers
        )
        assert response.status_code == 200, response.text
        stages.append(float(response.json()["timings"]["cas"]))

    p50 = statistics.median(stages)
    assert p50 < CAS_BUDGET_MS, f"cas p50 {p50:.1f} ms exceeds the {CAS_BUDGET_MS:.0f} ms budget"


async def test_latency_budget_book_resolution_under_20ms(
    perf_client: httpx.AsyncClient, perf_app: FastAPI, repo_root: Path
) -> None:
    """Resolving a book command stays inside its 20 ms budget with the LRU warm (İP-8 risk plan).

    ADR-0014 budgets the *resolution*, not the HTTP round trip, so the resolver is measured
    directly on the application's own database; the endpoint is exercised once to prove the two
    agree on the answer.
    """
    from aicls.book import BookResolver
    from aicls.db import BookRepository, session_scope
    from tests.integration.api.conftest import load_book

    load_book(perf_app, repo_root / "datasets" / "book")
    headers = await _signed_in(perf_client)
    over_http = await perf_client.post(
        "/book/resolve", json={"text": "türev tanımını getir"}, headers=headers
    )
    assert over_http.status_code == 200, over_http.text
    assert over_http.json()[0]["unit"]["id"] == "tanim-2.4"

    durations: list[float] = []
    with session_scope(perf_app.state.engine) as session:
        resolver = BookResolver(BookRepository(session))
        assert resolver.resolve_one("türev tanımını getir") is not None  # warm the LRU
        for _ in range(SAMPLES):
            started = time.perf_counter()
            resolved = resolver.resolve_one("türev tanımını getir")
            durations.append((time.perf_counter() - started) * 1000.0)
            assert resolved is not None and resolved.unit.id == "tanim-2.4"

    p50 = statistics.median(durations)
    assert p50 < BOOK_BUDGET_MS, (
        f"book resolution p50 {p50:.3f} ms exceeds the {BOOK_BUDGET_MS:.0f} ms budget"
    )
