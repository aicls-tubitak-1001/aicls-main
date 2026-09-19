"""The voice data-collection tab (İP-3, api.md §3.8, docs/guvenlik-gizlilik.md §2)."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import FastAPI
import httpx
import pytest

from aicls.api.participants import PARTICIPANT_TOKEN_HEADER
from aicls.asr.audio_io import write_wav_pcm16
from aicls.asr.synth import make_noise, make_speech_like
from aicls.core.settings import Settings, get_settings
from tests.integration.api.conftest import account, auth_header

pytestmark = pytest.mark.integration

#: Prompts seeded before a recording can be uploaded.
PROMPTS: tuple[dict[str, str], ...] = (
    {"text_tr": "x karenin türevini al", "scenario_tag": "turev"},
    {"text_tr": "sinüs x'in integralini hesapla", "scenario_tag": "integral"},
    {"text_tr": "türev tanımını getir", "scenario_tag": "kitap"},
)


def seed_prompts(app: FastAPI) -> None:
    """Insert the reading prompts the collection flow hands out."""
    from aicls.db import RecordingRepository, session_scope

    with session_scope(app.state.engine) as session:
        RecordingRepository(session).add_prompts(PROMPTS)


async def consent(client: httpx.AsyncClient) -> dict[str, str]:
    """Accept the current consent version and return the participant header."""
    current = await client.get("/contribute/consent/current")
    assert current.status_code == 200, current.text
    accepted = await client.post(
        "/contribute/consent",
        json={
            "version": current.json()["version"],
            "accepted": True,
            "demographics": {"age_band": "18-24", "gender": "f", "university": "mus"},
        },
    )
    assert accepted.status_code == 201, accepted.text
    body = accepted.json()
    assert body["speaker_pseudonym"].startswith("P")
    return {PARTICIPANT_TOKEN_HEADER: body["token"]}


@pytest.fixture
def blocked_settings(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> Iterator[Settings]:
    """Settings with the ethics approval reference cleared (the gate closed)."""
    monkeypatch.setenv("AICLS_ETHICS_APPROVAL_REF", "")
    get_settings.cache_clear()
    try:
        yield get_settings()
    finally:
        get_settings.cache_clear()


async def test_api_contribute_requires_consent_before_upload(
    client: httpx.AsyncClient, app: FastAPI
) -> None:
    """Without an accepted consent every participant endpoint answers 403 CONSENT_REQUIRED."""
    seed_prompts(app)
    prompts = await client.get("/contribute/prompts/next")
    assert prompts.status_code == 403
    assert prompts.json()["code"] == "CONSENT_REQUIRED"

    upload = await client.post(
        "/contribute/recordings",
        files={"audio": ("a.wav", write_wav_pcm16(make_speech_like()), "audio/wav")},
        data={"prompt_id": "01J000000000000000000000"},
    )
    assert upload.status_code == 403
    assert upload.json()["code"] == "CONSENT_REQUIRED"


async def test_api_contribute_upload_runs_qc_and_returns_reasons_tr(
    client: httpx.AsyncClient, app: FastAPI
) -> None:
    """Quality control is a 200 payload: accepted, or rejected with Turkish reasons."""
    seed_prompts(app)
    headers = await consent(client)
    prompts = await client.get("/contribute/prompts/next", params={"n": 2}, headers=headers)
    assert prompts.status_code == 200, prompts.text
    prompt_id = prompts.json()[0]["id"]

    good = await client.post(
        "/contribute/recordings",
        files={"audio": ("a.wav", write_wav_pcm16(make_speech_like()), "audio/wav")},
        data={"prompt_id": prompt_id},
        headers=headers,
    )
    assert good.status_code == 200, good.text
    body = good.json()
    assert body["accepted"] is True
    assert body["qc"]["passed"] is True
    assert body["qc"]["reasons_tr"] == []
    assert body["id"]

    bad_prompt = prompts.json()[1]["id"]
    bad = await client.post(
        "/contribute/recordings",
        files={
            "audio": (
                "b.wav",
                write_wav_pcm16(make_noise(duration_s=0.2, amplitude=0.3)),
                "audio/wav",
            )
        },
        data={"prompt_id": bad_prompt},
        headers=headers,
    )
    assert bad.status_code == 200, bad.text
    rejected = bad.json()
    assert rejected["accepted"] is False
    assert rejected["id"] is None
    assert rejected["qc"]["reasons_tr"], "a rejection must tell the participant what went wrong"
    assert all(isinstance(reason, str) for reason in rejected["qc"]["reasons_tr"])


async def test_api_contribute_progress_and_recording_ownership(
    client: httpx.AsyncClient, app: FastAPI
) -> None:
    """A participant sees only their own recordings and their own progress."""
    seed_prompts(app)
    headers = await consent(client)
    prompt_id = (await client.get("/contribute/prompts/next", headers=headers)).json()[0]["id"]
    stored = await client.post(
        "/contribute/recordings",
        files={"audio": ("a.wav", write_wav_pcm16(make_speech_like()), "audio/wav")},
        data={"prompt_id": prompt_id},
        headers=headers,
    )
    recording_id = stored.json()["id"]

    mine = await client.get(f"/contribute/recordings/{recording_id}", headers=headers)
    assert mine.status_code == 200
    assert mine.json()["accepted"] is True

    progress = await client.get("/contribute/me/progress", headers=headers)
    assert progress.status_code == 200
    assert progress.json()["n_accepted"] == 1
    assert progress.json()["seconds_accepted"] > 0

    other = await consent(client)
    denied = await client.get(f"/contribute/recordings/{recording_id}", headers=other)
    assert denied.status_code == 404

    removed = await client.delete(f"/contribute/recordings/{recording_id}", headers=headers)
    assert removed.status_code == 204
    gone = await client.get(f"/contribute/recordings/{recording_id}", headers=headers)
    assert gone.status_code == 404


async def test_api_contribute_withdraw_is_not_gated(
    client: httpx.AsyncClient, app: FastAPI
) -> None:
    """KVKK withdrawal works and makes the participant token stop working."""
    seed_prompts(app)
    headers = await consent(client)
    withdrawn = await client.post("/contribute/withdraw", headers=headers)
    assert withdrawn.status_code == 204
    after = await client.get("/contribute/me/progress", headers=headers)
    assert after.status_code == 403


async def test_api_contribute_rejects_identity_fields_in_demographics(
    client: httpx.AsyncClient,
) -> None:
    """``extra="forbid"``: a request smuggling a name fails validation (KVKK minimisation)."""
    current = await client.get("/contribute/consent/current")
    response = await client.post(
        "/contribute/consent",
        json={
            "version": current.json()["version"],
            "accepted": True,
            "demographics": {"age_band": "18-24", "name": "Ada", "email": "a@b.c"},
        },
    )
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


async def test_api_contribute_stats_is_staff_only(client: httpx.AsyncClient, app: FastAPI) -> None:
    """The dashboard is instructor/admin; the numbers come from the repository."""
    seed_prompts(app)
    headers = await consent(client)
    prompt_id = (await client.get("/contribute/prompts/next", headers=headers)).json()[0]["id"]
    await client.post(
        "/contribute/recordings",
        files={"audio": ("a.wav", write_wav_pcm16(make_speech_like()), "audio/wav")},
        data={"prompt_id": prompt_id},
        headers=headers,
    )
    _user, token = await account(client, app, "istatistik@ornek.edu.tr", "instructor")
    stats = await client.get("/contribute/stats", headers=auth_header(token))
    assert stats.status_code == 200, stats.text
    body = stats.json()
    assert body["utterances"] == 1
    assert body["speakers"] == 1
    assert body["by_university"] == {"mus": 1}
    assert body["qc_pass_rate"] == 1.0


async def test_api_contribute_blocked_without_ethics_ref_503(
    blocked_settings: Settings,
) -> None:
    """With ``AICLS_ETHICS_APPROVAL_REF`` empty the whole tab answers 503 GATE_BLOCKED."""
    from aicls.api import create_app

    app = create_app(blocked_settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver/api/v1"
        ) as gated:
            for method, path in (
                ("GET", "/contribute/consent/current"),
                ("GET", "/contribute/prompts/next"),
                ("GET", "/contribute/me/progress"),
            ):
                response = await gated.request(method, path)
                assert response.status_code == 503, f"{path}: {response.status_code}"
                body = response.json()
                assert body["code"] == "GATE_BLOCKED"
                assert body["setting"] == "AICLS_ETHICS_APPROVAL_REF"
            # health stays up, so the VM is still monitorable while the tab is closed
            assert (await gated.get("/health/live")).status_code == 200
