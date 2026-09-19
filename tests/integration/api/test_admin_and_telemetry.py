"""Feedback, SUS, latency telemetry and the administration surface (api.md §3.9–§3.11)."""

from __future__ import annotations

import io
from pathlib import Path
import zipfile

from fastapi import FastAPI
import httpx
import pytest

from tests.integration.api.conftest import account, auth_header

pytestmark = pytest.mark.integration

#: SUS answers that score 100 (odd items 5, even items 1) — the ADR-0015 definition.
SUS_PERFECT: list[int] = [5, 1, 5, 1, 5, 1, 5, 1, 5, 1]


@pytest.fixture
async def student(client: httpx.AsyncClient, app: FastAPI) -> dict[str, str]:
    """A signed-in student."""
    _user, token = await account(client, app, "geribildirim@ornek.edu.tr")
    return auth_header(token)


@pytest.fixture
async def admin(client: httpx.AsyncClient, app: FastAPI) -> dict[str, str]:
    """A signed-in administrator."""
    _user, token = await account(client, app, "yonetici@ornek.edu.tr", "admin")
    return auth_header(token)


async def test_api_feedback_sus_score_computed(
    client: httpx.AsyncClient, student: dict[str, str]
) -> None:
    """The SUS score follows the frozen ADR-0015 definition (contract target > 75)."""
    perfect = await client.post(
        "/feedback/sus", json={"answers": SUS_PERFECT, "cohort": "pilot"}, headers=student
    )
    assert perfect.status_code == 201, perfect.text
    assert perfect.json()["score"] == 100.0

    neutral = await client.post(
        "/feedback/sus", json={"answers": [3] * 10, "cohort": "pilot"}, headers=student
    )
    assert neutral.json()["score"] == 50.0

    invalid = await client.post(
        "/feedback/sus", json={"answers": [9] * 10, "cohort": "pilot"}, headers=student
    )
    assert invalid.status_code == 422
    assert invalid.json()["code"] == "VALIDATION_ERROR"

    too_few = await client.post(
        "/feedback/sus", json={"answers": [3] * 9, "cohort": "pilot"}, headers=student
    )
    assert too_few.status_code == 422


async def test_api_feedback_is_created_listed_and_triaged(
    client: httpx.AsyncClient, student: dict[str, str], admin: dict[str, str]
) -> None:
    """A student files feedback; staff list and triage it (İP-9)."""
    created = await client.post(
        "/feedback",
        json={"category": "critical", "text": "Sonuç görünmüyor", "page": "/workspace"},
        headers=student,
    )
    assert created.status_code == 201, created.text
    feedback_id = created.json()["id"]
    assert created.json()["status"] == "open"

    denied = await client.get("/feedback", headers=student)
    assert denied.status_code == 403

    listed = await client.get("/feedback", params={"category": "critical"}, headers=admin)
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()["items"]] == [feedback_id]

    triaged = await client.patch(
        f"/admin/feedback/{feedback_id}", json={"status": "triaged"}, headers=admin
    )
    assert triaged.status_code == 200
    assert triaged.json()["status"] == "triaged"


async def test_api_telemetry_latency_stores_client_and_server_ms(
    client: httpx.AsyncClient, student: dict[str, str], admin: dict[str, str]
) -> None:
    """The browser's render time is merged with the server stages and reported by the admin API."""
    workspace = await client.post("/workspaces", json={"title": "Ölçüm"}, headers=student)
    computed = await client.post(
        "/compute/text",
        json={"text": "x karenin türevini al", "workspace_id": workspace.json()["id"]},
        headers=student,
    )
    operation_id = computed.json()["operation_id"]

    sample = await client.post(
        "/telemetry/latency",
        json={
            "operation_id": operation_id,
            "modality": "text",
            "client_submit_ts": 1000.0,
            "client_render_ts": 1450.0,
            "render_ms": 22.5,
        },
        headers=student,
    )
    assert sample.status_code == 204, sample.text

    stats = await client.get("/admin/metrics/latency", headers=admin)
    assert stats.status_code == 200, stats.text
    body = stats.json()
    assert body["n"] == 1
    assert body["mean_ms"] == pytest.approx(450.0)
    assert body["p50_ms"] == pytest.approx(450.0)
    assert body["by_modality"]["text"] == pytest.approx(450.0)


async def test_api_telemetry_rejects_a_foreign_operation(
    client: httpx.AsyncClient, app: FastAPI, student: dict[str, str]
) -> None:
    """A telemetry sample cannot be attached to somebody else's operation."""
    workspace = await client.post("/workspaces", json={"title": "Ölçüm"}, headers=student)
    computed = await client.post(
        "/compute/text",
        json={"text": "x karenin türevini al", "workspace_id": workspace.json()["id"]},
        headers=student,
    )
    _user, other = await account(client, app, "baska@ornek.edu.tr")
    response = await client.post(
        "/telemetry/latency",
        json={
            "operation_id": computed.json()["operation_id"],
            "modality": "text",
            "client_submit_ts": 1.0,
            "client_render_ts": 2.0,
            "render_ms": 1.0,
        },
        headers=auth_header(other),
    )
    assert response.status_code == 404


async def test_api_admin_model_registry_activate_and_reload(
    client: httpx.AsyncClient, app: FastAPI, admin: dict[str, str]
) -> None:
    """Activating a checkpoint reloads the registry and writes an audit row."""
    from aicls.db import ModelVersionRepository, session_scope

    with session_scope(app.state.engine) as session:
        ModelVersionRepository(session).register(
            "ocr", "ocr-0.3.0", "models/ocr-0.3.0", "sha256:" + "0" * 64, {"cer": 0.09}
        )

    listed = await client.get("/admin/models", headers=admin)
    assert listed.status_code == 200
    assert listed.json()[0]["version"] == "ocr-0.3.0"
    assert listed.json()[0]["is_active"] is False

    activated = await client.post(
        "/admin/models/activate", json={"component": "ocr", "version": "ocr-0.3.0"}, headers=admin
    )
    assert activated.status_code == 204, activated.text

    again = await client.get("/admin/models", headers=admin)
    assert again.json()[0]["is_active"] is True

    audit = await client.get("/admin/audit", params={"action": "model_activate"}, headers=admin)
    assert audit.status_code == 200
    rows = audit.json()["items"]
    assert rows and rows[0]["target_id"] == "ocr-0.3.0"

    missing = await client.post(
        "/admin/models/activate", json={"component": "asr", "version": "yok"}, headers=admin
    )
    assert missing.status_code == 404


async def test_api_admin_users_listing_and_role_change(
    client: httpx.AsyncClient, app: FastAPI, admin: dict[str, str]
) -> None:
    """An administrator can page the accounts and change a role (audited)."""
    user, _token = await account(client, app, "terfi@ornek.edu.tr")
    listed = await client.get("/admin/users", params={"limit": 50}, headers=admin)
    assert listed.status_code == 200
    assert user["id"] in {item["id"] for item in listed.json()["items"]}

    changed = await client.patch(
        f"/admin/users/{user['id']}/role", json={"role": "instructor"}, headers=admin
    )
    assert changed.status_code == 204

    audit = await client.get("/admin/audit", params={"action": "role_change"}, headers=admin)
    assert audit.json()["items"][0]["detail"]["role"] == "instructor"


async def test_api_admin_qc_queue_and_review(
    client: httpx.AsyncClient, app: FastAPI, admin: dict[str, str]
) -> None:
    """A stored recording appears in the review queue and accepts a verdict."""
    from aicls.asr.audio_io import write_wav_pcm16
    from aicls.asr.synth import make_speech_like
    from tests.integration.api.test_contribute import consent, seed_prompts

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

    queue = await client.get("/admin/qc/queue", headers=admin)
    assert queue.status_code == 200
    assert [item["id"] for item in queue.json()] == [recording_id]

    reviewed = await client.post(
        f"/admin/qc/{recording_id}/review",
        json={"verdict": "fixed", "transcript_fix": "x karenin türevini al"},
        headers=admin,
    )
    assert reviewed.status_code == 204

    empty = await client.get("/admin/qc/queue", headers=admin)
    assert empty.json() == []
    done = await client.get("/admin/qc/queue", params={"status": "fixed"}, headers=admin)
    assert [item["id"] for item in done.json()] == [recording_id]


async def test_api_admin_contributions_export_writes_a_manifest(
    client: httpx.AsyncClient, app: FastAPI, admin: dict[str, str]
) -> None:
    """The export writes a speaker-disjoint manifest and reports its fingerprint."""
    from aicls.asr.audio_io import write_wav_pcm16
    from aicls.asr.synth import make_speech_like
    from tests.integration.api.test_contribute import consent, seed_prompts

    seed_prompts(app)
    headers = await consent(client)
    prompt_id = (await client.get("/contribute/prompts/next", headers=headers)).json()[0]["id"]
    await client.post(
        "/contribute/recordings",
        files={"audio": ("a.wav", write_wav_pcm16(make_speech_like()), "audio/wav")},
        data={"prompt_id": prompt_id},
        headers=headers,
    )

    response = await client.post("/admin/contributions/export", headers=admin)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["rows"] == 1
    assert body["speakers"] == 1
    assert body["fingerprint"].startswith("sha256:")
    assert Path(body["path"]).is_file()


async def test_api_admin_book_load_refuses_untagged_headings(
    client: httpx.AsyncClient, admin: dict[str, str], book_datasets: Path
) -> None:
    """``POST /admin/book/load`` is all-or-nothing: 100 % tagged or 422 (İP-8 criterion)."""
    good = io.BytesIO()
    with zipfile.ZipFile(good, "w") as archive:
        for name in ("sample/bolum-01-limit.md", "sample/bolum-02-turev.md"):
            archive.writestr(Path(name).name, (book_datasets / name).read_text(encoding="utf-8"))
    loaded = await client.post(
        "/admin/book/load",
        files={"file": ("kitap.zip", good.getvalue(), "application/zip")},
        headers=admin,
    )
    assert loaded.status_code == 200, loaded.text
    assert loaded.json()["tagged_ratio"] == 1.0
    assert loaded.json()["units"] > 0

    bad = io.BytesIO()
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr(
            "bolum-03-etiketsiz.md",
            (book_datasets / "invalid/bolum-03-etiketsiz.md").read_text(encoding="utf-8"),
        )
    refused = await client.post(
        "/admin/book/load",
        files={"file": ("kitap.zip", bad.getvalue(), "application/zip")},
        headers=admin,
    )
    assert refused.status_code == 422
    assert refused.json()["code"] in {"VALIDATION_ERROR", "PARSE_ERROR"}


async def test_api_admin_book_load_refuses_a_non_zip(
    client: httpx.AsyncClient, admin: dict[str, str]
) -> None:
    """A payload that is not a zip is an ``UPLOAD_INVALID``, not a stack trace."""
    response = await client.post(
        "/admin/book/load",
        files={"file": ("kitap.zip", b"bu bir zip degil", "application/zip")},
        headers=admin,
    )
    assert response.status_code == 400
    assert response.json()["reason"] == "not_a_zip"
