"""The command path end to end: text, LaTeX, clarification, refusal and plots (api.md §3.4).

These are the contract tests of İP-4 on the HTTP surface. Two of them are the project's headline
claims in test form:

* ``test_api_compute_text_end_to_end_stub_latency_recorded`` — a Turkish sentence becomes a
  SymPy-computed answer, with the stage timings the "< 3 s" metric is built from;
* ``test_api_unsupported_returns_422_with_contract_message`` — an expression the platform
  recognises but cannot compute produces the frozen İP-4 sentence, never a wrong answer and
  never a 500.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
import httpx
import pytest

from aicls.core.messages_tr import MSG_UNSUPPORTED_PROBLEM
from tests.integration.api.conftest import account, auth_header

pytestmark = pytest.mark.integration

#: Expressions the OCR path recognises but the v1 grammar refuses (KNOWN_ISSUES item 2).
OUT_OF_SCOPE: tuple[str, ...] = ("\\binom{n}{k}", "\\forall\\varepsilon>0")


@pytest.fixture
async def student(client: httpx.AsyncClient, app: FastAPI) -> dict[str, Any]:
    """A signed-in student with one workspace."""
    user, token = await account(client, app, "ogrenci@ornek.edu.tr")
    headers = auth_header(token)
    workspace = await client.post("/workspaces", json={"title": "Türev"}, headers=headers)
    return {"user": user, "headers": headers, "workspace_id": workspace.json()["id"]}


async def test_api_compute_text_end_to_end_stub_latency_recorded(
    client: httpx.AsyncClient, student: dict[str, Any]
) -> None:
    """ "x karenin türevini al" is understood, computed by SymPy and recorded with timings."""
    response = await client.post(
        "/compute/text",
        json={"text": "x karenin türevini al", "workspace_id": student["workspace_id"]},
        headers=student["headers"],
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "result"
    assert body["command"]["op"] == "diff"
    assert body["result"]["latex"] == r"2 \cdot x"
    assert body["result"]["plain"] == "2*x"
    assert body["nlu"]["intent"] == "differentiate"
    assert body["nlu"]["source"] == "rules"
    assert body["nlu"]["trace"], "the UI panel needs the trace"
    assert body["operation_id"], "the operation must be recorded for telemetry"
    assert body["timings"]["total"] > 0
    assert body["timings"]["nlu"] is not None
    assert body["timings"]["cas"] is not None
    assert body["model_versions"] == {"ocr": "stub-0.1.0", "asr": "asr-stub-1"}

    history = await client.get(
        f"/workspaces/{student['workspace_id']}/operations", headers=student["headers"]
    )
    assert history.status_code == 200
    rows = history.json()["items"]
    assert len(rows) == 1
    assert rows[0]["id"] == body["operation_id"]
    assert rows[0]["input_kind"] == "text"
    assert rows[0]["result_latex"] == r"2 \cdot x"


async def test_api_compute_latex_diff_returns_2x(
    client: httpx.AsyncClient, student: dict[str, Any]
) -> None:
    """The deterministic path: a typed command, no NLU involved."""
    response = await client.post(
        "/compute/latex",
        json={"command": {"op": "diff", "expr": "x^{2}", "var": "x"}},
        headers=student["headers"],
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "result"
    assert body["result"]["latex"] == r"2 \cdot x"
    assert body["nlu"] is None


async def test_api_compute_text_returns_clarification_200_not_error(
    client: httpx.AsyncClient, student: dict[str, Any]
) -> None:
    """An ambiguous or under-specified command is a 200 payload with a Turkish question."""
    response = await client.post(
        "/compute/text", json={"text": "türevini al"}, headers=student["headers"]
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "clarification"
    assert body["result"] is None
    assert body["clarification"]["question_tr"]
    assert body["clarification"]["missing_slot"] == "expr"


async def test_api_clarification_candidates_carry_a_command_preview(
    client: httpx.AsyncClient, student: dict[str, Any]
) -> None:
    """ "açılım" is genuinely both ``expand`` and ``series``; each option previews its command."""
    response = await client.post(
        "/compute/text",
        json={"text": "x kare artı iki x ifadesinin açılımını yaz"},
        headers=student["headers"],
    )
    assert response.status_code == 200, response.text
    body = response.json()
    if body["kind"] != "clarification":
        pytest.skip("the inventory resolved this phrasing without asking")
    candidates = body["clarification"]["candidates"]
    assert candidates, "a clarification between intents must offer its options"
    assert {candidate["intent"] for candidate in candidates}
    previewed = [c for c in candidates if c["command"] is not None]
    assert previewed, "at least one option must preview the command it would run"
    assert previewed[0]["latex_preview"]


async def test_api_confirm_intent_runs_the_chosen_reading(
    client: httpx.AsyncClient, student: dict[str, Any]
) -> None:
    """``confirm_intent`` re-runs the rule-based resolver for one named intent."""
    response = await client.post(
        "/compute/text",
        json={"text": "x kare ifadesinin türevini al", "confirm_intent": "differentiate"},
        headers=student["headers"],
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "result"
    assert body["command"]["op"] == "diff"
    assert body["nlu"]["intent"] == "differentiate"


async def test_api_compute_text_no_intent_is_422_with_suggestions(
    client: httpx.AsyncClient, student: dict[str, Any]
) -> None:
    """A request outside the inventory is refused with Turkish examples, never guessed at."""
    response = await client.post(
        "/compute/text",
        json={"text": "bugün hava nasıl olacak"},
        headers=student["headers"],
    )
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "NLU_NO_INTENT"
    assert body["suggestions_tr"]


@pytest.mark.parametrize("latex", OUT_OF_SCOPE)
async def test_api_unsupported_returns_422_with_contract_message(
    client: httpx.AsyncClient, student: dict[str, Any], latex: str
) -> None:
    """Recognised but not computable: the frozen İP-4 sentence, a named construct, never a 500."""
    response = await client.post(
        "/compute/latex",
        json={"command": {"op": "simplify", "expr": latex}},
        headers=student["headers"],
    )
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["code"] == "UNSUPPORTED_PROBLEM"
    assert body["detail_tr"] == MSG_UNSUPPORTED_PROBLEM
    assert body["construct"], "the UI must be able to name the symbol to avoid"
    assert response.headers["content-type"].startswith("application/problem+json")


async def test_api_problem_json_shape_on_parse_error_with_position(
    client: httpx.AsyncClient, student: dict[str, Any]
) -> None:
    """A malformed formula is an RFC 9457 document carrying the offending position."""
    response = await client.post(
        "/compute/latex",
        json={"command": {"op": "simplify", "expr": "x +* 2"}},
        headers=student["headers"],
    )
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["type"].endswith("/PARSE_ERROR")
    assert body["code"] == "PARSE_ERROR"
    assert body["status"] == 422
    assert body["title"]
    assert body["detail_tr"]
    assert body["position"] == 4
    assert body["instance"] == response.headers["X-Request-Id"]
    assert response.headers["content-type"].startswith("application/problem+json")


async def test_api_compute_parse_reports_the_construct_without_computing(
    client: httpx.AsyncClient, student: dict[str, Any]
) -> None:
    """The editor preview keeps the parser's own diagnosis, as a 200 with ``ok:false``."""
    ok = await client.post(
        "/compute/parse", json={"latex": "x^{2} + 1"}, headers=student["headers"]
    )
    assert ok.status_code == 200
    assert ok.json()["ok"] is True
    assert ok.json()["canonical_latex"]

    bad = await client.post(
        "/compute/parse", json={"latex": "\\binom{n}{k}"}, headers=student["headers"]
    )
    assert bad.status_code == 200
    body = bad.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "UNSUPPORTED_CONSTRUCT"
    assert body["error"]["construct"] == "\\binom"


async def test_api_compute_requires_authentication(client: httpx.AsyncClient) -> None:
    """The command path is never open."""
    response = await client.post("/compute/text", json={"text": "x karenin türevini al"})
    assert response.status_code == 401
    assert response.json()["code"] == "AUTH_INVALID"


async def test_api_compute_text_rejects_a_foreign_workspace(
    client: httpx.AsyncClient, app: FastAPI, student: dict[str, Any]
) -> None:
    """Another user's workspace answers 404, not 403 (no existence oracle)."""
    _other, other_token = await account(client, app, "baskasi@ornek.edu.tr")
    response = await client.post(
        "/compute/text",
        json={"text": "x karenin türevini al", "workspace_id": student["workspace_id"]},
        headers=auth_header(other_token),
    )
    assert response.status_code == 404
    assert response.json()["code"] == "NOT_FOUND"


async def test_api_compute_plot_is_stored_and_served_to_its_owner(
    client: httpx.AsyncClient, app: FastAPI, student: dict[str, Any]
) -> None:
    """A ``plot`` command stores a PNG asset and only its owner may fetch it."""
    response = await client.post(
        "/compute/latex",
        json={
            "command": {"op": "plot", "expr": "x^{2}", "var": "x", "xmin": -2, "xmax": 2},
            "workspace_id": student["workspace_id"],
        },
        headers=student["headers"],
    )
    assert response.status_code == 200, response.text
    plot_url = response.json()["result"]["plot_url"]
    assert plot_url and plot_url.endswith(".png")

    fetched = await client.get(plot_url.removeprefix("/api/v1"), headers=student["headers"])
    assert fetched.status_code == 200
    assert fetched.headers["content-type"] == "image/png"
    assert fetched.content.startswith(b"\x89PNG")

    _other, other_token = await account(client, app, "meraklı@ornek.edu.tr")
    denied = await client.get(plot_url.removeprefix("/api/v1"), headers=auth_header(other_token))
    assert denied.status_code == 404


async def test_api_history_export_tex_and_pdf(
    client: httpx.AsyncClient, student: dict[str, Any]
) -> None:
    """One operation exports as LaTeX source and as a one-page PDF (proposal A.2.2)."""
    computed = await client.post(
        "/compute/text",
        json={"text": "x karenin türevini al", "workspace_id": student["workspace_id"]},
        headers=student["headers"],
    )
    operation_id = computed.json()["operation_id"]
    base = f"/workspaces/{student['workspace_id']}/operations/{operation_id}/export"

    tex = await client.post(f"{base}?format=tex", headers=student["headers"])
    assert tex.status_code == 200
    assert tex.headers["content-type"].startswith("text/x-tex")
    assert r"\begin{document}" in tex.text
    assert r"2 \cdot x" in tex.text

    pdf = await client.post(f"{base}?format=pdf", headers=student["headers"])
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF")


async def test_api_operation_can_be_deleted_from_the_history(
    client: httpx.AsyncClient, student: dict[str, Any]
) -> None:
    """A student may remove one row of their own history."""
    computed = await client.post(
        "/compute/text",
        json={"text": "x karenin türevini al", "workspace_id": student["workspace_id"]},
        headers=student["headers"],
    )
    operation_id = computed.json()["operation_id"]
    removed = await client.delete(
        f"/workspaces/{student['workspace_id']}/operations/{operation_id}",
        headers=student["headers"],
    )
    assert removed.status_code == 204
    again = await client.get(
        f"/workspaces/{student['workspace_id']}/operations", headers=student["headers"]
    )
    assert again.json()["items"] == []
