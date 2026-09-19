"""The two WebSocket streams (api.md §4, ADR-0014).

httpx speaks HTTP only, so these tests use ``starlette.testclient.TestClient``, which runs the
same ASGI application in-process on a portal thread — still no server socket, still the same
``create_app`` the REST tests drive.

The stream timings are shortened through ``app.state.ws_timings`` so the suite does not spend
seconds waiting for a 400 ms debounce and a 1.5 s idle timeout; the production values live in
:class:`aicls.api.ws.common.WsTimings` and are asserted by ``test_ws_timings_defaults``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from fastapi import FastAPI
import numpy as np
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from aicls.api.security import hash_password
from aicls.api.ws.common import MAX_STRIKES, RATE_LIMIT_MESSAGES, WsTimings
from aicls.api.ws.protocol import WS_POLICY_VIOLATION, WS_TRY_AGAIN_LATER
from aicls.asr.types import Audio, Transcript
from aicls.db.enums import Role

pytestmark = [pytest.mark.integration, pytest.mark.ws]

#: PCM16 frame length used by the browser recorder (250 ms at 16 kHz).
FRAME_SAMPLES = 4000

#: Fast timings so the tests do not sleep for the production debounce budgets.
FAST = WsTimings(pen_up_debounce_ms=20, idle_final_ms=60, silence_ms=250, max_audio_s=30.0)


class FixedTranscriber:
    """A transcriber that always answers the same Turkish command (test double)."""

    model_version = "test-fixed"

    def warmup(self) -> None:
        """Nothing to load."""

    def transcribe(self, audio: Audio) -> Transcript:
        """Return a sentence the NLU layer certainly understands."""
        del audio
        return Transcript(
            text="x karenin türevini al",
            confidence=0.91,
            elapsed_ms=1.0,
            model_version=self.model_version,
        )


def stroke(n: int = 30, offset: int = 0) -> dict[str, Any]:
    """One ``stroke`` payload in the shape ``web/src/features/workspace`` sends."""
    return {
        "points": [
            {"x": 10.0 + offset + i, "y": 40.0 + (i % 5), "t": 1700000000000 + i * 8}
            for i in range(n)
        ],
        "canvas": {"w": 800, "h": 300},
    }


def pcm16(samples: np.ndarray) -> bytes:
    """Encode float samples as little-endian PCM16 bytes."""
    return np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def voiced_frame(seed: int = 0) -> bytes:
    """250 ms of a loud voiced-looking signal."""
    t = np.arange(FRAME_SAMPLES, dtype=np.float64) / 16000.0
    return pcm16(0.4 * np.sin(2 * np.pi * (180 + seed) * t))


def silent_frame() -> bytes:
    """250 ms of near-silence (below the endpointing threshold)."""
    return pcm16(np.zeros(FRAME_SAMPLES, dtype=np.float64))


@pytest.fixture
def ws_app(app: FastAPI) -> Iterator[TestClient]:
    """A ``TestClient`` on the application, with fast stream timings."""
    app.state.ws_timings = FAST
    with TestClient(app, base_url="http://testserver") as client:
        yield client


def ticket_for(client: TestClient, app: FastAPI, email: str = "ws@ornek.edu.tr") -> str:
    """Seed a user directly, log in and mint a WebSocket ticket."""
    from aicls.db import UserRepository, session_scope

    with session_scope(app.state.engine) as session:
        UserRepository(session).create(
            email, hash_password("Kalkulus2026!"), "WS Kullanıcı", Role.STUDENT
        )
    token = client.post(
        "/api/v1/auth/login", json={"email": email, "password": "Kalkulus2026!"}
    ).json()["access_token"]
    issued = client.post("/api/v1/auth/ws-ticket", headers={"Authorization": f"Bearer {token}"})
    assert issued.status_code == 200, issued.text
    return str(issued.json()["ticket"])


def test_ws_timings_defaults() -> None:
    """The production budgets are the ones ADR-0014 fixes."""
    defaults = WsTimings()
    assert defaults.pen_up_debounce_ms == 400
    assert defaults.idle_final_ms == 1500
    assert defaults.silence_ms == 600
    assert defaults.max_audio_s == 30.0


def test_ws_rejects_expired_ticket(ws_app: TestClient, app: FastAPI) -> None:
    """A missing, unknown or already-redeemed ticket closes the socket with 1008."""
    with pytest.raises(WebSocketDisconnect) as missing, ws_app.websocket_connect("/ws/v1/ink"):
        pass
    assert missing.value.code == WS_POLICY_VIOLATION

    with (
        pytest.raises(WebSocketDisconnect) as unknown,
        ws_app.websocket_connect("/ws/v1/ink?ticket=uydurma"),
    ):
        pass
    assert unknown.value.code == WS_POLICY_VIOLATION

    ticket = ticket_for(ws_app, app)
    with ws_app.websocket_connect(f"/ws/v1/ink?ticket={ticket}") as socket:
        socket.send_json({"type": "ping", "seq": 1, "payload": {}})
        assert socket.receive_json()["type"] == "pong"
    with (
        pytest.raises(WebSocketDisconnect) as reused,
        ws_app.websocket_connect(f"/ws/v1/ink?ticket={ticket}"),
    ):
        pass
    assert reused.value.code == WS_POLICY_VIOLATION


def test_ws_ink_stream_partial_then_final(ws_app: TestClient, app: FastAPI) -> None:
    """Pen-up debounce answers ``partial``; ``finalize`` answers ``final`` with an id."""
    ticket = ticket_for(ws_app, app)
    with ws_app.websocket_connect(f"/ws/v1/ink?ticket={ticket}") as socket:
        socket.send_json({"type": "stroke", "seq": 1, "payload": stroke()})
        socket.send_json({"type": "pen_up", "seq": 2, "payload": {}})
        partial = socket.receive_json()
        assert partial["type"] == "partial"
        assert partial["payload"]["latex"]
        assert 0.0 <= partial["payload"]["confidence"] <= 1.0

        socket.send_json({"type": "stroke", "seq": 3, "payload": stroke(offset=50)})
        socket.send_json({"type": "finalize", "seq": 4, "payload": {}})
        final = socket.receive_json()
        assert final["type"] == "final"
        assert final["seq"] == 4
        assert final["payload"]["latex"]
        assert final["payload"]["alternatives"]
        assert final["payload"]["recognition_id"]


def test_ws_ink_idle_produces_a_final(ws_app: TestClient, app: FastAPI) -> None:
    """Without ``finalize``, an idle connection still produces a ``final`` (api.md §4.1)."""
    ticket = ticket_for(ws_app, app)
    with ws_app.websocket_connect(f"/ws/v1/ink?ticket={ticket}") as socket:
        socket.send_json({"type": "stroke", "seq": 1, "payload": stroke()})
        socket.send_json({"type": "pen_up", "seq": 2, "payload": {}})
        assert socket.receive_json()["type"] == "partial"
        assert socket.receive_json()["type"] == "final"


def test_ws_ink_clear_and_unknown_type(ws_app: TestClient, app: FastAPI) -> None:
    """``clear`` drops the buffer; an unknown envelope is an ``error``, not a disconnect."""
    ticket = ticket_for(ws_app, app)
    with ws_app.websocket_connect(f"/ws/v1/ink?ticket={ticket}") as socket:
        socket.send_json({"type": "stroke", "seq": 1, "payload": stroke()})
        socket.send_json({"type": "clear", "seq": 2, "payload": {}})
        socket.send_json({"type": "uydurma", "seq": 3, "payload": {}})
        error = socket.receive_json()
        assert error["type"] == "error"
        assert error["payload"]["problem"]["code"] == "OCR_INPUT_INVALID"
        socket.send_json({"type": "ping", "seq": 4, "payload": {}})
        assert socket.receive_json()["type"] == "pong"


def test_ws_ink_slow_down_when_flooded(ws_app: TestClient, app: FastAPI) -> None:
    """Flooding answers ``slow_down`` and, on persistence, closes the socket (api.md §1, §4.1)."""
    ticket = ticket_for(ws_app, app)
    received: list[str] = []
    with (
        pytest.raises(WebSocketDisconnect) as closed,
        ws_app.websocket_connect(f"/ws/v1/ink?ticket={ticket}") as socket,
    ):
        for index in range(40):
            socket.send_json({"type": "ping", "seq": index, "payload": {}})
        while True:
            received.append(socket.receive_json()["type"])
    assert closed.value.code == WS_TRY_AGAIN_LATER
    assert received.count("pong") == RATE_LIMIT_MESSAGES, "the budget is 30 messages per window"
    assert received.count("slow_down") == MAX_STRIKES


def test_ws_audio_final_then_command_with_stub(ws_app: TestClient, app: FastAPI) -> None:
    """``stop`` produces ``final`` and then a ``command`` envelope carrying a full ComputeOut."""
    app.state.registry.set_transcriber(FixedTranscriber())
    ticket = ticket_for(ws_app, app)
    with ws_app.websocket_connect(f"/ws/v1/audio?ticket={ticket}") as socket:
        socket.send_json(
            {"type": "start", "seq": 1, "payload": {"sample_rate": 16000, "format": "pcm16"}}
        )
        for index in range(3):
            socket.send_bytes(voiced_frame(index))
        socket.send_json({"type": "stop", "seq": 2, "payload": {}})

        final = socket.receive_json()
        assert final["type"] == "final"
        assert final["payload"]["text"] == "x karenin türevini al"
        assert final["payload"]["confidence"] == pytest.approx(0.91)
        assert final["payload"]["transcript_id"]

        command = socket.receive_json()
        assert command["type"] == "command"
        outcome = command["payload"]["outcome"]
        assert outcome["kind"] == "result"
        assert outcome["result"]["latex"] == r"2 \cdot x"
        assert outcome["nlu"]["intent"] == "differentiate"


def test_ws_audio_energy_endpointing_ends_the_utterance(ws_app: TestClient, app: FastAPI) -> None:
    """Trailing silence ends the utterance without a ``stop`` (ADR-0014)."""
    app.state.registry.set_transcriber(FixedTranscriber())
    ticket = ticket_for(ws_app, app)
    with ws_app.websocket_connect(f"/ws/v1/audio?ticket={ticket}") as socket:
        socket.send_json(
            {"type": "start", "seq": 1, "payload": {"sample_rate": 16000, "format": "pcm16"}}
        )
        socket.send_bytes(voiced_frame())
        socket.send_bytes(silent_frame())
        assert socket.receive_json()["type"] == "final"
        assert socket.receive_json()["type"] == "command"


def test_ws_audio_backpressure(ws_app: TestClient, app: FastAPI) -> None:
    """A client that floods frames is told to slow down and then cut off, not left to fill RAM."""
    ticket = ticket_for(ws_app, app)
    received: list[str] = []
    with (
        pytest.raises(WebSocketDisconnect) as closed,
        ws_app.websocket_connect(f"/ws/v1/audio?ticket={ticket}") as socket,
    ):
        socket.send_json(
            {"type": "start", "seq": 1, "payload": {"sample_rate": 16000, "format": "pcm16"}}
        )
        for _ in range(40):
            socket.send_bytes(silent_frame())
        while True:
            received.append(socket.receive_json()["type"])
    assert closed.value.code == WS_TRY_AGAIN_LATER
    assert received.count("slow_down") == MAX_STRIKES, received


def test_ws_audio_frame_before_start_is_an_error(ws_app: TestClient, app: FastAPI) -> None:
    """Audio before ``start`` is refused, and the connection stays usable."""
    ticket = ticket_for(ws_app, app)
    with ws_app.websocket_connect(f"/ws/v1/audio?ticket={ticket}") as socket:
        socket.send_bytes(voiced_frame())
        error = socket.receive_json()
        assert error["type"] == "error"
        assert error["payload"]["problem"]["code"] == "ASR_INPUT_INVALID"
        assert error["payload"]["problem"]["reason"] == "start_expected"


def test_ws_audio_cancel_discards_the_buffer(ws_app: TestClient, app: FastAPI) -> None:
    """``cancel`` throws the utterance away: a following ``stop`` produces nothing."""
    ticket = ticket_for(ws_app, app)
    with ws_app.websocket_connect(f"/ws/v1/audio?ticket={ticket}") as socket:
        socket.send_json(
            {"type": "start", "seq": 1, "payload": {"sample_rate": 16000, "format": "pcm16"}}
        )
        socket.send_bytes(voiced_frame())
        socket.send_json({"type": "cancel", "seq": 2, "payload": {}})
        socket.send_json({"type": "stop", "seq": 3, "payload": {}})
        socket.send_json({"type": "ping", "seq": 4, "payload": {}})
        assert socket.receive_json()["type"] == "pong"
