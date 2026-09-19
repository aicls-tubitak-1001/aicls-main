"""``/ws/v1/ink`` — recognition that overlaps writing (api.md §4.1, ADR-0014).

The stream exists for one contractual reason: with a beam-searching OCR model, starting
recognition only after the last pen-up cannot fit "command → rendered result < 3 s". So every
pen-up ships its trace, the server waits ``pen_up_debounce_ms`` for the next stroke and then
answers ``partial``; ``finalize`` (or ``idle_final_ms`` of silence) answers ``final`` with a
``recognition_id`` the client can correct through ``POST /ocr/feedback``.

Recognition is serialised per connection with an :class:`asyncio.Semaphore`, so a fast writer
can never run two beams at once on the same socket.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocket, WebSocketDisconnect

from aicls.api.ws.common import SLOW_DOWN_RETRY_MS, MessageRate, WsTimings, authenticate
from aicls.api.ws.protocol import (
    WS_TRY_AGAIN_LATER,
    WsEnvelope,
    envelope,
    error_envelope,
    slow_down,
)
from aicls.core.errors import AiclsError
from aicls.core.logging import get_logger
from aicls.core.timing import StageTimer
from aicls.db.engine import session_scope
from aicls.ocr.canvas import ink_from_canvas_json
from aicls.ocr.types import MAX_POINTS_PER_INK, MAX_STROKES_PER_INK, OcrInputError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aicls.api.pipeline import PipelineService

#: Client message types this stream understands.
ACCEPTED: frozenset[str] = frozenset({"stroke", "pen_up", "clear", "finalize", "ping"})

_logger = get_logger(__name__)


class InkBuffer:
    """The strokes collected on one connection, with the api.md §1 budgets applied."""

    def __init__(self) -> None:
        self.strokes: list[dict[str, Any]] = []
        self.canvas: dict[str, Any] = {}
        self.points = 0

    def add(self, payload: dict[str, Any]) -> None:
        """Append one completed trace.

        Raises:
            OcrInputError: If the stroke is malformed or a budget is exceeded.
        """
        points = payload.get("points")
        if not isinstance(points, list) or not points:
            raise OcrInputError(details={"reason": "stroke_points_type"})
        if len(self.strokes) >= MAX_STROKES_PER_INK:
            raise OcrInputError(
                details={"reason": "too_many_strokes", "limit": MAX_STROKES_PER_INK}
            )
        if self.points + len(points) > MAX_POINTS_PER_INK:
            raise OcrInputError(details={"reason": "too_many_points", "limit": MAX_POINTS_PER_INK})
        canvas = payload.get("canvas")
        if isinstance(canvas, dict):
            self.canvas = dict(canvas)
        self.strokes.append({"points": points})
        self.points += len(points)

    def clear(self) -> None:
        """Drop everything collected so far (the ``clear`` envelope)."""
        self.strokes.clear()
        self.points = 0

    @property
    def empty(self) -> bool:
        """Whether nothing has been drawn yet."""
        return not self.strokes

    def payload(self) -> dict[str, Any]:
        """The Canvas JSON document the recogniser is given."""
        document: dict[str, Any] = {"version": 1, "strokes": self.strokes}
        if self.canvas:
            document["canvas"] = self.canvas
        return document


async def ink_stream(websocket: WebSocket) -> None:
    """Serve one ``/ws/v1/ink`` connection until the client disconnects."""
    app = websocket.app
    tickets = app.state.tickets
    timings: WsTimings = app.state.ws_timings
    pipeline: PipelineService = app.state.pipeline
    engine = app.state.engine
    try:
        ticket = await authenticate(websocket, tickets)
    except AiclsError:
        return

    buffer = InkBuffer()
    rate = MessageRate()
    lock = asyncio.Semaphore(1)
    deadline: float | None = None
    stage = "partial"
    loop = asyncio.get_running_loop()

    async def recognise(seq: int, final: bool) -> None:
        """Run one recognition and answer ``partial`` or ``final``."""
        if buffer.empty:
            return
        async with lock:
            timer = StageTimer()
            try:
                ink = ink_from_canvas_json(buffer.payload())
                outcome = await run_in_threadpool(
                    _recognise, pipeline, engine, ink, ticket.user_id, final, timer
                )
            except AiclsError as exc:
                await websocket.send_json(error_envelope(exc, seq))
                return
        recognition = outcome["recognition"]
        if final:
            await websocket.send_json(
                envelope(
                    "final",
                    {
                        "latex": recognition.latex,
                        "alternatives": [list(item) for item in recognition.alternatives],
                        "recognition_id": outcome["recognition_id"],
                        "confidence": recognition.confidence,
                    },
                    seq,
                )
            )
        else:
            await websocket.send_json(
                envelope(
                    "partial",
                    {"latex": recognition.latex, "confidence": recognition.confidence},
                    seq,
                )
            )

    try:
        while True:
            timeout = None if deadline is None else max(0.0, deadline - loop.time())
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=timeout)
            except TimeoutError:
                deadline_stage, deadline = stage, None
                await recognise(0, final=deadline_stage == "final")
                if deadline_stage == "partial":
                    stage, deadline = "final", loop.time() + timings.idle_final_ms / 1000.0
                continue

            if not rate.allow():
                await websocket.send_json(slow_down(SLOW_DOWN_RETRY_MS))
                if rate.exhausted:
                    await websocket.close(code=WS_TRY_AGAIN_LATER, reason="rate")
                    return
                continue

            try:
                message = WsEnvelope.parse(raw)
            except AiclsError as exc:
                await websocket.send_json(error_envelope(exc))
                continue
            if message.type not in ACCEPTED:
                await websocket.send_json(
                    error_envelope(
                        OcrInputError(details={"reason": "unknown_type", "type": message.type}),
                        message.seq,
                    )
                )
                continue

            if message.type == "ping":
                await websocket.send_json(message.reply("pong"))
            elif message.type == "clear":
                buffer.clear()
                deadline = None
            elif message.type == "stroke":
                try:
                    buffer.add(message.payload)
                except AiclsError as exc:
                    await websocket.send_json(error_envelope(exc, message.seq))
                    continue
                deadline = None
            elif message.type == "pen_up":
                stage = "partial"
                deadline = loop.time() + timings.pen_up_debounce_ms / 1000.0
            elif message.type == "finalize":
                deadline = None
                await recognise(message.seq, final=True)
    except WebSocketDisconnect:  # pragma: no cover - normal client close
        _logger.debug("api.ws.ink.disconnect", user_id=ticket.user_id)


def _recognise(
    pipeline: PipelineService,
    engine: Any,
    ink: Any,
    user_id: str,
    persist: bool,
    timer: StageTimer,
) -> dict[str, Any]:
    """Run the recogniser in a worker thread; only a ``final`` is written to the database."""
    from aicls.db.repositories.identity import UserRepository

    if not persist:
        outcome = pipeline.run_ink(ink, None, session=None, timer=timer)
        return {"recognition": outcome.recognition, "recognition_id": None}
    with session_scope(engine) as session:
        user = UserRepository(session).get(user_id)
        outcome = pipeline.run_ink(ink, user, session=session, timer=timer)
        return {"recognition": outcome.recognition, "recognition_id": outcome.operation_id}


__all__ = ["ACCEPTED", "InkBuffer", "ink_stream"]
