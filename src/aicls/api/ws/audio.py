"""``/ws/v1/audio`` — streamed speech with energy endpointing (api.md §4.2, ADR-0014).

The client announces ``start{sample_rate:16000, format:"pcm16"}`` and then pushes 250 ms binary
PCM16 frames. The server buffers them (capped at ``max_audio_s``), watches the trailing energy
and ends the utterance after ``silence_ms`` of silence — or immediately on ``stop``. The
transcript goes out as ``final``, and because the whole point of the stream is to keep the "< 3 s
to a rendered result" budget, NLU and the CAS then run on the *same* connection and the outcome
arrives as a ``command`` envelope carrying a complete ``ComputeOut``.

Whisper produces no streaming partial, so ``partial`` is emitted only for recordings longer than
:data:`PARTIAL_AFTER_S`, from the audio buffered so far.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocket, WebSocketDisconnect

from aicls.api.responses import compute_out
from aicls.api.ws.common import SLOW_DOWN_RETRY_MS, MessageRate, WsTimings, authenticate
from aicls.api.ws.protocol import (
    WS_TRY_AGAIN_LATER,
    WsEnvelope,
    envelope,
    error_envelope,
    slow_down,
)
from aicls.asr.errors import invalid_audio
from aicls.asr.types import TARGET_SAMPLE_RATE, Audio
from aicls.core.errors import AiclsError
from aicls.core.logging import get_logger
from aicls.core.timing import StageTimer
from aicls.db.engine import session_scope
from aicls.nlu.types import ParseContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aicls.api.pipeline import PipelineService

#: Client message types this stream understands (binary frames carry the audio).
ACCEPTED: frozenset[str] = frozenset({"start", "stop", "cancel", "ping"})

#: Bytes per PCM16 sample.
SAMPLE_WIDTH = 2

#: Recording length after which a chunk-based ``partial`` is emitted.
PARTIAL_AFTER_S = 5.0

#: RMS below which a window counts as silence (full-scale float).
SILENCE_RMS = 0.01

#: Length of the analysis window used for endpointing, in seconds.
ENERGY_WINDOW_S = 0.2

_logger = get_logger(__name__)


class AudioBuffer:
    """PCM16 bytes collected on one connection, capped at ``max_seconds``."""

    def __init__(self, sample_rate: int = TARGET_SAMPLE_RATE, max_seconds: float = 30.0) -> None:
        self.sample_rate = sample_rate
        self.max_bytes = int(max_seconds * sample_rate * SAMPLE_WIDTH)
        self._chunks: list[bytes] = []
        self.size = 0
        self.had_speech = False

    def add(self, frame: bytes) -> None:
        """Append one binary frame.

        Raises:
            AsrInputError: If the buffer would exceed the 30 s cap.
        """
        if self.size + len(frame) > self.max_bytes:
            raise invalid_audio("buffer_overflow", limit=self.max_bytes)
        self._chunks.append(frame)
        self.size += len(frame)

    def clear(self) -> None:
        """Drop the buffered audio (``cancel`` or after a ``final``)."""
        self._chunks.clear()
        self.size = 0
        self.had_speech = False

    @property
    def duration_s(self) -> float:
        """Length of the buffered audio in seconds."""
        return self.size / float(self.sample_rate * SAMPLE_WIDTH)

    @property
    def empty(self) -> bool:
        """Whether nothing has been buffered."""
        return self.size == 0

    def samples(self) -> np.ndarray:
        """Buffered audio as float32 in ``[-1, 1]``."""
        raw = b"".join(self._chunks)
        usable = len(raw) - (len(raw) % SAMPLE_WIDTH)
        pcm = np.frombuffer(raw[:usable], dtype="<i2")
        return (pcm.astype(np.float32) / 32768.0).astype(np.float32)

    def audio(self) -> Audio:
        """Buffered audio as an :class:`~aicls.asr.types.Audio`."""
        return Audio(self.samples(), self.sample_rate)

    def trailing_rms(self, window_s: float = ENERGY_WINDOW_S) -> float:
        """Root-mean-square energy of the last ``window_s`` seconds."""
        samples = self.samples()
        if samples.size == 0:
            return 0.0
        window = max(1, int(window_s * self.sample_rate))
        tail = samples[-window:]
        return float(np.sqrt(np.mean(np.square(tail.astype(np.float64)))))


def is_endpoint(buffer: AudioBuffer, silence_ms: int) -> bool:
    """Whether the trailing silence is long enough to end the utterance.

    Args:
        buffer: The connection's audio buffer.
        silence_ms: Trailing silence that ends an utterance (600 ms per ADR-0014).

    Returns:
        ``True`` when speech was heard and the last ``silence_ms`` are below
        :data:`SILENCE_RMS`.
    """
    if not buffer.had_speech or buffer.duration_s < silence_ms / 1000.0:
        return False
    return buffer.trailing_rms(silence_ms / 1000.0) < SILENCE_RMS


async def audio_stream(websocket: WebSocket) -> None:
    """Serve one ``/ws/v1/audio`` connection until the client disconnects."""
    app = websocket.app
    timings: WsTimings = app.state.ws_timings
    pipeline: PipelineService = app.state.pipeline
    engine = app.state.engine
    try:
        ticket = await authenticate(websocket, app.state.tickets)
    except AiclsError:
        return

    buffer = AudioBuffer(max_seconds=timings.max_audio_s)
    rate = MessageRate()
    started = False
    partial_sent = False

    async def finish(seq: int) -> None:
        """Transcribe the buffer, answer ``final`` and then run NLU + CAS as ``command``."""
        if buffer.empty:
            return
        timer = StageTimer()
        audio = buffer.audio()
        buffer.clear()
        try:
            outcome = await run_in_threadpool(
                _transcribe_and_command, pipeline, engine, audio, ticket.user_id, timer
            )
        except AiclsError as exc:
            await websocket.send_json(error_envelope(exc, seq))
            return
        transcript = outcome["transcript"]
        await websocket.send_json(
            envelope(
                "final",
                {
                    "text": transcript.text,
                    "confidence": transcript.confidence,
                    "transcript_id": outcome["transcript_id"],
                },
                seq,
            )
        )
        if outcome["problem"] is not None:
            await websocket.send_json(envelope("error", {"problem": outcome["problem"]}, seq))
            return
        await websocket.send_json(envelope("command", {"outcome": outcome["compute"]}, seq))

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":  # pragma: no cover - client close
                return
            if not rate.allow():
                await websocket.send_json(slow_down(SLOW_DOWN_RETRY_MS))
                if rate.exhausted:
                    await websocket.close(code=WS_TRY_AGAIN_LATER, reason="rate")
                    return
                continue

            frame = message.get("bytes")
            if frame is not None:
                if not started:
                    await websocket.send_json(error_envelope(invalid_audio("start_expected")))
                    continue
                try:
                    buffer.add(frame)
                except AiclsError as exc:
                    await websocket.send_json(error_envelope(exc))
                    buffer.clear()
                    continue
                if buffer.trailing_rms() >= SILENCE_RMS:
                    buffer.had_speech = True
                if not partial_sent and buffer.duration_s >= PARTIAL_AFTER_S:
                    partial_sent = True
                    await websocket.send_json(
                        envelope("partial", {"text": "", "duration_s": buffer.duration_s})
                    )
                if is_endpoint(buffer, timings.silence_ms):
                    partial_sent = False
                    await finish(0)
                continue

            raw = message.get("text")
            if raw is None:  # pragma: no cover - starlette sends text or bytes
                continue
            try:
                parsed = WsEnvelope.parse(raw)
            except AiclsError as exc:
                await websocket.send_json(error_envelope(exc))
                continue
            if parsed.type not in ACCEPTED:
                await websocket.send_json(
                    error_envelope(invalid_audio("unknown_type", type=parsed.type), parsed.seq)
                )
                continue
            if parsed.type == "ping":
                await websocket.send_json(parsed.reply("pong"))
            elif parsed.type == "start":
                started = True
                partial_sent = False
                buffer.clear()
                buffer.sample_rate = int(parsed.payload.get("sample_rate", TARGET_SAMPLE_RATE))
            elif parsed.type == "cancel":
                buffer.clear()
                partial_sent = False
            elif parsed.type == "stop":
                partial_sent = False
                await finish(parsed.seq)
    except WebSocketDisconnect:  # pragma: no cover - normal client close
        _logger.debug("api.ws.audio.disconnect", user_id=ticket.user_id)


def _transcribe_and_command(
    pipeline: PipelineService, engine: Any, audio: Audio, user_id: str, timer: StageTimer
) -> dict[str, Any]:
    """Transcribe, then run the text path; a refusal travels as a problem, not an exception."""
    from aicls.db.repositories.identity import UserRepository

    with session_scope(engine) as session:
        user = UserRepository(session).get(user_id)
        spoken = pipeline.run_audio(
            audio, user, None, None, session=session, timer=timer, to_command=False
        )
        transcript = spoken.transcript
        try:
            outcome = pipeline.run_text(
                transcript.text if transcript is not None else "",
                ParseContext(modality="voice"),
                user,
                None,
                session=session,
                timer=timer,
            )
        except AiclsError as exc:
            return {
                "transcript": transcript,
                "transcript_id": spoken.operation_id,
                "compute": None,
                "problem": exc.to_problem(),
            }
        return {
            "transcript": transcript,
            "transcript_id": spoken.operation_id,
            "compute": compute_out(outcome, timer).model_dump(mode="json"),
            "problem": None,
        }


__all__ = [
    "ACCEPTED",
    "ENERGY_WINDOW_S",
    "PARTIAL_AFTER_S",
    "SAMPLE_WIDTH",
    "SILENCE_RMS",
    "AudioBuffer",
    "audio_stream",
    "is_endpoint",
]
