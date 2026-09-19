"""Typed WebSocket envelopes (api.md §4, ADR-0014).

Every message on both streams is ``{type, seq, payload}``. ``seq`` is incremented by the client
and echoed by the server in the envelope that answers it, so a client can match a ``final`` to
the ``finalize`` that asked for it without keeping a separate correlation channel.

The envelope is deliberately *not* a discriminated union over payload models: the two streams
share the frame, and validating the payload inside each handler keeps one unknown message from
tearing down a connection that is otherwise healthy.
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aicls.core import messages_tr
from aicls.core.errors import AiclsError, ErrorCode

#: Client → server message types of ``/ws/v1/ink``.
InkClientType: TypeAlias = Literal["stroke", "pen_up", "clear", "finalize", "ping"]

#: Server → client message types of ``/ws/v1/ink``.
InkServerType: TypeAlias = Literal["partial", "final", "error", "slow_down", "pong"]

#: Client → server message types of ``/ws/v1/audio``.
AudioClientType: TypeAlias = Literal["start", "stop", "cancel", "ping"]

#: Server → client message types of ``/ws/v1/audio``.
AudioServerType: TypeAlias = Literal["partial", "final", "command", "error", "slow_down", "pong"]

#: Every accepted client message type (a stricter check happens per stream).
CLIENT_TYPES: frozenset[str] = frozenset(
    {"stroke", "pen_up", "clear", "finalize", "ping", "start", "stop", "cancel"}
)

#: WebSocket close code used when a ticket is missing, expired or already redeemed.
WS_POLICY_VIOLATION = 1008

#: WebSocket close code used when the client keeps flooding after ``slow_down``.
WS_TRY_AGAIN_LATER = 1013

#: Largest text frame accepted from a client (a stroke batch, not a file).
MAX_FRAME_CHARS = 512 * 1024


class WsEnvelope(BaseModel):
    """One WebSocket frame: ``{type, seq, payload}``.

    Attributes:
        type: Message type; the stream handler decides which types it understands.
        seq: Client-incremented sequence number, echoed by the answering envelope.
        payload: Free-form message body, validated by the handler.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(min_length=1, max_length=32)
    seq: int = Field(default=0, ge=0, le=2**31 - 1)
    payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def parse(cls, raw: str) -> WsEnvelope:
        """Decode one text frame.

        Raises:
            aicls.core.errors.AiclsError: ``VALIDATION_ERROR`` for an oversized frame, invalid
                JSON or a frame that is not an envelope.
        """
        if len(raw) > MAX_FRAME_CHARS:
            raise AiclsError(
                ErrorCode.VALIDATION_ERROR,
                messages_tr.MSG_VALIDATION_ERROR,
                {"reason": "frame_too_large", "limit": MAX_FRAME_CHARS},
            )
        try:
            return cls.model_validate_json(raw)
        except ValidationError as exc:
            raise AiclsError(
                ErrorCode.VALIDATION_ERROR,
                messages_tr.MSG_VALIDATION_ERROR,
                {"reason": "bad_envelope", "errors": exc.errors(include_url=False)[:4]},
            ) from exc

    def reply(self, type_: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build an answering frame that carries this envelope's ``seq``."""
        return {"type": type_, "seq": self.seq, "payload": payload or {}}


def envelope(type_: str, payload: dict[str, Any] | None = None, seq: int = 0) -> dict[str, Any]:
    """Build a server → client frame."""
    return {"type": type_, "seq": seq, "payload": payload or {}}


def error_envelope(error: AiclsError, seq: int = 0) -> dict[str, Any]:
    """Wrap an error as an ``error`` frame carrying the RFC 9457 problem document."""
    return envelope("error", {"problem": error.to_problem()}, seq)


def slow_down(retry_after_ms: int, seq: int = 0) -> dict[str, Any]:
    """Build the backpressure frame (api.md §4.1)."""
    return envelope("slow_down", {"retry_after_ms": int(retry_after_ms)}, seq)


__all__ = [
    "CLIENT_TYPES",
    "MAX_FRAME_CHARS",
    "WS_POLICY_VIOLATION",
    "WS_TRY_AGAIN_LATER",
    "AudioClientType",
    "AudioServerType",
    "InkClientType",
    "InkServerType",
    "WsEnvelope",
    "envelope",
    "error_envelope",
    "slow_down",
]
