"""WebSocket streams: typed envelopes, ink recognition and speech (api.md §4, ADR-0014)."""

from __future__ import annotations

from aicls.api.ws.audio import AudioBuffer, audio_stream, is_endpoint
from aicls.api.ws.common import MessageRate, WsTimings, authenticate
from aicls.api.ws.ink import InkBuffer, ink_stream
from aicls.api.ws.protocol import (
    WS_POLICY_VIOLATION,
    WS_TRY_AGAIN_LATER,
    WsEnvelope,
    envelope,
    error_envelope,
    slow_down,
)

__all__ = [
    "WS_POLICY_VIOLATION",
    "WS_TRY_AGAIN_LATER",
    "AudioBuffer",
    "InkBuffer",
    "MessageRate",
    "WsEnvelope",
    "WsTimings",
    "audio_stream",
    "authenticate",
    "envelope",
    "error_envelope",
    "ink_stream",
    "is_endpoint",
    "slow_down",
]
