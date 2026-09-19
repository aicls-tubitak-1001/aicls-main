"""Shared WebSocket plumbing: ticket authentication and per-connection backpressure.

Both streams open the same way — ``?ticket=<value>`` from ``POST /auth/ws-ticket``, redeemed
once — and both protect the server the same way: a per-connection message-rate budget that
answers ``slow_down`` and, if the client keeps pushing, closes the socket (api.md §1, §4.1).
A rate budget is the honest backpressure signal here, because recognition itself is fast enough
with a stub that a queue would never build up in a test.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import time

from starlette.websockets import WebSocket

from aicls.api.errors import AuthError
from aicls.api.security import Ticket, WsTicketStore
from aicls.api.ws.protocol import WS_POLICY_VIOLATION

#: Messages one connection may send inside :data:`RATE_WINDOW_S` before ``slow_down``.
RATE_LIMIT_MESSAGES = 30

#: Length of the message-rate window in seconds.
RATE_WINDOW_S = 1.0

#: ``retry_after_ms`` advertised by a ``slow_down`` envelope.
SLOW_DOWN_RETRY_MS = 250

#: Consecutive ``slow_down`` envelopes tolerated before the connection is closed.
MAX_STRIKES = 5


@dataclass(frozen=True, slots=True)
class WsTimings:
    """Debounce and endpointing budgets, overridable per application (ADR-0014).

    Attributes:
        pen_up_debounce_ms: Idle time after ``pen_up`` before a partial recognition runs.
        idle_final_ms: Idle time after which a ``final`` is produced without ``finalize``.
        silence_ms: Trailing silence that ends an utterance on the audio stream.
        max_audio_s: Hard cap on one buffered utterance.
    """

    pen_up_debounce_ms: int = 400
    idle_final_ms: int = 1500
    silence_ms: int = 600
    max_audio_s: float = 30.0


class MessageRate:
    """Sliding-window message counter for one connection.

    Args:
        limit: Messages allowed inside ``window_s``.
        window_s: Window length in seconds.
        clock: Monotonic seconds source; injectable for tests.
    """

    def __init__(
        self,
        limit: int = RATE_LIMIT_MESSAGES,
        window_s: float = RATE_WINDOW_S,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._limit = limit
        self._window_s = window_s
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self._hits: deque[float] = deque()
        self.strikes = 0

    def allow(self) -> bool:
        """Record one message; ``False`` when the budget is exhausted."""
        now = self._clock()
        cutoff = now - self._window_s
        while self._hits and self._hits[0] <= cutoff:
            self._hits.popleft()
        if len(self._hits) >= self._limit:
            self.strikes += 1
            return False
        self._hits.append(now)
        self.strikes = 0
        return True

    @property
    def exhausted(self) -> bool:
        """Whether the client ignored :data:`MAX_STRIKES` consecutive ``slow_down`` envelopes."""
        return self.strikes >= MAX_STRIKES


async def authenticate(websocket: WebSocket, tickets: WsTicketStore) -> Ticket:
    """Accept the connection only when it carries a live, unused ticket.

    Args:
        websocket: The incoming connection (not yet accepted).
        tickets: The application's ticket store.

    Returns:
        The redeemed :class:`~aicls.api.security.Ticket`.

    Raises:
        AuthError: If the ticket is missing, unknown, expired or already used. The socket is
            closed with 1008 before the error propagates.
    """
    value = websocket.query_params.get("ticket", "")
    try:
        ticket = tickets.consume(value) if value else None
    except AuthError:
        ticket = None
    if ticket is None:
        await websocket.close(code=WS_POLICY_VIOLATION, reason="ticket_invalid")
        raise AuthError(details={"reason": "ticket_invalid"})
    await websocket.accept()
    return ticket


__all__ = [
    "MAX_STRIKES",
    "RATE_LIMIT_MESSAGES",
    "RATE_WINDOW_S",
    "SLOW_DOWN_RETRY_MS",
    "MessageRate",
    "WsTimings",
    "authenticate",
]
