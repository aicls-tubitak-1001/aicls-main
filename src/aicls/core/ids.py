"""Identifiers: monotonic ULIDs and short content hashes.

ULIDs (26 Crockford base32 characters, 48-bit millisecond timestamp + 80-bit randomness) are the
primary keys and request ids of the platform (docs/mimari/api.md §1). Generation is monotonic
within a process: two ids created in the same millisecond, or after the wall clock steps back,
still sort in creation order. No third-party ULID package is used.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time

from aicls.core import messages_tr
from aicls.core.errors import InputValidationError

ULID_LENGTH = 26
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_DECODE = {ch: i for i, ch in enumerate(_CROCKFORD)}
_TIME_CHARS = 10
_RAND_CHARS = 16
_MAX_TIME = (1 << 48) - 1
_MAX_RAND = (1 << 80) - 1


def _encode(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        chars.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


class UlidGenerator:
    """Thread-safe monotonic ULID source.

    Ids created in the same millisecond, or after the wall clock steps backwards, reuse the last
    timestamp and increment the random component so they still sort in creation order. The
    module-level :func:`ulid` uses one shared instance; tests create their own for isolation.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_time = -1
        self._last_rand = 0

    def __call__(self, *, timestamp_ms: int | None = None) -> str:
        """Return a new ULID (see :func:`ulid`)."""
        now = int(time.time() * 1000) if timestamp_ms is None else timestamp_ms
        if now < 0 or now > _MAX_TIME:
            raise InputValidationError(
                messages_tr.msg("INVALID_TIMESTAMP", value=now), details={"timestamp_ms": now}
            )
        with self._lock:
            if now <= self._last_time:
                now = self._last_time
                rand = self._last_rand + 1
                if rand > _MAX_RAND:  # pragma: no cover - astronomically unlikely
                    now += 1
                    rand = int.from_bytes(os.urandom(10), "big")
            else:
                rand = int.from_bytes(os.urandom(10), "big")
            self._last_time, self._last_rand = now, rand
        return _encode(now, _TIME_CHARS) + _encode(rand, _RAND_CHARS)


_default_generator = UlidGenerator()


def ulid(*, timestamp_ms: int | None = None) -> str:
    """Return a new monotonic ULID from the process-wide generator.

    Args:
        timestamp_ms: Milliseconds since the Unix epoch; defaults to the current time. Because
            the shared generator is monotonic, an explicit value earlier than the last issued id
            is raised to that id's timestamp — use :class:`UlidGenerator` for fixed timestamps.

    Raises:
        InputValidationError: If ``timestamp_ms`` is outside the 48-bit range.
    """
    return _default_generator(timestamp_ms=timestamp_ms)


def is_ulid(value: str) -> bool:
    """True when ``value`` is a syntactically valid upper-case ULID."""
    return (
        isinstance(value, str)
        and len(value) == ULID_LENGTH
        and all(ch in _DECODE for ch in value)
        and value[0] <= "7"
    )


def ulid_timestamp_ms(value: str) -> int:
    """Extract the millisecond timestamp encoded in a ULID.

    Raises:
        InputValidationError: If ``value`` is not a valid ULID.
    """
    if not is_ulid(value):
        raise InputValidationError(
            messages_tr.msg("INVALID_ULID", value=value), details={"value": value}
        )
    result = 0
    for ch in value[:_TIME_CHARS]:
        result = (result << 5) | _DECODE[ch]
    return result


def short_hash(s: str, n: int = 8) -> str:
    """First ``n`` hex characters of the SHA-256 of ``s`` (UTF-8).

    Used for cache keys, run names and config fingerprints where a full digest is unwieldy.

    Raises:
        InputValidationError: If ``n`` is not between 1 and 64.
    """
    if not 1 <= n <= 64:
        raise InputValidationError(messages_tr.msg("INVALID_HASH_LENGTH", n=n), details={"n": n})
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


__all__ = ["ULID_LENGTH", "UlidGenerator", "is_ulid", "short_hash", "ulid", "ulid_timestamp_ms"]
