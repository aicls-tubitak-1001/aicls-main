"""ULID generation (monotonic, sortable, 26-char Crockford base32) and short hashes."""

from __future__ import annotations

import re
import threading

import pytest

from aicls.core.errors import AiclsError
from aicls.core.ids import (
    ULID_LENGTH,
    UlidGenerator,
    is_ulid,
    short_hash,
    ulid,
    ulid_timestamp_ms,
)

ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


def test_ulid_shape() -> None:
    u = ulid()
    assert len(u) == ULID_LENGTH == 26
    assert ULID_RE.match(u)
    assert is_ulid(u)
    assert not is_ulid(u.lower() + "x")
    assert not is_ulid("01J8R2ZQ7G4V0X9K3M2N5P6Q7I")  # I is not a Crockford digit
    assert not is_ulid("")


def test_ulid_monotonic_within_same_millisecond() -> None:
    gen = UlidGenerator()
    ts = 1_700_000_000_000
    a = gen(timestamp_ms=ts)
    b = gen(timestamp_ms=ts)
    c = gen(timestamp_ms=ts)
    assert a < b < c
    assert a[:10] == b[:10] == c[:10]
    assert ulid_timestamp_ms(a) == ts


def test_ulid_sortable_across_time() -> None:
    gen = UlidGenerator()
    earlier = gen(timestamp_ms=1_700_000_000_000)
    later = gen(timestamp_ms=1_700_000_000_001)
    assert earlier < later
    assert ulid_timestamp_ms(later) - ulid_timestamp_ms(earlier) == 1


def test_ulid_clock_going_backwards_stays_monotonic() -> None:
    gen = UlidGenerator()
    a = gen(timestamp_ms=1_700_000_005_000)
    b = gen(timestamp_ms=1_700_000_000_000)  # clock stepped back
    assert b > a
    assert ulid_timestamp_ms(b) == ulid_timestamp_ms(a)


def test_module_level_ulid_is_monotonic_against_explicit_past_timestamp() -> None:
    first = ulid()
    past = ulid(timestamp_ms=1_700_000_000_000)
    assert past > first


def test_ulid_unique_under_threads() -> None:
    out: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        local = [ulid() for _ in range(200)]
        with lock:
            out.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(out)) == len(out) == 800


def test_ulid_rejects_out_of_range_timestamp() -> None:
    with pytest.raises(AiclsError):
        ulid(timestamp_ms=-1)
    with pytest.raises(AiclsError):
        ulid(timestamp_ms=1 << 48)


def test_ulid_timestamp_ms_rejects_invalid() -> None:
    with pytest.raises(AiclsError):
        ulid_timestamp_ms("not-a-ulid")


def test_short_hash_is_prefix_of_sha256_and_stable() -> None:
    assert short_hash("abc") == "ba7816bf"
    assert short_hash("abc", 16) == "ba7816bf8f01cfea"
    assert short_hash("İstanbul") == short_hash("İstanbul")
    assert short_hash("a") != short_hash("b")


def test_short_hash_rejects_bad_length() -> None:
    with pytest.raises(AiclsError):
        short_hash("abc", 0)
    with pytest.raises(AiclsError):
        short_hash("abc", 65)
