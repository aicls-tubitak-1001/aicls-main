"""Error contract (``UNSUPPORTED_PROBLEM`` / ``CAS_TIMEOUT``) and the LRU result cache."""

from __future__ import annotations

import pytest

from aicls.core import messages_tr
from aicls.core.errors import AiclsError, ErrorCode
from aicls.mathengine import CasTimeout, Diff, LruCache, Result, UnsupportedProblem
from aicls.mathengine.cache import DEFAULT_CACHE_SIZE
from aicls.mathengine.errors import error_to_payload, payload_to_error


def _result(latex: str = "2 x") -> Result:
    return Result(kind="expr", latex=latex, plain="2*x", srepr="Mul(Integer(2), x)")


# ---------------------------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------------------------
def test_unsupported_problem_carries_the_contract_message() -> None:
    error = UnsupportedProblem()
    assert error.message_tr == messages_tr.MSG_UNSUPPORTED_PROBLEM
    assert error.code is ErrorCode.UNSUPPORTED_PROBLEM
    assert error.http_status == 422
    assert error.to_problem()["detail_tr"] == "Bu problem tipi şu an için desteklenmemektedir."


def test_unsupported_problem_reason_is_machine_readable_only() -> None:
    error = UnsupportedProblem(reason="unevaluated_integral", details={"op": "integrate"})
    assert error.reason == "unevaluated_integral"
    assert error.details["reason"] == "unevaluated_integral"
    assert error.details["op"] == "integrate"
    # The Turkish text never leaks the internal reason.
    assert error.message_tr == messages_tr.MSG_UNSUPPORTED_PROBLEM


def test_cas_timeout_message_names_the_budget() -> None:
    error = CasTimeout(timeout_s=2.5)
    assert error.code is ErrorCode.CAS_TIMEOUT
    assert error.http_status == 504
    assert "2.5" in error.message_tr
    assert error.details["timeout_s"] == 2.5
    assert CasTimeout().message_tr == messages_tr.MSG_CAS_TIMEOUT


@pytest.mark.parametrize(
    "error",
    [
        UnsupportedProblem(reason="conditionset"),
        CasTimeout(timeout_s=2.5),
        AiclsError(ErrorCode.PARSE_ERROR, details={"position": 3}),
    ],
)
def test_errors_survive_the_worker_round_trip(error: AiclsError) -> None:
    """Sandbox workers cannot pickle our exceptions; they ship a payload instead."""
    restored = payload_to_error(error_to_payload(error))
    assert type(restored) is type(error)
    assert restored.code is error.code
    assert restored.message_tr == error.message_tr
    assert restored.details == error.details


def test_unknown_error_payload_degrades_to_internal() -> None:
    restored = payload_to_error({"code": "NO_SUCH_CODE", "message_tr": "x", "details": {}})
    assert restored.code is ErrorCode.INTERNAL


def test_core_errors_keep_their_class_across_the_pipe() -> None:
    """A worker's ``InputValidationError`` must not arrive as a bare ``AiclsError``."""
    from aicls.core.errors import InputValidationError, InternalError, NotFoundError

    for error in (InputValidationError("a"), NotFoundError("b"), InternalError("c")):
        restored = payload_to_error(error_to_payload(error))
        assert type(restored) is type(error)
        assert restored.message_tr == error.message_tr
        assert restored.http_status == error.http_status


def test_error_payload_is_json_safe() -> None:
    payload = error_to_payload(UnsupportedProblem(details={"value": object()}))
    assert isinstance(payload["details"]["value"], str)


def test_error_payload_flattens_nested_containers() -> None:
    payload = error_to_payload(
        UnsupportedProblem(
            details={
                "nested": {"n": 1, "obj": object()},
                "items": [1, "two", object()],
                "unique": {3, 4},
                "none": None,
                "flag": True,
            }
        )
    )
    details = payload["details"]
    assert details["nested"]["n"] == 1
    assert isinstance(details["nested"]["obj"], str)
    assert details["items"][:2] == [1, "two"]
    assert sorted(details["unique"]) == [3, 4]
    assert details["none"] is None
    assert details["flag"] is True


# ---------------------------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------------------------
def test_cache_default_size_follows_adr_0016() -> None:
    assert DEFAULT_CACHE_SIZE == 4096
    assert LruCache().maxsize == DEFAULT_CACHE_SIZE


def test_cache_returns_stored_result_and_counts_hits() -> None:
    cache = LruCache(maxsize=4)
    key = "k"
    assert cache.get(key) is None
    cache.put(key, _result())
    stored = cache.get(key)
    assert stored is not None
    assert stored.latex == "2 x"
    assert cache.stats() == {"hits": 1, "misses": 1, "size": 1, "maxsize": 4}


def test_cache_evicts_least_recently_used() -> None:
    cache = LruCache(maxsize=2)
    cache.put("a", _result("a"))
    cache.put("b", _result("b"))
    assert cache.get("a") is not None  # "a" becomes the most recent
    cache.put("c", _result("c"))
    assert cache.get("b") is None
    assert cache.get("a") is not None
    assert cache.get("c") is not None
    assert cache.size == 2


def test_cache_put_refreshes_an_existing_key() -> None:
    cache = LruCache(maxsize=2)
    cache.put("a", _result("first"))
    cache.put("a", _result("second"))
    stored = cache.get("a")
    assert stored is not None and stored.latex == "second"
    assert cache.size == 1


def test_cache_clear_resets_entries_and_counters() -> None:
    cache = LruCache(maxsize=2)
    cache.put("a", _result())
    cache.get("a")
    cache.clear()
    assert cache.size == 0
    assert cache.stats()["hits"] == 0


def test_cache_rejects_a_non_positive_size() -> None:
    from aicls.core.errors import InputValidationError

    with pytest.raises(InputValidationError) as excinfo:
        LruCache(maxsize=0)
    assert excinfo.value.details["field"] == "maxsize"


def test_cache_exposes_its_counters_and_container_protocol() -> None:
    cache = LruCache(maxsize=2)
    assert (cache.hits, cache.misses, len(cache)) == (0, 0, 0)
    assert "a" not in cache
    cache.put("a", _result())
    assert "a" in cache
    assert len(cache) == 1
    cache.get("a")
    cache.get("missing")
    assert (cache.hits, cache.misses) == (1, 1)


def test_cache_key_of_a_command_is_the_canonical_json() -> None:
    from aicls.mathengine import canonical_key

    assert canonical_key(Diff(op="diff", expr="x^{2}")).startswith("{")
