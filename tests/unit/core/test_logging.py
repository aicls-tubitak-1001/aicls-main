"""structlog configuration: JSON output, request_id binding, level filtering."""

from __future__ import annotations

from collections.abc import Iterator
import io
import json

import pytest
import structlog

from aicls.core import logging as core_logging
from aicls.core.errors import AiclsError
from aicls.core.logging import (
    bind_request_id,
    clear_request_context,
    configure_logging,
    get_logger,
    request_context,
)


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    clear_request_context()
    core_logging.reset_logging()
    yield
    clear_request_context()
    core_logging.reset_logging()


def _configure(level: str = "INFO") -> io.StringIO:
    stream = io.StringIO()
    configure_logging(level=level, stream=stream, json_output=True)
    return stream


def test_json_line_with_request_id() -> None:
    stream = _configure()
    bind_request_id("01J8R2ZQ7G4V0X9K3M2N5P6Q7R")
    log = get_logger("aicls.test")
    log.info("hello", stage="nlu", ms=12.3)
    record = json.loads(stream.getvalue().strip().splitlines()[-1])
    assert record["event"] == "hello"
    assert record["request_id"] == "01J8R2ZQ7G4V0X9K3M2N5P6Q7R"
    assert record["stage"] == "nlu"
    assert record["ms"] == 12.3
    assert record["level"] == "info"
    assert record["logger"] == "aicls.test"
    assert record["timestamp"].endswith("Z")


def test_level_filtering() -> None:
    stream = _configure("WARNING")
    log = get_logger("x")
    log.info("dropped")
    log.warning("kept")
    lines = stream.getvalue().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "kept"


def test_request_context_manager_binds_and_unbinds() -> None:
    stream = _configure()
    log = get_logger("x")
    with request_context("abc") as rid:
        assert rid == "abc"
        log.info("inside")
    log.info("outside")
    inside, outside = (json.loads(ln) for ln in stream.getvalue().strip().splitlines())
    assert inside["request_id"] == "abc"
    assert "request_id" not in outside


def test_request_context_generates_ulid_when_absent() -> None:
    with request_context() as rid:
        assert len(rid) == 26


def test_configure_is_idempotent_and_reconfigurable() -> None:
    s1 = _configure()
    get_logger("x").info("one")
    s2 = _configure()
    get_logger("x").info("two")
    assert "one" in s1.getvalue() and "two" not in s1.getvalue()
    assert "two" in s2.getvalue()


def test_get_logger_autoconfigures_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AICLS_LOG_LEVEL", "ERROR")
    log = get_logger("auto")
    assert core_logging.is_configured()
    assert core_logging.current_level() == "ERROR"
    with structlog.testing.capture_logs() as captured:
        log.error("boom", code="INTERNAL")
    assert captured[0]["event"] == "boom"


def test_console_renderer_when_not_json() -> None:
    stream = io.StringIO()
    configure_logging(level="INFO", stream=stream, json_output=False)
    get_logger("x").info("plain")
    assert "plain" in stream.getvalue()
    assert not stream.getvalue().lstrip().startswith("{")


def test_invalid_level_rejected() -> None:
    with pytest.raises(AiclsError):
        configure_logging(level="LOUD")


def test_log_error_helper_serialises_aicls_error() -> None:
    stream = _configure()
    err = AiclsError("NOT_FOUND", details={"id": "x"})
    core_logging.log_error(get_logger("x"), err, request_id="r1")
    record = json.loads(stream.getvalue().strip().splitlines()[-1])
    assert record["code"] == "NOT_FOUND"
    assert record["http_status"] == 404
    assert record["request_id"] == "r1"
    assert record["details"] == {"id": "x"}
    assert record["level"] == "warning"
    err500 = AiclsError("INTERNAL")
    core_logging.log_error(get_logger("x"), err500)
    record = json.loads(stream.getvalue().strip().splitlines()[-1])
    assert record["level"] == "error"
