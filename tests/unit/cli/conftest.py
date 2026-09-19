"""Fixtures shared by the CLI tests.

Every test drives the real Typer application through ``typer.testing.CliRunner``; nothing is
mocked away except the one function that would block on a socket (``aicls.cli.serve.run_server``).

The database URL is a **file** under ``tmp_path``, not ``:memory:``: each CLI invocation builds
and disposes its own engine, so an in-memory database would vanish between two commands and
``aicls models import`` followed by ``aicls models list`` could never be tested end to end.

No test ever sets a real ``AICLS_GATE_*`` value in the process environment or writes a row into
``docs/gelistirme/onay-defteri.md`` (ADR-0012). The gate tests assert refusals.
"""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[3]

#: The committed gate-free OCR smoke configuration (2 steps, 64 synthetic samples, CPU).
OCR_SMOKE_CONFIG = REPO_ROOT / "training" / "configs" / "ocr" / "smoke_synthetic.yaml"

#: A real-data configuration; running it needs the TRAINING gate.
OCR_REAL_CONFIG = REPO_ROOT / "training" / "configs" / "ocr" / "offline_resnet34_tfm.yaml"


@pytest.fixture
def runner() -> CliRunner:
    """A Typer CLI runner (stdout and stderr are kept apart by click 8.2+)."""
    return CliRunner()


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    """A file-backed SQLite URL that survives between CLI invocations."""
    return f"sqlite+pysqlite:///{(tmp_path / 'aicls-cli-test.db').as_posix()}"


@pytest.fixture
def cli_env(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, db_url: str, data_root: Path
) -> Iterator[dict[str, str]]:
    """Hermetic environment for a CLI invocation: temp data root, file database, gates empty."""
    from aicls.core.settings import get_settings

    monkeypatch.setenv("AICLS_DB_URL", db_url)
    get_settings.cache_clear()
    try:
        yield {**test_env, "AICLS_DB_URL": db_url, "AICLS_DATA_ROOT": str(data_root)}
    finally:
        get_settings.cache_clear()


@pytest.fixture
def app_module() -> Any:
    """The root Typer application under test."""
    from aicls.cli.main import app

    return app


def payload_of(output: str) -> Any:
    """Parse the command's own JSON document out of stdout.

    In a real ``aicls`` process stdout carries the payload and nothing else: the root callback
    redirects structured logging to stderr before any service module is imported. Inside the test
    process a service module may already have been imported by an earlier test, with its
    module-level logger bound to the stream that was configured then, so a stray single-line log
    record can still land here. Every top-level document is decoded and the one that is not a
    structlog record (``event`` + ``level``) is the command's payload.
    """
    decoder = json.JSONDecoder()
    documents: list[Any] = []
    index = 0
    while index < len(output):
        while index < len(output) and output[index].isspace():
            index += 1
        if index >= len(output):
            break
        document, index = decoder.raw_decode(output, index)
        documents.append(document)
    payloads = [
        document
        for document in documents
        if not (isinstance(document, dict) and {"event", "level"} <= set(document))
    ]
    if len(payloads) != 1:
        raise AssertionError("expected exactly one command payload on stdout:\n" + output)
    return payloads[0]


def problem_of(stderr: str) -> dict[str, Any]:
    """Pick the RFC 9457 problem document out of stderr (log lines share the stream)."""
    for line in reversed(stderr.strip().splitlines()):
        try:
            document = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(document, dict) and "detail_tr" in document:
            return document
    raise AssertionError(f"no problem document on stderr:\n{stderr}")


def write_yaml(path: Path, text: str) -> Path:
    """Write a UTF-8, LF-terminated YAML file and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path
