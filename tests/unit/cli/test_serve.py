"""``aicls serve``: the resolved server configuration, without opening a socket.

Only :func:`aicls.cli.serve.run_server` is replaced — the seam that would block on a port. The
profile decision, the environment export the ``--reload`` worker depends on and the loopback
default are all asserted on the real command.
"""

from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

from tests.unit.cli.conftest import payload_of


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Capture the :class:`aicls.cli.serve.ServeSpec` instead of starting uvicorn."""
    from aicls.cli import serve as serve_module

    calls: list[object] = []
    monkeypatch.setattr(serve_module, "run_server", calls.append)
    return calls


pytestmark = pytest.mark.usefixtures("cli_env")


def test_serve_defaults_to_loopback(
    runner: CliRunner, app_module: object, started: list[object]
) -> None:
    """A development server binds 127.0.0.1 unless someone asks for something wider."""
    result = runner.invoke(app_module, ["serve", "--json"])
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["host"] == "127.0.0.1"
    assert payload["port"] == 8000
    assert payload["app"] == "aicls.api.app:create_app"
    assert payload["started"] is True
    assert len(started) == 1


def test_serve_profile_reaches_the_reload_worker(
    runner: CliRunner, app_module: object, started: list[object]
) -> None:
    """``--profile`` is exported: a reload worker is a fresh process and reads the environment."""
    result = runner.invoke(
        app_module, ["serve", "--profile", "collect", "--port", "8123", "--reload", "--json"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["profile"] == "collect"
    assert payload["port"] == 8123
    assert payload["reload"] is True
    assert os.environ["AICLS_PROFILE"] == "collect"
    spec = started[0]
    assert spec.profile == "collect"  # type: ignore[attr-defined]
    assert spec.reload is True  # type: ignore[attr-defined]


def test_serve_dry_run_does_not_start(
    runner: CliRunner, app_module: object, started: list[object]
) -> None:
    """``--dry-run`` resolves and prints the configuration without binding anything."""
    result = runner.invoke(app_module, ["serve", "--dry-run", "--json"])
    assert result.exit_code == 0
    assert payload_of(result.stdout)["started"] is False
    assert started == []


def test_serve_refuses_an_unknown_profile(
    runner: CliRunner, app_module: object, started: list[object]
) -> None:
    """Only the two profiles of :class:`aicls.core.settings.Settings` exist."""
    result = runner.invoke(app_module, ["serve", "--profile", "everything"])
    assert result.exit_code == 1
    assert "unknown_profile" in result.stderr
    assert started == []
