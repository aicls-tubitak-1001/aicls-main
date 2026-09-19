"""The root application: version, help, the error boundary and the output conventions.

The exit-code mapping is the contract the rest of the platform relies on: ``3`` means "an
approval gate refused" (ADR-0012) and nothing else does, ``1`` is every other typed failure and
``2`` stays click's own usage error.
"""

from __future__ import annotations

import sys

import pytest
from typer.testing import CliRunner

from tests.unit.cli.conftest import payload_of, problem_of

pytestmark = pytest.mark.usefixtures("cli_env")


def test_entry_point_target_is_importable() -> None:
    """``pyproject.toml`` declares ``aicls = "aicls.cli.main:app"``; it must exist and be callable."""
    from aicls.cli.main import app

    assert callable(app)


def test_importing_the_cli_does_not_import_torch() -> None:
    """Service imports are lazy: ``aicls --help`` must not pay for the ML stack."""
    import os
    import subprocess

    from tests.unit.cli.conftest import REPO_ROOT

    code = (
        "import sys; import aicls.cli.main; "
        "print('torch' in sys.modules, 'fastapi' in sys.modules, 'sympy' in sys.modules)"
    )
    completed = subprocess.run(  # fixed argv, no shell, no user input
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert completed.stdout.strip() == "False False False"


def test_version_flag(runner: CliRunner, app_module: object) -> None:
    """``--version`` prints the package version and exits 0."""
    from aicls import __version__

    result = runner.invoke(app_module, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_no_arguments_shows_help(runner: CliRunner, app_module: object) -> None:
    """A bare ``aicls`` prints the help instead of doing something surprising."""
    result = runner.invoke(app_module, [])
    assert result.exit_code != 0
    assert "Usage: aicls" in result.stdout


@pytest.mark.parametrize(
    "group",
    [
        "serve",
        "db",
        "data",
        "train",
        "profile",
        "eval",
        "book",
        "models",
        "contrib",
        "nlu",
        "report",
    ],
)
def test_every_command_group_is_mounted(runner: CliRunner, app_module: object, group: str) -> None:
    """Every command group of the module specification is reachable and documented."""
    result = runner.invoke(app_module, [group, "--help"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Usage:" in result.stdout


def test_unknown_command_is_a_usage_error(runner: CliRunner, app_module: object) -> None:
    """click owns exit code 2 for usage problems; the platform codes are 1 and 3."""
    result = runner.invoke(app_module, ["nosuchcommand"])
    assert result.exit_code == 2


def test_typed_error_maps_to_exit_one_and_a_turkish_message(
    runner: CliRunner, app_module: object
) -> None:
    """An :class:`AiclsError` becomes its exit code plus the Turkish user message on stderr."""
    result = runner.invoke(app_module, ["data", "verify", "nosuchdataset"])
    assert result.exit_code == 1
    assert "NOT_FOUND" in result.stderr
    assert "bulunamadı" in result.stderr  # aicls.core.messages_tr, not an English string


def test_typed_error_is_a_problem_document_under_json(
    runner: CliRunner, app_module: object
) -> None:
    """``--json`` turns the failure into the same RFC 9457 document the HTTP API returns."""
    result = runner.invoke(app_module, ["data", "verify", "nosuchdataset", "--json"])
    assert result.exit_code == 1
    problem = problem_of(result.stderr)
    assert problem["code"] == "NOT_FOUND"
    assert problem["status"] == 404
    assert "detail_tr" in problem


def test_json_output_goes_to_stdout_alone(runner: CliRunner, app_module: object) -> None:
    """Log lines share the process, so ``--json`` output must be parseable on its own."""
    result = runner.invoke(app_module, ["nlu", "try", "x karenin türevini al", "--json"])
    assert result.exit_code == 0, result.stdout + result.stderr
    payload_of(result.stdout)  # must parse without stripping anything


def test_jsonable_handles_the_project_value_types() -> None:
    """One encoder covers pydantic models, dataclasses, paths, datetimes and enums."""
    import datetime as dt
    from pathlib import Path

    from aicls.cli.common import jsonable
    from aicls.db.enums import EvalKind

    value = jsonable(
        {
            "path": Path("a/b"),
            "when": dt.datetime(2026, 9, 8, 12, 0, tzinfo=dt.UTC),
            "kind": EvalKind.CAS,
            "items": (1, 2),
            "tags": {"b", "a"},
        }
    )
    assert value["path"] == "a/b"
    assert value["when"].startswith("2026-09-08T12:00:00")
    assert value["kind"] == "cas"
    assert value["items"] == [1, 2]
    assert value["tags"] == ["a", "b"]
