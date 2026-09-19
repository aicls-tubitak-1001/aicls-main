"""``aicls book``: the İP-8 content graph, with 100 % tagging as a gate rather than a statistic.

The committed sample chapters (``datasets/book/sample``) are the fixture, so the test also proves
that what is in the repository still satisfies the criterion the work package promises.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.unit.cli.conftest import REPO_ROOT, payload_of

SAMPLE_DIR = REPO_ROOT / "datasets" / "book" / "sample"

pytestmark = pytest.mark.usefixtures("cli_env")


def test_book_validate_passes_on_the_committed_sample(
    runner: CliRunner, app_module: object
) -> None:
    """Every unit-looking heading in the sample chapters is tagged (İP-8: 100 %)."""
    result = runner.invoke(app_module, ["book", "validate", str(SAMPLE_DIR), "--json"])
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["ok"] is True
    assert payload["untagged"] == []
    assert payload["units"] > 0


def test_book_validate_fails_on_an_untagged_heading(
    runner: CliRunner, app_module: object, tmp_path: Path
) -> None:
    """A heading that produces no unit must fail the command, not lower an average."""
    chapter = tmp_path / "bolum-99.md"
    chapter.write_text(
        "# Bölüm 99\n\n## Tanım. Bir şey\n\nGövde metni.\n",
        encoding="utf-8",
        newline="\n",
    )
    result = runner.invoke(app_module, ["book", "validate", str(tmp_path), "--json"])
    assert result.exit_code == 1
    payload = payload_of(result.stdout)
    assert payload["ok"] is False
    assert payload["untagged"]


def test_book_coverage_reports_per_file_and_overall(runner: CliRunner, app_module: object) -> None:
    """Coverage is reported per chapter and combined, against the contractual target of 1.0."""
    result = runner.invoke(app_module, ["book", "coverage", str(SAMPLE_DIR), "--json"])
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["coverage"] == 1.0
    assert payload["target"] == 1.0
    assert payload["files"]


def test_book_load_stores_units_and_relations(
    runner: CliRunner, app_module: object, db_url: str
) -> None:
    """Loading writes units, relations and generated aliases through the real migrations."""
    result = runner.invoke(
        app_module, ["book", "load", str(SAMPLE_DIR), "--db-url", db_url, "--json"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["units"] > 0
    assert payload["aliases"] > 0
    assert payload["tagged_ratio"] == 1.0


def test_book_export_from_a_directory(
    runner: CliRunner, app_module: object, tmp_path: Path
) -> None:
    """JSON-LD and Turtle are both produced, from a source directory without a database."""
    jsonld = tmp_path / "book.json"
    ttl = tmp_path / "book.ttl"
    result = runner.invoke(
        app_module,
        [
            "book",
            "export",
            "--dir",
            str(SAMPLE_DIR),
            "--jsonld",
            str(jsonld),
            "--ttl",
            str(ttl),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["units"] > 0
    document = json.loads(jsonld.read_text(encoding="utf-8"))
    assert "@context" in document and "@graph" in document
    assert ttl.read_text(encoding="utf-8").strip()


def test_book_export_from_the_database(
    runner: CliRunner, app_module: object, tmp_path: Path, db_url: str
) -> None:
    """After a load, the same graph comes back out of the database."""
    runner.invoke(app_module, ["book", "load", str(SAMPLE_DIR), "--db-url", db_url])
    jsonld = tmp_path / "from-db.json"
    result = runner.invoke(
        app_module, ["book", "export", "--jsonld", str(jsonld), "--db-url", db_url, "--json"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["source"] == "database"
    assert payload["units"] > 0


def test_book_export_without_a_target_is_refused(runner: CliRunner, app_module: object) -> None:
    """An export that writes nothing is a mistake, not a no-op."""
    result = runner.invoke(app_module, ["book", "export", "--dir", str(SAMPLE_DIR)])
    assert result.exit_code == 1
    assert "no_export_target" in result.stderr
