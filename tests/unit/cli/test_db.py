"""``aicls db``: migrations and reference data (ADR-0003).

The schema is only ever created by the migrations, so ``upgrade`` is asserted through Alembic's
own ``alembic_version`` stamp rather than through the presence of a table, and ``downgrade base``
must actually take it away again.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.unit.cli.conftest import payload_of

pytestmark = pytest.mark.usefixtures("cli_env")


def test_db_upgrade_stamps_head(runner: CliRunner, app_module: object, db_url: str) -> None:
    """``aicls db upgrade`` brings an empty database to the newest revision."""
    from aicls.db import head_revision

    result = runner.invoke(app_module, ["db", "upgrade", "--url", db_url, "--json"])
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["revision_before"] is None
    assert payload["revision"] == head_revision()


def test_db_downgrade_to_base_drops_the_schema(
    runner: CliRunner, app_module: object, db_url: str
) -> None:
    """``downgrade base`` unstamps the database; the pair is what makes a migration reviewable."""
    runner.invoke(app_module, ["db", "upgrade", "--url", db_url])
    result = runner.invoke(app_module, ["db", "downgrade", "base", "--url", db_url, "--json"])
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["revision"] is None


def test_db_revision_writes_a_script(
    runner: CliRunner, app_module: object, db_url: str, tmp_path: Path
) -> None:
    """``aicls db revision -m`` produces a migration file (written into a scratch directory here)."""
    versions = tmp_path / "versions"
    result = runner.invoke(
        app_module,
        [
            "db",
            "revision",
            "-m",
            "cli smoke",
            "--url",
            db_url,
            "--no-autogenerate",
            "--version-path",
            str(versions),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["files"], "a revision command that writes nothing is a silent failure"
    written = list(versions.glob("*.py"))
    assert written and "cli smoke" in written[0].read_text(encoding="utf-8")


def test_db_seed_is_idempotent(runner: CliRunner, app_module: object, db_url: str) -> None:
    """Seeding stores the consent versions and reading prompts; running it twice adds nothing."""
    first = runner.invoke(app_module, ["db", "seed", "--url", db_url, "--json"])
    assert first.exit_code == 0, first.stdout + first.stderr
    payload = payload_of(first.stdout)
    assert payload["consent_versions"] >= 1
    assert payload["prompts_added"] > 0
    assert payload["prompts_total"] == payload["prompts_added"]

    second = runner.invoke(app_module, ["db", "seed", "--url", db_url, "--json"])
    assert second.exit_code == 0
    repeat = payload_of(second.stdout)
    assert repeat["consent_added"] == 0
    assert repeat["prompts_added"] == 0
    assert repeat["prompts_total"] == payload["prompts_total"]


def test_seeded_prompts_come_from_the_command_inventory() -> None:
    """Reading prompts are the utterances the NLU layer supports, never a separate list."""
    from aicls.cli.db import prompt_rows
    from aicls.nlu import load_inventory

    rows = prompt_rows(2)
    inventory = load_inventory()
    known = {example.text for intent in inventory.intents for example in intent.examples}
    assert rows
    assert {row["text_tr"] for row in rows} <= known
    assert {row["intent"] for row in rows} <= set(inventory.intent_ids)


def test_db_seed_leaves_the_database_at_head(
    runner: CliRunner, app_module: object, db_url: str
) -> None:
    """A command that needs the schema migrates it first rather than failing on a fresh database."""
    from aicls.db import current_revision, head_revision, make_engine

    result = runner.invoke(app_module, ["db", "seed", "--url", db_url])
    assert result.exit_code == 0, result.stdout + result.stderr
    engine = make_engine(db_url)
    try:
        assert current_revision(engine) == head_revision()
    finally:
        engine.dispose()
