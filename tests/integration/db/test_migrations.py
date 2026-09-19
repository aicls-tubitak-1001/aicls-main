"""Migration tests required by modules/db.json and ADR-0003.

``test_alembic_upgrade_head_on_sqlite_and_mysql_dialect_compile`` proves both halves of the
"SQLite in development, MySQL in production" decision on a machine with no MySQL server: the
migrations really run on SQLite, and every one of them compiles under the MySQL dialect through
Alembic's offline (``--sql``) mode.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import Engine, inspect

from aicls.db import TABLE_NAMES, make_engine, upgrade_head
from aicls.db.migrate import (
    INITIAL_REVISION,
    current_revision,
    diff_metadata,
    downgrade,
    head_revision,
    offline_sql,
    script_directory,
)

pytestmark = pytest.mark.integration

MYSQL_URL = "mysql+pymysql://aicls_app:secret@127.0.0.1:3306/aicls?charset=utf8mb4"


def _table_names(engine: Engine) -> set[str]:
    return set(inspect(engine).get_table_names())


def test_alembic_upgrade_head_on_sqlite_and_mysql_dialect_compile(engine: Engine) -> None:
    """Migrations run on SQLite and the same revisions compile for MySQL without a server."""
    tables = _table_names(engine)
    assert set(TABLE_NAMES) <= tables
    assert "alembic_version" in tables
    assert current_revision(engine) == head_revision()

    sql = offline_sql(MYSQL_URL)
    for name in TABLE_NAMES:
        assert f"CREATE TABLE {name}" in sql, name
    assert "ENGINE=InnoDB" in sql
    assert "CHARSET=utf8mb4" in sql
    assert "COLLATE utf8mb4_0900_ai_ci" in sql
    # The one MySQL-specific object (ADR-0003 rule 7) is emitted only for MySQL.
    assert "CREATE FULLTEXT INDEX ftx_book_units_search_text" in sql
    assert "FULLTEXT" not in offline_sql("sqlite+pysqlite:///:memory:")


def test_migrations_head_base_head_sqlite(tmp_path: object) -> None:
    """head -> base -> head leaves exactly the same set of tables (downgrade is complete)."""
    engine = make_engine("sqlite+pysqlite:///:memory:")
    try:
        upgrade_head(engine)
        after_first = _table_names(engine)
        assert set(TABLE_NAMES) <= after_first

        downgrade(engine, "base")
        after_base = _table_names(engine)
        assert after_base - {"alembic_version"} == set()
        assert current_revision(engine) is None

        upgrade_head(engine)
        assert _table_names(engine) == after_first
        assert current_revision(engine) == head_revision()
    finally:
        engine.dispose()


def test_autogenerate_diff_is_empty(engine: Engine) -> None:
    """The migrated schema and the ORM metadata agree (no forgotten revision)."""
    assert diff_metadata(engine) == []


def test_initial_revision_is_the_only_root() -> None:
    """Exactly one linear history rooted at revision 0001 (one revision per pull request)."""
    scripts = script_directory()
    revisions = list(scripts.walk_revisions())
    assert scripts.get_current_head() is not None
    assert [rev.revision for rev in revisions][-1] == INITIAL_REVISION
    assert all(rev.is_branch_point is False for rev in revisions)


def test_upgrade_head_accepts_a_url(tmp_path: object) -> None:
    """``upgrade_head`` also accepts a plain URL (the ``aicls db upgrade`` path)."""
    from pathlib import Path

    db_path = Path(str(tmp_path)) / "aicls-test.db"
    url = f"sqlite+pysqlite:///{db_path.as_posix()}"
    upgrade_head(url)
    engine = make_engine(url)
    try:
        assert set(TABLE_NAMES) <= _table_names(engine)
    finally:
        engine.dispose()


@pytest.mark.mysql
def test_migrations_mysql_service() -> None:
    """Nightly: the full head -> base -> head cycle against a real ``mysql:8`` service.

    Skipped unless ``AICLS_TEST_MYSQL_URL`` is set (tests/conftest.py marker policy).
    """
    url = os.environ["AICLS_TEST_MYSQL_URL"]
    engine = make_engine(url)
    try:
        upgrade_head(engine)
        assert set(TABLE_NAMES) <= _table_names(engine)
        assert diff_metadata(engine) == []
        downgrade(engine, "base")
        assert set(TABLE_NAMES) & _table_names(engine) == set()
        upgrade_head(engine)
        assert current_revision(engine) == head_revision()
    finally:
        engine.dispose()


def test_diff_excludes_exactly_the_index_the_migration_creates() -> None:
    """``diff_metadata`` leaves out one reflected index by name; that name must be the one
    migration 0001 creates under its MySQL guard.

    The two constants live in different files on purpose (a migration must not import the
    live models), so this is the only thing holding them together. If somebody renames the
    index in the migration, the nightly MySQL job fails again with ``remove_index`` and the
    reason is this line, not the design.
    """
    from pathlib import Path

    from aicls.db.models.book import FULLTEXT_INDEX_NAME

    migration = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "aicls"
        / "db"
        / "migrations"
        / "versions"
        / "0001_initial_schema.py"
    )
    assert f'FULLTEXT_INDEX = "{FULLTEXT_INDEX_NAME}"' in migration.read_text(encoding="utf-8")
