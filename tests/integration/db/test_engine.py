"""Engine factory and session scope (ADR-0003 §1)."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Engine, inspect, select
from sqlalchemy.pool import StaticPool

from aicls.core.errors import InputValidationError
from aicls.db import (
    User,
    UserRepository,
    dispose_engines,
    get_engine,
    get_session,
    make_engine,
    session_scope,
    set_default_engine,
    upgrade_head,
)
from aicls.db.engine import (
    MYSQL_POOL_RECYCLE_S,
    SQLITE_BUSY_TIMEOUT_MS,
    engine_options,
)

pytestmark = pytest.mark.integration


def _pragma(engine: Engine, name: str) -> object:
    with engine.connect() as connection:
        cursor = connection.connection.cursor()
        try:
            cursor.execute(f"PRAGMA {name}")  # raw-sql-ok: reads back the pragma under test
            row = cursor.fetchone()
        finally:
            cursor.close()
    return row[0] if row else None


def test_sqlite_file_engine_enables_wal_and_foreign_keys(tmp_path: Path) -> None:
    """A file-backed SQLite engine runs in WAL mode with foreign keys enforced."""
    url = f"sqlite+pysqlite:///{(tmp_path / 'aicls.db').as_posix()}"
    engine = make_engine(url)
    try:
        assert str(_pragma(engine, "journal_mode")).lower() == "wal"
        assert int(_pragma(engine, "foreign_keys")) == 1
        assert int(_pragma(engine, "busy_timeout")) == SQLITE_BUSY_TIMEOUT_MS
    finally:
        engine.dispose()


def test_memory_engine_uses_one_shared_connection(engine: Engine) -> None:
    """``:memory:`` uses ``StaticPool`` so the migrated schema is visible to every session."""
    assert int(_pragma(engine, "foreign_keys")) == 1
    with session_scope(engine) as first:
        UserRepository(first).create("a@example.edu.tr", "h", "A")
    with session_scope(engine) as second:
        assert second.scalars(select(User)).one().email == "a@example.edu.tr"


def test_foreign_keys_are_enforced_on_sqlite(engine: Engine) -> None:
    """A dangling foreign key is rejected (``PRAGMA foreign_keys=ON``)."""
    from sqlalchemy.exc import IntegrityError

    from aicls.db import Workspace

    with pytest.raises(IntegrityError), session_scope(engine) as session:
        session.add(Workspace(owner_id="01" + "0" * 24, title="orphan"))
        session.flush()


def test_make_engine_rejects_empty_and_unsupported_urls() -> None:
    """Only SQLite and MySQL URLs are accepted (ADR-0003 rejected PostgreSQL)."""
    with pytest.raises(InputValidationError):
        make_engine("")
    with pytest.raises(InputValidationError):
        make_engine("postgresql+psycopg://user@localhost/aicls")


def test_mysql_engine_is_configured_for_utf8mb4_without_connecting() -> None:
    """The MySQL branch sets utf8mb4, strict SQL mode and pre-ping (no connection is opened)."""
    url, options = engine_options("mysql+pymysql://u:p@127.0.0.1/aicls")
    assert url.get_backend_name() == "mysql"
    assert options["connect_args"]["charset"] == "utf8mb4"
    assert options["connect_args"]["sql_mode"] == "STRICT_TRANS_TABLES"
    assert options["pool_pre_ping"] is True
    assert options["pool_recycle"] == MYSQL_POOL_RECYCLE_S

    _, memory_options = engine_options("sqlite+pysqlite:///:memory:")
    assert memory_options["poolclass"] is StaticPool


def test_session_scope_rolls_back_on_error(engine: Engine) -> None:
    """An exception inside the scope rolls the transaction back."""
    with pytest.raises(RuntimeError), session_scope(engine) as session:
        UserRepository(session).create("boom@example.edu.tr", "h", "B")
        raise RuntimeError("boom")
    with session_scope(engine) as session:
        assert UserRepository(session).get_by_email("boom@example.edu.tr") is None


def test_get_session_dependency_uses_the_default_engine(engine: Engine) -> None:
    """``set_default_engine`` redirects the FastAPI dependency at a test database."""
    set_default_engine(engine)
    try:
        assert get_engine() is engine
        generator = get_session()
        session = next(generator)
        UserRepository(session).create("dep@example.edu.tr", "h", "D")
        with pytest.raises(StopIteration):
            next(generator)
        with session_scope(engine) as check:
            assert UserRepository(check).get_by_email("dep@example.edu.tr") is not None
    finally:
        set_default_engine(None)


def test_get_engine_caches_per_url_and_dispose_clears_it(tmp_path: Path) -> None:
    """Engines are cached per URL and released by ``dispose_engines``."""
    url = f"sqlite+pysqlite:///{(tmp_path / 'cached.db').as_posix()}"
    first = get_engine(url)
    assert get_engine(url) is first
    upgrade_head(first)
    assert "users" in inspect(first).get_table_names()
    dispose_engines()
    assert get_engine(url) is not first
    dispose_engines()
