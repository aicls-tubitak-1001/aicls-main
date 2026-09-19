"""Engine factory, session scope and the FastAPI session dependency (ADR-0003).

One factory builds both supported backends from a URL:

* **SQLite** (dev/test) — ``PRAGMA foreign_keys=ON`` on every connection, WAL journal and a
  5 s busy timeout for file databases, ``StaticPool`` for ``:memory:`` so a whole test shares
  one connection.
* **MySQL 8** (prod) — ``utf8mb4``, ``sql_mode=STRICT_TRANS_TABLES`` and ``pool_pre_ping`` so a
  recycled connection never surfaces as a request error.

Everything else in the platform obtains its :class:`~sqlalchemy.orm.Session` from
:func:`session_scope` or the FastAPI dependency :func:`get_session`; repositories are the only
place that builds queries (docs/mimari/veri-modeli.md §1 rule 8).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from aicls.core.errors import InputValidationError
from aicls.core.logging import get_logger
from aicls.core.messages_tr import msg
from aicls.core.settings import get_settings

#: SQLite busy timeout in milliseconds (WAL still serialises writers).
SQLITE_BUSY_TIMEOUT_MS = 5000
#: Seconds after which a pooled MySQL connection is recycled.
MYSQL_POOL_RECYCLE_S = 3600

_engines: dict[str, Engine] = {}
_default_engine: Engine | None = None


def _is_memory_sqlite(url: URL) -> bool:
    """True when ``url`` addresses an in-memory SQLite database."""
    database = url.database or ""
    return database in ("", ":memory:") or "mode=memory" in database


def _install_sqlite_pragmas(engine: Engine, *, memory: bool) -> None:
    """Enable foreign keys on every connection and WAL for file databases."""

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")  # raw-sql-ok: SQLite pragma, not a query
            cursor.execute(
                f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}"  # raw-sql-ok: SQLite pragma
            )
            if not memory:
                cursor.execute("PRAGMA journal_mode=WAL")  # raw-sql-ok: SQLite pragma
                cursor.execute("PRAGMA synchronous=NORMAL")  # raw-sql-ok: SQLite pragma
        finally:
            cursor.close()


def engine_options(url: str, *, echo: bool = False) -> tuple[URL, dict[str, Any]]:
    """Return the parsed URL and the ``create_engine`` keyword arguments AI-CLS uses for it.

    Separated from :func:`make_engine` so the backend-specific settings (utf8mb4, strict SQL
    mode, ``StaticPool`` for ``:memory:``) can be asserted without opening a connection.

    Raises:
        InputValidationError: If ``url`` is empty or names an unsupported backend.
    """
    if not url or not url.strip():
        raise InputValidationError(msg("VALIDATION_ERROR"), details={"field": "db_url"})
    parsed = make_url(url)
    backend = parsed.get_backend_name()
    kwargs: dict[str, Any] = {"echo": echo, "future": True}
    if backend == "sqlite":
        kwargs["connect_args"] = {"check_same_thread": False}
        if _is_memory_sqlite(parsed):
            kwargs["poolclass"] = StaticPool
        return parsed, kwargs
    if backend == "mysql":
        kwargs["connect_args"] = {
            "charset": "utf8mb4",
            "sql_mode": "STRICT_TRANS_TABLES",
            "use_unicode": True,
        }
        kwargs["pool_recycle"] = MYSQL_POOL_RECYCLE_S
        kwargs["pool_pre_ping"] = True
        return parsed, kwargs
    raise InputValidationError(
        msg("VALIDATION_ERROR"),
        details={"field": "db_url", "backend": backend, "allowed": ["sqlite", "mysql"]},
    )


def make_engine(url: str, *, echo: bool = False) -> Engine:
    """Create a configured :class:`~sqlalchemy.Engine` for ``url``.

    Args:
        url: SQLAlchemy URL, e.g. ``"sqlite+pysqlite:///:memory:"`` or
            ``"mysql+pymysql://aicls_app:...@127.0.0.1/aicls?charset=utf8mb4"``.
        echo: Log every emitted statement (development only).

    Returns:
        A new engine; the caller owns its lifetime. Use :func:`get_engine` for the shared one.

    Raises:
        InputValidationError: If ``url`` is empty or names an unsupported backend.
    """
    parsed, kwargs = engine_options(url, echo=echo)
    engine = create_engine(parsed, **kwargs)
    if parsed.get_backend_name() == "sqlite":
        _install_sqlite_pragmas(engine, memory=_is_memory_sqlite(parsed))
    return engine


def get_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    """Return the process-wide engine for ``url`` (defaults to ``settings.db_url``).

    Engines are cached per URL so the API and the CLI share one connection pool. A test may
    install its own with :func:`set_default_engine`.
    """
    if url is None and _default_engine is not None:
        return _default_engine
    resolved = url if url is not None else get_settings().db_url
    engine = _engines.get(resolved)
    if engine is None:
        engine = make_engine(resolved, echo=echo)
        _engines[resolved] = engine
        get_logger(__name__).debug("engine_created", backend=make_url(resolved).get_backend_name())
    return engine


def set_default_engine(engine: Engine | None) -> None:
    """Install (or clear with ``None``) the engine returned by :func:`get_engine` and
    :func:`get_session`.

    The API lifespan and the test fixtures use this to point the whole process at one
    ``:memory:`` database.
    """
    global _default_engine  # noqa: PLW0603 - module-level singleton by design
    _default_engine = engine


def dispose_engines() -> None:
    """Dispose and forget every cached engine (shutdown and test teardown)."""
    global _default_engine  # noqa: PLW0603 - module-level singleton by design
    for engine in _engines.values():
        engine.dispose()
    _engines.clear()
    _default_engine = None


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """Yield a :class:`~sqlalchemy.orm.Session` that commits on success and rolls back on error.

    Args:
        engine: Engine to bind; defaults to :func:`get_engine`.

    Yields:
        A session with ``expire_on_commit=False`` so returned ORM objects stay readable after
        the block exits.
    """
    with Session(engine if engine is not None else get_engine(), expire_on_commit=False) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session (``Depends(get_session)``)."""
    with session_scope() as session:
        yield session


__all__ = [
    "MYSQL_POOL_RECYCLE_S",
    "SQLITE_BUSY_TIMEOUT_MS",
    "dispose_engines",
    "engine_options",
    "get_engine",
    "get_session",
    "make_engine",
    "session_scope",
    "set_default_engine",
]
