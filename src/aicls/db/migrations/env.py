"""Alembic environment for AI-CLS (ADR-0003).

Three ways of running are supported:

* **online, own engine** — the URL comes from ``config.attributes["url"]``, the
  ``sqlalchemy.url`` option or ``AICLS_DB_URL`` via :mod:`aicls.core.settings`;
* **online, caller's connection** — :func:`aicls.db.migrate.upgrade_head` passes an open
  ``Connection`` (or ``Engine``) in ``config.attributes["connection"]``, which is what makes
  migrations work against a shared ``sqlite:///:memory:`` test database;
* **offline** (``--sql``) — DDL is rendered for a dialect without connecting, which is how the
  MySQL compile test checks every migration without a MySQL server.

``render_as_batch`` is enabled for SQLite so that ``ALTER``-poor SQLite tables are recreated;
the naming convention lives on :attr:`aicls.db.base.Base.metadata`.
"""

from __future__ import annotations

from typing import Any

from alembic import context
from sqlalchemy import Connection, Engine, engine_from_config, pool

from aicls.db import models as _models  # noqa: F401 - imported for metadata side effects
from aicls.db.base import Base

config = context.config
target_metadata = Base.metadata


def _resolve_url() -> str:
    """Return the database URL: config attribute, ini option, then settings."""
    url = config.attributes.get("url")
    if url:
        return str(url)
    ini_url = config.get_main_option("sqlalchemy.url", None)
    if ini_url:
        return str(ini_url)
    from aicls.core.settings import get_settings

    return get_settings().db_url


def _configure_kwargs(dialect_name: str) -> dict[str, Any]:
    """Shared ``context.configure`` options for both modes."""
    return {
        "target_metadata": target_metadata,
        "render_as_batch": dialect_name == "sqlite",
        "compare_type": True,
        "compare_server_default": True,
        "include_schemas": False,
    }


def run_migrations_offline() -> None:
    """Render migrations as SQL for the URL's dialect without connecting."""
    url = _resolve_url()
    dialect = url.split(":", 1)[0].split("+", 1)[0]
    context.configure(
        url=url,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        **_configure_kwargs(dialect),
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_with_connection(connection: Connection) -> None:
    """Configure the migration context on an open connection and run it."""
    context.configure(connection=connection, **_configure_kwargs(connection.dialect.name))
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection (supplied by the caller or built here)."""
    supplied = config.attributes.get("connection")
    if isinstance(supplied, Connection):
        _run_with_connection(supplied)
        return
    if isinstance(supplied, Engine):
        with supplied.connect() as connection:
            _run_with_connection(connection)
            connection.commit()
        return
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _resolve_url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        _run_with_connection(connection)
        connection.commit()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
