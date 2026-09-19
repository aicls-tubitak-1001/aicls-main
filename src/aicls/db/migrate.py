"""Programmatic Alembic helpers (``aicls db upgrade``, tests, API startup).

Migrations — not ``metadata.create_all`` — are the only way a schema is created, in every
environment including the in-memory test database, so every pull request exercises them
(ADR-0003). :func:`upgrade_head` therefore accepts an open :class:`~sqlalchemy.Engine` or
:class:`~sqlalchemy.Connection` as well as a URL: a shared ``sqlite:///:memory:`` engine holds
the whole database in one connection, and building a second engine from the URL would migrate a
different, empty database.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, Engine

from aicls.db.base import Base
from aicls.db.models.book import FULLTEXT_INDEX_NAME

#: Directory holding ``env.py`` and ``versions/``.
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
#: Revision identifier of the initial schema.
INITIAL_REVISION = "0001"


def alembic_config(target: str | Engine | Connection | None = None) -> Config:
    """Build an Alembic :class:`~alembic.config.Config` bound to the packaged migrations.

    Args:
        target: A database URL, an :class:`~sqlalchemy.Engine` or an open
            :class:`~sqlalchemy.Connection`. ``None`` leaves the URL to ``env.py`` (which falls
            back to ``AICLS_DB_URL``).

    Returns:
        A configuration whose ``script_location`` is this package's ``migrations`` directory, so
        it works regardless of the current working directory.
    """
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.set_main_option("version_path_separator", "os")
    if isinstance(target, str):
        config.attributes["url"] = target
    elif isinstance(target, Connection | Engine):
        config.attributes["connection"] = target
    return config


def script_directory() -> ScriptDirectory:
    """Return the :class:`~alembic.script.ScriptDirectory` of the packaged migrations."""
    return ScriptDirectory.from_config(alembic_config())


def head_revision() -> str | None:
    """Return the identifier of the newest revision on disk."""
    return script_directory().get_current_head()


def upgrade_head(url: str | Engine | Connection) -> None:
    """Upgrade the target database to ``head``.

    Args:
        url: Database URL, engine or open connection (see the module docstring).
    """
    command.upgrade(alembic_config(url), "head")


def downgrade(url: str | Engine | Connection, revision: str = "base") -> None:
    """Downgrade the target database to ``revision`` (``"base"`` drops every table)."""
    command.downgrade(alembic_config(url), revision)


def current_revision(target: Engine | Connection) -> str | None:
    """Return the revision stamped in ``alembic_version``, or ``None`` when unstamped."""
    if isinstance(target, Engine):
        with target.connect() as connection:
            return MigrationContext.configure(connection).get_current_revision()
    return MigrationContext.configure(target).get_current_revision()


def diff_metadata(target: Engine | Connection) -> list[Any]:
    """Return Alembic's autogenerate diff between the database and the ORM metadata.

    An empty list means every model change has a migration — asserted by
    ``test_autogenerate_diff_is_empty``.
    """
    if isinstance(target, Engine):
        with target.connect() as connection:
            return diff_metadata(connection)
    context = MigrationContext.configure(
        target,
        opts={
            "compare_type": True,
            "compare_server_default": True,
            "render_as_batch": target.dialect.name == "sqlite",
            "include_object": _include_in_diff,
        },
    )
    return list(compare_metadata(context, Base.metadata))


def _include_in_diff(
    obj: Any, name: str | None, type_: str, reflected: bool, compare_to: Any
) -> bool:
    """Alembic ``include_object`` hook: everything except the MySQL-only FULLTEXT index.

    The index is created by migration 0001 under a dialect guard and is deliberately not
    declared on the model (``aicls.db.models.book``), so against a real MySQL the reflected
    schema carries one index the metadata lacks and autogenerate reports ``remove_index``.
    That is the design, not drift. SQLite never has the index, which is why the local test
    passed and only the nightly MySQL job saw it (2026-09-10).
    """
    return not (type_ == "index" and reflected and name == FULLTEXT_INDEX_NAME)


def offline_sql(url: str, *, revision: str = "head") -> str:
    """Render every migration as SQL for ``url``'s dialect without connecting.

    This is how ``test_alembic_upgrade_head_on_sqlite_and_mysql_dialect_compile`` proves that all
    migrations compile under MySQL on a machine that has no MySQL server.

    Args:
        url: A database URL; only its dialect matters.
        revision: Target revision, ``"head"`` by default.

    Returns:
        The generated DDL script.
    """
    from contextlib import redirect_stdout
    import io

    config = alembic_config(url)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        command.upgrade(config, revision, sql=True)
    return buffer.getvalue()


__all__ = [
    "INITIAL_REVISION",
    "MIGRATIONS_DIR",
    "alembic_config",
    "current_revision",
    "diff_metadata",
    "downgrade",
    "head_revision",
    "offline_sql",
    "script_directory",
    "upgrade_head",
]
