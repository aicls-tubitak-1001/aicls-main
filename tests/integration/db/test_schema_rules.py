"""Dialect-neutrality rules of docs/mimari/veri-modeli.md §1, enforced mechanically.

These are the guards that keep the SQLite development database and the MySQL production database
from drifting apart (ADR-0003): an unbounded ``String`` would compile to ``TEXT`` and could not
be indexed on MySQL, a native enum or a server-side default would behave differently on each
backend, and an unnamed constraint would break Alembic's batch recreation of SQLite tables.
"""

from __future__ import annotations

import pytest
from sqlalchemy import (
    CHAR,
    JSON,
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    Integer,
    SmallInteger,
    String,
    Text,
)
from sqlalchemy.schema import CreateIndex, CreateTable
from sqlalchemy.sql.schema import Table

from aicls.db import TABLE_NAMES, Base
from aicls.db.base import ULID_LENGTH
from aicls.db.enums import ENUM_LENGTH

pytestmark = pytest.mark.integration

ALLOWED_TYPES = (
    String,
    Text,
    Integer,
    SmallInteger,
    Float,
    Boolean,
    DateTime,
    Date,
    JSON,
    Enum,
)
#: The single unit-identifier primary key that is not a ULID (``book_units.id``).
NON_ULID_PK = {"book_units"}
#: Maximum bytes MySQL allows in one index (utf8mb4 = 4 bytes per character).
MYSQL_INDEX_BYTE_LIMIT = 3072


def _tables() -> list[Table]:
    return list(Base.metadata.sorted_tables)


def test_schema_rules_all_strings_have_length() -> None:
    """Every ``String`` column declares a length (rule 2); ``Text`` columns are never indexed."""
    offenders = [
        f"{table.name}.{column.name}"
        for table in _tables()
        for column in table.columns
        if isinstance(column.type, String)
        and not isinstance(column.type, Text)
        and not column.type.length
    ]
    assert offenders == []

    indexed_text = [
        f"{table.name}.{column.name}"
        for table in _tables()
        for index in table.indexes
        for column in index.columns
        if isinstance(column.type, Text)
    ]
    assert indexed_text == []


def test_schema_rules_every_table_is_declared_once() -> None:
    """The metadata holds exactly the tables of docs/mimari/veri-modeli.md §3."""
    assert {table.name for table in _tables()} == set(TABLE_NAMES)


def test_schema_rules_primary_keys_are_ulid_char26() -> None:
    """Every primary key is a single ``CHAR(26)`` ULID string; no autoincrement integers."""
    for table in _tables():
        pk_columns = list(table.primary_key.columns)
        assert len(pk_columns) == 1, table.name
        column = pk_columns[0]
        assert not isinstance(column.type, Integer), table.name
        if table.name in NON_ULID_PK:
            assert column.type.length == 64
            continue
        assert isinstance(column.type, CHAR), table.name
        assert column.type.length == ULID_LENGTH, table.name


def test_schema_rules_enums_are_not_native_and_are_checked() -> None:
    """Enums compile to ``VARCHAR(32)`` with a named CHECK constraint (rule 3)."""
    for table in _tables():
        for column in table.columns:
            if isinstance(column.type, Enum):
                assert column.type.native_enum is False, f"{table.name}.{column.name}"
                assert column.type.create_constraint is True, f"{table.name}.{column.name}"
                assert column.type.length == ENUM_LENGTH, f"{table.name}.{column.name}"


def test_schema_rules_datetimes_are_naive_utc_without_server_defaults() -> None:
    """No timezone-aware column and no server-side default or ``ON UPDATE`` (rules 5 and 8)."""
    for table in _tables():
        for column in table.columns:
            if isinstance(column.type, DateTime):
                assert column.type.timezone is False, f"{table.name}.{column.name}"
            assert column.server_default is None, f"{table.name}.{column.name}"
            assert column.server_onupdate is None, f"{table.name}.{column.name}"


def test_schema_rules_only_portable_column_types_are_used() -> None:
    """Every column uses one of the ten portable types (rule 4: JSON via ``sa.JSON``)."""
    for table in _tables():
        for column in table.columns:
            assert isinstance(column.type, ALLOWED_TYPES), (
                f"{table.name}.{column.name}: {column.type!r}"
            )


def test_schema_rules_constraints_and_indexes_follow_the_naming_convention() -> None:
    """Every constraint and index carries a convention-shaped, MySQL-legal name."""
    prefixes = ("pk_", "uq_", "ck_", "fk_", "ix_")
    for table in _tables():
        for constraint in table.constraints:
            assert constraint.name is not None, table.name
            name = str(constraint.name)
            assert name.startswith(prefixes), name
            assert len(name) <= 64, name
        for index in table.indexes:
            assert index.name is not None and str(index.name).startswith("ix_")
            assert len(str(index.name)) <= 64, index.name


def test_schema_rules_indexed_string_columns_fit_the_mysql_key_limit() -> None:
    """No index exceeds MySQL's 3072-byte key limit with 4-byte utf8mb4 characters."""
    for table in _tables():
        for index in table.indexes:
            width = sum(
                (column.type.length or 0) * 4
                for column in index.columns
                if isinstance(column.type, String)
            )
            assert width <= MYSQL_INDEX_BYTE_LIMIT, f"{table.name}.{index.name}: {width} bytes"


def test_schema_rules_every_table_compiles_for_both_dialects() -> None:
    """Each ``CREATE TABLE``/``CREATE INDEX`` compiles under SQLite and MySQL."""
    from sqlalchemy.dialects import mysql, sqlite

    for table in _tables():
        for dialect in (sqlite.dialect(), mysql.dialect()):
            ddl = str(CreateTable(table).compile(dialect=dialect))
            assert ddl.startswith("\nCREATE TABLE") or ddl.startswith("CREATE TABLE")
            for index in table.indexes:
                assert "CREATE INDEX" in str(CreateIndex(index).compile(dialect=dialect))


def test_schema_rules_fulltext_index_is_not_in_the_metadata() -> None:
    """The MySQL FULLTEXT index lives only in the migration, so SQLite autogenerate is clean."""
    book_units = Base.metadata.tables["book_units"]
    assert all("fulltext" not in str(index.name).lower() for index in book_units.indexes)
    assert all("ftx_" not in str(index.name) for index in book_units.indexes)
