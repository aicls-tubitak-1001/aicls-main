"""Fixtures for the database integration tests.

The schema is always created by ``alembic upgrade head`` — never ``metadata.create_all`` — so
every test run exercises the migrations (ADR-0003, docs/gelistirme/test-stratejisi.md §4).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from aicls.db import (
    ConsentVersion,
    PromptSentence,
    Speaker,
    User,
    Workspace,
    make_engine,
    session_scope,
    upgrade_head,
)

MEMORY_URL = "sqlite+pysqlite:///:memory:"


@pytest.fixture
def engine() -> Iterator[Engine]:
    """A migrated in-memory SQLite engine (``StaticPool``: one shared connection)."""
    created = make_engine(MEMORY_URL)
    upgrade_head(created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """A session on the migrated in-memory database, committed at the end of the test."""
    with session_scope(engine) as db_session:
        yield db_session


@pytest.fixture
def user(session: Session) -> User:
    """A persisted student user."""
    from aicls.db import UserRepository

    return UserRepository(session).create("ada@example.edu.tr", "argon2id$hash", "Ada Lovelace")


@pytest.fixture
def workspace(session: Session, user: User) -> Workspace:
    """A workspace owned by :func:`user`."""
    from aicls.db import WorkspaceRepository

    return WorkspaceRepository(session).create(user.id, "Türev çalışması")


@pytest.fixture
def consent(session: Session) -> ConsentVersion:
    """A stored consent version (``v1``)."""
    from aicls.db import RecordingRepository

    return RecordingRepository(session).upsert_consent_version(
        "v1", "Katılım onamı metni.", sha256="0" * 64, ethics_approval_ref="TEST"
    )


@pytest.fixture
def speaker(session: Session, consent: ConsentVersion) -> Speaker:
    """A pseudonymous speaker who accepted ``consent``."""
    from aicls.db import RecordingRepository

    return RecordingRepository(session).create_speaker(
        consent.id, age_band="18-24", gender="f", university="mus"
    )


@pytest.fixture
def prompts(session: Session) -> list[PromptSentence]:
    """Three active prompt sentences."""
    from aicls.db import RecordingRepository

    repo = RecordingRepository(session)
    repo.add_prompts(
        [
            {"text_tr": "x karenin türevini al", "scenario_tag": "turev", "intent": "derivative"},
            {"text_tr": "limit x sıfıra giderken sinüs x bölü x", "scenario_tag": "limit"},
            {"text_tr": "türev tanımını getir", "scenario_tag": "kitap"},
        ]
    )
    return repo.next_prompts("nobody", 10)
