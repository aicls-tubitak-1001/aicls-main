"""Fixtures for the ``aicls.collect`` unit tests.

The collection flow is domain logic on top of a real ``RecordingRepository``, so these tests use
a migrated in-memory SQLite database (``alembic upgrade head``, never ``metadata.create_all`` —
ADR-0003) and a temporary ``AICLS_DATA_ROOT`` outside OneDrive. Audio bytes are always synthetic
(``aicls.asr.synth``); no recording is ever committed to the repository (KVKK).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from aicls.asr import Audio, make_speech_like, make_tone, write_wav_pcm16
from aicls.collect import Demographics, accept_consent, current_consent
from aicls.core.settings import Settings, get_settings
from aicls.db import (
    PromptSentence,
    RecordingRepository,
    Speaker,
    make_engine,
    session_scope,
    upgrade_head,
)

MEMORY_URL = "sqlite+pysqlite:///:memory:"

#: Prompt sentences used by the tests (reading scenarios of İP-3).
PROMPT_ROWS: list[dict[str, object]] = [
    {"text_tr": "x karenin türevini al", "scenario_tag": "turev", "intent": "derivative"},
    {"text_tr": "iki x üzeri üçün türevini al", "scenario_tag": "turev", "intent": "derivative"},
    {"text_tr": "x karenin integralini al", "scenario_tag": "integral", "intent": "integral"},
    {"text_tr": "limit x sıfıra giderken sinüs x bölü x", "scenario_tag": "limit"},
    {"text_tr": "türev tanımını getir", "scenario_tag": "kitap", "intent": "book_fetch"},
]


def good_wav() -> bytes:
    """PCM16 WAV bytes of an utterance-shaped signal that passes quality control."""
    return write_wav_pcm16(make_speech_like(duration_s=2.0))


def bad_wav() -> bytes:
    """PCM16 WAV bytes of a steady tone: no noise floor, so the SNR check rejects it."""
    return write_wav_pcm16(make_tone(duration_s=2.0, noise_amplitude=0.0))


def short_wav() -> bytes:
    """PCM16 WAV bytes shorter than the minimum accepted duration."""
    return write_wav_pcm16(make_speech_like(duration_s=0.4, burst_s=0.1, gap_s=0.05))


def silent_wav(duration_s: float = 2.0) -> bytes:
    """PCM16 WAV bytes of digital silence (fails the silence-ratio check)."""
    return write_wav_pcm16(Audio(np.zeros(int(duration_s * 16000), dtype=np.float32), 16000))


@pytest.fixture
def collect_settings(test_env: dict[str, str], data_root: Path) -> Iterator[Settings]:
    """``Settings`` built from the hermetic test environment, with the cache reset around it."""
    get_settings.cache_clear()
    try:
        yield get_settings()
    finally:
        get_settings.cache_clear()


@pytest.fixture
def engine(collect_settings: Settings) -> Iterator[Engine]:
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
def repo(session: Session) -> RecordingRepository:
    """The voice data-collection repository under test."""
    return RecordingRepository(session)


@pytest.fixture
def prompts(repo: RecordingRepository) -> list[PromptSentence]:
    """Five active prompt sentences spread over four reading scenarios."""
    repo.add_prompts(PROMPT_ROWS)
    return repo.next_prompts("nobody", 50)


@pytest.fixture
def demographics() -> Demographics:
    """A minimal band-only demographics record."""
    return Demographics(age_band="18-24", gender="f", university="mus")


@pytest.fixture
def speaker(repo: RecordingRepository, demographics: Demographics) -> Speaker:
    """A consented, pseudonymous speaker (``P00001``)."""
    return accept_consent(
        repo,
        user_id=None,
        version=current_consent().version,
        demographics=demographics,
        ip_hash="a" * 64,
    )
