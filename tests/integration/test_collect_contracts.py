"""Cross-module contracts of ``aicls.collect`` — the boundaries, not each side's fixtures.

Wave 1 lost a nested field between the web Canvas payload and ``aicls.ocr`` because both sides
only ever tested their own fixtures (KNOWN_ISSUES.md §"Cross-module contract testing"). The
collection flow sits on three such boundaries, and each one is asserted here against the *real*
neighbouring implementation:

``aicls.collect`` → ``aicls.asr``
    ``read_wav_pcm16`` → ``run_qc`` → ``write_wav_pcm16`` must round-trip, and the quality report
    must keep the field names the API response schema promises.

``aicls.collect`` → ``aicls.db``
    ``RecordingRepository.add`` lifts ``passed``/``snr_db``/``clipping_ratio``/``silence_ratio``
    from the *top level* of the quality document into columns, while the API wants them nested
    under ``checks``. Every string written must also fit its column.

``aicls.collect`` → ``aicls.api`` / ``aicls.data``
    The payload shapes of ``GET /contribute/consent/current``, ``POST /contribute/recordings``
    and the training manifest, which downstream packages code against.
"""

from __future__ import annotations

from collections.abc import Iterator
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from aicls.asr import QcThresholds, make_speech_like, read_wav_pcm16, run_qc, write_wav_pcm16
from aicls.collect import (
    MANIFEST_FIELDS,
    SPLIT_NAMES,
    Demographics,
    accept_consent,
    accept_recording,
    current_consent,
    export_manifest,
    next_prompts,
    withdraw,
)
from aicls.collect.service import MAX_ETHICS_REF_LENGTH, qc_document
from aicls.core.settings import Settings, get_settings
from aicls.db import (
    Asset,
    ConsentVersion,
    PromptSentence,
    Recording,
    RecordingRepository,
    Speaker,
    make_engine,
    session_scope,
    upgrade_head,
)

pytestmark = pytest.mark.integration

MEMORY_URL = "sqlite+pysqlite:///:memory:"

PROMPT_ROWS = [
    {"text_tr": "x karenin türevini al", "scenario_tag": "turev", "intent": "derivative"},
    {"text_tr": "x karenin integralini al", "scenario_tag": "integral", "intent": "integral"},
]


@pytest.fixture
def collect_settings(test_env: dict[str, str], data_root: Path) -> Iterator[Settings]:
    """``Settings`` built from the hermetic test environment."""
    get_settings.cache_clear()
    try:
        yield get_settings()
    finally:
        get_settings.cache_clear()


@pytest.fixture
def session(collect_settings: Settings) -> Iterator[Session]:
    """A session on a migrated in-memory database."""
    engine: Engine = make_engine(MEMORY_URL)
    upgrade_head(engine)
    try:
        with session_scope(engine) as db_session:
            yield db_session
    finally:
        engine.dispose()


@pytest.fixture
def repo(session: Session) -> RecordingRepository:
    return RecordingRepository(session)


@pytest.fixture
def prompts(repo: RecordingRepository) -> list[PromptSentence]:
    repo.add_prompts(PROMPT_ROWS)
    return repo.next_prompts("nobody", 10)


@pytest.fixture
def speaker(repo: RecordingRepository) -> Speaker:
    return accept_consent(
        repo,
        user_id=None,
        version=current_consent().version,
        demographics=Demographics(age_band="18-24", gender="f", university="mus"),
        ip_hash=None,
    )


class TestAsrBoundary:
    """What ``aicls.asr`` hands over is exactly what the collection flow stores."""

    def test_wav_round_trip_survives_the_store(
        self, repo: RecordingRepository, speaker: Speaker, prompts: list[PromptSentence]
    ) -> None:
        original = make_speech_like(duration_s=2.0)
        decision = accept_recording(
            repo,
            speaker_id=speaker.id,
            prompt_id=prompts[0].id,
            wav=write_wav_pcm16(original),
        )
        assert decision.accepted is True
        assert decision.path is not None
        restored = read_wav_pcm16(decision.path.read_bytes())
        assert restored.sr == original.sr == 16000
        assert restored.n_samples == original.n_samples
        assert decision.duration_s == pytest.approx(original.duration_s)

    def test_quality_document_satisfies_both_consumers(self) -> None:
        report = run_qc(make_speech_like(duration_s=2.0), QcThresholds())
        document = qc_document(report)

        # (a) the API response contract (docs/mimari/api.md §3.8)
        assert set(report.as_dict()) == {"passed", "checks", "reasons_tr"}
        assert document["checks"] == report.as_dict()["checks"]
        assert document["reasons_tr"] == []

        # (b) the columns RecordingRepository.add lifts out of the top level
        for column in ("passed", "snr_db", "clipping_ratio", "silence_ratio"):
            assert column in document, column
        assert document["snr_db"] == report.value("snr_db")

    def test_measurements_reach_their_columns(
        self, repo: RecordingRepository, speaker: Speaker, prompts: list[PromptSentence]
    ) -> None:
        decision = accept_recording(
            repo,
            speaker_id=speaker.id,
            prompt_id=prompts[0].id,
            wav=write_wav_pcm16(make_speech_like(duration_s=2.0)),
        )
        row = repo.session.get(Recording, decision.recording_id)
        assert row is not None
        assert row.snr_db == pytest.approx(decision.qc.value("snr_db"))
        assert row.silence_ratio == pytest.approx(decision.qc.value("silence_ratio"))
        assert row.clipping_ratio == pytest.approx(decision.qc.value("clipping_ratio"))
        assert row.snr_db > 0.0  # a stored recording always has a measured SNR


class TestDbBoundary:
    """Everything written fits its column, and the repository sees the shapes it expects."""

    def test_written_strings_fit_their_columns(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
        collect_settings: Settings,
    ) -> None:
        consent = current_consent()
        assert len(consent.version) <= ConsentVersion.__table__.c.version.type.length
        assert len(consent.sha256) <= ConsentVersion.__table__.c.sha256.type.length
        assert (
            len(collect_settings.ethics_approval_ref)
            <= ConsentVersion.__table__.c.ethics_approval_ref.type.length
            == MAX_ETHICS_REF_LENGTH
        )

        decision = accept_recording(
            repo,
            speaker_id=speaker.id,
            prompt_id=prompts[0].id,
            wav=write_wav_pcm16(make_speech_like(duration_s=2.0)),
        )
        assert decision.rel_path is not None
        assert len(decision.rel_path) <= Asset.__table__.c.path.type.length
        assert len(speaker.pseudonym) == Speaker.__table__.c.pseudonym.type.length
        assert len(decision.sha256) <= Recording.__table__.c.sha256.type.length

    def test_repository_view_matches_the_collect_policy(
        self, repo: RecordingRepository, speaker: Speaker, prompts: list[PromptSentence]
    ) -> None:
        for prompt in prompts:
            accept_recording(
                repo,
                speaker_id=speaker.id,
                prompt_id=prompt.id,
                wav=write_wav_pcm16(make_speech_like(duration_s=2.0)),
            )
        assert len(repo.exportable()) == 2
        stats = repo.stats()
        assert stats.utterances == 2
        assert stats.speakers == 1
        assert stats.qc_pass_rate == 1.0
        assert stats.by_university == {"mus": 2}

        withdraw(repo, speaker.pseudonym)
        assert repo.exportable() == []
        assert repo.stats().utterances == 0

    def test_prompt_balancing_uses_the_repository_ordering(
        self, repo: RecordingRepository, speaker: Speaker, prompts: list[PromptSentence]
    ) -> None:
        suggested = next_prompts(repo, speaker.pseudonym, n=2)
        assert [p.id for p in suggested] == [p.id for p in repo.next_prompts(speaker.id, 2)]
        assert {p.scenario_tag for p in suggested} == {"turev", "integral"}


class TestApiPayloadContracts:
    """The dictionaries downstream packages deserialise."""

    def test_consent_payload(self) -> None:
        payload = current_consent().as_dict()
        assert set(payload) == {"version", "text_tr", "sha256", "effective_from"}
        assert json.loads(json.dumps(payload, ensure_ascii=False))["sha256"]

    def test_recording_payload(
        self, repo: RecordingRepository, speaker: Speaker, prompts: list[PromptSentence]
    ) -> None:
        payload = accept_recording(
            repo,
            speaker_id=speaker.id,
            prompt_id=prompts[0].id,
            wav=write_wav_pcm16(make_speech_like(duration_s=2.0)),
        ).as_dict()
        assert set(payload) == {"id", "accepted", "qc", "duration_s"}
        assert set(payload["qc"]) == {"passed", "checks", "reasons_tr"}
        assert set(payload["qc"]["checks"][0]) == {"name", "value", "threshold", "passed"}
        assert json.loads(json.dumps(payload, ensure_ascii=False))["accepted"] is True

    def test_manifest_row_contract(
        self,
        repo: RecordingRepository,
        speaker: Speaker,
        prompts: list[PromptSentence],
        data_root: Path,
        tmp_path: Path,
    ) -> None:
        accept_recording(
            repo,
            speaker_id=speaker.id,
            prompt_id=prompts[0].id,
            wav=write_wav_pcm16(make_speech_like(duration_s=2.0)),
        )
        out = tmp_path / "manifest.jsonl"
        report = export_manifest(repo, out=out, data_root=data_root)
        (row,) = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

        assert list(row) == list(MANIFEST_FIELDS)  # order is part of the contract
        assert (data_root / str(row["path"])).is_file()
        assert row["split"] in SPLIT_NAMES
        assert row["text"] == prompts[0].text_tr
        assert row["consent_version"] == current_consent().version
        assert report.fingerprint.startswith("sha256:")

        splits = json.loads(report.splits_path.read_text(encoding="utf-8"))
        assert set(splits) == {"version", "seed", "ratios", "updated_at", "speakers"}
        assert splits["speakers"] == {speaker.pseudonym: row["split"]}


class TestLayering:
    """ADR-0002: the collection flow stays torch-free even though it uses ``aicls.asr``."""

    def test_importing_collect_does_not_load_torch(self, repo_root: Path) -> None:
        # aicls.asr keeps every torch/transformers import inside a function, so importing the
        # collection flow must not pull the ML stack into the API process. import-linter cannot
        # see that on its own (it walks the static graph), which is why this runtime check
        # exists next to the "allow_indirect_imports" note in docs/mimari/moduller.md.
        environment = dict(os.environ, PYTHONPATH=str(repo_root / "src"))
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, aicls.collect;"
                "print(sorted(m for m in sys.modules if m in {'torch', 'transformers'}))",
            ],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "[]"
