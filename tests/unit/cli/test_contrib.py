"""``aicls contrib``: manifest export, KVKK withdrawal and the İP-3 dashboard figures.

The corpus itself is never in the repository (docs/ml/veri-yonetimi.md), so these tests drive an
empty, freshly migrated database: what they assert is the shape of the contract — a manifest with
its split file and fingerprint, a withdrawal that keeps the consent record, and statistics that
report zero rather than crash.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.unit.cli.conftest import payload_of

pytestmark = pytest.mark.usefixtures("cli_env")


def test_contrib_stats_on_an_empty_corpus(
    runner: CliRunner, app_module: object, db_url: str
) -> None:
    """The dashboard reports zeros on an empty database instead of failing."""
    result = runner.invoke(app_module, ["contrib", "stats", "--db-url", db_url, "--json"])
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["utterances"] == 0
    assert payload["speakers"] == 0
    assert payload["hours_accepted"] == 0.0


def test_contrib_export_manifest_writes_the_split_file(
    runner: CliRunner, app_module: object, tmp_path: Path, db_url: str
) -> None:
    """The export always refreshes the speaker-disjoint split file beside the manifest."""
    out = tmp_path / "manifests" / "audio_commands.jsonl"
    result = runner.invoke(
        app_module,
        ["contrib", "export-manifest", "--out", str(out), "--db-url", db_url, "--json"],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["rows"] == 0
    assert Path(payload["path"]) == out
    assert Path(payload["splits_path"]).name == "audio_commands_splits.json"
    assert payload["fingerprint"].startswith("sha256:")
    assert out.is_file()


def test_contrib_export_manifest_refuses_a_non_jsonl_target(
    runner: CliRunner, app_module: object, tmp_path: Path, db_url: str
) -> None:
    """The trainer reads JSONL; a ``.csv`` target is a typo worth failing on."""
    result = runner.invoke(
        app_module,
        [
            "contrib",
            "export-manifest",
            "--out",
            str(tmp_path / "audio.csv"),
            "--db-url",
            db_url,
        ],
    )
    assert result.exit_code == 1
    assert "VALIDATION_ERROR" in result.stderr


def test_contrib_withdraw_unknown_speaker_is_not_found(
    runner: CliRunner, app_module: object, db_url: str
) -> None:
    """A withdrawal for a speaker who does not exist must not report a silent success."""
    result = runner.invoke(
        app_module, ["contrib", "withdraw", "--speaker", "P00012", "--db-url", db_url]
    )
    assert result.exit_code == 1
    assert "NOT_FOUND" in result.stderr


def test_contrib_withdraw_keeps_the_consent_record(
    runner: CliRunner, app_module: object, db_url: str, data_root: Path
) -> None:
    """KVKK: the audio goes, the proof that consent was given and withdrawn stays."""
    from aicls.collect import Demographics, accept_consent, current_consent
    from aicls.db import RecordingRepository, make_engine, session_scope, upgrade_head

    engine = make_engine(db_url)
    try:
        upgrade_head(engine)
        with session_scope(engine) as session:
            repo = RecordingRepository(session)
            consent = current_consent()
            repo.upsert_consent_version(
                consent.version,
                consent.text_tr,
                sha256=consent.sha256,
                effective_from=consent.effective_from,
                ethics_approval_ref="TEST",
            )
            speaker = accept_consent(
                repo,
                user_id=None,
                version=consent.version,
                demographics=Demographics(age_band="18-24", gender="f", university="mus"),
            )
            pseudonym = speaker.pseudonym
    finally:
        engine.dispose()

    result = runner.invoke(
        app_module,
        [
            "contrib",
            "withdraw",
            "--speaker",
            pseudonym,
            "--data-root",
            str(data_root),
            "--db-url",
            db_url,
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["speaker_pseudonym"] == pseudonym
    assert payload["consent_kept"] is True
    assert payload["recordings_deleted"] == 0
