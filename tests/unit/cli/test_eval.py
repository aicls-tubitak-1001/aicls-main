"""``aicls eval``: the only place where :mod:`aicls.eval` and :mod:`aicls.db` meet.

The contract asserted here is provenance, not accuracy. A number that reaches ``eval_runs``
without the commit it was measured at, the fingerprint of the corpus it was measured on and the
report's own ``citable`` verdict is useless for a TÜBİTAK progress report, so every write is
checked for all three.

The İP-3 numbers in these runs come from **stub** backends (there is no trained checkpoint and
every download and training gate is closed); the reports say so, and so do the assertions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.unit.cli.conftest import payload_of

pytestmark = pytest.mark.usefixtures("cli_env")


@pytest.mark.timeout(300)
def test_cli_eval_cas_writes_report_with_fingerprint(
    runner: CliRunner, app_module: object, tmp_path: Path, db_url: str
) -> None:
    """The CAS run writes a fingerprinted report and records it in ``eval_runs`` intact.

    The run pushes the whole curriculum set (over a thousand rows) through SymPy in this
    process: 9.9 s on the workstation, and past the 60 s default on GitHub's two-vCPU ubuntu
    runner with two xdist workers sharing it (2026-09-10). The budget above is for that
    machine; the test is not slow by design and stays in the PR selection because it is the
    only one that drives the ``aicls eval cas`` door end to end.
    """
    out = tmp_path / "metrics"
    result = runner.invoke(
        app_module, ["eval", "cas", "--out", str(out), "--db-url", db_url, "--json"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)

    report = payload["reports"][0]
    assert report["kind"] == "cas"
    assert report["n"] > 1000, "the IP-4 curriculum set is the corpus"
    assert report["dataset_fingerprint"].startswith("sha256:")
    assert report["metrics"]["correct_rate"] >= 0.0
    written = Path(report["path"])
    assert written.is_file()
    document = json.loads(written.read_text(encoding="utf-8"))
    assert document["dataset_fingerprint"] == report["dataset_fingerprint"]

    row = payload["eval_runs"][0]
    assert row["dataset_fingerprint"] == report["dataset_fingerprint"]
    assert row["git_sha"] == report["git_sha"]
    assert row["is_reportable"] == report["citable"], "citable is the report's own verdict"
    assert row["report_path"] == report["path"]

    # `aicls eval` never promotes: latest.json stays the reviewer's decision.
    assert not (out / "latest.json").exists()


def test_eval_persists_into_the_eval_runs_table(
    runner: CliRunner, app_module: object, tmp_path: Path, db_url: str
) -> None:
    """The row is really in the table, with the fingerprint and the citable flag preserved."""
    from aicls.db import EvalRunRepository, make_engine, session_scope

    result = runner.invoke(
        app_module, ["eval", "book", "--out", str(tmp_path), "--db-url", db_url, "--json"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    expected = payload_of(result.stdout)["reports"][0]

    engine = make_engine(db_url)
    try:
        with session_scope(engine) as session:
            rows = EvalRunRepository(session).list("book")
            assert len(rows) == 1
            row = rows[0]
            assert row.dataset_fingerprint == expected["dataset_fingerprint"]
            assert row.is_reportable is expected["citable"]
            assert row.n == expected["n"]
    finally:
        engine.dispose()


def test_eval_from_db_lists_recorded_runs(
    runner: CliRunner, app_module: object, tmp_path: Path, db_url: str
) -> None:
    """``--from-db`` reads back what was recorded instead of measuring again."""
    runner.invoke(app_module, ["eval", "nlu", "--out", str(tmp_path), "--db-url", db_url])
    result = runner.invoke(app_module, ["eval", "nlu", "--from-db", "--db-url", db_url, "--json"])
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["count"] == 1
    assert payload["runs"][0]["kind"] == "nlu"
    assert payload["runs"][0]["dataset_fingerprint"].startswith("sha256:")


def test_eval_from_db_refuses_a_malformed_since(
    runner: CliRunner, app_module: object, db_url: str
) -> None:
    """``--since`` is an ISO instant; a typo must not silently list everything."""
    result = runner.invoke(
        app_module, ["eval", "nlu", "--from-db", "--since", "yesterday", "--db-url", db_url]
    )
    assert result.exit_code == 1
    assert "since_not_iso8601" in result.stderr


def test_eval_no_db_skips_the_write(
    runner: CliRunner, app_module: object, tmp_path: Path, db_url: str
) -> None:
    """``--no-db`` writes the report file and nothing else."""
    result = runner.invoke(
        app_module, ["eval", "nlu", "--out", str(tmp_path), "--no-db", "--db-url", db_url, "--json"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert payload_of(result.stdout)["eval_runs"] == []


def test_eval_marks_a_stub_run_as_a_wiring_check(
    runner: CliRunner, app_module: object, tmp_path: Path, db_url: str
) -> None:
    """The OCR harness runs the stub recogniser; the output must say so, not quote a CER."""
    result = runner.invoke(
        app_module, ["eval", "ocr", "--out", str(tmp_path), "--db-url", db_url, "--json"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    report = payload_of(result.stdout)["reports"][0]
    assert report["stub"] is True
    assert "WIRING CHECK" in report["notes"]


def test_eval_refuses_split_for_a_kind_without_one(
    runner: CliRunner, app_module: object, db_url: str
) -> None:
    """A flag that would silently do nothing is refused instead."""
    result = runner.invoke(
        app_module, ["eval", "cas", "--split", "test", "--db-url", db_url, "--no-db"]
    )
    assert result.exit_code == 1
    assert "split_not_applicable" in result.stderr


def test_eval_refuses_model_for_a_kind_without_one(
    runner: CliRunner, app_module: object, db_url: str
) -> None:
    """Only the OCR and ASR evaluations can be model-backed."""
    result = runner.invoke(
        app_module, ["eval", "parser", "--model", "0.1.0", "--db-url", db_url, "--no-db"]
    )
    assert result.exit_code == 1
    assert "model_not_applicable" in result.stderr


def test_eval_refuses_an_unregistered_model_version(
    runner: CliRunner, app_module: object, db_url: str
) -> None:
    """``--model`` must name a checkpoint the registry knows about."""
    result = runner.invoke(app_module, ["eval", "ocr", "--model", "9.9.9", "--db-url", db_url])
    assert result.exit_code == 1
    assert "model_version_not_registered" in result.stderr


def test_eval_refuses_a_model_version_that_is_not_promoted(
    runner: CliRunner, app_module: object, db_url: str
) -> None:
    """The request path and the evaluation must not disagree about which checkpoint serves."""
    _register_version(db_url, active=False)
    result = runner.invoke(app_module, ["eval", "ocr", "--model", "0.1.0", "--db-url", db_url])
    assert result.exit_code == 1
    assert "model_version_not_active" in result.stderr


def test_eval_model_backed_run_reports_the_backend_it_actually_used(
    runner: CliRunner, app_module: object, tmp_path: Path, db_url: str
) -> None:
    """With no checkpoint on disk the registry serves the stub, and the report says ``stub``."""
    _register_version(db_url, active=True)
    result = runner.invoke(
        app_module,
        [
            "eval",
            "ocr",
            "--model",
            "0.1.0",
            "--split",
            "test",
            "--out",
            str(tmp_path),
            "--db-url",
            db_url,
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    report = payload_of(result.stdout)["reports"][0]
    assert report["name"] == "ocr_test"
    assert report["stub"] is True, "no checkpoint exists: the gates are closed"
    assert report["model_version"] != "0.1.0", "the report names the backend that really ran"


def _register_version(db_url: str, *, active: bool, component: str = "ocr") -> None:
    """Put one version into ``model_versions`` (its path deliberately does not exist)."""
    from aicls.db import ModelVersionRepository, make_engine, session_scope, upgrade_head

    engine = make_engine(db_url)
    try:
        upgrade_head(engine)
        with session_scope(engine) as session:
            repo = ModelVersionRepository(session)
            repo.register(component, "0.1.0", f"/nonexistent/{component}-0.1.0", "0" * 64)
            if active:
                repo.activate(component, "0.1.0")
    finally:
        engine.dispose()


def test_eval_asr_model_backed_run_uses_the_registry_transcriber(
    runner: CliRunner, app_module: object, tmp_path: Path, db_url: str
) -> None:
    """The ASR route resolves its backend the same way, over the speaker-disjoint synthetic split."""
    _register_version(db_url, active=True, component="asr")
    result = runner.invoke(
        app_module,
        [
            "eval",
            "asr",
            "--model",
            "0.1.0",
            "--out",
            str(tmp_path),
            "--db-url",
            db_url,
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    report = payload_of(result.stdout)["reports"][0]
    assert report["name"] == "asr_test"
    assert report["stub"] is True
    assert report["dataset_fingerprint"].startswith("sha256:")


def test_eval_refuses_an_unknown_split(runner: CliRunner, app_module: object, db_url: str) -> None:
    """Only train/valid/test exist; a typo must not be read as "the default"."""
    result = runner.invoke(
        app_module, ["eval", "ocr", "--split", "holdout", "--model", "0.1.0", "--db-url", db_url]
    )
    assert result.exit_code == 1
    assert "unknown_split" in result.stderr


def test_eval_split_without_a_model_is_refused(
    runner: CliRunner, app_module: object, db_url: str
) -> None:
    """The harness corpora are fixed; ``--split`` only means something for a model-backed run."""
    result = runner.invoke(
        app_module, ["eval", "ocr", "--split", "test", "--db-url", db_url, "--no-db"]
    )
    assert result.exit_code == 1
    assert "split_requires_model" in result.stderr
