"""Cross-module contracts of :mod:`aicls.cli` (wave-5 integration gate).

The CLI is the top of the layering (ADR-0002) and the second of **two** entry points to the
training gate — ``training/scripts/train.py`` is the first. Two doors onto one lock is the seam
neither owner could test alone, so it is tested here: the same configuration, through both doors,
must reach the same verdict, name the same gate and resolve to the same run.

Also covered:

* the package's lazy surface — ``import aicls.cli`` must cost nothing, because ``aicls --help``
  is on every developer's critical path and the command modules sit on top of torch and SymPy;
* every command in the tree answers ``--help`` (a missing service import would only surface at
  runtime otherwise);
* the eval → ``eval_runs`` round trip is covered by ``tests/unit/cli/test_eval.py`` against the
  real harness and a real database; what is added here is that the *reportability* verdict a
  written report carries survives into the table it is read back from.

No test here opens a gate, sets ``AICLS_GATE_*`` or writes to the approval ledger; the gate tests
assert refusals only (ADR-0012).
"""

from __future__ import annotations

from collections.abc import Iterator
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import pytest
from typer.testing import CliRunner

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "training" / "configs"
OCR_SMOKE_CONFIG = CONFIG_ROOT / "ocr" / "smoke_synthetic.yaml"
OCR_REAL_CONFIG = CONFIG_ROOT / "ocr" / "offline_resnet34_tfm.yaml"
ASR_SMOKE_CONFIG = CONFIG_ROOT / "asr" / "smoke_synthetic.yaml"

#: Exit code every refused gated action returns (``aicls.core.errors.GateError``).
EXIT_GATE = 3

#: Command groups mounted on the root application.
COMMAND_NAMES: tuple[str, ...] = (
    "serve",
    "db",
    "data",
    "train",
    "profile",
    "eval",
    "book",
    "models",
    "contrib",
    "nlu",
    "report",
)


@pytest.fixture
def runner() -> CliRunner:
    """A Typer CLI runner (click 8.2+ keeps stdout and stderr apart)."""
    return CliRunner()


@pytest.fixture
def app_module() -> Any:
    """The root Typer application under test."""
    from aicls.cli.main import app

    return app


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    """A file-backed SQLite URL that survives between CLI invocations."""
    return f"sqlite+pysqlite:///{(tmp_path / 'aicls-integration.db').as_posix()}"


@pytest.fixture
def cli_env(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, db_url: str, data_root: Path
) -> Iterator[dict[str, str]]:
    """Hermetic environment for a CLI invocation: temp data root, file database, gates empty."""
    from aicls.core.settings import get_settings

    monkeypatch.setenv("AICLS_DB_URL", db_url)
    get_settings.cache_clear()
    try:
        yield {**test_env, "AICLS_DB_URL": db_url, "AICLS_DATA_ROOT": str(data_root)}
    finally:
        get_settings.cache_clear()


def _payload(output: str) -> Any:
    """Pick the command's own JSON document out of stdout (log records may share the stream)."""
    decoder = json.JSONDecoder()
    documents: list[Any] = []
    index = 0
    while index < len(output):
        while index < len(output) and output[index].isspace():
            index += 1
        if index >= len(output):
            break
        document, index = decoder.raw_decode(output, index)
        documents.append(document)
    payloads = [
        document
        for document in documents
        if not (isinstance(document, dict) and {"event", "level"} <= set(document))
    ]
    if len(payloads) != 1:
        raise AssertionError("expected exactly one command payload on stdout:\n" + output)
    return payloads[0]


# ==============================================================================================
# the package surface
# ==============================================================================================
def test_importing_aicls_cli_costs_nothing() -> None:
    """``import aicls.cli`` must not drag in typer, torch, FastAPI or SymPy.

    Checked in a subprocess: an earlier test in this session has certainly imported all of them.
    """
    code = (
        "import sys, json; import aicls.cli; "
        "print(json.dumps(sorted(m for m in sys.modules "
        "if m in {'typer', 'click', 'torch', 'fastapi', 'sympy', 'sqlalchemy'})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "SYSTEMROOT": "C:\\Windows", "PATH": ""},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip()) == []


def test_the_lazy_app_is_the_real_application_and_the_submodule_survives() -> None:
    """``aicls.cli.app`` resolves to the Typer app; ``aicls.cli.main`` stays the submodule."""
    import typer

    import aicls.cli
    import aicls.cli.main

    assert isinstance(aicls.cli.app, typer.Typer)
    assert aicls.cli.app is aicls.cli.main.app
    assert aicls.cli.main.__name__ == "aicls.cli.main"
    with pytest.raises(AttributeError):
        _ = aicls.cli.does_not_exist  # type: ignore[attr-defined]


def test_the_lazy_package_lists_what_it_exports() -> None:
    """``dir(aicls.cli)`` must mention the lazily resolved name; tab-completion relies on it."""
    import aicls.cli

    assert "app" in dir(aicls.cli)


def test_gate_status_reports_the_lock_without_asking_for_it(
    test_env: dict[str, str], tmp_path: Path
) -> None:
    """The helper both entry points share reports, never approves (ADR-0012).

    It must list every missing condition and refuse an unknown gate name rather than answer
    "nothing missing". All four gates carry an approval since 2026-09-08, so the full-triple case
    is exercised against a ledger that does not exist rather than against a gate nobody approved;
    the repository ledger then carries the case that matters more: an approved row does not
    remove the other two conditions.
    """
    from aicls.core.errors import InputValidationError
    from aicls.core.gates import gate_status

    assert set(gate_status("TRAINING", ledger_path=tmp_path / "absent.md")) >= {
        "cli_flag",
        "env",
        "ledger",
    }
    assert set(gate_status("TRAINING")) >= {"cli_flag", "env"}
    assert "cli_flag" not in gate_status("TRAINING", cli_flag=True)
    assert "env" in gate_status("TRAINING", cli_flag=True)
    # a ledger that does not exist is a closed gate, not an open one
    assert "ledger" in gate_status("TRAINING", ledger_path=tmp_path / "absent.md")
    with pytest.raises(InputValidationError):
        gate_status("NO_SUCH_GATE")  # type: ignore[arg-type]


def test_every_command_group_answers_help(runner: CliRunner, app_module: object) -> None:
    """A command whose service import is broken only fails at runtime; ``--help`` catches it."""
    root = runner.invoke(app_module, ["--help"])
    assert root.exit_code == 0, root.stdout + root.stderr
    for name in COMMAND_NAMES:
        result = runner.invoke(app_module, [name, "--help"])
        assert result.exit_code == 0, f"{name} --help failed:\n{result.stdout}{result.stderr}"


# ==============================================================================================
# two entry points, one gate
# ==============================================================================================
def test_both_entry_points_refuse_the_same_real_config(
    runner: CliRunner, app_module: object, cli_env: dict[str, str]
) -> None:
    """``aicls train`` and ``training/scripts/train.py`` refuse identically, with exit 3.

    Both must name the ``TRAINING`` gate and the same three missing conditions. A difference
    here would mean one door onto the lock is weaker than the other.
    """
    import importlib.util

    cli_result = runner.invoke(app_module, ["train", "--config", str(OCR_REAL_CONFIG)])
    assert cli_result.exit_code == EXIT_GATE, cli_result.stdout + cli_result.stderr
    assert "TRAINING" in cli_result.stderr

    spec = importlib.util.spec_from_file_location(
        "aicls_script_train_contract", REPO_ROOT / "training" / "scripts" / "train.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = script
    try:
        spec.loader.exec_module(script)
        code = script.main(["--config", str(OCR_REAL_CONFIG), "--no-queue"])
    finally:
        sys.modules.pop(spec.name, None)
    assert code == EXIT_GATE

    # Compare the two doors against each other rather than against a hardcoded triple: which
    # conditions are outstanding changes as the ledger changes, but the two entry points must
    # never disagree about it, and every outstanding condition must appear in the refusal the
    # operator actually reads.
    from aicls.core.gates import gate_status

    missing = script.gate_status("TRAINING", cli_flag=False)
    assert set(missing) == set(gate_status("TRAINING", cli_flag=False)), missing
    assert "cli_flag" in missing and "env" in missing, missing
    for condition in missing:
        assert condition in cli_result.stderr, f"the CLI refusal does not name {condition}"


def test_both_entry_points_agree_the_smoke_config_is_gate_free(
    runner: CliRunner, app_module: object, cli_env: dict[str, str]
) -> None:
    """The gate-free verdict and the resolved run must be identical through both doors."""
    import importlib.util

    cli_result = runner.invoke(
        app_module, ["train", "--config", str(OCR_SMOKE_CONFIG), "--dry-run", "--json"]
    )
    assert cli_result.exit_code == 0, cli_result.stdout + cli_result.stderr
    cli_plan = _payload(cli_result.stdout)

    spec = importlib.util.spec_from_file_location(
        "aicls_script_train_contract2", REPO_ROOT / "training" / "scripts" / "train.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = script
    try:
        spec.loader.exec_module(script)
        parser = script.build_parser()
        args = parser.parse_args(["--config", str(OCR_SMOKE_CONFIG), "--dry-run", "--no-queue"])
        cfg, config_path, notes = script.resolve_config(args)
        args.notes = notes
        script_plan = script.build_plan(cfg, config_path, args).to_dict()
    finally:
        sys.modules.pop(spec.name, None)

    assert cli_plan["gate_free"] is True
    assert script_plan["gate_required"] is False
    assert cli_plan["config_hash"] == script_plan["config_hash"]
    assert cli_plan["dataset"] == script_plan["dataset"] == "synthetic_ink"
    assert cli_plan["wiring_check"] is script_plan["wiring_check"] is True


def test_synthetic_flag_rewrites_the_dataset_rather_than_skipping_the_gate(
    runner: CliRunner, app_module: object, cli_env: dict[str, str]
) -> None:
    """``--synthetic`` on a real-data config must change *what is trained*, not the verdict.

    Read literally, "``--synthetic`` means no gate" would let anyone train on MathWriting by
    passing a flag. Both entry points instead replace ``data.dataset`` with the family's
    synthetic sibling and let ``TrainConfig.is_gate_free`` decide from the dataset name.
    """
    without = runner.invoke(
        app_module, ["train", "--config", str(OCR_REAL_CONFIG), "--dry-run", "--json"]
    )
    assert without.exit_code == 0, without.stdout + without.stderr
    plan_without = _payload(without.stdout)
    assert plan_without["dataset"] == "mathwriting"
    assert plan_without["gate_free"] is False
    # A dry run reports the lock instead of demanding it: that is how an operator produces the
    # plan the approval PR has to carry (docs/gelistirme/onay-defteri.md). The ledger condition
    # is satisfied for TRAINING since 2026-09-08; the flag and the environment variable are not,
    # and a dry run must still say so rather than presenting the run as ready.
    assert set(plan_without["missing_conditions"]) >= {"cli_flag", "env"}

    with_flag = runner.invoke(
        app_module,
        ["train", "--config", str(OCR_REAL_CONFIG), "--synthetic", "--dry-run", "--json"],
    )
    assert with_flag.exit_code == 0, with_flag.stdout + with_flag.stderr
    plan = _payload(with_flag.stdout)
    assert plan["dataset"] == "synthetic_ink"
    assert plan["gate_free"] is True


def test_cli_training_run_produces_the_documented_run_directory(
    runner: CliRunner, app_module: object, cli_env: dict[str, str], tmp_path: Path
) -> None:
    """``aicls train --synthetic`` really trains on CPU and leaves a complete run directory.

    WIRING CHECK: two optimizer steps of a randomly initialised model over generated scribbles.
    The assertions are about plumbing and labelling, never about the numbers.
    """
    run_dir = tmp_path / "cli-run"
    result = runner.invoke(
        app_module,
        [
            "train",
            "--config",
            str(OCR_SMOKE_CONFIG),
            "--synthetic",
            "--run-dir",
            str(run_dir),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = _payload(result.stdout)
    assert payload["plan"]["gate_free"] is True
    report = payload["report"]
    assert report["wiring_check"] is True
    assert report["steps"] >= 1
    # `citable` is not asserted: it follows the checkout, so requiring False here made the test
    # pass only on a dirty tree. `wiring_check` is the flag that says this number is not evidence.

    for name in ("run_manifest.json", "metrics.jsonl", "env.json", "fit_report.json"):
        assert (run_dir / name).is_file(), name
    assert sorted((run_dir / "checkpoints").glob("step-*")), "no checkpoint was written"


@pytest.mark.skipif(
    importlib.util.find_spec("accelerate") is None,
    reason="Seq2SeqTrainingArguments cannot be built without accelerate (gated ml extra)",
)
def test_cli_asr_smoke_config_runs_without_weights_and_without_a_gate(
    runner: CliRunner, app_module: object, cli_env: dict[str, str], tmp_path: Path
) -> None:
    """``training/configs/asr/smoke_synthetic.yaml`` is gate-free *and* weight-free — both hold.

    Wave 5 shipped that configuration with ``model.base_dir: null``, which
    :func:`aicls.ml.registry.build_whisper` supports and ``WhisperFineTuner`` refused, so the
    only ASR configuration that needs no approval could not run through this command at all.
    Verified for real on an RTX 5090 (cu130 container, transformers 4.57.6) on 2026-09-08.

    WIRING CHECK: a randomly initialised Whisper over generated sine sweeps, scored through a
    character tokenizer. ``wer = 1.0`` here is the plumbing working, not an ASR measurement.
    """
    run_dir = tmp_path / "cli-asr-run"
    result = runner.invoke(
        app_module,
        [
            "train",
            "--config",
            str(ASR_SMOKE_CONFIG),
            "--synthetic",
            "--run-dir",
            str(run_dir),
            "--override",
            "data.n_samples=8",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = _payload(result.stdout)
    assert payload["plan"]["gate_free"] is True
    report = payload["report"]
    assert report["global_step"] >= 1
    # The old assertion here was `citable is False`, justified as "no git state and random
    # weights". Only the second half is a property of the run: random weights make the WER below
    # meaningless whatever git says, and on a clean checkout there IS git state, so the assertion
    # failed for a reason that had nothing to do with the ASR path it was testing.
    assert set(report["metrics"]) >= {"wer", "wer_raw"}
    assert report["metrics"]["wer"] >= 0.0  # a random Whisper transcribes nothing; not evidence

    assert (run_dir / "run_manifest.json").is_file()
    checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint-*"))
    assert checkpoints, "Hugging Face wrote no checkpoint"
    for name in ("config.yaml", "run_manifest.json", "metrics.json"):
        assert (checkpoints[-1] / name).is_file(), name


# ==============================================================================================
# eval -> eval_runs
# ==============================================================================================
def test_a_written_report_and_its_database_row_agree_on_reportability(
    runner: CliRunner, app_module: object, cli_env: dict[str, str], tmp_path: Path, db_url: str
) -> None:
    """The report file, the CLI payload and the ``eval_runs`` row carry one verdict.

    ``is_reportable`` is the report's own recomputed ``citable`` flag — never the caller's wish —
    and the fingerprint that identifies the corpus must survive the round trip, otherwise a
    stored run cannot be compared with a later one.
    """
    from aicls.db import EvalRunRepository, make_engine, session_scope

    out = tmp_path / "metrics"
    result = runner.invoke(
        app_module, ["eval", "parser", "--out", str(out), "--db-url", db_url, "--json"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    report = _payload(result.stdout)["reports"][0]

    document = json.loads(Path(report["path"]).read_text(encoding="utf-8"))
    assert document["dataset_fingerprint"] == report["dataset_fingerprint"]
    assert document["citable"] == report["citable"]

    engine = make_engine(db_url)
    try:
        with session_scope(engine) as session:
            rows = EvalRunRepository(session).list("parser")
            assert len(rows) == 1
            assert rows[0].dataset_fingerprint == document["dataset_fingerprint"]
            assert rows[0].is_reportable is document["citable"]
            assert rows[0].report_path == report["path"]
    finally:
        engine.dispose()


def test_a_trained_checkpoint_can_be_registered_and_listed(
    runner: CliRunner, app_module: object, cli_env: dict[str, str], tmp_path: Path, db_url: str
) -> None:
    """A checkpoint the trainer wrote is registrable by ``aicls models import``, digest and all.

    This is the ml -> cli -> db seam nothing crossed: ``aicls models import`` verifies each
    entry's SHA-256 against the bytes on disk, and the digest an operator would copy into
    ``models/registry.yaml`` comes from the ``checkpoint.json`` that
    :class:`aicls.ml.checkpoint.CheckpointSaver` wrote. If the two ever disagreed — a prefix, a
    different file, a different hash function — every promotion would fail at the last step.
    """
    run_dir = tmp_path / "run"
    trained = runner.invoke(
        app_module,
        [
            "train",
            "--config",
            str(OCR_SMOKE_CONFIG),
            "--run-dir",
            str(run_dir),
            "--no-val",
            "--json",
        ],
    )
    assert trained.exit_code == 0, trained.stdout + trained.stderr

    checkpoints = run_dir / "checkpoints"
    step_dir = sorted(checkpoints.glob("step-*"))[-1]
    meta = json.loads((step_dir / "checkpoint.json").read_text(encoding="utf-8"))

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "schema_version: 1\n"
        "models:\n"
        "  - id: ocr-wiring-check\n"
        "    kind: ocr\n"
        f"    path: {step_dir.name}\n"
        f"    checkpoint: {meta['model_file']}\n"
        f"    sha256: {meta['model_sha256']}\n"
        "    version: 0.0.1-wiring\n"
        "    notes: WIRING CHECK, randomly initialised weights on synthetic ink\n",
        encoding="utf-8",
        newline="\n",
    )

    imported = runner.invoke(
        app_module,
        [
            "models",
            "import",
            str(registry),
            "--models-root",
            str(checkpoints),
            "--db-url",
            db_url,
            "--json",
        ],
    )
    assert imported.exit_code == 0, imported.stdout + imported.stderr
    entry = _payload(imported.stdout)["imported"][0]
    assert entry["version"] == "0.0.1-wiring"
    assert entry["config_hash"], "the run manifest's config hash must travel with the model"

    listed = runner.invoke(
        app_module, ["models", "list", "--component", "ocr", "--db-url", db_url, "--json"]
    )
    assert listed.exit_code == 0, listed.stdout + listed.stderr
    versions = [row["version"] for row in _payload(listed.stdout)["versions"]]
    assert "0.0.1-wiring" in versions


# -------------------------------------------------------------------------------- the two doors
#
# `aicls train` and `training/scripts/train.py` drive the same Trainer, and they have drifted
# apart three separate times, each found only when somebody needed the missing half:
#
#   * --resume existed only on the script, so a nine-hour run interrupted at step 18000 could not
#     be continued with the command that started it;
#   * the same override took --set at one door and --override at the other;
#   * the single-GPU queue lock was taken only by the script, so the protection against two
#     trainings on one card could be walked past by using the other command.
#
# The first two cost an operator time. The third cost the protection itself, silently. This is
# the test that would have caught all three on the commit that introduced them.
_SCRIPT_ONLY = {
    # --config is required at the CLI, so "which shipped smoke config does --synthetic use"
    # is a question only the script can be asked.
    "--family",
}
_CLI_ONLY = {
    # typer renders the boolean pair as --val/--no-val; argparse takes the negative alone. Same
    # capability, two spellings of it, and the negative half is present at both doors.
    "--val",
}


def _long_options(text: str) -> set[str]:
    """Every long option a ``--help`` output offers."""
    return set(re.findall(r"--[a-z][a-z-]+", text))


def _script_help() -> str:
    """``training/scripts/train.py --help``, wide enough that no option name is wrapped away."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "training" / "scripts" / "train.py"), "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "COLUMNS": "200"},
        check=False,
        cwd=REPO_ROOT,
    )
    return result.stdout + result.stderr


def test_both_doors_onto_the_trainer_offer_the_same_flags(
    runner: CliRunner, app_module: Any
) -> None:
    """Neither door may grow a flag the other lacks without saying so here.

    The two allowlists are the "why". Adding to one is a deliberate statement that a capability
    cannot exist at the other door; anything else is the drift this test exists to stop, and it
    now fails on the commit that introduces it rather than months later, when somebody reaches
    for the missing half and finds out the hard way.
    """
    cli_help = runner.invoke(app_module, ["train", "--help"], env={"COLUMNS": "200"}).stdout
    cli = _long_options(cli_help)
    script = _long_options(_script_help())
    assert {"--resume", "--config"} <= cli, "the CLI help could not be read"
    assert {"--resume", "--config"} <= script, "the script help could not be read"

    missing_from_cli = script - cli - _SCRIPT_ONLY
    missing_from_script = cli - script - _CLI_ONLY

    assert not missing_from_cli, (
        f"training/scripts/train.py offers {sorted(missing_from_cli)} and `aicls train` does not; "
        "add it there too, or to _SCRIPT_ONLY with the reason it cannot exist at the CLI"
    )
    assert not missing_from_script, (
        f"`aicls train` offers {sorted(missing_from_script)} and training/scripts/train.py does "
        "not; add it there too, or to _CLI_ONLY with the reason"
    )


def test_the_two_doors_agree_on_the_flags_that_protect_the_gpu_and_the_record() -> None:
    """The four that matter most, named rather than left to a set difference.

    A missing flag is usually a nuisance. These four missing are a run that cannot be continued,
    a GPU that can be double-booked, and a number whose provenance nobody can check.
    """
    cli_source = (REPO_ROOT / "src" / "aicls" / "cli" / "train.py").read_text(encoding="utf-8")
    script_source = (REPO_ROOT / "training" / "scripts" / "train.py").read_text(encoding="utf-8")

    for flag in ("--resume", "--force-queue", "--no-queue", "--allow-dirty"):
        assert flag in cli_source, f"{flag} is missing from `aicls train`"
        assert flag in script_source, f"{flag} is missing from training/scripts/train.py"
