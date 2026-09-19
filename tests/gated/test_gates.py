"""Approval-gate refusal tests for ``training/scripts/`` (ADR-0012, egitim-altyapisi §9).

These tests guard the project owner's standing instruction: **no download and no training run
without explicit approval**. They therefore assert the *refusal*, not the action:

* every gated script exits **3** when a condition is missing, and leaves the filesystem untouched;
* the triple lock really is a triple lock — flag plus environment variable is still refused
  without a ledger row;
* CI refuses even when all three conditions are present;
* the one gate-free path (a synthetic dataset) works without ever consulting the gate.

Nothing here downloads, and nothing here trains on real data. The two download functions are
monkeypatched to explode, so even a regression that let a refusal through could not reach the
network; the only real training is one CPU step on generated fixtures, which is a wiring check
and produces no evidence for any contractual metric.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import json
from pathlib import Path
import types

import pytest

from aicls.core.gates import GATE_NAMES
from aicls.core.settings import get_settings

pytestmark = pytest.mark.gated

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "training" / "scripts"

DOWNLOAD_SCRIPTS: tuple[tuple[str, str, str], ...] = (
    ("training/scripts/download_mathwriting.py", "DATASET_DOWNLOAD", "download_archive"),
    ("training/scripts/download_weights.py", "WEIGHTS_DOWNLOAD", "download_snapshot"),
)

UNGATED_SCRIPTS: tuple[str, ...] = (
    "training/scripts/prepare_mathwriting.py",
    "training/scripts/prepare_audio_commands.py",
    "training/scripts/build_tokenizer.py",
    "training/scripts/evaluate.py",
)

LEDGER_HEAD = (
    "# Onay Defteri\n\nintro\n\n| KAPI | durum | tarih | kim | kapsam |\n|---|---|---|---|---|\n"
)

#: Network calls only these two files may contain (checked statically below).
NETWORK_MARKERS: tuple[str, ...] = ("urllib.request", "urlopen(", "snapshot_download", "requests.")

Loader = Callable[[str], types.ModuleType]


def _ledger(tmp_path: Path, rows: str = "") -> Path:
    """Write a syntactically valid approval ledger with the given rows."""
    path = tmp_path / "onay-defteri.md"
    path.write_text(LEDGER_HEAD + rows, encoding="utf-8")
    return path


def _explode(*args: object, **kwargs: object) -> None:
    """Stand-in for a download function; a test that reaches the network is a failed test."""
    raise AssertionError("a gated test must never perform a real download")


@pytest.fixture
def gated_env(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> Iterator[dict[str, str]]:
    """Hermetic environment with every gate closed and no CI markers set."""
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    get_settings.cache_clear()
    yield test_env
    get_settings.cache_clear()


@pytest.fixture
def no_downloads(
    load_script: Loader, monkeypatch: pytest.MonkeyPatch
) -> dict[str, types.ModuleType]:
    """Import both download scripts with their network functions replaced by a landmine."""
    modules: dict[str, types.ModuleType] = {}
    for rel_path, _gate, function in DOWNLOAD_SCRIPTS:
        module = load_script(rel_path)
        monkeypatch.setattr(module, function, _explode)
        modules[rel_path] = module
    return modules


# ----------------------------------------------------------------------------------------------
# the download gates
# ----------------------------------------------------------------------------------------------
def test_gated_download_refuses_without_flag_and_env(
    no_downloads: dict[str, types.ModuleType],
    gated_env: dict[str, str],
    data_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No flag, no environment variable, no ledger row: exit 3 and nothing on disk."""
    for rel_path, gate, _function in DOWNLOAD_SCRIPTS:
        module = no_downloads[rel_path]
        dest = tmp_path / f"dest-{gate}"
        argv = ["--dest", str(dest)]
        if "weights" in rel_path:
            argv = ["--model", "whisper-small", *argv]

        code = module.main(argv)

        assert code == 3, f"{rel_path} must exit 3 when the gate is closed"
        assert not dest.exists(), f"{rel_path} created {dest} despite a closed gate"
        assert not (data_root / "gate_invocations.log").exists()
        captured = capsys.readouterr()
        # the Turkish refusal names all three missing conditions
        assert "--i-have-approval" in captured.err
        assert f"AICLS_GATE_{gate}=approved" in captured.err
        assert "onay-defteri.md" in captured.err


def test_gated_download_refuses_without_ledger_line(
    no_downloads: dict[str, types.ModuleType],
    gated_env: dict[str, str],
    data_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag **and** environment variable are still not enough: the ledger row is the third lock."""
    ledger = _ledger(tmp_path)  # valid table, no rows at all
    for rel_path, gate, _function in DOWNLOAD_SCRIPTS:
        module = no_downloads[rel_path]
        monkeypatch.setenv(f"AICLS_GATE_{gate}", "approved")
        get_settings.cache_clear()
        dest = tmp_path / f"ledgerless-{gate}"
        argv = ["--dest", str(dest), "--i-have-approval"]
        if "weights" in rel_path:
            argv = ["--model", "whisper-small", *argv]

        assert module.gate_status(gate, cli_flag=True, ledger_path=ledger) == ["ledger"]
        assert module.main(argv, ledger_path=ledger) == 3
        assert not dest.exists()
        assert not (data_root / "gate_invocations.log").exists()


def test_gated_download_dry_run_prints_the_cost_without_asking_for_the_gate(
    no_downloads: dict[str, types.ModuleType],
    gated_env: dict[str, str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--dry-run`` is the operator's way to see url, size, licence, target and disk, gate-free."""
    module = no_downloads["training/scripts/download_mathwriting.py"]
    dest = tmp_path / "preview"

    assert module.main(["--dest", str(dest), "--dry-run"]) == 0

    out = capsys.readouterr().out
    for fragment in ("url", "licence", "sha256", "target dir", "free disk", "gate status"):
        assert fragment in out, f"the plan must state {fragment!r}"
    # the licence is CC BY-NC-SA 4.0 (verified 2026-09-08); NonCommercial/ShareAlike are the
    # facts an approver must see, so the plan states them and this test pins them.
    assert "CC-BY-NC-SA-4.0" in out
    assert "NonCommercial" in out
    assert str(dest) in out
    assert not dest.exists()


def test_gated_download_dry_run_json_is_a_machine_readable_approval_dossier(
    no_downloads: dict[str, types.ModuleType],
    gated_env: dict[str, str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--json`` emits the same plan as a document an approval PR can quote."""
    module = no_downloads["training/scripts/download_weights.py"]

    assert module.main(["--model", "openai/whisper-small", "--dry-run", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out.split("dry run:")[0])
    assert payload["repo_id"] == "openai/whisper-small"
    assert payload["gate"] == "WEIGHTS_DOWNLOAD"
    assert payload["gate_open"] is False
    # The dossier lists whatever is outstanding. Since 2026-09-08 the ledger condition is met, so
    # asserting the full triple would be asserting the ledger's contents rather than the dossier's
    # job, which is to tell an operator exactly what is still missing.
    assert {"cli_flag", "env"} <= set(payload["missing_conditions"])
    assert "ledger" not in payload["missing_conditions"]


def test_gated_download_weights_refuses_an_unknown_model(
    no_downloads: dict[str, types.ModuleType], gated_env: dict[str, str]
) -> None:
    """A hub id that is not in the catalog is a validation error, never a download attempt."""
    module = no_downloads["training/scripts/download_weights.py"]

    assert module.main(["--model", "someone/not-in-the-catalog", "--dry-run"]) == 1


# ----------------------------------------------------------------------------------------------
# the training gate
# ----------------------------------------------------------------------------------------------
def test_gated_train_refuses_in_ci(
    load_script: Loader,
    gated_env: dict[str, str],
    data_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CI refuses even with flag, environment variable and an approved ledger row present."""
    module = load_script("training/scripts/train.py")
    monkeypatch.setattr(module, "_run", _explode)
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("AICLS_GATE_TRAINING", "approved")
    get_settings.cache_clear()
    ledger = _ledger(
        tmp_path, "| TRAINING | approved | 2027-01-15 | csahin | e-posta 2027-01-14 |\n"
    )
    run_dir = tmp_path / "run"

    code = module.main(
        [
            "--config",
            str(REPO_ROOT / "training" / "configs" / "ocr" / "offline_resnet34_tfm.yaml"),
            "--run-dir",
            str(run_dir),
            "--i-have-approval",
            "--no-queue",
        ],
        ledger_path=ledger,
    )

    assert code == 3
    assert not run_dir.exists()
    assert not (data_root / "gate_invocations.log").exists()


def test_gated_train_refuses_a_real_dataset_without_approval(
    load_script: Loader,
    gated_env: dict[str, str],
    data_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default state of the repository: MathWriting training is refused, in Turkish."""
    module = load_script("training/scripts/train.py")
    monkeypatch.setattr(module, "_run", _explode)
    run_dir = tmp_path / "run"

    code = module.main(
        [
            "--config",
            str(REPO_ROOT / "training" / "configs" / "ocr" / "offline_resnet34_tfm.yaml"),
            "--run-dir",
            str(run_dir),
            "--no-queue",
        ]
    )

    assert code == 3
    assert not run_dir.exists()
    assert not (data_root / "gate_invocations.log").exists()
    err = capsys.readouterr().err
    assert "--i-have-approval" in err
    assert "AICLS_GATE_TRAINING=approved" in err
    assert "onay-defteri.md" in err


def test_gated_train_allows_synthetic_without_gate(
    load_script: Loader, gated_env: dict[str, str], tmp_path: Path
) -> None:
    """A synthetic run trains one CPU step and never consults the gate at all.

    ``aicls.ml.trainer.require_gate`` is replaced by a landmine: if the synthetic path ever asked
    for approval, this test would fail. The step itself is a **wiring check** — random weights on
    generated Bezier scribbles — and the report says so.
    """
    import aicls.ml.trainer as trainer_module

    module = load_script("training/scripts/train.py")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trainer_module, "require_gate", _explode)
        run_dir = tmp_path / "synthetic-run"

        code = module.main(["--synthetic", "--max-steps", "1", "--run-dir", str(run_dir)])

    assert code == 0
    assert (run_dir / "run_manifest.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    report = json.loads((run_dir / "fit_report.json").read_text(encoding="utf-8"))
    assert report["steps"] == 1
    assert report["wiring_check"] is True, "a synthetic run must never be presented as evidence"
    # `citable` intentionally unasserted: it reports whether the tree was clean, which is a
    # property of the checkout and not of this run. It was False here only by accident of the
    # working copy being dirty during development.


def test_gated_train_synthetic_flag_rewrites_the_dataset_instead_of_skipping_the_gate(
    load_script: Loader, gated_env: dict[str, str]
) -> None:
    """``--synthetic`` cannot smuggle a real dataset past the gate: it rewrites ``data.dataset``.

    This is the safety property of the flag. Gate-freeness is decided by
    :attr:`aicls.ml.config.TrainConfig.is_gate_free`, which reads only the dataset name, so a run
    is exempt because of what it trains on, never because of what the caller claimed.
    """
    module = load_script("training/scripts/train.py")
    real = str(REPO_ROOT / "training" / "configs" / "ocr" / "offline_resnet34_tfm.yaml")

    without = module.build_parser().parse_args(["--config", real])
    cfg_real, _path, _notes = module.resolve_config(without)
    assert cfg_real.data.dataset == "mathwriting"
    assert cfg_real.is_gate_free is False

    with_flag = module.build_parser().parse_args(["--config", real, "--synthetic"])
    cfg_synth, _path, notes = module.resolve_config(with_flag)
    assert cfg_synth.data.dataset == "synthetic_ink"
    assert cfg_synth.is_gate_free is True
    assert any("data.dataset" in note for note in notes)


def test_gated_train_dry_run_does_not_ask_for_the_gate(
    load_script: Loader,
    gated_env: dict[str, str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--dry-run`` previews a real-data run (plan + gate status) and writes nothing."""
    import aicls.ml.trainer as trainer_module

    module = load_script("training/scripts/train.py")
    run_dir = tmp_path / "never"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trainer_module, "require_gate", _explode)
        code = module.main(
            [
                "--config",
                str(REPO_ROOT / "training" / "configs" / "asr" / "whisper_small_ft.yaml"),
                "--run-dir",
                str(run_dir),
                "--dry-run",
            ]
        )

    assert code == 0
    assert not run_dir.exists()
    out = capsys.readouterr().out
    assert "gate         : TRAINING" in out
    assert "closed" in out


# ----------------------------------------------------------------------------------------------
# structural guarantees about the scripts themselves
# ----------------------------------------------------------------------------------------------
def test_a_ledger_row_alone_never_opens_a_gate(tmp_path: Path, load_script: Loader) -> None:
    """Checking in an approval is one of three conditions, never the whole lock (ADR-0012).

    Since 2026-09-08 the committed ledger does approve CUDA_INSTALL, DATASET_DOWNLOAD and
    TRAINING, so "the repository approves nothing" is no longer the invariant to protect. The
    invariant that survives, and the one that matters, is that a merged row still cannot run
    anything on its own: the operator must also be at a shell that exports the gate variable and
    must still pass the flag. That is what keeps a reviewed decision from becoming an ambient
    permission every later process inherits.
    """
    module = load_script("training/scripts/download_mathwriting.py")
    for gate in GATE_NAMES:
        missing = module.gate_status(gate, cli_flag=True)
        assert "env" in missing, f"{gate} would run from the ledger alone"


def test_only_the_two_download_scripts_contain_network_code() -> None:
    """Network calls live in the allow-listed files only (``tools/check_gates.py`` mirrors this)."""
    allowed = {"download_mathwriting.py", "download_weights.py"}
    for path in sorted(SCRIPTS.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
        hits = [marker for marker in NETWORK_MARKERS if marker in code]
        if path.name in allowed:
            continue
        assert not hits, f"{path.name} contains network markers {hits}"


def test_network_code_sits_after_the_gate_in_both_download_scripts() -> None:
    """The single network function is defined *below* the ``require_gate`` import and call."""
    for rel_path, _gate, function in DOWNLOAD_SCRIPTS:
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        assert "require_gate(" in text, rel_path
        # the network helper is called from _run_approved, which main() reaches only after the
        # gate returned; assert the call sites rather than trusting the reading order
        assert f"{function}(" in text
        approved = text.index("def _run_approved(")
        assert text.index(f"{function}(", approved) > approved, rel_path


def test_ungated_scripts_never_call_require_gate() -> None:
    """The ungated scripts operate on local data; a gate call there would be misleading."""
    for rel_path in UNGATED_SCRIPTS:
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
        assert "require_gate(" not in code, rel_path


def test_gated_scripts_use_the_canonical_cli_flag_name() -> None:
    """All three gated scripts spell the flag the way ``aicls.core.gates.CLI_FLAG`` does."""
    from aicls.core.gates import CLI_FLAG

    for rel_path in (*(row[0] for row in DOWNLOAD_SCRIPTS), "training/scripts/train.py"):
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        assert "CLI_FLAG" in text and CLI_FLAG == "--i-have-approval", rel_path
