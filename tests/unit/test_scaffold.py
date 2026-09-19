"""Scaffold contract tests (ADR-0010, ADR-0011, ADR-0012).

These names are contractual (scaffold.json ``tests_required``):

* ``test_poe_tasks_contain_no_gated_commands``
* ``test_install_ml_script_contains_sm120_check``
* ``test_gate_ledger_format``
* ``test_check_env_warns_on_onedrive_path``
* ``test_check_gates_tool_flags_hub_id_usage``
"""

from __future__ import annotations

from collections.abc import Callable
import datetime as dt
from pathlib import Path
import re
import tomllib
import types

import pytest

GATE_NAMES = {"DATASET_DOWNLOAD", "WEIGHTS_DOWNLOAD", "CUDA_INSTALL", "TRAINING"}
LEDGER_HEADER = "| KAPI | durum | tarih | kim | kapsam |"
LEDGER_SEPARATOR_RE = re.compile(r"^\|(?:\s*:?-+:?\s*\|){5}$")

# Substrings that must never appear in a poe task command (ADR-0012).
GATED_COMMAND_PATTERNS = (
    "download_mathwriting",
    "download_weights",
    "download_dataset",
    "install_ml",
    "aicls train",
    "cli.main train",
    "scripts/train.py",
    "--i-have-approval",
    "-IHaveApproval",
    "hf_hub_download",
    "snapshot_download",
    "from_pretrained",
    "whl/cu1",
    "pip-compile",  # network; lockfile refresh is a separate chore PR (requirements/README.md)
)


def _iter_task_commands(tasks: dict[str, object]) -> list[tuple[str, str]]:
    """Flatten poe task definitions into ``(task_name, command_text)`` pairs."""
    out: list[tuple[str, str]] = []
    for name, spec in tasks.items():
        if isinstance(spec, str):
            out.append((name, spec))
            continue
        assert isinstance(spec, dict), name
        for key in ("cmd", "shell", "script", "expr"):
            if key in spec:
                out.append((name, str(spec[key])))
        if "sequence" in spec:
            for item in spec["sequence"]:  # type: ignore[union-attr]
                out.append(
                    (
                        name,
                        str(item)
                        if not isinstance(item, dict)
                        else str(item.get("cmd", item.get("ref", ""))),
                    )
                )
    return out


# --------------------------------------------------------------------------------------------
# poe tasks
# --------------------------------------------------------------------------------------------
def test_poe_tasks_contain_no_gated_commands(repo_root: Path) -> None:
    data = tomllib.loads((repo_root / "poe_tasks.toml").read_text(encoding="utf-8"))
    tasks = data["tasks"]
    assert tasks, "no tasks defined"
    commands = _iter_task_commands(tasks)
    assert commands
    offenders = [
        (name, cmd, pat)
        for name, cmd in commands
        for pat in GATED_COMMAND_PATTERNS
        if pat.lower() in cmd.lower()
    ]
    assert not offenders, f"gated/networked commands in poe tasks: {offenders}"
    # No task may be named like a gated action either.
    assert not {n for n in tasks if n in {"train", "download", "install-ml", "lock"}}
    # The tasks the docs promise exist (docs/gelistirme/is-akisi.md §8 subset for this phase).
    expected = {
        "env-check",
        "lint",
        "fmt",
        "type",
        "imports",
        "test",
        "test-fast",
        "test-all",
        "cov",
        "ci",
        "dev",
        "docs",
        "docs-build",
        "db-upgrade",
        "db-revision",
        "openapi",
        "eval",
        "report-check",
        "fixtures",
        "new-adr",
        "new-exp",
    }
    assert expected <= set(tasks), expected - set(tasks)
    # Sanity: the env block sets PYTHONUTF8 so Windows consoles handle Turkish output.
    assert data["env"]["PYTHONUTF8"] == "1"


# --------------------------------------------------------------------------------------------
# install_ml.ps1 (gate 3)
# --------------------------------------------------------------------------------------------
def test_install_ml_script_contains_sm120_check(repo_root: Path) -> None:
    text = (repo_root / "scripts" / "install_ml.ps1").read_text(encoding="utf-8")
    assert "get_device_capability()" in text
    assert "(12, 0)" in text, "must verify Blackwell sm_120 compute capability"
    assert "download.pytorch.org/whl/cu128" in text, "must pin the cu128 wheel index"
    assert "AICLS_GATE_CUDA_INSTALL" in text
    assert "IHaveApproval" in text
    assert "onay-defteri.md" in text, "must read the approval ledger"
    assert "exit 3" in text, "refused gate must exit with code 3 (ADR-0012)"
    assert "bfloat16" in text, "bf16 matmul smoke test required (kurulum.md RTX 5090 notu)"
    assert "gate_invocations.log" in text
    # The gate check must precede any pip install in the script body.
    assert text.index("exit 3") < text.index('pip", "install"')


# --------------------------------------------------------------------------------------------
# approval ledger format
# --------------------------------------------------------------------------------------------
def _parse_ledger(text: str) -> list[dict[str, str]]:
    """Parse the ledger table; raises AssertionError on format violations."""
    lines = text.splitlines()
    header_idx = next((i for i, ln in enumerate(lines) if ln.strip() == LEDGER_HEADER), None)
    assert header_idx is not None, f"ledger header must be exactly {LEDGER_HEADER!r}"
    assert LEDGER_SEPARATOR_RE.match(lines[header_idx + 1].strip()), (
        "5-column separator row expected"
    )
    rows: list[dict[str, str]] = []
    for ln in lines[header_idx + 2 :]:
        s = ln.strip()
        if not s.startswith("|"):
            break
        cells = [c.strip() for c in s.strip("|").split("|")]
        assert len(cells) == 5, f"row must have 5 cells: {s!r}"
        gate, status, date, who, scope = cells
        assert gate in GATE_NAMES, f"unknown gate {gate!r}"
        assert status in {"approved", "revoked"}, f"status must be approved|revoked: {status!r}"
        dt.date.fromisoformat(date)  # YYYY-MM-DD
        assert who, "kim column must not be empty"
        assert scope, "kapsam column must not be empty"
        rows.append({"gate": gate, "status": status, "date": date, "who": who, "scope": scope})
    return rows


def test_gate_ledger_format(repo_root: Path) -> None:
    text = (repo_root / "docs" / "gelistirme" / "onay-defteri.md").read_text(encoding="utf-8")
    rows = _parse_ledger(text)
    # Approvals exist from 2026-09-08 onwards, so "the ledger is empty" is no longer the
    # invariant. What must hold for every recorded approval is that it is attributable: a row
    # nobody signed, or one that does not say where the approval came from and what it covers,
    # is exactly the row an auditor cannot check.
    for row in rows:
        if row["status"] != "approved":
            continue
        assert row["who"].strip(), f"approved row for {row['gate']} has no signatory"
        scope = row["scope"].strip()
        assert len(scope) > 40, f"approved row for {row['gate']} has a scope too thin to audit"
        assert "onay:" in scope.lower(), (
            f"approved row for {row['gate']} does not name the source of the approval"
        )
    # Parser sanity on a synthetic ledger with approve + revoke.
    sample = (
        "# x\n\n"
        f"{LEDGER_HEADER}\n|---|---|---|---|---|\n"
        "| CUDA_INSTALL | approved | 2027-01-10 | csahin | e-posta 2027-01-09; cu128, sürücü 610.88 |\n"
        "| CUDA_INSTALL | revoked | 2027-02-01 | csahin | toplantı kararı |\n\n"
        "sonraki paragraf\n"
    )
    parsed = _parse_ledger(sample)
    assert [r["status"] for r in parsed] == ["approved", "revoked"]
    with pytest.raises(AssertionError):
        _parse_ledger(sample.replace("| revoked |", "| pending |"))
    with pytest.raises(ValueError):
        _parse_ledger(sample.replace("2027-02-01", "01.02.2027"))


# --------------------------------------------------------------------------------------------
# check_env (ADR-0011)
# --------------------------------------------------------------------------------------------
def test_check_env_warns_on_onedrive_path(
    load_script: Callable[[str], types.ModuleType], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("ONEDRIVE", "ONEDRIVECOMMERCIAL", "ONEDRIVECONSUMER"):
        monkeypatch.delenv(key, raising=False)
    env = load_script("tools/check_env.py")
    onedrive_repo = Path(r"C:\Users\cagri\OneDrive\Masaüstü\fırat\PROJE\ABDULLAH_HOCA_1001\ai-cls")
    assert env.is_under_onedrive(onedrive_repo)
    assert env.is_under_onedrive(Path(r"C:\Users\x\OneDrive - Universite\repo"))
    assert not env.is_under_onedrive(Path(r"C:\Users\cagri\.venvs\ai-cls"))
    assert not env.is_under_onedrive(tmp_path)

    findings = env.check_onedrive_paths(
        repo_root=onedrive_repo,
        venv_prefix=Path(r"C:\Users\cagri\.venvs\ai-cls"),
        data_root=str(onedrive_repo / "data"),
        models_root=None,
        pycache_prefix=str(tmp_path / "pycache"),
    )
    by_check = {f.check: f for f in findings}
    assert by_check["onedrive.repo"].level == "warn"
    assert "OneDrive" in by_check["onedrive.repo"].message
    assert by_check["onedrive.data_root"].level == "error", "data root under OneDrive is an error"
    assert by_check["onedrive.venv"].level == "ok"

    clean = env.check_onedrive_paths(
        repo_root=tmp_path / "src" / "ai-cls",
        venv_prefix=tmp_path / "venv",
        data_root=str(tmp_path / "data"),
        models_root=str(tmp_path / "data" / "models"),
        pycache_prefix=str(tmp_path / "pycache"),
    )
    assert all(f.level != "error" for f in clean)
    assert not any("OneDrive" in f.message and f.level == "warn" for f in clean)

    # The ENV-based detection also works when the path is not literally named OneDrive.
    monkeypatch.setenv("ONEDRIVE", str(tmp_path / "synced"))
    (tmp_path / "synced" / "repo").mkdir(parents=True)
    assert env.is_under_onedrive(tmp_path / "synced" / "repo")


# --------------------------------------------------------------------------------------------
# check_gates (ADR-0012)
# --------------------------------------------------------------------------------------------
def test_check_gates_tool_flags_hub_id_usage(
    load_script: Callable[[str], types.ModuleType], tmp_path: Path
) -> None:
    gates = load_script("tools/check_gates.py")
    src = tmp_path / "src"
    (src / "aicls" / "asr").mkdir(parents=True)
    bad = src / "aicls" / "asr" / "whisper_transcriber.py"
    bad.write_text(
        "from transformers import WhisperForConditionalGeneration\n"
        "\n"
        "def load():\n"
        '    return WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")\n'
        "\n"
        "def enc():\n"
        "    import torchvision\n"
        '    return torchvision.models.resnet34(weights="DEFAULT")\n'
        "\n"
        "def fetch():\n"
        '    url = "https://huggingface.co/openai/whisper-small/resolve/main/model.safetensors"\n'
        "    return url\n",
        encoding="utf-8",
    )
    hits = gates.scan_file(bad, root=tmp_path)
    rules = {h.rule for h in hits}
    assert "hub-id" in rules, "from_pretrained('openai/...') must be flagged"
    assert "torchvision-weights" in rules, "weights='DEFAULT' must be flagged"
    assert "url-literal" in rules, "huggingface.co URL literal must be flagged"
    assert {h.line_no for h in hits} == {4, 8, 11}

    good = src / "aicls" / "asr" / "ok.py"
    good.write_text(
        "from pathlib import Path\n"
        '# from_pretrained("openai/whisper-small")  <- comment, not code\n'
        "def load(path: Path):\n"
        "    return Model.from_pretrained(path)\n"
        "def enc():\n"
        "    return torchvision.models.resnet34(weights=None)\n"
        "def build(weights=None):\n"
        "    return weights\n"
        'x = requests.get("https://example.org")  # gate-ok: reviewed, not a model download\n',
        encoding="utf-8",
    )
    assert gates.scan_file(good, root=tmp_path) == []

    # Gated download scripts are exempt by path.
    gated_dir = tmp_path / "training" / "scripts"
    gated_dir.mkdir(parents=True)
    gated = gated_dir / "download_weights.py"
    gated.write_text('snapshot_download("openai/whisper-small")\n', encoding="utf-8")
    assert gates.scan_file(gated, root=tmp_path) == []
    # ...but the same code elsewhere is not.
    other = gated_dir / "prepare_audio_commands.py"
    other.write_text('snapshot_download("openai/whisper-small")\n', encoding="utf-8")
    assert [h.rule for h in gates.scan_file(other, root=tmp_path)] == ["hf-download"]

    # CLI: exit 1 with hits, 0 without.
    assert gates.main([str(bad), "--root", str(tmp_path)]) == 1
    assert gates.main([str(good), "--root", str(tmp_path)]) == 0


def test_check_gates_passes_on_repository(
    load_script: Callable[[str], types.ModuleType], repo_root: Path
) -> None:
    gates = load_script("tools/check_gates.py")
    hits = gates.scan_tree(repo_root)
    assert hits == [], "\n".join(h.render(repo_root) for h in hits)


# --------------------------------------------------------------------------------------------
# misc scaffold invariants
# --------------------------------------------------------------------------------------------
def test_package_version_is_single_source(repo_root: Path) -> None:
    import aicls

    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+)?", aicls.__version__)
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    assert "version" in pyproject["project"]["dynamic"]
    assert pyproject["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "aicls.__version__"}


def test_all_subpackages_importable_and_empty() -> None:
    import importlib

    for name in (
        "core",
        "db",
        "latex2sympy",
        "mathengine",
        "nlu",
        "ocr",
        "asr",
        "book",
        "collect",
        "data",
        "eval",
        "ml",
        "api",
        "cli",
    ):
        mod = importlib.import_module(f"aicls.{name}")
        assert mod.__doc__, name


def test_env_example_lists_every_gate_variable(repo_root: Path) -> None:
    text = (repo_root / ".env.example").read_text(encoding="utf-8")
    for gate in GATE_NAMES:
        assert re.search(rf"^AICLS_GATE_{gate}=\s*$", text, flags=re.MULTILINE), gate
    assert "AICLS_JWT_SECRET=" in text
    assert "AICLS_DATA_ROOT=" in text
    assert "AICLS_ETHICS_APPROVAL_REF=" in text


def test_ci_workflow_never_sets_gates_and_runs_required_steps(repo_root: Path) -> None:
    text = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for gate in GATE_NAMES:
        assert re.search(rf'AICLS_GATE_{gate}:\s*""', text), (
            f"{gate} must be explicitly empty in CI"
        )
    assert "approved" not in text
    for step in (
        "poe lint",
        "poe type",
        "poe imports",
        "poe gates",
        "poe test",
        "coverage_gate.py",
        "mkdocs build --strict",
        "check_docs_map.py",
        "check_gates.py",
    ):
        assert step in text, step
    assert "windows-latest" in text and "ubuntu-latest" in text


def test_coverage_gate_reads_tiers_and_evaluates(
    load_script: Callable[[str], types.ModuleType], repo_root: Path, tmp_path: Path
) -> None:
    gate = load_script("scripts/coverage_gate.py")
    tiers = gate.load_thresholds(repo_root / "pyproject.toml")
    assert tiers["overall"] == 85 and tiers["aicls.latex2sympy"] == 95 and tiers["aicls.ocr"] == 70
    xml = tmp_path / "coverage.xml"
    xml.write_text(
        '<?xml version="1.0" ?><coverage><packages><package name="aicls.core">'
        '<classes><class filename="src/aicls/core/a.py"><lines>'
        '<line number="1" hits="1"/><line number="2" hits="1"/><line number="3" hits="0"/>'
        "</lines></class></classes></package></packages></coverage>\n",
        encoding="utf-8",
    )
    stats = gate.collect_module_stats(xml)
    assert stats["aicls.core.a"].valid == 3 and stats["aicls.core.a"].covered == 2
    _, failures = gate.evaluate(
        stats, {"overall": 50, "aicls.core": 90, "aicls.nlu": 95}, require_all=False
    )
    assert failures == ["aicls.core: 66.7% < 90.0%"]
    _, failures = gate.evaluate(stats, {"aicls.nlu": 95}, require_all=True)
    assert failures and "no measured lines" in failures[0]


def test_lint_no_raw_sql_flags_text_and_allows_typed_api(
    load_script: Callable[[str], types.ModuleType], tmp_path: Path
) -> None:
    lint = load_script("tools/lint_no_raw_sql.py")
    bad = tmp_path / "repo.py"
    bad.write_text(
        'from sqlalchemy import text\nrows = session.execute(text("SELECT * FROM users WHERE name = :n"))\n'
        'session.execute("DELETE FROM users")\n',
        encoding="utf-8",
    )
    problems = lint.scan_text(bad.read_text(encoding="utf-8"), Path("repo.py"))
    assert len(problems) == 2
    good = 'stmt = select(User).where(User.name == name)\nlog.info("Update cache for %s", name)\n'
    assert lint.scan_text(good, Path("ok.py")) == []
    mig = tmp_path / "migrations" / "versions" / "0001_x.py"
    assert lint.is_exempt(mig)


def test_docs_map_tool_parses_table_and_passes(
    load_script: Callable[[str], types.ModuleType], repo_root: Path
) -> None:
    tool = load_script("tools/check_docs_map.py")
    rows = tool.parse_rows(
        (repo_root / "docs" / "raporlama" / "ip-eslestirme.md").read_text(encoding="utf-8")
    )
    assert len(rows) >= 20
    assert any(r.planned for r in rows)
    assert any("src/aicls/latex2sympy/" in r.paths for r in rows)
    assert tool.main(["--root", str(repo_root)]) == 0
