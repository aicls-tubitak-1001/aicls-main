"""Approval gates (ADR-0012): triple lock, ledger parsing, invocation log."""

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path

import pytest

from aicls.core.errors import AiclsError, ErrorCode
from aicls.core.gates import (
    GATE_NAMES,
    GateEntry,
    GateError,
    GateLedgerError,
    is_gate_approved_in_ledger,
    read_gate_ledger,
    require_gate,
)
from aicls.core.settings import Settings, get_settings

LEDGER_HEAD = (
    "# Onay Defteri\n\nintro\n\n| KAPI | durum | tarih | kim | kapsam |\n|---|---|---|---|---|\n"
)


def _ledger(tmp_path: Path, rows: str) -> Path:
    p = tmp_path / "onay-defteri.md"
    p.write_text(LEDGER_HEAD + rows, encoding="utf-8")
    return p


@pytest.fixture
def approved_ledger(tmp_path: Path) -> Path:
    return _ledger(
        tmp_path,
        "| TRAINING | approved | 2027-01-15 | csahin | e-posta 2027-01-14; MathWriting v2024.09 |\n",
    )


@pytest.fixture
def gate_env(test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    get_settings.cache_clear()
    yield Settings()
    get_settings.cache_clear()


# ---------------------------------------------------------------------------------------------
# ledger parsing
# ---------------------------------------------------------------------------------------------
#: Words that turn a scope into a blanket permission. A row an auditor cannot check is worse
#: than no row, because it looks like process while granting everything.
_UNBOUNDED_SCOPE_WORDS = ("serbest", "sınırsız", "sinirsiz", "her şey", "hepsi", "unlimited", "any")


def test_every_approval_in_the_ledger_is_bounded_and_attributable() -> None:
    """An approval must say where it came from and where it stops.

    All four gates now carry an approval, so "this gate is closed" is no longer the invariant to
    protect. What survives, and what the ledger format exists for, is that each row can be
    audited: it names the person who recorded it, cites the instruction it rests on, and states
    limits. The coordinator's approval for WEIGHTS_DOWNLOAD was given verbally as "serbest" -
    free - and that word is exactly what may not end up in the scope column, because a permission
    with no stated boundary cannot be reviewed, revoked in part, or checked against what was
    actually done.
    """
    entries = read_gate_ledger()
    assert all(isinstance(e, GateEntry) for e in entries)
    for entry in entries:
        if entry.status != "approved":
            continue
        assert entry.who.strip(), f"{entry.gate} was approved by nobody"
        scope = entry.scope.strip()
        assert "onay:" in scope.lower(), f"{entry.gate} scope does not cite its source"
        assert len(scope) > 60, f"{entry.gate} scope is too thin to audit: {scope!r}"
        # Only the LIMITS half is checked for blanket words. The citation half legitimately
        # quotes what was said, and what was said may well have been "serbest"; the row's job is
        # to turn that into something bounded, not to hide it.
        limits = scope.lower().split("onay:")[0]
        bare = {w.strip(" .,;:\"'()") for w in limits.split()}
        blanket = [w for w in _UNBOUNDED_SCOPE_WORDS if w in bare]
        assert not blanket, f"{entry.gate} scope is unbounded ({blanket}); state the limits"


def test_read_gate_ledger_parses_rows(approved_ledger: Path) -> None:
    entries = read_gate_ledger(approved_ledger)
    assert entries == [
        GateEntry(
            gate="TRAINING",
            status="approved",
            date="2027-01-15",
            who="csahin",
            scope="e-posta 2027-01-14; MathWriting v2024.09",
        )
    ]
    assert is_gate_approved_in_ledger("TRAINING", entries)
    assert not is_gate_approved_in_ledger("CUDA_INSTALL", entries)


def test_ledger_latest_row_wins_revoked(tmp_path: Path) -> None:
    p = _ledger(
        tmp_path,
        "| TRAINING | approved | 2027-01-15 | csahin | ok |\n"
        "| TRAINING | revoked | 2027-02-01 | csahin | iptal |\n",
    )
    assert not is_gate_approved_in_ledger("TRAINING", read_gate_ledger(p))


def test_ledger_stops_at_first_non_table_line(tmp_path: Path) -> None:
    p = _ledger(
        tmp_path,
        "| TRAINING | approved | 2027-01-15 | csahin | ok |\n\nSonraki paragraf | değil |\n",
    )
    assert len(read_gate_ledger(p)) == 1


@pytest.mark.parametrize(
    "row",
    [
        "| FOO | approved | 2027-01-15 | csahin | ok |\n",
        "| TRAINING | maybe | 2027-01-15 | csahin | ok |\n",
        "| TRAINING | approved | 15.01.2027 | csahin | ok |\n",
        "| TRAINING | approved | 2027-01-15 |  | ok |\n",
        "| TRAINING | approved | 2027-01-15 | csahin |  |\n",
        "| TRAINING | approved | 2027-01-15 | csahin |\n",
    ],
)
def test_ledger_malformed_rows_raise(tmp_path: Path, row: str) -> None:
    p = _ledger(tmp_path, row)
    with pytest.raises(GateLedgerError) as ei:
        read_gate_ledger(p)
    assert ei.value.code is ErrorCode.VALIDATION_ERROR


def test_ledger_missing_header_raises(tmp_path: Path) -> None:
    p = tmp_path / "x.md"
    p.write_text("# nothing here\n", encoding="utf-8")
    with pytest.raises(GateLedgerError):
        read_gate_ledger(p)


def test_ledger_missing_separator_raises(tmp_path: Path) -> None:
    p = tmp_path / "x.md"
    p.write_text("| KAPI | durum | tarih | kim | kapsam |\n| a | b |\n", encoding="utf-8")
    with pytest.raises(GateLedgerError):
        read_gate_ledger(p)


def test_ledger_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AiclsError) as ei:
        read_gate_ledger(tmp_path / "absent.md")
    assert ei.value.code is ErrorCode.NOT_FOUND


# ---------------------------------------------------------------------------------------------
# triple lock
# ---------------------------------------------------------------------------------------------
def test_gate_names() -> None:
    assert GATE_NAMES == ("DATASET_DOWNLOAD", "WEIGHTS_DOWNLOAD", "CUDA_INSTALL", "TRAINING")


def test_require_gate_refuses_without_flag(
    gate_env: Settings, approved_ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AICLS_GATE_TRAINING", "approved")
    with pytest.raises(GateError) as ei:
        require_gate("TRAINING", cli_flag=False, ledger_path=approved_ledger)
    err = ei.value
    assert err.exit_code == 3
    assert err.code is ErrorCode.GATE_BLOCKED
    assert err.http_status == 503
    assert err.details["gate"] == "TRAINING"
    assert err.details["missing"] == ["cli_flag"]
    assert "--i-have-approval" in err.message_tr


def test_require_gate_refuses_without_env(gate_env: Settings, approved_ledger: Path) -> None:
    with pytest.raises(GateError) as ei:
        require_gate("TRAINING", cli_flag=True, ledger_path=approved_ledger)
    assert ei.value.details["missing"] == ["env"]
    assert "AICLS_GATE_TRAINING" in ei.value.message_tr


def test_require_gate_refuses_env_value_other_than_approved(
    gate_env: Settings, approved_ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AICLS_GATE_TRAINING", "yes")
    with pytest.raises(GateError) as ei:
        require_gate("TRAINING", cli_flag=True, ledger_path=approved_ledger)
    assert ei.value.details["missing"] == ["env"]


def test_require_gate_refuses_without_ledger_row(
    gate_env: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AICLS_GATE_TRAINING", "approved")
    empty = _ledger(tmp_path, "")
    with pytest.raises(GateError) as ei:
        require_gate("TRAINING", cli_flag=True, ledger_path=empty)
    assert ei.value.details["missing"] == ["ledger"]
    assert "onay-defteri.md" in ei.value.message_tr


def test_require_gate_refuses_with_all_three_missing_and_lists_them(
    gate_env: Settings, tmp_path: Path
) -> None:
    empty = _ledger(tmp_path, "")
    with pytest.raises(GateError) as ei:
        require_gate("DATASET_DOWNLOAD", cli_flag=False, ledger_path=empty)
    assert ei.value.details["missing"] == ["cli_flag", "env", "ledger"]
    # the message enumerates all three conditions (ADR-0012: eksik koşullar mesajda)
    m = ei.value.message_tr
    assert "--i-have-approval" in m and "AICLS_GATE_DATASET_DOWNLOAD" in m and "onay-defteri" in m


def test_require_gate_refuses_against_repo_ledger_by_default(
    gate_env: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every gate in the committed ledger is now approved, so this can no longer be tested by
    # picking an unapproved one. Point the DEFAULT ledger path at an empty ledger instead: that
    # tests the mechanism - require_gate consults the repository ledger when none is passed -
    # rather than testing today's contents, which is what made the old version go stale.
    empty = _ledger(tmp_path, "")
    monkeypatch.setattr("aicls.core.gates.default_ledger_path", lambda: empty)
    monkeypatch.setenv("AICLS_GATE_CUDA_INSTALL", "approved")
    with pytest.raises(GateError) as ei:
        require_gate("CUDA_INSTALL", cli_flag=True)
    assert "ledger" in ei.value.details["missing"]


def test_require_gate_refuses_in_ci_even_when_all_three_hold(
    gate_env: Settings, approved_ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AICLS_GATE_TRAINING", "approved")
    monkeypatch.setenv("CI", "true")
    with pytest.raises(GateError) as ei:
        require_gate("TRAINING", cli_flag=True, ledger_path=approved_ledger)
    assert ei.value.details["missing"] == ["ci"]


def test_require_gate_passes_with_triple_lock_and_logs_invocation(
    gate_env: Settings, approved_ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AICLS_GATE_TRAINING", "approved")
    require_gate("TRAINING", cli_flag=True, ledger_path=approved_ledger)
    log = Path(os.environ["AICLS_DATA_ROOT"]) / "gate_invocations.log"
    assert log.is_file()
    line = log.read_text(encoding="utf-8").strip().splitlines()[-1]
    assert '"gate": "TRAINING"' in line
    assert '"who"' in line and '"at"' in line and '"config_hash"' in line and '"git_sha"' in line
    # second invocation appends
    require_gate("TRAINING", cli_flag=True, ledger_path=approved_ledger)
    assert len(log.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_require_gate_accepts_explicit_settings(
    gate_env: Settings, approved_ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AICLS_GATE_TRAINING", "approved")
    custom = Settings()
    require_gate("TRAINING", cli_flag=True, ledger_path=approved_ledger, settings=custom)


def test_require_gate_unknown_name_rejected(gate_env: Settings, approved_ledger: Path) -> None:
    with pytest.raises(AiclsError) as ei:
        require_gate("DELETE_EVERYTHING", cli_flag=True, ledger_path=approved_ledger)  # type: ignore[arg-type]  # deliberate misuse
    assert ei.value.code is ErrorCode.VALIDATION_ERROR


def test_gate_error_problem_details(gate_env: Settings, tmp_path: Path) -> None:
    empty = _ledger(tmp_path, "")
    with pytest.raises(GateError) as ei:
        require_gate("WEIGHTS_DOWNLOAD", cli_flag=False, ledger_path=empty)
    problem = ei.value.to_problem()
    assert problem["status"] == 503
    assert problem["code"] == "GATE_BLOCKED"
    assert problem["gate"] == "WEIGHTS_DOWNLOAD"


# ---------------------------------------------------------------------------------------------
# invocation-log helpers (best-effort git SHA without spawning git)
# ---------------------------------------------------------------------------------------------
def test_git_sha_from_detached_head(tmp_path: Path) -> None:
    from aicls.core.gates import _git_sha

    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("abc123\n", encoding="utf-8")
    assert _git_sha(tmp_path) == "abc123"


def test_git_sha_from_ref_file_and_packed_refs(tmp_path: Path) -> None:
    from aicls.core.gates import _git_sha

    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git / "refs" / "heads" / "main").write_text("deadbeef\n", encoding="utf-8")
    assert _git_sha(tmp_path) == "deadbeef"
    (git / "refs" / "heads" / "main").unlink()
    (git / "packed-refs").write_text(
        "# pack-refs\ncafebabe refs/heads/main\n0000 refs/heads/other\n", encoding="utf-8"
    )
    assert _git_sha(tmp_path) == "cafebabe"
    (git / "packed-refs").write_text("0000 refs/heads/other\n", encoding="utf-8")
    assert _git_sha(tmp_path) is None


def test_git_sha_without_git_dir(tmp_path: Path) -> None:
    from aicls.core.gates import _git_sha

    assert _git_sha(tmp_path) is None


def test_is_ci_environment_reads_mapping() -> None:
    from aicls.core.gates import is_ci_environment

    assert is_ci_environment({"CI": "true"})
    assert is_ci_environment({"GITHUB_ACTIONS": "1"})
    assert not is_ci_environment({"CI": "false"})
    assert not is_ci_environment({})
