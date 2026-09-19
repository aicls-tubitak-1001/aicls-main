"""``aicls report``: the gate between a measurement and a citable number (ADR-0015 §4).

``check`` is the one that has to be strict. A number measured on a commit that is not an ancestor
of ``HEAD`` describes code nobody can check out, so it must fail — and it must fail loudly enough
that the reason appears in the output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.unit.cli.conftest import payload_of

pytestmark = pytest.mark.usefixtures("cli_env")

#: A well-formed commit hash that no checkout has ever pointed at.
ORPHAN_SHA = "0123456789abcdef0123456789abcdef01234567"


def _report(**overrides: object) -> object:
    """A minimal, valid, **reportable** :class:`aicls.eval.EvalReport`.

    ``n`` clears :data:`aicls.eval.MIN_VAL_SAMPLES` because most tests here promote this report
    and then read a number back out of ``latest.json``. A report that cannot be reported
    publishes ``null`` for every metric, which would make those assertions test the wiring-check
    path by accident. ``n: 10`` used to be enough only because a finished report never judged
    its own sample count; the contract asks for CAS accuracy over at least a thousand problems.
    """
    from aicls.eval.report import EvalReport

    fields: dict[str, object] = {
        "name": "cas_curriculum",
        "kind": "cas",
        "git_sha": ORPHAN_SHA,
        "dirty": False,
        "citable": True,
        "dataset_fingerprint": "sha256:" + "a" * 64,
        "dataset_version": "curriculum-2026.09-v1",
        "n": 1000,
        "metrics": {"correct_rate": 1.0},
    }
    fields.update(overrides)
    return EvalReport(**fields)  # type: ignore[arg-type]  # a test fixture, validated by pydantic


def test_cli_report_check_fails_on_non_ancestor(
    runner: CliRunner, app_module: object, tmp_path: Path
) -> None:
    """A promoted number whose commit is not an ancestor of HEAD may not be cited."""
    from aicls.eval.report import promote

    promote([_report()], tmp_path)
    result = runner.invoke(
        app_module, ["report", "check", "--reports-dir", str(tmp_path), "--json"]
    )
    assert result.exit_code == 1
    payload = payload_of(result.stdout)
    assert payload["ok"] is False
    assert any("not an ancestor" in problem for problem in payload["problems"])


def test_report_check_fails_on_a_dirty_run(
    runner: CliRunner, app_module: object, tmp_path: Path
) -> None:
    """A dirty tree means the measured code is not what is committed."""
    from aicls.eval.report import promote

    promote([_report(dirty=True, citable=True)], tmp_path)
    result = runner.invoke(
        app_module, ["report", "check", "--reports-dir", str(tmp_path), "--json"]
    )
    assert result.exit_code == 1
    problems = payload_of(result.stdout)["problems"]
    assert any("dirty working tree" in problem for problem in problems)
    assert any("citable is false" in problem for problem in problems)


def test_report_check_passes_on_an_empty_pointer(
    runner: CliRunner, app_module: object, tmp_path: Path
) -> None:
    """Nothing promoted yet is a green build, not a failure (CI must stay usable)."""
    result = runner.invoke(
        app_module, ["report", "check", "--reports-dir", str(tmp_path), "--json"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["ok"] is True
    assert payload["kinds"] == []


def test_report_promote_writes_latest(
    runner: CliRunner, app_module: object, tmp_path: Path
) -> None:
    """Promotion is the only way a number reaches ``latest.json``."""
    from aicls.eval.report import write_report

    path = write_report(_report(), tmp_path)  # type: ignore[arg-type]
    result = runner.invoke(
        app_module, ["report", "promote", str(path), "--out", str(tmp_path), "--json"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["kinds"] == ["cas"]
    document = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert document["cas"]["dataset_fingerprint"].startswith("sha256:")
    assert document["cas"]["correct_rate"] == 1.0


def test_report_promote_refuses_two_reports_of_one_kind(
    runner: CliRunner, app_module: object, tmp_path: Path
) -> None:
    """Two reports of the same kind: keeping the last one silently would publish a guess."""
    from aicls.eval.report import write_report

    first = write_report(_report(name="cas_a"), tmp_path)  # type: ignore[arg-type]
    second = write_report(_report(name="cas_b"), tmp_path)  # type: ignore[arg-type]
    result = runner.invoke(
        app_module, ["report", "promote", str(first), str(second), "--out", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "VALIDATION_ERROR" in result.stderr


def test_report_promote_refuses_a_missing_file(
    runner: CliRunner, app_module: object, tmp_path: Path
) -> None:
    """A path typo must not quietly promote nothing."""
    result = runner.invoke(
        app_module, ["report", "promote", str(tmp_path / "nope.json"), "--out", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "NOT_FOUND" in result.stderr


def test_report_render_writes_the_generated_table(
    runner: CliRunner, app_module: object, tmp_path: Path
) -> None:
    """The Markdown table is generated from the report files; no number is typed by hand."""
    from aicls.eval.report import write_report

    write_report(_report(), tmp_path)  # type: ignore[arg-type]
    result = runner.invoke(
        app_module, ["report", "render", "--reports-dir", str(tmp_path), "--json"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["count"] == 1
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "correct_rate" in text


def test_report_render_skips_profile_and_pointer_files(tmp_path: Path) -> None:
    """``profile_*.json`` (throughput) and ``latest.json`` are not evaluation reports."""
    from aicls.cli.report import load_reports
    from aicls.eval.report import promote, write_report

    write_report(_report(), tmp_path)  # type: ignore[arg-type]
    promote([_report()], tmp_path)  # type: ignore[list-item]
    (tmp_path / "profile_ocr.json").write_text('{"steps_per_s": 1.0}', encoding="utf-8")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    reports = load_reports(tmp_path)
    assert [report.kind for report in reports] == ["cas"]


def test_report_check_refuses_a_promoted_wiring_check(
    runner: CliRunner, app_module: object, tmp_path: Path
) -> None:
    """A wiring check may sit in a run directory; it may not sit in the file the docs read.

    Its metric values are already ``null``, but the entry itself has to be refused, and the
    refused kinds have to be named in the payload so an operator can see which they are.
    """
    from aicls.eval.report import assess_wiring_check, promote

    verdict = assess_wiring_check(steps=20, warmup_steps=2000, n_val=16)
    promote(
        [
            _report(
                kind="ocr",
                name="ocr_excerpt",
                metrics={"cer": 40.4},
                breakdown={"wiring_check": verdict.to_dict()},
            )
        ],
        tmp_path,
    )
    result = runner.invoke(
        app_module, ["report", "check", "--reports-dir", str(tmp_path), "--json"]
    )

    assert result.exit_code == 1
    payload = payload_of(result.stdout)
    assert payload["wiring_checks"] == ["ocr"]
    assert any("wiring check" in problem for problem in payload["problems"])
    assert any("step_budget_too_small" in problem for problem in payload["problems"])


def test_report_check_names_no_wiring_check_for_a_measured_run(
    runner: CliRunner, app_module: object, tmp_path: Path
) -> None:
    """The other direction: a measured entry is never listed as a wiring check."""
    from aicls.eval.report import promote

    promote([_report()], tmp_path)
    result = runner.invoke(
        app_module, ["report", "check", "--reports-dir", str(tmp_path), "--json"]
    )

    payload = payload_of(result.stdout)
    assert payload["wiring_checks"] == []
    assert not any("wiring check" in problem for problem in payload["problems"])
