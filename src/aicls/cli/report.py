"""``aicls report`` — the gate between a measurement and a citable number (ADR-0015 §4).

``reports/metrics/<name>_<gitsha7>.json`` is a measurement. ``reports/metrics/latest.json`` is
what the MkDocs macros read, so a number reaches the documentation only by being *promoted* into
it, and ``check`` re-verifies everything that has been:

* the commit must be ``HEAD`` or an ancestor of it — a number measured on a branch that was never
  merged describes code nobody can check out;
* the tree must have been clean and the report must say it is citable;
* the normaliser versions must still be the current ones, or the metric definition has moved
  under the number;
* the definitions reference must be there.

``render`` regenerates the Markdown table from the report files, so no number in the
documentation is ever typed by a human.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from aicls.cli.common import cli_logger, emit, fail, use_json

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aicls.eval.report import EvalReport

app = typer.Typer(
    name="report",
    help="Check, promote and render the evaluation reports (ADR-0015).",
    no_args_is_help=True,
)

#: File names under ``reports/metrics`` that are not evaluation reports.
NON_REPORT_NAMES = ("latest.json",)

#: Prefix of the throughput reports ``aicls profile`` writes into the same directory.
PROFILE_PREFIX = "profile_"


def _reports_dir(given: Path | None) -> Path:
    """``reports/metrics`` of the checkout, unless the caller named a directory."""
    from aicls.core.paths import find_repo_root
    from aicls.eval.report import DEFAULT_OUT_DIR

    return Path(given) if given is not None else find_repo_root() / DEFAULT_OUT_DIR


def load_reports(directory: Path) -> list[EvalReport]:
    """Load every :class:`aicls.eval.EvalReport` in ``directory``, in file order.

    ``latest.json`` and the ``profile_*.json`` throughput reports are skipped, and a file that
    does not validate is logged and skipped rather than aborting the render: an unreadable file
    must not hide the reports that are readable.
    """
    from aicls.eval.report import EvalReport

    reports: list[EvalReport] = []
    for path in sorted(Path(directory).glob("*.json")):
        if path.name in NON_REPORT_NAMES or path.name.startswith(PROFILE_PREFIX):
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            reports.append(EvalReport.model_validate(document))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            cli_logger(__name__).warning(
                "cli.report.unreadable", path=str(path), error=str(exc)[:200]
            )
    return reports


def _load_named(paths: Sequence[Path]) -> list[EvalReport]:
    """Load the named report files.

    Raises:
        aicls.core.errors.NotFoundError: If one of them does not exist.
        aicls.core.errors.InputValidationError: If one is not a valid report document — a file
            that cannot be validated must never be promoted.
    """
    from aicls.core.errors import InputValidationError, NotFoundError
    from aicls.core.messages_tr import MSG_VALIDATION_ERROR, msg
    from aicls.eval.report import EvalReport

    reports: list[EvalReport] = []
    for path in paths:
        target = Path(path)
        if not target.is_file():
            raise NotFoundError(
                msg("FILE_NOT_FOUND", path=str(target)), details={"path": str(target)}
            )
        try:
            reports.append(
                EvalReport.model_validate(json.loads(target.read_text(encoding="utf-8")))
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise InputValidationError(
                MSG_VALIDATION_ERROR,
                details={"reason": "report_invalid", "path": str(target), "error": str(exc)[:200]},
            ) from exc
    return reports


@app.command("check")
def check(
    latest: Annotated[
        Path | None,
        typer.Option("--latest", help="Pointer file; reports/metrics/latest.json by default."),
    ] = None,
    reports_dir: Annotated[
        Path | None, typer.Option("--reports-dir", help="Report directory.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Re-verify every promoted number; exit ``1`` when one may not be cited.

    The ancestry is read from ``.git`` without running ``git`` (:func:`aicls.eval.known_shas`), so
    the check works offline. That set is an approximation of ``git rev-list HEAD`` — it holds what
    this checkout recorded — which is why a CI job passes the real ancestry instead.

    A promoted **wiring check** is refused here as well: its metric values are already ``null``,
    but the entry itself must not be allowed to stand in the file the documentation reads
    (:func:`aicls.eval.report.check_promoted`). The refused kinds are listed as
    ``wiring_checks`` in the ``--json`` payload.
    """
    from aicls.core.paths import find_repo_root
    from aicls.eval.provenance import known_shas, read_head
    from aicls.eval.report import LATEST_FILENAME, WIRING_CHECK, check_promoted, load_latest

    use_json(json_output)
    root = find_repo_root()
    directory = _reports_dir(reports_dir)
    pointer = Path(latest) if latest is not None else directory / LATEST_FILENAME
    promoted = load_latest(pointer)
    head_sha, branch = read_head(root)
    ancestors = known_shas(root)
    problems = check_promoted(promoted, head_sha=head_sha, ancestors=ancestors)
    wiring = sorted(
        kind
        for kind, entry in promoted.items()
        if isinstance(entry, dict) and entry.get("measurement") == WIRING_CHECK
    )
    payload = {
        "latest": pointer.as_posix(),
        "head_sha": head_sha,
        "branch": branch,
        "known_commits": len(ancestors),
        "kinds": sorted(promoted),
        "wiring_checks": wiring,
        "problems": problems,
        "ok": not problems,
    }
    lines = [
        f"{pointer} ({len(promoted)} promoted kinds), HEAD {head_sha[:7] or 'unknown'}",
        *(f"  {problem}" for problem in problems),
    ]
    if problems:
        fail(payload, *lines, f"report-check FAILED with {len(problems)} problem(s)")
    emit(payload, *lines, "report-check passed")


@app.command("promote")
def promote(
    files: Annotated[
        list[Path], typer.Argument(help="Report files to promote, one per evaluation kind.")
    ],
    out: Annotated[
        Path | None, typer.Option("--out", help="Directory holding latest.json.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Merge the given reports into ``latest.json`` — the file the documentation reads.

    Raises:
        aicls.core.errors.InputValidationError: If two files carry the same evaluation kind;
            keeping the last one silently would publish a number nobody chose.
    """
    from aicls.eval.report import promote as do_promote

    use_json(json_output)
    directory = _reports_dir(out)
    reports = _load_named(files)
    path = do_promote(reports, directory)
    payload = {
        "path": path.as_posix(),
        "kinds": [report.kind for report in reports],
        "citable": {report.kind: report.citable for report in reports},
        "count": len(reports),
    }
    emit(
        payload,
        *(
            f"{report.kind}: {report.name} ({'citable' if report.citable else 'NOT citable'})"
            for report in reports
        ),
        f"wrote {path}",
    )


@app.command("render")
def render(
    reports_dir: Annotated[
        Path | None, typer.Option("--reports-dir", help="Directory holding the report files.")
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Markdown file; <reports-dir>/summary.md.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Regenerate the Markdown results table from the report files."""
    from aicls.eval.report import render_markdown

    use_json(json_output)
    directory = _reports_dir(reports_dir)
    reports = load_reports(directory)
    target = Path(out) if out is not None else directory / "summary.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    text = render_markdown(reports)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    payload: dict[str, Any] = {
        "reports_dir": directory.as_posix(),
        "path": target.as_posix(),
        "reports": [
            {"name": report.name, "kind": report.kind, "n": report.n} for report in reports
        ],
        "count": len(reports),
    }
    emit(payload, f"rendered {len(reports)} report(s) into {target}")


__all__ = [
    "NON_REPORT_NAMES",
    "PROFILE_PREFIX",
    "app",
    "check",
    "load_reports",
    "promote",
    "render",
]
