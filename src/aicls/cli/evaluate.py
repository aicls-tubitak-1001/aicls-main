"""``aicls eval`` — run one evaluation and persist it (ADR-0015, docs/ml/degerlendirme.md).

This is the only place where :mod:`aicls.eval` and :mod:`aicls.db` meet. ``aicls.eval`` may not
import ``aicls.db`` for the numbers it produces and the API may not import ``aicls.eval`` at all;
the CLI sits above both, so the write into ``eval_runs`` happens here.

What the write must preserve, because a number without them is worthless in a progress report:

* ``git_sha`` and ``dirty`` — which commit produced the number and whether the tree was clean;
* ``dataset_fingerprint`` and ``dataset_version`` — which bytes it was measured on;
* ``citable`` — stored as ``eval_runs.is_reportable``. It is **recomputed** by
  :class:`aicls.eval.EvalReport` (a dirty or commit-less run is never citable), so the flag in the
  database is the report's own verdict, never the caller's wish.

``aicls eval`` never promotes. ``latest.json`` and ``summary.md`` are what the MkDocs macros read,
and only a reviewer moves a number there — with ``aicls report promote``. The harness writes both
as part of a run, so this command restores them afterwards unless ``--promote`` is given.

``--model`` is not cosmetic. It runs the OCR/ASR evaluation against the checkpoint the
:class:`aicls.api.model_registry.ModelRegistry` actually serves, over the same synthetic corpus
the harness uses, and the report keeps whatever ``model_version`` that registry reports. When no
checkpoint is promoted (the state today, with the download and training gates closed) the
registry hands back the deterministic stub and the report says ``stub: true`` — the flag never
turns a stub run into a model run on paper.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import datetime as dt
from enum import StrEnum
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Annotated, Any

import typer

from aicls.cli.common import cli_logger, db_session, emit, resolve_db_url, use_json

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aicls.eval.report import EvalReport

HELP = "Run an evaluation, write its report and record it in eval_runs."

#: Model-registry component behind each evaluation kind that can be model-backed.
MODEL_COMPONENTS: dict[str, str] = {"ocr": "ocr", "asr": "asr"}

#: Splits ``--split`` accepts.
SPLIT_NAMES = ("train", "valid", "test")

#: Default split when a model-backed run does not name one.
DEFAULT_SPLIT = "test"


class EvalKindChoice(StrEnum):
    """Evaluation kinds ``aicls eval`` accepts (``all`` runs the whole harness)."""

    PARSER = "parser"
    CAS = "cas"
    NLU = "nlu"
    OCR = "ocr"
    ASR = "asr"
    BOOK = "book"
    LATENCY = "latency"
    ALL = "all"


@contextmanager
def preserved(paths: Sequence[Path]) -> Iterator[None]:
    """Restore ``paths`` to their previous content (or absence) when the block ends.

    Used to keep ``reports/metrics/latest.json`` and ``summary.md`` out of an unreviewed run:
    the harness always writes them, and promotion is a separate, deliberate decision.
    """
    saved = {path: (path.read_bytes() if path.is_file() else None) for path in paths}
    try:
        yield
    finally:
        for path, content in saved.items():
            if content is None:
                path.unlink(missing_ok=True)
            elif path.read_bytes() != content:
                path.write_bytes(content)


def report_payload(report: EvalReport, path: Path | None) -> dict[str, Any]:
    """The JSON shape ``aicls eval`` prints for one report."""
    return {
        "name": report.name,
        "kind": report.kind,
        "n": report.n,
        "metrics": dict(report.metrics),
        "ci95": {key: list(value) for key, value in report.ci95.items()},
        "git_sha": report.git_sha,
        "dirty": report.dirty,
        "citable": report.citable,
        "dataset_fingerprint": report.dataset_fingerprint,
        "dataset_version": report.dataset_version,
        "model_version": report.model_version,
        "normalizer_versions": dict(report.normalizer_versions),
        "definitions_ref": report.definitions_ref,
        "stub": bool(report.breakdown.get("stub", False)),
        "notes": report.breakdown.get("notes", ""),
        "path": path.as_posix() if path is not None else None,
    }


def evaluate(
    kind: Annotated[EvalKindChoice, typer.Argument(help="Which evaluation to run.")],
    *,
    model: Annotated[
        str,
        typer.Option("--model", help="Measure the promoted ocr/asr checkpoint of this version."),
    ] = "",
    split: Annotated[
        str,
        typer.Option(
            "--split",
            help="Label the model-backed report with this split name (ocr/asr only). It does "
            "NOT choose the data: the harness corpora are fixed. To measure a real held-out "
            "split, use training/scripts/evaluate.py --shards ... --split test.",
        ),
    ] = "",
    out: Annotated[
        Path | None, typer.Option("--out", help="Report directory; reports/metrics by default.")
    ] = None,
    from_db: Annotated[
        bool, typer.Option("--from-db", help="List recorded runs instead of running anything.")
    ] = False,
    since: Annotated[
        str, typer.Option("--since", help="With --from-db: ISO date/time lower bound.")
    ] = "",
    limit: Annotated[int, typer.Option("--limit", help="With --from-db: maximum rows.")] = 20,
    db_url: Annotated[str, typer.Option("--db-url", help="Database URL; AICLS_DB_URL.")] = "",
    persist: Annotated[
        bool, typer.Option("--db/--no-db", help="Record the run in the eval_runs table.")
    ] = True,
    promote: Annotated[
        bool,
        typer.Option("--promote", help="Also rewrite latest.json (see: aicls report promote)."),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Run an evaluation and record it, or list what has been recorded.

    Raises:
        aicls.core.errors.InputValidationError: If ``--split`` or ``--model`` is given for a kind
            that has no such notion, or the requested version is not the promoted one.
        aicls.core.errors.NotFoundError: If ``--model`` names an unregistered version.
    """
    from aicls.core.paths import find_repo_root
    from aicls.eval.harness import KINDS, default_paths, run_all

    use_json(json_output)
    if from_db:
        _list_runs(kind, since=since, limit=limit, db_url=db_url)
        return

    selected = tuple(KINDS) if kind is EvalKindChoice.ALL else (kind.value,)
    _check_flags(selected, model=model, split=split)
    root = find_repo_root()
    target = Path(out) if out is not None else default_paths(root)["out_dir"]
    target.mkdir(parents=True, exist_ok=True)
    keep = () if promote else (target / "latest.json", target / "summary.md")

    if model:
        report, path = _run_model_backed(
            selected[0],
            root=root,
            out_dir=target,
            split=split or DEFAULT_SPLIT,
            model=model,
            db_url=db_url,
        )
        reports: list[EvalReport] = [report]
        files: list[Path | None] = [path]
    else:
        with preserved(keep):
            output = run_all(root=root, out_dir=target, kinds=selected, write=True)
        reports = list(output.reports)
        files = list(output.files) or [None] * len(reports)

    stored = _persist(reports, files, db_url=db_url) if persist else []
    payload = {
        "kind": kind.value,
        "out": target.as_posix(),
        "promoted": bool(promote),
        "requested": {"model": model or None, "split": split or None},
        "reports": [
            report_payload(report, path) for report, path in zip(reports, files, strict=True)
        ],
        "eval_runs": stored,
    }
    emit(payload, *_lines(payload))


def _lines(payload: dict[str, Any]) -> list[str]:
    """Human-readable rendering of a completed evaluation."""
    lines: list[str] = []
    for entry in payload["reports"]:
        stub = " [STUB: wiring check, not evidence]" if entry["stub"] else ""
        cite = "citable" if entry["citable"] else "NOT citable"
        lines.append(f"{entry['kind']}: n={entry['n']} {entry['metrics']}{stub}")
        lines.append(f"  {cite}, fingerprint {entry['dataset_fingerprint'] or '(none)'}")
        if entry["path"]:
            lines.append(f"  wrote {entry['path']}")
    for row in payload["eval_runs"]:
        lines.append(f"  eval_runs id={row['id']} is_reportable={row['is_reportable']}")
    return lines


def _check_flags(kinds: Sequence[str], *, model: str, split: str) -> None:
    """Refuse ``--model``/``--split`` where they have no meaning.

    A flag that silently does nothing is worse than one that says so: these numbers end up in a
    progress report, and "I passed ``--split test``" must never be a false memory.
    """
    from aicls.core.errors import InputValidationError
    from aicls.core.messages_tr import MSG_VALIDATION_ERROR

    unsupported = [name for name in kinds if name not in MODEL_COMPONENTS]
    if (model or split) and unsupported:
        raise InputValidationError(
            MSG_VALIDATION_ERROR,
            details={
                "reason": "model_not_applicable" if model else "split_not_applicable",
                "kinds": unsupported,
                "supported": sorted(MODEL_COMPONENTS),
            },
        )
    if split and split not in SPLIT_NAMES:
        raise InputValidationError(
            MSG_VALIDATION_ERROR,
            details={"reason": "unknown_split", "split": split, "expected": list(SPLIT_NAMES)},
        )
    if split and not model:
        raise InputValidationError(
            MSG_VALIDATION_ERROR,
            details={
                "reason": "split_requires_model",
                "hint": "the harness corpora are fixed; --split selects a model-backed run",
            },
        )


def _require_promoted(component: str, version: str, db_url: str) -> None:
    """Check that ``version`` is registered **and** promoted for ``component``.

    Raises:
        aicls.core.errors.NotFoundError: If no such version is registered.
        aicls.core.errors.InputValidationError: If it is registered but not active — the request
            path and the evaluation must not disagree about which checkpoint is in service.
    """
    from aicls.core.errors import InputValidationError, NotFoundError
    from aicls.core.messages_tr import MSG_NOT_FOUND, MSG_VALIDATION_ERROR
    from aicls.db import ModelVersionRepository

    with db_session(db_url) as session:
        repo = ModelVersionRepository(session)
        rows = {row.version: row for row in repo.list_versions(component)}
        row = rows.get(version)
        if row is None:
            raise NotFoundError(
                MSG_NOT_FOUND,
                details={
                    "reason": "model_version_not_registered",
                    "component": component,
                    "version": version,
                    "registered": sorted(rows),
                },
            )
        if not row.is_active:
            raise InputValidationError(
                MSG_VALIDATION_ERROR,
                details={
                    "reason": "model_version_not_active",
                    "component": component,
                    "version": version,
                    "hint": f"aicls models activate --component {component} --version {version}",
                },
            )


def _run_model_backed(
    kind: str, *, root: Path, out_dir: Path, split: str, model: str, db_url: str
) -> tuple[EvalReport, Path]:
    """Evaluate the promoted OCR/ASR checkpoint over the harness's synthetic corpus.

    The corpus, its fingerprint and its dataset version come from :mod:`aicls.eval.harness`, so a
    model-backed run is comparable with the stub baseline row for row. The backend comes from
    :class:`aicls.api.model_registry.ModelRegistry`, which returns the deterministic stub whenever
    ``AICLS_MODELS_ENABLED`` is off, no version is promoted or the checkpoint directory is gone —
    and the report then carries ``stub=True`` and the stub's own ``model_version``.
    """
    from aicls.api.model_registry import ModelRegistry
    from aicls.core.settings import get_settings
    from aicls.db import make_engine, session_scope
    from aicls.eval.harness import (
        DATASET_VERSIONS,
        DEFAULT_ASR_ROWS,
        DEFAULT_OCR_REPEATS,
        build_asr_dataset,
        build_ocr_samples,
        default_paths,
        synthetic_fingerprint,
    )
    from aicls.eval.provenance import git_info
    from aicls.eval.report import write_report
    from aicls.eval.runners import run_asr_eval, run_ocr_eval
    from aicls.ocr import CURRICULUM_LATEX

    component = MODEL_COMPONENTS[kind]
    _require_promoted(component, model, db_url)
    settings = get_settings()
    engine = make_engine(resolve_db_url(db_url))
    try:
        registry = ModelRegistry(settings, lambda: session_scope(engine))
        git = git_info(root)
        paths = default_paths(root)
        if kind == "ocr":
            recognizer = registry.get_recognizer()
            fingerprint = synthetic_fingerprint(
                {
                    "generator": "aicls.eval.harness.build_ocr_samples",
                    "labels": list(CURRICULUM_LATEX),
                    "repeats": DEFAULT_OCR_REPEATS,
                    "seed": 1234,
                }
            )
            report = run_ocr_eval(
                recognizer,
                build_ocr_samples(repeats=DEFAULT_OCR_REPEATS),
                git=git,
                dataset_fingerprint=fingerprint,
                dataset_version=DATASET_VERSIONS["ocr"],
                model_version=registry.version("ocr"),
                failures_dir=out_dir,
                name=f"ocr_{split}",
                stub=registry.state("ocr") != "loaded",
                # The corpus here is always generated (this function's own docstring says so),
                # whatever the backend is. Without this a real recogniser on a clean tree
                # publishes a CER over ink this repository drew, as a measurement.
                synthetic=True,
            )
        else:
            with tempfile.TemporaryDirectory(prefix="aicls-eval-") as work:
                dataset, texts = build_asr_dataset(
                    paths["commands"], n=DEFAULT_ASR_ROWS, work_dir=Path(work)
                )
                report = run_asr_eval(
                    registry.get_transcriber(),
                    dataset,
                    git=git,
                    corpus=paths["commands"],
                    dataset_fingerprint=synthetic_fingerprint(
                        {
                            "generator": "aicls.eval.harness.build_asr_dataset",
                            "texts_from": paths["commands"].name,
                            "rows": DEFAULT_ASR_ROWS,
                            "splits": dataset.splits_fingerprint,
                            "seed": 1234,
                        }
                    ),
                    dataset_version=DATASET_VERSIONS["asr"],
                    model_version=registry.version("asr"),
                    failures_dir=out_dir,
                    name=f"asr_{split}",
                    stub=registry.state("asr") != "loaded",
                    synthetic=True,
                )
            cli_logger(__name__).debug("cli.eval.asr_corpus", sentences=len(texts))
    finally:
        engine.dispose()
    cli_logger(__name__).info(
        "cli.eval.model_backed",
        kind=kind,
        requested=model,
        served=registry.version(component),
        state=registry.state(component),
    )
    return report, write_report(report, out_dir)


def _persist(
    reports: Sequence[EvalReport], files: Sequence[Path | None], *, db_url: str
) -> list[dict[str, Any]]:
    """Insert every report into ``eval_runs`` and return the stored identities."""
    from aicls.db import EvalRunRepository

    stored: list[dict[str, Any]] = []
    with db_session(db_url) as session:
        repo = EvalRunRepository(session)
        for report, path in zip(reports, files, strict=True):
            document = report.to_dict()
            if path is not None:
                document["report_path"] = path.as_posix()
            # citable is the report's own recomputed verdict, never the caller's claim.
            row = repo.add(document, is_reportable=report.citable)
            session.flush()
            stored.append(
                {
                    "id": row.id,
                    "kind": row.kind.value,
                    "name": row.name,
                    "git_sha": row.git_sha,
                    "dirty": row.dirty,
                    "dataset_fingerprint": row.dataset_fingerprint,
                    "dataset_version": row.dataset_version,
                    "is_reportable": row.is_reportable,
                    "report_path": row.report_path,
                }
            )
    cli_logger(__name__).info("cli.eval.persisted", rows=len(stored))
    return stored


def _list_runs(kind: EvalKindChoice, *, since: str, limit: int, db_url: str) -> None:
    """Print recorded evaluation runs (``--from-db``)."""
    from aicls.core.errors import InputValidationError
    from aicls.core.messages_tr import MSG_VALIDATION_ERROR
    from aicls.db import EvalRunRepository

    lower: dt.datetime | None = None
    if since:
        try:
            lower = dt.datetime.fromisoformat(since)
        except ValueError as exc:
            raise InputValidationError(
                MSG_VALIDATION_ERROR, details={"reason": "since_not_iso8601", "value": since}
            ) from exc
    wanted = None if kind is EvalKindChoice.ALL else kind.value
    with db_session(db_url) as session:
        rows = EvalRunRepository(session).list(wanted, lower, limit=limit)
        runs = [
            {
                "id": row.id,
                "kind": row.kind.value,
                "name": row.name,
                "n": row.n,
                "metrics": dict(row.metrics_json),
                "git_sha": row.git_sha,
                "dirty": row.dirty,
                "dataset_fingerprint": row.dataset_fingerprint,
                "dataset_version": row.dataset_version,
                "model_version": row.model_version,
                "is_reportable": row.is_reportable,
                "report_path": row.report_path,
                "created_at": row.created_at,
            }
            for row in rows
        ]
    emit(
        {"kind": wanted, "since": since or None, "count": len(runs), "runs": runs},
        *(
            f"{row['created_at']} {row['kind']} n={row['n']} "
            f"reportable={row['is_reportable']} {row['metrics']}"
            for row in runs
        ),
    )


__all__ = [
    "DEFAULT_SPLIT",
    "HELP",
    "MODEL_COMPONENTS",
    "SPLIT_NAMES",
    "EvalKindChoice",
    "evaluate",
    "preserved",
    "report_payload",
]
