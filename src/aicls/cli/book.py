"""``aicls book`` — load, validate and export the textbook knowledge graph (İP-8, ADR-0019).

İP-8 promises **100 % tagged content**, so the coverage check is a gate, not a statistic:
``validate`` exits non-zero and names every heading that produced no content unit, and ``load``
refuses the whole directory rather than storing a partially tagged chapter (that is
:func:`aicls.book.require_full_coverage`'s contract, applied inside
:func:`aicls.book.load_markdown_dir`).

``export`` writes the same graph in the two serialisations the proposal names, JSON-LD and
Turtle, either from the database or straight from a source directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from aicls.cli.common import db_session, emit, fail, use_json

app = typer.Typer(
    name="book",
    help="Load, validate and export the textbook content graph (IP-8).",
    no_args_is_help=True,
)


def _tag(directory: Path) -> list[Any]:
    """Tag every Markdown chapter of ``directory`` (sorted by file name)."""
    from aicls.book import tag_markdown_dir

    return tag_markdown_dir(Path(directory))


@app.command("load")
def load(
    directory: Annotated[Path, typer.Argument(help="Directory of tagged Markdown chapters.")],
    db_url: Annotated[str, typer.Option("--db-url", help="Database URL; AICLS_DB_URL.")] = "",
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Tag a chapter directory and store its units, relations and aliases.

    Raises:
        aicls.book.UntaggedHeadingError: If any heading is untagged; nothing is stored, because
            İP-8 measures tagging coverage as an all-or-nothing criterion.
    """
    from aicls.book import load_markdown_dir
    from aicls.db import BookRepository

    use_json(json_output)
    with db_session(db_url) as session:
        report = load_markdown_dir(Path(directory), BookRepository(session))
    payload = {
        "directory": Path(directory).as_posix(),
        "units": report.units,
        "triples": report.triples,
        "aliases": report.aliases,
        "tagged_ratio": report.tagged_ratio,
    }
    emit(
        payload,
        f"loaded {report.units} units, {report.triples} relations, {report.aliases} aliases",
        f"tagging coverage {report.tagged_ratio:.4f}",
    )


@app.command("validate")
def validate(
    directory: Annotated[Path, typer.Argument(help="Directory of Markdown chapters to check.")],
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Check that every unit-looking heading was tagged; exit ``1`` when one was not."""
    from aicls.book import untagged_headings

    use_json(json_output)
    results = _tag(directory)
    spans = untagged_headings(*results)
    units = sum(len(result.nodes) for result in results)
    payload = {
        "directory": Path(directory).as_posix(),
        "files": len(results),
        "units": units,
        "untagged": [
            {"source": span.source, "line": span.line, "text": span.text, "kind": span.kind}
            for span in spans
        ],
        "ok": not spans,
    }
    lines = [f"{len(results)} files, {units} units, {len(spans)} untagged headings"]
    lines.extend(f"  {span.source}:{span.line} [{span.kind}] {span.text}" for span in spans)
    if spans:
        fail(payload, *lines, "validation FAILED (IP-8 requires 100 % tagging)")
    emit(payload, *lines)


@app.command("coverage")
def coverage(
    directory: Annotated[Path, typer.Argument(help="Directory of Markdown chapters.")],
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Report the tagging coverage per file and overall (the İP-8 ``100 %`` criterion)."""
    from aicls.book import coverage as file_coverage

    use_json(json_output)
    results = _tag(directory)
    per_file = [
        {
            "source": result.source,
            "chapter": result.chapter,
            "units": len(result.nodes),
            "untagged": len(result.untagged_spans),
            "coverage": file_coverage(result),
        }
        for result in results
    ]
    tagged = sum(entry["units"] for entry in per_file)
    untagged = sum(entry["untagged"] for entry in per_file)
    overall = 1.0 if tagged + untagged == 0 else tagged / (tagged + untagged)
    payload = {
        "directory": Path(directory).as_posix(),
        "files": per_file,
        "units": tagged,
        "untagged": untagged,
        "coverage": overall,
        "target": 1.0,
    }
    emit(
        payload,
        *(
            f"{entry['source']}: {entry['units']} units, coverage {entry['coverage']:.4f}"
            for entry in per_file
        ),
        f"overall coverage {overall:.4f} (target 1.0000)",
    )


@app.command("export")
def export(
    jsonld: Annotated[
        Path | None, typer.Option("--jsonld", help="Write the JSON-LD document here.")
    ] = None,
    ttl: Annotated[Path | None, typer.Option("--ttl", help="Write the Turtle graph here.")] = None,
    directory: Annotated[
        Path | None,
        typer.Option("--dir", help="Export a source directory instead of the database."),
    ] = None,
    db_url: Annotated[str, typer.Option("--db-url", help="Database URL; AICLS_DB_URL.")] = "",
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Export the content graph as JSON-LD and/or Turtle.

    Raises:
        aicls.core.errors.InputValidationError: If neither ``--jsonld`` nor ``--ttl`` is given —
            an export that writes nothing is a mistake, not a no-op.
    """
    import json as json_module

    from aicls.book import export_jsonld, export_turtle
    from aicls.core.errors import InputValidationError
    from aicls.core.messages_tr import MSG_VALIDATION_ERROR

    use_json(json_output)
    if jsonld is None and ttl is None:
        raise InputValidationError(
            MSG_VALIDATION_ERROR,
            details={"reason": "no_export_target", "expected": "--jsonld and/or --ttl"},
        )
    if directory is not None:
        results = _tag(directory)
        units = [node for result in results for node in result.nodes]
        triples = [triple for result in results for triple in result.triples]
        source = Path(directory).as_posix()
    else:
        from aicls.book import triples_from_repo, units_from_repo
        from aicls.db import BookRepository

        with db_session(db_url) as session:
            repo = BookRepository(session)
            units = units_from_repo(repo)
            triples = triples_from_repo(repo)
        source = "database"

    written: dict[str, str] = {}
    if jsonld is not None:
        document = export_jsonld(units, triples)
        target = Path(jsonld)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json_module.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
        written["jsonld"] = target.as_posix()
    if ttl is not None:
        target = Path(ttl)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(export_turtle(units, triples))
        written["ttl"] = target.as_posix()

    payload = {"source": source, "units": len(units), "triples": len(triples), "written": written}
    emit(
        payload,
        f"exported {len(units)} units and {len(triples)} relations from {source}",
        *(f"wrote {path}" for path in written.values()),
    )


__all__ = ["app", "coverage", "export", "load", "validate"]
