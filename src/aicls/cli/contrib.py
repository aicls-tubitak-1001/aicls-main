"""``aicls contrib`` — the voice data-collection corpus (İP-3, KVKK, docs/ml/veri-yonetimi.md).

Three operations, each with a rule that is not negotiable:

* ``export-manifest`` writes the ASR training manifest with **no identity field** in it and
  refreshes the speaker-disjoint split file, freezing the speakers that already have a split so a
  test speaker can never drift into training between two exports;
* ``withdraw`` deletes a participant's audio and derived features and keeps the proof of consent
  — that asymmetry is the KVKK requirement, and it is why the command reports both numbers;
* ``stats`` is the dashboard figure (hours accepted, speakers, QC pass rate) the 50-hour target of
  İP-3 is tracked against.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from aicls.cli.common import db_session, emit, jsonable, use_json

app = typer.Typer(
    name="contrib",
    help="Export, withdraw from and summarise the voice contribution corpus (IP-3).",
    no_args_is_help=True,
)


@app.command("export-manifest")
def export_manifest(
    out: Annotated[Path, typer.Option("--out", help="Destination JSONL manifest.")],
    source: Annotated[
        str, typer.Option("--source", help="Provenance label: contrib, internal or tts.")
    ] = "contrib",
    seed: Annotated[int, typer.Option("--seed", help="Seed ordering newly split speakers.")] = 1234,
    splits_path: Annotated[
        Path | None,
        typer.Option("--splits", help="Split file to freeze against; next to --out by default."),
    ] = None,
    data_root: Annotated[
        Path | None, typer.Option("--data-root", help="Audio root; AICLS_DATA_ROOT by default.")
    ] = None,
    verify_files: Annotated[
        bool,
        typer.Option("--verify-files/--no-verify-files", help="Skip rows whose audio is missing."),
    ] = True,
    db_url: Annotated[str, typer.Option("--db-url", help="Database URL; AICLS_DB_URL.")] = "",
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Write the pseudonymised ASR manifest and refresh the frozen speaker split."""
    from aicls.collect import export_manifest as export
    from aicls.db import RecordingRepository

    use_json(json_output)
    with db_session(db_url) as session:
        report = export(
            RecordingRepository(session),
            out=Path(out),
            split_seed=seed,
            data_root=data_root,
            splits_path=splits_path,
            source=source,
            verify_files=verify_files,
        )
    payload = jsonable(report)
    emit(
        payload,
        f"{report.rows} rows from {report.speakers} speakers ({report.seconds / 3600:.2f} h)",
        f"splits {report.by_split} (frozen speakers: {report.frozen_speakers})",
        f"fingerprint {report.fingerprint}",
        f"wrote {report.path} and {report.splits_path}",
    )


@app.command("withdraw")
def withdraw(
    speaker: Annotated[str, typer.Option("--speaker", help="Speaker pseudonym or ULID (P00012).")],
    data_root: Annotated[
        Path | None, typer.Option("--data-root", help="Audio root; AICLS_DATA_ROOT by default.")
    ] = None,
    db_url: Annotated[str, typer.Option("--db-url", help="Database URL; AICLS_DB_URL.")] = "",
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Withdraw a participant's consent: delete the recordings, keep the consent record.

    Raises:
        aicls.core.errors.NotFoundError: If no such speaker exists.
    """
    from aicls.collect import withdraw as do_withdraw
    from aicls.db import RecordingRepository

    use_json(json_output)
    with db_session(db_url) as session:
        report = do_withdraw(RecordingRepository(session), speaker, data_root=data_root)
    payload = jsonable(report)
    emit(
        payload,
        f"{report.speaker_pseudonym}: {report.recordings_deleted} recordings and "
        f"{report.files_deleted} files deleted ({report.bytes_deleted} bytes)",
        "consent record kept (KVKK: the proof of consent is not erased)",
    )


@app.command("stats")
def stats(
    db_url: Annotated[str, typer.Option("--db-url", help="Database URL; AICLS_DB_URL.")] = "",
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Print the contribution dashboard figures (hours, speakers, QC pass rate)."""
    from aicls.db import RecordingRepository

    use_json(json_output)
    with db_session(db_url) as session:
        figures = RecordingRepository(session).stats()
    payload = jsonable(figures)
    emit(
        payload,
        f"{figures.hours_accepted:.2f} h accepted, {figures.utterances} utterances, "
        f"{figures.speakers} speakers",
        f"QC pass rate {figures.qc_pass_rate:.4f}",
        f"by university {figures.by_university}",
    )


__all__ = ["app", "export_manifest", "stats", "withdraw"]
