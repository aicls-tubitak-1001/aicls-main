"""``aicls data`` — dataset identity, splits and shard preparation (docs/ml/veri-yonetimi.md).

Every command here is *descriptive*: it hashes, splits or converts data that is already on disk.
**Nothing in this module downloads anything** (ADR-0012) — the catalog entry printed by
``aicls data verify`` names the approval gate a download would sit behind, and the gated
``training/scripts/download_*.py`` are the only place a fetch may ever live.

* ``verify`` re-hashes a data root against its ``manifest.json`` and exits non-zero when a file
  is missing, changed or unexpected: the fingerprint quoted in a report must still describe the
  bytes on disk.
* ``manifest`` produces that manifest in the first place.
* ``split`` builds writer-/speaker-disjoint, stratified splits and freezes the previous
  assignment, so a test group can never migrate into training between two runs.
* ``prepare-mathwriting`` turns an InkML corpus into memory-mappable shards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from aicls.cli.common import cli_logger, emit, fail, parse_ratios, read_jsonl, split_csv, use_json

app = typer.Typer(
    name="data",
    help="Verify, describe, split and prepare datasets (no downloads: ADR-0012).",
    no_args_is_help=True,
)

#: Splits ``prepare-mathwriting`` converts when the caller names none.
DEFAULT_MATHWRITING_SPLITS = ("train", "valid", "test")

#: Default ``(train, valid, test)`` shares of ``aicls data split``.
DEFAULT_RATIOS = (0.8, 0.1, 0.1)


def _catalog_summary(name: str) -> dict[str, Any] | None:
    """Catalog facts about ``name`` (licence, gate, expected size), or ``None`` when unknown."""
    from aicls.core.errors import AiclsError
    from aicls.data import catalog_entry

    try:
        entry = catalog_entry(name)
    except AiclsError:
        return None
    return {
        "version": entry.version,
        "license": entry.license,
        "gate": entry.gate,
        "expected_bytes": entry.expected_bytes,
        "source": entry.source,
    }


@app.command("verify")
def verify(
    name: Annotated[str, typer.Argument(help="Dataset name, e.g. 'mathwriting'.")],
    root: Annotated[
        Path | None,
        typer.Option("--root", help="Data root; AICLS_DATA_ROOT/<name> by default."),
    ] = None,
    manifest_path: Annotated[
        Path | None,
        typer.Option("--manifest", help="Manifest file; <root>/manifest.json by default."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Re-hash a data root and compare it with its manifest.

    Exits ``1`` when anything is missing, changed or unexpected, or when the recomputed
    fingerprint differs from the one reports cite.
    """
    from aicls.core.settings import get_settings
    from aicls.data import MANIFEST_FILENAME, load_manifest

    use_json(json_output)
    base = Path(root) if root is not None else get_settings().data_root / name
    target = Path(manifest_path) if manifest_path is not None else base / MANIFEST_FILENAME
    manifest = load_manifest(target)
    report = manifest.verify(base)
    payload = {
        "name": name,
        "root": base.as_posix(),
        "manifest": target.as_posix(),
        "dataset_version": manifest.version,
        "fingerprint": manifest.fingerprint,
        "n_files": manifest.n_files,
        "total_bytes": manifest.total_bytes,
        "ok": report.ok,
        "checked": report.checked,
        "missing": report.missing,
        "changed": report.changed,
        "extra": report.extra,
        "fingerprint_matches": report.fingerprint_matches,
        "catalog": _catalog_summary(name),
    }
    lines = [
        f"{name} ({manifest.version}) at {base}",
        f"fingerprint {manifest.fingerprint}",
        report.summary(),
    ]
    if not report.ok:
        fail(payload, *lines, "verification FAILED")
    emit(payload, *lines)


@app.command("manifest")
def manifest(
    root: Annotated[Path, typer.Argument(help="Data root to hash.")],
    name: Annotated[str, typer.Option("--name", help="Dataset name recorded in the manifest.")],
    license_: Annotated[
        str, typer.Option("--license", help="SPDX identifier or 'internal'.")
    ] = "internal",
    source: Annotated[
        str, typer.Option("--source", help="Where the data came from; 'repo://' for own data.")
    ] = "repo://",
    version: Annotated[
        str, typer.Option("--version", help="Version label; <name>-<YYYY.MM> by default.")
    ] = "",
    out: Annotated[
        Path | None, typer.Option("--out", help="Output file or directory; the root by default.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Hash every file below ``root`` and write ``manifest.json``."""
    from aicls.data import DataManifest

    use_json(json_output)
    document = DataManifest.build(
        Path(root), name, license=license_, source_url=source, version=version or None
    )
    written = document.save(Path(out) if out is not None else Path(root))
    payload = {
        "name": document.name,
        "version": document.version,
        "license": document.license,
        "source_url": document.source_url,
        "fingerprint": document.fingerprint,
        "n_files": document.n_files,
        "total_bytes": document.total_bytes,
        "path": written.as_posix(),
    }
    emit(
        payload,
        f"{document.name} ({document.version}): {document.n_files} files, "
        f"{document.total_bytes} bytes",
        f"fingerprint {document.fingerprint}",
        f"wrote {written}",
    )


@app.command("split")
def split(
    manifest_file: Annotated[Path, typer.Argument(help="JSONL manifest with one row per sample.")],
    *,
    group: Annotated[
        str,
        typer.Option("--group", help="Row field that must never be split (speaker, writer_id)."),
    ] = "speaker",
    stratify: Annotated[
        str,
        typer.Option("--stratify", help="Comma-separated fields to balance: university,gender"),
    ] = "",
    id_field: Annotated[str, typer.Option("--id-field", help="Row field holding the id.")] = "id",
    ratios: Annotated[
        str, typer.Option("--ratios", help="train,valid,test shares (default 0.8,0.1,0.1).")
    ] = "",
    seed: Annotated[int, typer.Option("--seed", help="Shuffle seed.")] = 1234,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Split file; <manifest dir>/audio_commands_splits.json."),
    ] = None,
    freeze: Annotated[
        bool,
        typer.Option(
            "--freeze/--no-freeze",
            help="Keep the split of every group already present in the output file.",
        ),
    ] = True,
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Build group-disjoint, stratified train/valid/test splits from a JSONL manifest.

    Raises:
        aicls.core.errors.InputValidationError: If a row has no id, or the group/stratify fields
            are missing from a row — a silently dropped row would change a reported corpus size.
    """
    from aicls.core.errors import InputValidationError
    from aicls.core.messages_tr import MSG_VALIDATION_ERROR
    from aicls.data import SPLITS_FILENAME, make_splits, save_splits

    use_json(json_output)
    source = Path(manifest_file)
    rows = read_jsonl(source)
    fields = split_csv(stratify)
    ids: list[str] = []
    groups: dict[str, str] = {}
    strata: dict[str, tuple[str, ...]] = {}
    for index, row in enumerate(rows):
        sample_id = str(row.get(id_field, "")).strip()
        if not sample_id:
            raise InputValidationError(
                MSG_VALIDATION_ERROR,
                details={"reason": "missing_id_field", "field": id_field, "row": index},
            )
        if group not in row:
            raise InputValidationError(
                MSG_VALIDATION_ERROR,
                details={"reason": "missing_group_field", "field": group, "row": index},
            )
        ids.append(sample_id)
        groups[sample_id] = str(row[group])
        strata[sample_id] = tuple(str(row.get(field, "")) for field in fields)

    target = Path(out) if out is not None else source.parent / SPLITS_FILENAME
    frozen = target if freeze and target.is_file() else None
    splits = make_splits(
        ids,
        group_key=lambda sample_id: groups[sample_id],
        stratify=(lambda sample_id: strata[sample_id]) if fields else None,
        ratios=parse_ratios(ratios, DEFAULT_RATIOS),  # type: ignore[arg-type]  # validated to 3
        seed=seed,
        frozen=frozen,
    )
    written = save_splits(splits, target)
    payload = {
        "manifest": source.as_posix(),
        "path": written.as_posix(),
        "rows": len(ids),
        "groups": len(set(groups.values())),
        "group_key": group,
        "stratify": list(fields),
        "sizes": splits.sizes(),
        "fingerprint": splits.fingerprint,
        "seed": seed,
        "frozen_from": frozen.as_posix() if frozen is not None else None,
    }
    emit(
        payload,
        f"{len(ids)} rows in {len(set(groups.values()))} groups -> {splits.sizes()}",
        f"fingerprint {splits.fingerprint}",
        f"wrote {written}",
    )


@app.command("prepare-mathwriting")
def prepare_mathwriting(
    root: Annotated[Path, typer.Argument(help="Extracted MathWriting corpus root (InkML).")],
    out: Annotated[Path, typer.Option("--out", help="Shard root to write.")],
    splits: Annotated[
        list[str] | None,
        typer.Option("--split", help="Split to convert; repeatable (train, valid, test, ...)."),
    ] = None,
    shard_size: Annotated[int, typer.Option("--shard-size", help="Samples per shard.")] = 2048,
    limit: Annotated[
        int, typer.Option("--limit", help="Stop after this many samples per split (0 = all).")
    ] = 0,
    render: Annotated[
        bool, typer.Option("--render/--no-render", help="Also store a rasterised image.")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Convert an InkML corpus into memory-mappable ``npz`` shards, once.

    The official split assignment is read from the corpus layout and written next to the shards
    when it can be determined; it is never invented, because MathWriting's own test split is
    what the İP-3 CER number will be measured on.
    """
    from aicls.core.errors import AiclsError
    from aicls.data import mathwriting_official, prepare_shards, save_splits

    use_json(json_output)
    corpus = Path(root)
    destination = Path(out)
    selected = tuple(splits) if splits else DEFAULT_MATHWRITING_SPLITS
    manifests: list[dict[str, Any]] = []
    for name in selected:
        shard_manifest = prepare_shards(
            corpus,
            destination,
            split=name,
            shard_size=shard_size,
            render=render,
            limit=limit or None,
        )
        manifests.append(
            {
                "split": name,
                "samples": shard_manifest.n_samples,
                "shards": len(shard_manifest.shards),
                "fingerprint": shard_manifest.fingerprint,
            }
        )
    official: str | None = None
    try:
        official = save_splits(
            mathwriting_official(corpus), destination / "mathwriting_splits.json"
        ).as_posix()
    except AiclsError as exc:  # the corpus may hold a single split directory
        cli_logger(__name__).warning("cli.data.official_splits_unavailable", error=str(exc)[:200])
    payload = {
        "root": corpus.as_posix(),
        "out": destination.as_posix(),
        "splits": manifests,
        "official_splits": official,
        "render": render,
    }
    emit(
        payload,
        *(
            f"{entry['split']}: {entry['samples']} samples in {entry['shards']} shards"
            for entry in manifests
        ),
        f"wrote {destination}",
    )


__all__ = [
    "DEFAULT_MATHWRITING_SPLITS",
    "DEFAULT_RATIOS",
    "app",
    "manifest",
    "prepare_mathwriting",
    "split",
    "verify",
]
