"""InkML -> memory-mapped ``npz`` shards for MathWriting (**ungated**).

Ungated on purpose: this script only reads data that is *already on disk*. Getting that data
there is the ``DATASET_DOWNLOAD`` gate's job (``training/scripts/download_mathwriting.py``); if
the source is missing, this script says so and exits 1 instead of fetching anything.

It is deliberately thin. All the work lives in :mod:`aicls.data.shards`, which is unit-tested
with synthetic InkML (``test_prepare_mathwriting_shard_roundtrip_on_synthetic_inkml``): parsing
630 000 XML files once per epoch is an I/O wall, so the corpus is converted once into
uncompressed ``npz`` shards that later runs read through a memory map.

**Two input paths, and the second one is the cheap one:**

* ``--src <dir>`` reads an extracted corpus tree (the historical path);
* ``--archive <file.tgz>`` streams the InkML members **straight out of the released archive**.
  Nothing is extracted: the ~20 GB tree and the 630 000 small-file writes never happen, which
  matters because ``AICLS_DATA_ROOT`` lives on a ``drvfs`` mount where creating that many small
  files is pathologically slow. All requested splits are produced in a single pass, because a
  gzip stream cannot be seeked and five passes would decompress 2.9 GiB five times.

Measured 2026-09-08 in the project's GPU container with the data root on a ``drvfs`` mount, on a
20 000-file synthetic fixture: extract 166 s + prepare 135 s = **301 s** for the directory path,
**6.5 s** for the archive path, identical samples and identical fingerprints in all five splits.
That is an I/O measurement of two code paths on synthetic InkML — not a result about MathWriting.

The two paths order samples by whatever the source gives (sorted path vs. archive order). ``tar``
writes a sorted directory walk, so the shards normally come out byte-identical, but an archive
packed in another order would shift the shard boundaries and change the fingerprint.
``shards_manifest.json`` records which source was used in ``source_root``, so a report always
names the exact preparation behind its CER number.

Usage (from the repository root, project venv)::

    python training/scripts/prepare_mathwriting.py --dry-run
    python training/scripts/prepare_mathwriting.py --archive --manifest      # default archive path
    python training/scripts/prepare_mathwriting.py --archive D:/aicls-data/mathwriting/excerpt.tgz
    python training/scripts/prepare_mathwriting.py --src D:/aicls-data/mathwriting/raw --split train

Every split gets ``<out>/<split>/<split>-NNNNN.npz`` plus ``shards_manifest.json`` whose
``fingerprint`` is what an OCR report cites as the identity of its training data. Those per-split
files are also what a training run reads back: :func:`aicls.ml.manifest.collect_data_fingerprints`
turns them into the ``data_fingerprints`` block of ``run_manifest.json`` (one
``mathwriting.<split>`` entry each, plus one aggregate ``mathwriting`` key from
:func:`aicls.data.shards.shards_root_fingerprint`). Nothing else has to be written for a run to
name its data — ``--manifest`` is the separate, and much more expensive, whole-root
:class:`~aicls.data.manifest.DataManifest` used by ``aicls data verify``.

Exit codes: ``0`` success or dry run, ``1`` a typed AI-CLS error (missing source, empty corpus).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:  # plain checkout without an editable install
    sys.path.insert(0, str(_SRC))

from aicls.core.errors import AiclsError, NotFoundError  # noqa: E402
from aicls.core.logging import get_logger  # noqa: E402
from aicls.core.messages_tr import msg  # noqa: E402
from aicls.core.settings import get_settings  # noqa: E402
from aicls.data import (  # noqa: E402
    MATHWRITING_SPLITS,
    DataManifest,
    ShardManifest,
    catalog_entry,
    prepare_shards_from_source,
)
from aicls.data.shards import (  # noqa: E402
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SHARD_SIZE,
    shards_root_fingerprint,
)
from aicls.data.sources import (  # noqa: E402
    DEFAULT_COUNT_CAP,
    DirectoryInkmlSource,
    InkmlSource,
    TarInkmlSource,
)

_log = get_logger(__name__)

#: Gate that has to be open before the *source* data can exist at all.
SOURCE_GATE = "DATASET_DOWNLOAD"

#: Catalog key describing the corpus (licence and archive name come from there).
CATALOG_NAME = "mathwriting"

#: Default corpus sub-directory produced by ``download_mathwriting.py --extract``.
DEFAULT_SRC_SUBDIR = "mathwriting/raw"

#: Directory the downloaded archive is written to.
DEFAULT_ARCHIVE_SUBDIR = "mathwriting"

#: Default shards root read by ``data.root`` in the OCR configs.
DEFAULT_OUT_SUBDIR = "mathwriting/shards"

#: Upper bound on the files counted per split during ``--dry-run`` (a preview, not a census).
DRY_RUN_SCAN_CAP = DEFAULT_COUNT_CAP

#: Name of the dataset-level manifest written with ``--manifest``.
MANIFEST_NAME = "mathwriting-shards"

#: ``--archive`` without a value: use :func:`default_archive_path`. An empty string is a safe
#: sentinel because a real archive path can never be empty.
ARCHIVE_DEFAULT = ""


def default_archive_path() -> Path:
    """Where ``download_mathwriting.py`` leaves the archive (``AICLS_DATA_ROOT/mathwriting``)."""
    entry = catalog_entry(CATALOG_NAME)
    return get_settings().data_root / DEFAULT_ARCHIVE_SUBDIR / entry.archive_name


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """What one split would cost: where its InkML lives and how many documents were seen.

    Attributes:
        split: Split name.
        source: Where the documents are read from (a directory path, or ``tar:<archive>!<split>``).
        target: Directory the shards are written to.
        n_files: Documents counted, capped at :data:`DRY_RUN_SCAN_CAP`.
        capped: Whether counting stopped at the cap, so ``n_files`` is a lower bound.
        available: Whether the split exists in the source at all.
    """

    split: str
    source: str
    target: Path
    n_files: int
    capped: bool
    available: bool

    def render(self) -> str:
        """One line of the dry-run plan."""
        count = f"{self.n_files}{'+' if self.capped else ''}"
        state = "ok" if self.available else "MISSING"
        return f"  {self.split:10s} {count:>8s} inkml  {state:8s} {self.source} -> {self.target}"


def open_source(path: Path, *, default_split: str | None = None) -> InkmlSource:
    """Open ``path`` as a directory corpus or a tar archive.

    Args:
        path: An extracted corpus root or a ``.tgz``/``.tar`` archive.
        default_split: Split assigned to archive members that sit outside a split directory.

    Returns:
        The matching :class:`~aicls.data.sources.InkmlSource`.

    Raises:
        NotFoundError: If ``path`` does not exist; ``details["gate"]`` names the gate that has to
            be approved before the data can be there at all.
    """
    path = Path(path)
    if path.is_dir():
        return DirectoryInkmlSource(path)
    if path.is_file():
        return TarInkmlSource(path, default_split=default_split)
    raise NotFoundError(
        msg("FILE_NOT_FOUND", path=str(path)),
        details={
            "path": str(path),
            "reason": "inkml_source_missing",
            "gate": SOURCE_GATE,
            "hint": "python training/scripts/download_mathwriting.py --excerpt --dry-run",
        },
    )


def plan_splits(
    source: InkmlSource | None, path: Path, out: Path, splits: list[str]
) -> list[SplitPlan]:
    """Describe every requested split without writing anything.

    Args:
        source: The opened source, or ``None`` when ``path`` does not exist yet.
        path: The source path the operator named (used for the plan of a missing source).
        out: Shards root.
        splits: Requested split names.

    Returns:
        One :class:`SplitPlan` per split, in the requested order.
    """
    if source is None:
        return [
            SplitPlan(
                split=split,
                source=str(path),
                target=out / split,
                n_files=0,
                capped=False,
                available=False,
            )
            for split in splits
        ]
    counts = source.count_splits(splits, cap=DRY_RUN_SCAN_CAP)
    return [
        SplitPlan(
            split=split,
            source=counts[split].source,
            target=out / split,
            n_files=counts[split].n_files,
            capped=counts[split].capped,
            available=counts[split].available,
        )
        for split in splits
    ]


def convert(
    source: InkmlSource,
    out: Path,
    splits: list[str],
    *,
    shard_size: int = DEFAULT_SHARD_SIZE,
    render: bool = False,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    stroke_width: int = 2,
    limit: int | None = None,
    skip_invalid: bool = False,
) -> dict[str, ShardManifest]:
    """Convert every requested split in one pass; a thin wrapper over the data layer.

    Args:
        source: Directory or archive source.
        out: Shards root.
        splits: Split names to convert.
        shard_size: Samples per shard file.
        render: Also store a rasterised image per sample (offline path).
        image_size: ``(height, width)`` of the rendered images.
        stroke_width: Pen width used when rendering.
        limit: Stop after this many documents per split (development aid).
        skip_invalid: Count an unreadable InkML document as skipped instead of failing.

    Returns:
        One :class:`~aicls.data.shards.ShardManifest` per split.
    """
    manifests = prepare_shards_from_source(
        source,
        out,
        splits=splits,
        shard_size=shard_size,
        render=render,
        image_size=image_size,
        stroke_width=stroke_width,
        limit=limit,
        skip_invalid=skip_invalid,
    )
    for split, manifest in manifests.items():
        _log.info(
            "training.prepare_mathwriting.split_done",
            split=split,
            samples=manifest.n_samples,
            skipped=manifest.n_skipped,
            shards=manifest.n_shards,
            source=manifest.source_root,
            fingerprint=manifest.fingerprint,
        )
    return manifests


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the script."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="prepare_mathwriting.py",
        description="Convert MathWriting InkML into npz shards (ungated; local data only).",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=settings.data_root / DEFAULT_SRC_SUBDIR,
        help="extracted InkML corpus root (default AICLS_DATA_ROOT/mathwriting/raw)",
    )
    parser.add_argument(
        "--archive",
        nargs="?",
        const=ARCHIVE_DEFAULT,
        default=None,
        metavar="TGZ",
        help=(
            "stream InkML straight out of this .tgz instead of an extracted tree; "
            "bare --archive uses AICLS_DATA_ROOT/mathwriting/<catalog archive name>"
        ),
    )
    parser.add_argument(
        "--archive-default-split",
        default=None,
        choices=list(MATHWRITING_SPLITS),
        help="split for archive members outside a split directory (flat archives)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=settings.data_root / DEFAULT_OUT_SUBDIR,
        help="shards root (default AICLS_DATA_ROOT/mathwriting/shards)",
    )
    parser.add_argument(
        "--split",
        dest="splits",
        nargs="+",
        default=list(MATHWRITING_SPLITS),
        choices=list(MATHWRITING_SPLITS),
        help="splits to convert (default: all official MathWriting splits)",
    )
    parser.add_argument(
        "--shard-size", type=int, default=DEFAULT_SHARD_SIZE, help="samples per shard file"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="store a rasterised image per sample (offline path, no re-render at train time)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=list(DEFAULT_IMAGE_SIZE),
        help="rendered image size",
    )
    parser.add_argument("--stroke-width", type=int, default=2, help="pen width when rendering")
    parser.add_argument("--limit", type=int, default=None, help="stop after N samples per split")
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="log and count unreadable InkML documents instead of failing the whole run",
    )
    parser.add_argument(
        "--manifest",
        action="store_true",
        help="also hash the shards root into manifest.json (dataset identity for reports)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and exit 0, writing nothing"
    )
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    return parser


def resolve_source_path(args: argparse.Namespace) -> tuple[Path, bool]:
    """Return the path to read and whether it is an archive (``--archive`` wins over ``--src``).

    Args:
        args: The parsed command line.

    Returns:
        ``(path, is_archive)``. A bare ``--archive`` resolves to :func:`default_archive_path`.
    """
    if args.archive is None:
        return Path(args.src), False
    if str(args.archive) == ARCHIVE_DEFAULT:
        return default_archive_path(), True
    return Path(str(args.archive)), True


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command line (``sys.argv[1:]`` when omitted).

    Returns:
        Process exit code: 0 ok/dry run, 1 typed error.
    """
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    image_size = (int(args.image_size[0]), int(args.image_size[1]))
    source_path, is_archive = resolve_source_path(args)

    source: InkmlSource | None = None
    plans: list[SplitPlan]
    try:
        if source_path.exists():
            source = open_source(source_path, default_split=args.archive_default_split)
        plans = plan_splits(source, source_path, out, list(args.splits))
    except AiclsError as exc:  # a corrupt archive: say so, do not call it "missing"
        print(f"error: {exc}", file=sys.stderr)
        print(f"details: {exc.details}", file=sys.stderr)
        return int(exc.exit_code)

    print(f"plan: prepare mathwriting shards (ungated; source gate {SOURCE_GATE})")
    print(f"  mode        : {'archive stream (nothing extracted)' if is_archive else 'directory'}")
    print(f"  source      : {source_path}{'' if source is not None else '   (MISSING)'}")
    print(f"  shards root : {out}")
    print(f"  shard size  : {args.shard_size}   render: {bool(args.render)}  size: {image_size}")
    for plan in plans:
        print(plan.render())
    if args.dry_run:
        print("dry run: nothing was written.")
        return 0

    try:
        if source is None:
            # raises NotFoundError naming DATASET_DOWNLOAD and the download command
            source = open_source(source_path, default_split=args.archive_default_split)
        summary = _convert_all(args, source, source_path, out, image_size)
    except AiclsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(f"details: {exc.details}", file=sys.stderr)
        return int(exc.exit_code)

    print(json.dumps(summary, indent=2, ensure_ascii=False) if args.json else _render(summary))
    return 0


def _convert_all(
    args: argparse.Namespace,
    source: InkmlSource,
    source_path: Path,
    out: Path,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """Convert every requested split in one pass and collect the summary document."""
    splits = list(args.splits)
    print(f"converting {', '.join(splits)} from {source.kind} source ...")
    manifests = convert(
        source,
        out,
        splits,
        shard_size=int(args.shard_size),
        render=bool(args.render),
        image_size=image_size,
        stroke_width=int(args.stroke_width),
        limit=args.limit,
        skip_invalid=bool(args.skip_invalid),
    )
    rows: dict[str, Any] = {
        split: {
            "samples": manifest.n_samples,
            "skipped": manifest.n_skipped,
            "shards": manifest.n_shards,
            "has_images": manifest.has_images,
            "source": manifest.source_root,
            "fingerprint": manifest.fingerprint,
        }
        for split, manifest in manifests.items()
    }
    summary: dict[str, Any] = {
        "source": str(source_path),
        "source_kind": source.kind,
        "out": str(out),
        "splits": rows,
        # The identity a run manifest records for this root without re-hashing a byte; it is
        # derived from the per-split fingerprints above, so it is printable here for free.
        "root_fingerprint": shards_root_fingerprint(
            {split: manifest.fingerprint for split, manifest in manifests.items()}
        ),
    }
    if args.manifest:
        entry = catalog_entry(CATALOG_NAME)
        data_manifest = DataManifest.build(
            out,
            MANIFEST_NAME,
            license=entry.license,
            source_url=f"file://{source_path.as_posix()}",
        )
        summary["license"] = entry.license
        summary["manifest"] = str(data_manifest.save(out))
        summary["fingerprint"] = data_manifest.fingerprint
    return summary


def _render(summary: dict[str, Any]) -> str:
    """Human-readable summary of a completed conversion."""
    lines = [f"shards written to {summary['out']} (source: {summary['source_kind']})"]
    for split, row in summary["splits"].items():
        lines.append(
            f"  {split:10s} {row['samples']:>8d} samples  {row['shards']:>4d} shards  "
            f"skipped {row['skipped']:<5d} {row['fingerprint']}"
        )
    if summary.get("root_fingerprint"):
        lines.append(
            f"  data_fingerprints['mathwriting'] = {summary['root_fingerprint']} "
            "(what run_manifest.json will record for this root)"
        )
    if "manifest" in summary:
        lines.append(
            f"  manifest: {summary['manifest']} ({summary['fingerprint']}, "
            f"licence {summary['license']})"
        )
    lines.append(
        "next: python training/scripts/build_tokenizer.py "
        f"--shards {summary['out']} --out {Path(summary['out']).parent / 'vocab.json'}"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
