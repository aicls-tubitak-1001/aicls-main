"""InkML -> memory-mapped ``npz`` shards (docs/ml/egitim-altyapisi.md §2).

Parsing 630 000 XML files once per epoch is an I/O wall on Windows, so the corpus is converted
**once** into a handful of uncompressed ``.npz`` shards and read back through a memory map: the
operating system pages in only the strokes a batch actually touches, and several dataloader
workers share one copy.

Layout of a shard (all arrays uncompressed, so they can be mapped):

===================  ==========================  ==================================================
array                shape / dtype               meaning
===================  ==========================  ==================================================
``points``           ``(P, 3) float32``          every ``x, y, t`` sample of the shard, back to back
``stroke_ptr``       ``(S + 1,) int64``          stroke ``j`` is ``points[stroke_ptr[j]:...[j+1]]``
``sample_ptr``       ``(n + 1,) int64``          sample ``i`` owns strokes ``sample_ptr[i]:..``
``labels``           ``(n,) <U``                 normalised LaTeX label
``writer_ids``       ``(n,) <U``                 writer id (writer-disjoint splits)
``sample_ids``       ``(n,) <U``                 source file stem
``is_synthetic``     ``(n,) bool``               MathWriting synthetic subset flag
``images``           ``(n, H, W) uint8``         optional pre-rendered raster (offline path)
===================  ==========================  ==================================================

Every shard is hashed into a :class:`ShardManifest` (``shards_manifest.json``) whose fingerprint
goes into the OCR report, so a CER number always names the exact shards it was measured on. That
file is written **per split**, next to the split's shards; :func:`shard_fingerprints` and
:func:`shards_root_fingerprint` are the readers that turn a prepared root back into the
``{split: fingerprint}`` map a run manifest records, and they are the only place that knows this
layout — :class:`aicls.ml.manifest.RunManifest` asks here instead of guessing a file name.

Where the InkML comes from is a separate question, answered by :mod:`aicls.data.sources`:
:func:`prepare_shards` reads an extracted directory, :func:`prepare_shards_from_source` reads any
:class:`~aicls.data.sources.InkmlSource` — including the released ``mathwriting-2024.tgz``
streamed member by member, which skips both the ~20 GB extracted tree and the 630 000 small-file
writes that a ``drvfs`` data root cannot absorb. Sample order follows the source (sorted path vs.
archive order); ``source_root`` records which one produced the shards, because a fingerprint is
only comparable with another fingerprint made the same way.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import struct
from types import TracebackType
from typing import Any, ClassVar
import zipfile

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from aicls.core.errors import AiclsError
from aicls.core.hashing import FINGERPRINT_PREFIX, sha256_file, sha256_json
from aicls.core.logging import get_logger
from aicls.data.errors import invalid_data, missing_path
from aicls.data.sources import DirectoryInkmlSource, InkmlSource
from aicls.ocr import Ink, LatexTokenizer, Stroke, read_inkml, render_ink
from aicls.ocr.curriculum import DEFAULT_EXTRA_TOKENS
from aicls.ocr.tokenizer import join_tokens, tokenize_latex
from aicls.ocr.types import OcrInputError

_log = get_logger(__name__)

#: Name of the manifest written next to the shards of one split.
SHARD_MANIFEST_FILENAME = "shards_manifest.json"

#: Default number of samples per shard.
DEFAULT_SHARD_SIZE = 4096

#: Default raster size when ``render=True`` (matches ``aicls.ocr.DEFAULT_RENDER_SIZE``).
DEFAULT_IMAGE_SIZE: tuple[int, int] = (96, 384)

#: Splits whose samples are flagged ``is_synthetic``.
SYNTHETIC_SPLITS: frozenset[str] = frozenset({"synthetic"})

_NPY_MAGIC = b"\x93NUMPY"
_ZIP_LOCAL_HEADER = b"PK\x03\x04"


def normalize_label(latex: str) -> str:
    """Canonical spacing of a LaTeX label (vocabulary independent).

    The MathWriting labels use inconsistent spacing; the shard stores the tokenizer's normal form
    so that ``decode(encode(label)) == label`` holds and CER is computed on one canonical string
    (docs/ml/degerlendirme.md, ``latex_v1``).

    Args:
        latex: Raw label.

    Returns:
        The normalised label (``""`` stays ``""``).
    """
    return join_tokens(tokenize_latex(latex)) if latex else ""


class ShardEntry(BaseModel):
    """One shard file inside a :class:`ShardManifest`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    n_samples: int = Field(ge=0)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)


class ShardManifest(BaseModel):
    """Manifest of the shards of one split (``shards_manifest.json``)."""

    model_config = ConfigDict(extra="forbid")

    SCHEMA_VERSION: ClassVar[int] = 1

    dataset: str = "mathwriting"
    split: str
    shard_size: int = Field(gt=0)
    n_samples: int = Field(ge=0)
    n_skipped: int = Field(default=0, ge=0)
    shards: list[ShardEntry] = Field(default_factory=list)
    has_images: bool = False
    image_size: tuple[int, int] | None = None
    source_root: str = ""
    created_at: datetime
    fingerprint: str = ""
    schema_version: int = SCHEMA_VERSION

    @property
    def n_shards(self) -> int:
        """Number of shard files."""
        return len(self.shards)

    def to_json(self) -> str:
        """Serialise to pretty JSON (LF, UTF-8, trailing newline)."""
        payload: dict[str, Any] = json.loads(self.model_dump_json())
        return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"

    def save(self, directory: Path) -> Path:
        """Write ``shards_manifest.json`` into ``directory`` and return the path."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / SHARD_MANIFEST_FILENAME
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(self.to_json())
        return path


@dataclass(frozen=True, slots=True)
class ShardSample:
    """One sample read back from a shard.

    Attributes:
        sample_id: Source file stem.
        strokes: One ``(N, 3)`` float32 array per stroke (views into the memory map).
        label: Normalised LaTeX label.
        writer_id: Writer identifier, ``""`` when the corpus does not provide one.
        is_synthetic: Whether the sample comes from MathWriting's synthetic subset.
        image: Pre-rendered ``(H, W)`` uint8 raster, or ``None``.
    """

    sample_id: str
    strokes: list[np.ndarray]
    label: str
    writer_id: str
    is_synthetic: bool
    image: np.ndarray | None = None

    def to_ink(self) -> Ink:
        """Rebuild an :class:`~aicls.ocr.types.Ink` (copies the arrays)."""
        return Ink(
            strokes=[Stroke(points) for points in self.strokes if points.shape[0] > 0],
            label=self.label or None,
            writer_id=self.writer_id or None,
            meta={"source": "shard", "sample_id": self.sample_id},
        )


class ShardReader:
    """Random access to one shard, memory-mapped by default.

    Args:
        path: The ``.npz`` shard.
        mmap: Map the arrays instead of reading them into memory. Falls back to a plain read
            for arrays that cannot be mapped (compressed or Fortran-ordered members).

    Raises:
        NotFoundError: If ``path`` does not exist.
    """

    def __init__(self, path: Path, *, mmap: bool = True) -> None:
        self._path = Path(path)
        if not self._path.is_file():
            raise missing_path(self._path, "shard_missing")
        self._mmap = mmap
        self._cache: dict[str, np.ndarray] = {}
        self._names: tuple[str, ...] = tuple(_zip_members(self._path))
        self._sample_ptr = self.array("sample_ptr")
        self._stroke_ptr = self.array("stroke_ptr")

    # ------------------------------------------------------------------ context manager
    def __enter__(self) -> ShardReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Drop the mapped arrays (the OS releases the mapping)."""
        self._cache.clear()

    # ------------------------------------------------------------------ access
    @property
    def path(self) -> Path:
        """Shard path."""
        return self._path

    @property
    def has_images(self) -> bool:
        """Whether the shard carries pre-rendered rasters."""
        return "images" in self._names

    def array(self, name: str) -> np.ndarray:
        """Return one shard array (memory-mapped when possible, cached per reader).

        Raises:
            InputValidationError: If the shard has no such array.
        """
        if name in self._cache:
            return self._cache[name]
        if name not in self._names:
            raise invalid_data("shard_array_missing", array=name, available=list(self._names))
        array: np.ndarray | None = None
        if self._mmap:
            array = _memmap_npz_member(self._path, name)
        if array is None:
            with np.load(self._path) as handle:
                array = np.asarray(handle[name])
        self._cache[name] = array
        return array

    def __len__(self) -> int:
        return int(self._sample_ptr.shape[0] - 1)

    def sample(self, index: int) -> ShardSample:
        """Return sample ``index`` of this shard.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if not 0 <= index < len(self):
            raise IndexError(f"shard sample index out of range: {index}")
        points = self.array("points")
        first, last = int(self._sample_ptr[index]), int(self._sample_ptr[index + 1])
        strokes = [
            np.asarray(points[int(self._stroke_ptr[j]) : int(self._stroke_ptr[j + 1])])
            for j in range(first, last)
        ]
        image = None
        if self.has_images:
            image = np.asarray(self.array("images")[index])
        return ShardSample(
            sample_id=str(self.array("sample_ids")[index]),
            strokes=strokes,
            label=str(self.array("labels")[index]),
            writer_id=str(self.array("writer_ids")[index]),
            is_synthetic=bool(self.array("is_synthetic")[index]),
            image=image,
        )


class _SplitShardWriter:
    """Buffers the samples of one split and flushes them into ``<out>/<split>/<split>-NNNNN.npz``.

    Split out of :func:`prepare_shards` so that a *single* pass over a tar archive can feed
    several splits at once: one writer per split, each with its own buffer, counters and
    manifest. The directory path uses exactly the same writer, so the two sources cannot drift.
    """

    def __init__(
        self,
        out: Path,
        split: str,
        *,
        shard_size: int,
        synthetic: bool,
        render: bool,
        image_size: tuple[int, int],
        stroke_width: int,
    ) -> None:
        self.split = split
        self.directory = Path(out) / split
        self.n_samples = 0
        self.n_skipped = 0
        self._shard_size = shard_size
        self._synthetic = synthetic
        self._render = render
        self._image_size = image_size
        self._stroke_width = stroke_width
        self._entries: list[ShardEntry] = []
        self._buffer: list[tuple[str, Ink]] = []
        self._opened = False

    def add(self, sample_id: str, ink: Ink) -> None:
        """Add one parsed ink; empty inks are counted as skipped and never reach a shard."""
        if ink.is_empty:
            self.skip(sample_id, reason="empty_ink")
            return
        self._buffer.append((sample_id, ink))
        self.n_samples += 1
        if len(self._buffer) >= self._shard_size:
            self._flush()

    def skip(self, sample_id: str, *, reason: str) -> None:
        """Record a document that produced no sample (empty ink, unreadable InkML)."""
        self.n_skipped += 1
        _log.warning(
            "data.shards.sample_skipped", split=self.split, sample_id=sample_id, reason=reason
        )

    def finish(self, *, source_label: str) -> ShardManifest:
        """Flush the tail, write ``shards_manifest.json`` and return it."""
        self._flush()
        manifest = ShardManifest(
            split=self.split,
            shard_size=self._shard_size,
            n_samples=self.n_samples,
            n_skipped=self.n_skipped,
            shards=self._entries,
            has_images=self._render,
            image_size=(
                (int(self._image_size[0]), int(self._image_size[1])) if self._render else None
            ),
            source_root=source_label,
            created_at=datetime.now(UTC),
            fingerprint=FINGERPRINT_PREFIX
            + sha256_json([[e.name, e.n_samples, e.sha256] for e in self._entries]),
        )
        manifest.save(self.directory)
        _log.info(
            "data.shards.prepared",
            split=self.split,
            samples=self.n_samples,
            shards=len(self._entries),
            skipped=self.n_skipped,
            source=source_label,
            fingerprint=manifest.fingerprint,
        )
        return manifest

    def _flush(self) -> None:
        """Write the buffered samples as one shard file (no-op on an empty buffer)."""
        if not self._buffer:
            return
        if not self._opened:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._opened = True
        name = f"{self.split}-{len(self._entries):05d}.npz"
        path = self.directory / name
        _write_shard(
            path,
            self._buffer,
            synthetic=self._synthetic,
            render=self._render,
            image_size=self._image_size,
            stroke_width=self._stroke_width,
        )
        self._entries.append(
            ShardEntry(
                name=name,
                n_samples=len(self._buffer),
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
        self._buffer = []


def prepare_shards_from_source(
    source: InkmlSource,
    out: Path,
    *,
    splits: Sequence[str],
    shard_size: int = DEFAULT_SHARD_SIZE,
    render: bool = False,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    stroke_width: int = 2,
    is_synthetic: bool | None = None,
    limit: int | None = None,
    skip_invalid: bool = False,
) -> dict[str, ShardManifest]:
    """Convert every requested split of an :class:`~aicls.data.sources.InkmlSource` into shards.

    All requested splits are produced in **one** pass over the source, which is what makes the
    tar path affordable: a ``.tgz`` cannot be seeked, so preparing five splits with five calls
    would decompress 3 GB five times.

    Args:
        source: Where the InkML comes from (directory tree or tar archive).
        out: Shards root; split ``s`` is written to ``<out>/<s>/``.
        splits: Split names to convert.
        shard_size: Samples per shard file.
        render: Also store a rasterised image per sample (offline path without re-rendering).
        image_size: ``(height, width)`` of the rendered images.
        stroke_width: Pen width used when rendering.
        is_synthetic: Override the synthetic flag for **every** requested split; by default a
            split is synthetic exactly when it is in :data:`SYNTHETIC_SPLITS`.
        limit: Stop after this many documents per split (development aid).
        skip_invalid: Count an unreadable InkML document as skipped instead of failing the run.
            Off by default, so a corrupt corpus is loud; a 630 000-file archive is the case where
            turning it on is worth the (logged) loss.

    Returns:
        One :class:`ShardManifest` per requested split, in the requested order. A split the
        source does not hold gets an empty manifest and a warning.

    Raises:
        InputValidationError: If ``splits`` is empty, ``shard_size`` is not positive, or **no**
            requested split held a single InkML document.
        OcrInputError: On an unreadable InkML document while ``skip_invalid`` is false.
    """
    wanted = tuple(dict.fromkeys(splits))
    if not wanted:
        raise invalid_data("no_splits_requested")
    if shard_size <= 0:
        raise invalid_data("shard_size_not_positive", shard_size=shard_size)

    writers = {
        split: _SplitShardWriter(
            out,
            split,
            shard_size=shard_size,
            synthetic=(split in SYNTHETIC_SPLITS if is_synthetic is None else bool(is_synthetic)),
            render=render,
            image_size=image_size,
            stroke_width=stroke_width,
        )
        for split in wanted
    }

    n_seen = 0
    for split, record in source.iter_records(wanted, limit=limit):
        writer = writers[split]
        n_seen += 1
        try:
            ink = read_inkml(record.payload)
        except OcrInputError as exc:
            if not skip_invalid:
                raise
            writer.skip(record.sample_id, reason=str(exc.details.get("reason", "invalid_inkml")))
            continue
        writer.add(record.sample_id, ink)

    # ``limit=0`` deliberately reads nothing, so an empty pass is the requested outcome there
    # and not a mis-pointed source (this is what the directory path has always done).
    if n_seen == 0 and limit != 0:
        raise invalid_data(
            "no_inkml_files",
            root=source.source_label(wanted[0]),
            source_kind=source.kind,
            splits=list(wanted),
        )

    manifests: dict[str, ShardManifest] = {}
    for split, writer in writers.items():
        if writer.n_samples == 0 and writer.n_skipped == 0:
            _log.warning("data.shards.split_empty", split=split, source=source.source_label(split))
        manifests[split] = writer.finish(source_label=source.source_label(split))
    return manifests


def prepare_shards(
    inkml_root: Path,
    out: Path,
    *,
    split: str,
    shard_size: int = DEFAULT_SHARD_SIZE,
    render: bool = False,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    stroke_width: int = 2,
    is_synthetic: bool | None = None,
    limit: int | None = None,
    skip_invalid: bool = False,
) -> ShardManifest:
    """Convert one split of an extracted InkML directory into memory-mappable ``npz`` shards.

    A thin wrapper over :func:`prepare_shards_from_source` with a
    :class:`~aicls.data.sources.DirectoryInkmlSource`; kept because it is the documented entry
    point of the extracted-corpus path and is what every existing caller uses.

    Args:
        inkml_root: Corpus root; ``<inkml_root>/<split>`` is used when that directory exists.
        out: Shards root; files are written to ``<out>/<split>/``.
        split: Split name (``train``/``valid``/``test``/``symbols``/``synthetic``).
        shard_size: Samples per shard file.
        render: Also store a rasterised image per sample (offline path without re-rendering).
        image_size: ``(height, width)`` of the rendered images.
        stroke_width: Pen width used when rendering.
        is_synthetic: Override the synthetic flag; defaults to ``split == "synthetic"``.
        limit: Stop after this many samples (development aid).
        skip_invalid: Count an unreadable InkML file as skipped instead of failing the run.

    Returns:
        The written :class:`ShardManifest` (also saved as ``shards_manifest.json``).

    Raises:
        NotFoundError: If ``inkml_root`` does not exist.
        InputValidationError: If ``shard_size`` is not positive or no InkML file was found.
    """
    manifests = prepare_shards_from_source(
        DirectoryInkmlSource(inkml_root),
        out,
        splits=(split,),
        shard_size=shard_size,
        render=render,
        image_size=image_size,
        stroke_width=stroke_width,
        is_synthetic=is_synthetic,
        limit=limit,
        skip_invalid=skip_invalid,
    )
    return manifests[split]


def load_shard_manifest(root: Path, split: str) -> ShardManifest:
    """Read ``<root>/<split>/shards_manifest.json`` (``<root>`` itself is also accepted).

    Raises:
        NotFoundError: If no manifest is found.
        InputValidationError: If the manifest is malformed.
    """
    root = Path(root)
    candidates = (
        root / split / SHARD_MANIFEST_FILENAME,
        root / SHARD_MANIFEST_FILENAME,
        root,
    )
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise missing_path(root / split / SHARD_MANIFEST_FILENAME, "shard_manifest_missing")
    return read_shard_manifest(path)


def read_shard_manifest(path: Path) -> ShardManifest:
    """Parse one ``shards_manifest.json`` file.

    Args:
        path: The manifest file itself (no directory search; see :func:`load_shard_manifest`).

    Returns:
        The parsed :class:`ShardManifest`.

    Raises:
        InputValidationError: If it is not valid JSON or not a valid manifest document.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise invalid_data("shard_manifest_not_json", path=str(path), error=str(exc)) from exc
    try:
        return ShardManifest.model_validate(payload)
    except ValueError as exc:
        raise invalid_data("shard_manifest_invalid", path=str(path), error=str(exc)) from exc


# ------------------------------------------------------------------- dataset identity of a root
def discover_shard_splits(root: Path) -> tuple[str, ...]:
    """Split names prepared under ``root``, sorted.

    A split counts as prepared when ``<root>/<split>/shards_manifest.json`` exists — the layout
    :func:`prepare_shards_from_source` writes. A single-split root (``<root>/shards_manifest
    .json``, the fallback :func:`load_shard_manifest` accepts) contributes the split name
    recorded inside that file.

    Args:
        root: Shards root.

    Returns:
        The split names; empty when ``root`` is not a directory or holds no manifest. This never
        raises: it runs while a training run is starting up, where an absent or unreadable data
        root is a fact to record in the run manifest, not a reason to stop the run.
    """
    root = Path(root)
    if not root.is_dir():
        return ()
    found: set[str] = set()
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / SHARD_MANIFEST_FILENAME).is_file():
            found.add(child.name)
    single = root / SHARD_MANIFEST_FILENAME
    if single.is_file():
        try:
            found.add(read_shard_manifest(single).split)
        except AiclsError as exc:
            _log.warning("data.shards.manifest_unreadable", path=str(single), reason=str(exc))
    return tuple(sorted(found))


def shard_fingerprints(
    root: Path, splits: Sequence[str] | None = None, *, verify: bool = False
) -> dict[str, str]:
    """Fingerprint of every prepared split under ``root``, keyed by split name.

    The fingerprint is the one :func:`prepare_shards_from_source` computed while writing the
    shards (``sha256`` over the ``(shard name, samples, sha256)`` triples), so by default this is
    a cheap read of a handful of small JSON files — not a re-hash of the shard bytes. It is what
    ties a training run to the exact data it opened (:class:`aicls.ml.manifest.RunManifest`).

    **The default does not re-read the shards, and that limit is deliberate.** Re-preparing the
    corpus changes the recorded fingerprint, which is the case that matters; editing a shard file
    in place behind the manifest's back does not, because nothing re-hashed it. Verifying that at
    the start of every run would mean reading 2.9 GiB before the first optimizer step. Pass
    ``verify=True`` (or run ``aicls data verify`` against a
    :class:`~aicls.data.manifest.DataManifest`) when the bytes themselves have to be checked.

    Args:
        root: Shards root.
        splits: Restrict the result to these split names; by default every split
            :func:`discover_shard_splits` finds is reported.
        verify: Re-hash every shard file and drop a split whose bytes no longer match the digest
            its manifest recorded. Costs a full read of the split.

    Returns:
        ``{split: "sha256:<hex>"}`` in split order. A split with no manifest, an unreadable one,
        one whose manifest names a different split, or (under ``verify``) one whose bytes moved is
        logged and left out rather than raising: a partially prepared root must still produce a
        run manifest that says what it found.
    """
    root = Path(root)
    wanted = tuple(dict.fromkeys(splits)) if splits is not None else discover_shard_splits(root)
    fingerprints: dict[str, str] = {}
    for split in wanted:
        try:
            manifest = load_shard_manifest(root, split)
        except AiclsError as exc:
            _log.info(
                "data.shards.fingerprint_missing", root=str(root), split=split, reason=str(exc)
            )
            continue
        if manifest.split != split:
            # `load_shard_manifest` falls back to the root manifest, which would otherwise label
            # one split's fingerprint with another split's name -- a provenance lie.
            _log.warning(
                "data.shards.split_mismatch",
                root=str(root),
                requested=split,
                found=manifest.split,
            )
            continue
        if not manifest.fingerprint:
            _log.warning("data.shards.fingerprint_empty", root=str(root), split=split)
            continue
        if verify and not _shards_match(root / split, manifest):
            continue
        fingerprints[split] = manifest.fingerprint
    return dict(sorted(fingerprints.items()))


def _shards_match(directory: Path, manifest: ShardManifest) -> bool:
    """Whether every shard file below ``directory`` still hashes to its recorded digest."""
    for entry in manifest.shards:
        path = directory / entry.name
        if not path.is_file():
            _log.warning(
                "data.shards.shard_missing", split=manifest.split, shard=entry.name, path=str(path)
            )
            return False
        if sha256_file(path) != entry.sha256:
            _log.warning(
                "data.shards.fingerprint_stale",
                split=manifest.split,
                shard=entry.name,
                recorded=entry.sha256,
                reason="the shard bytes changed after preparation; re-run the preparation",
            )
            return False
    return True


def shards_root_fingerprint(fingerprints: Mapping[str, str]) -> str:
    """One ``"sha256:<hex>"`` identity for a set of per-split fingerprints.

    This is the single value a report can cite as *the* dataset
    (:attr:`aicls.eval.report.EvalReport.dataset_fingerprint`) when the corpus is a prepared
    shards root rather than one JSONL file. It hashes the sorted ``(split, fingerprint)`` pairs,
    so it changes when any recorded split changes and is independent of file-system order.

    Args:
        fingerprints: Per-split fingerprints, e.g. from :func:`shard_fingerprints`.

    Returns:
        The aggregate fingerprint, or ``""`` for an empty mapping — a fingerprint of nothing is
        not recorded.
    """
    if not fingerprints:
        return ""
    return FINGERPRINT_PREFIX + sha256_json(
        [[split, fingerprints[split]] for split in sorted(fingerprints)]
    )


def build_vocab_from_shards(
    root: Path,
    out: Path,
    *,
    splits: tuple[str, ...] = ("train",),
    min_freq: int = 1,
    extra_tokens: tuple[str, ...] = DEFAULT_EXTRA_TOKENS,
) -> Path:
    """Build ``vocab.json`` from the labels of the given splits.

    Args:
        root: Shards root.
        out: Target ``vocab.json`` path.
        splits: Splits whose labels feed the vocabulary.
        min_freq: Drop tokens rarer than this.
        extra_tokens: Tokens always kept (digits, letters, operators).

    Returns:
        The path that was written.

    Raises:
        NotFoundError: If a split has no shard manifest.
    """
    labels: list[str] = []
    for split in splits:
        manifest = load_shard_manifest(root, split)
        for entry in manifest.shards:
            with ShardReader(Path(root) / split / entry.name) as reader:
                labels.extend(str(value) for value in reader.array("labels"))
    tokenizer = LatexTokenizer.build(labels, min_freq=min_freq, extra_tokens=extra_tokens)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out)
    _log.info("data.shards.vocab_built", path=str(out), vocab_size=tokenizer.vocab_size)
    return out


# ------------------------------------------------------------------------------------ internals
def _write_shard(
    path: Path,
    samples: list[tuple[str, Ink]],
    *,
    synthetic: bool,
    render: bool,
    image_size: tuple[int, int],
    stroke_width: int,
) -> None:
    """Write one uncompressed npz shard (uncompressed so that it can be memory-mapped)."""
    points: list[np.ndarray] = []
    stroke_ptr: list[int] = [0]
    sample_ptr: list[int] = [0]
    labels: list[str] = []
    writers: list[str] = []
    ids: list[str] = []
    images: list[np.ndarray] = []

    offset = 0
    for sample_id, ink in samples:
        for stroke in ink.strokes:
            points.append(np.asarray(stroke.points, dtype=np.float32))
            offset += int(stroke.points.shape[0])
            stroke_ptr.append(offset)
        sample_ptr.append(len(stroke_ptr) - 1)
        labels.append(normalize_label(ink.label or ""))
        writers.append(ink.writer_id or "")
        ids.append(sample_id)
        if render:
            raster = render_ink(ink, image_size, stroke_width=stroke_width)
            images.append(np.round(raster * 255.0).astype(np.uint8))

    arrays: dict[str, np.ndarray] = {
        "points": (
            np.concatenate(points, axis=0) if points else np.zeros((0, 3), dtype=np.float32)
        ),
        "stroke_ptr": np.asarray(stroke_ptr, dtype=np.int64),
        "sample_ptr": np.asarray(sample_ptr, dtype=np.int64),
        "labels": np.asarray(labels, dtype=np.str_),
        "writer_ids": np.asarray(writers, dtype=np.str_),
        "sample_ids": np.asarray(ids, dtype=np.str_),
        "is_synthetic": np.full(len(samples), bool(synthetic), dtype=np.bool_),
    }
    if render:
        arrays["images"] = np.stack(images, axis=0)
    with path.open("wb") as handle:
        # savez's keyword arguments are the array names; the stub types **kwds against
        # `allow_pickle: bool`, which is why this call needs an ignore.
        np.savez(handle, **arrays)  # type: ignore[arg-type]


def _zip_members(path: Path) -> list[str]:
    """Array names inside an npz file (without the ``.npy`` suffix)."""
    with zipfile.ZipFile(path) as archive:
        return [name.removesuffix(".npy") for name in archive.namelist()]


def _member_data_offset(path: Path, member: str) -> int | None:
    """Byte offset of an uncompressed zip member's payload, or ``None`` when it is compressed."""
    with zipfile.ZipFile(path) as archive:
        try:
            info = archive.getinfo(member)
        except KeyError:  # pragma: no cover - callers check membership first
            return None
        if info.compress_type != zipfile.ZIP_STORED:
            return None
        header_offset = info.header_offset
    with path.open("rb") as handle:
        handle.seek(header_offset)
        header = handle.read(30)
        if len(header) < 30 or header[:4] != _ZIP_LOCAL_HEADER:  # pragma: no cover - defensive
            return None
        name_len, extra_len = struct.unpack("<HH", header[26:30])
    return header_offset + 30 + name_len + extra_len


def _memmap_npz_member(path: Path, name: str) -> np.ndarray | None:
    """Memory-map one array of an uncompressed npz, or ``None`` when that is not possible."""
    offset = _member_data_offset(path, f"{name}.npy")
    if offset is None:
        return None
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            if handle.read(6) != _NPY_MAGIC:  # pragma: no cover - defensive
                return None
            handle.seek(offset)
            # numpy < 2.4 ships these helpers without annotations and numpy 2.4 annotates them,
            # so a `type: ignore[no-untyped-call]` is required under one and an error under the
            # other (warn_unused_ignores). Binding them to Callable[..., Any] first satisfies
            # both. They are public API of numpy.lib.format and the only supported way to read
            # an .npy header.
            read_magic: Callable[..., Any] = np.lib.format.read_magic
            read_header_1_0: Callable[..., Any] = np.lib.format.read_array_header_1_0
            read_header_2_0: Callable[..., Any] = np.lib.format.read_array_header_2_0
            version = read_magic(handle)
            if version == (1, 0):
                shape, fortran, dtype = read_header_1_0(handle)
            elif version == (2, 0):  # pragma: no cover - only for >64 KiB headers
                shape, fortran, dtype = read_header_2_0(handle)
            else:  # pragma: no cover - future npy versions
                return None
            data_offset = handle.tell()
        if fortran or dtype.hasobject:  # pragma: no cover - we never write such arrays
            return None
        return np.memmap(path, dtype=dtype, mode="r", offset=data_offset, shape=shape)
    except (OSError, ValueError):  # pragma: no cover - fall back to a plain read
        _log.warning("data.shards.mmap_failed", path=str(path), array=name)
        return None


__all__ = [
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_SHARD_SIZE",
    "SHARD_MANIFEST_FILENAME",
    "SYNTHETIC_SPLITS",
    "ShardEntry",
    "ShardManifest",
    "ShardReader",
    "ShardSample",
    "build_vocab_from_shards",
    "discover_shard_splits",
    "load_shard_manifest",
    "normalize_label",
    "prepare_shards",
    "read_shard_manifest",
    "shard_fingerprints",
    "shards_root_fingerprint",
]
