"""Where InkML documents come from: an extracted directory tree or a tar archive read as a stream.

MathWriting ships as one ``mathwriting-2024.tgz`` holding roughly 630 000 tiny InkML files. On
the machines this project runs on, materialising that tree is the expensive part, not the
parsing: ``AICLS_DATA_ROOT`` sits on a ``drvfs`` mount (WSL2 -> Windows ``D:``) where sequential
I/O is fine but creating hundreds of thousands of small files is pathological, and the extracted
tree costs ~20 GB that nothing ever reads again after sharding.

So :func:`aicls.data.shards.prepare_shards_from_source` does not take a directory; it takes an
:class:`InkmlSource`. Two implementations exist:

* :class:`DirectoryInkmlSource` — the historical path, ``rglob("*.inkml")`` under a corpus root
  (``<root>/<split>`` when that sub-directory exists). Unchanged behaviour, including the
  sorted-by-path order.
* :class:`TarInkmlSource` — reads InkML members **straight out of the archive** in one sequential
  pass and never writes an extracted file. A ``.tgz`` cannot be seeked, so the pass is shared:
  :meth:`InkmlSource.iter_records` takes *all* requested splits and yields ``(split, record)``
  pairs in archive order, which is why the shard writer keeps one buffer per split.

Two things worth knowing before quoting a fingerprint:

* **Sample order is whatever the source gives.** The directory source sorts by path; the tar
  source follows archive order. ``tar`` writes a sorted directory walk, so in practice the two
  orders agree and the shards come out byte-identical — measured on a 20 000-file synthetic
  fixture, all five splits produced the same ``ShardManifest.fingerprint`` through both paths.
  That is an observation about how the archive happens to be packed, **not a guarantee**: an
  archive written in another order gives different shard boundaries and therefore a different
  fingerprint. ``source_root`` records which source was used (``tar:<archive>!<split>``), so a
  report can always name its input exactly.
* **Nothing here parses XML.** A record carries a path or the raw bytes; both are what
  :func:`aicls.ocr.read_inkml` accepts. Member payloads are read with a hard cap
  (:data:`aicls.ocr.MAX_INKML_BYTES`), so a hostile or corrupt archive cannot make this module
  allocate an arbitrary buffer.

Measured 2026-09-08 in the project's GPU container, ``AICLS_DATA_ROOT`` on a ``drvfs`` mount,
20 000 synthetic InkML files in five splits: extracting onto the mount took 166 s and preparing
from the extracted tree another 135 s (301 s total), while streaming the same archive took
**6.5 s** and wrote no InkML file at all. Those are I/O numbers for these two code paths on this
filesystem; they say nothing about MathWriting's content and are evidence for no metric.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import tarfile

from aicls.core.logging import get_logger
from aicls.data.errors import invalid_data, missing_path
from aicls.data.splits import MATHWRITING_SPLITS
from aicls.ocr import MAX_INKML_BYTES

_log = get_logger(__name__)

#: Suffix of an InkML document (compared case-insensitively).
INKML_SUFFIX = ".inkml"

#: Archive suffixes :func:`open_inkml_source` recognises as a tar (any compression).
TAR_SUFFIXES: tuple[str, ...] = (
    ".tgz",
    ".tar",
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
    ".tbz2",
    ".txz",
)

#: Default upper bound for a counting scan; a preview must not walk 630 000 members twice.
DEFAULT_COUNT_CAP = 20_000


@dataclass(frozen=True, slots=True)
class InkmlRecord:
    """One InkML document offered by a source.

    Attributes:
        sample_id: Stem of the document, used as the shard's ``sample_ids`` entry.
        payload: A filesystem path (directory source) or the raw document bytes (tar source);
            both are accepted by :func:`aicls.ocr.read_inkml`.
        member: Where the document came from, for log lines and error details.
    """

    sample_id: str
    payload: Path | bytes
    member: str = ""


@dataclass(frozen=True, slots=True)
class SplitCount:
    """Result of a counting scan for one split (the dry-run preview).

    Attributes:
        split: Split name.
        source: Where its documents come from (directory path or ``tar:<archive>!<split>``).
        n_files: Documents counted, at most the requested cap.
        capped: Whether counting stopped at the cap, so ``n_files`` is a lower bound.
        available: Whether the split exists in the source at all.
    """

    split: str
    source: str
    n_files: int
    capped: bool
    available: bool


class InkmlSource(ABC):
    """A read-only supply of InkML documents grouped into splits.

    Implementations must be re-iterable: :meth:`iter_records` and :meth:`count_splits` may each
    be called more than once and open their own handles.
    """

    @property
    @abstractmethod
    def kind(self) -> str:
        """``"directory"`` or ``"tar"`` — recorded in log lines and dry-run output."""

    @abstractmethod
    def source_label(self, split: str) -> str:
        """Identity of one split's data, stored in ``ShardManifest.source_root``."""

    @abstractmethod
    def count_splits(
        self, splits: Sequence[str], *, cap: int = DEFAULT_COUNT_CAP
    ) -> dict[str, SplitCount]:
        """Count the InkML documents of each split **without reading their payloads**.

        Args:
            splits: Split names to count.
            cap: Stop counting a split after this many documents.

        Returns:
            One :class:`SplitCount` per requested split, in the requested order.
        """

    @abstractmethod
    def iter_records(
        self, splits: Sequence[str], *, limit: int | None = None
    ) -> Iterator[tuple[str, InkmlRecord]]:
        """Yield ``(split, record)`` for every requested split.

        Args:
            splits: Split names to read; anything else in the source is ignored.
            limit: Stop after this many records **per split** (a development aid). The limit is
                applied to documents read, not to samples written, so empty inks still count.

        Yields:
            ``(split, record)`` pairs. The tar source yields them in archive order, interleaving
            splits; the directory source yields one split after another.
        """


class DirectoryInkmlSource(InkmlSource):
    """InkML files below an extracted corpus root.

    ``<root>/<split>`` is used when that sub-directory exists, otherwise ``<root>`` itself — the
    historical behaviour of :func:`aicls.data.shards.prepare_shards`. Note the consequence: a
    *flat* directory answers every split with the same files, so ask a flat directory for one
    split at a time.

    Args:
        root: The corpus root.

    Raises:
        NotFoundError: If ``root`` is not a directory.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        if not self._root.is_dir():
            raise missing_path(self._root, "inkml_root_missing")

    @property
    def kind(self) -> str:
        """``"directory"``."""
        return "directory"

    @property
    def root(self) -> Path:
        """The corpus root this source reads."""
        return self._root

    def split_dir(self, split: str) -> Path:
        """Directory holding ``split`` (``<root>/<split>`` when it exists, else ``<root>``)."""
        candidate = self._root / split
        return candidate if candidate.is_dir() else self._root

    def source_label(self, split: str) -> str:
        """The directory the split is read from."""
        return str(self.split_dir(split))

    def count_splits(
        self, splits: Sequence[str], *, cap: int = DEFAULT_COUNT_CAP
    ) -> dict[str, SplitCount]:
        """Count ``*.inkml`` files per split, stopping at ``cap``."""
        counts: dict[str, SplitCount] = {}
        for split in dict.fromkeys(splits):
            directory = self.split_dir(split)
            total = 0
            capped = False
            if directory.is_dir():
                for _ in directory.rglob(f"*{INKML_SUFFIX}"):
                    total += 1
                    if total >= cap:
                        capped = True
                        break
            counts[split] = SplitCount(
                split=split,
                source=str(directory),
                n_files=total,
                capped=capped,
                available=directory.is_dir(),
            )
        return counts

    def iter_records(
        self, splits: Sequence[str], *, limit: int | None = None
    ) -> Iterator[tuple[str, InkmlRecord]]:
        """Yield the files of each split in sorted-by-path order."""
        for split in dict.fromkeys(splits):
            directory = self.split_dir(split)
            if not directory.is_dir():  # pragma: no cover - split_dir falls back to the root
                continue
            files = sorted(directory.rglob(f"*{INKML_SUFFIX}"))
            if limit is not None:
                files = files[: max(0, limit)]
            for path in files:
                yield split, InkmlRecord(sample_id=path.stem, payload=path, member=str(path))


class TarInkmlSource(InkmlSource):
    """InkML documents read straight out of a tar archive, never extracted to disk.

    The archive is opened in ``"r|*"`` (stream) mode, so a gzip-compressed ``.tgz`` is
    decompressed exactly once per pass and no temporary tree is written. Members are matched to
    splits by the nearest recognised split directory in their path, which is how the released
    archive is laid out (``mathwriting-2024/train/<id>.inkml``); the top-level directory name is
    irrelevant.

    Args:
        archive: Path to the ``.tgz``/``.tar`` file.
        known_splits: Directory names recognised as split names.
        default_split: Split assigned to InkML members that sit outside any split directory;
            ``None`` ignores them (the default, so a stray file cannot silently join a split).
        max_member_bytes: Hard cap on the bytes read from one member. A larger member is read
            truncated and logged, so :func:`aicls.ocr.read_inkml` refuses it with
            ``reason="too_large"`` instead of this module allocating an unbounded buffer.

    Raises:
        NotFoundError: If ``archive`` is not a file.
    """

    def __init__(
        self,
        archive: Path,
        *,
        known_splits: Sequence[str] = MATHWRITING_SPLITS,
        default_split: str | None = None,
        max_member_bytes: int = MAX_INKML_BYTES,
    ) -> None:
        self._archive = Path(archive)
        if not self._archive.is_file():
            raise missing_path(self._archive, "inkml_archive_missing")
        if int(max_member_bytes) <= 0:
            raise invalid_data("max_member_bytes_not_positive", max_member_bytes=max_member_bytes)
        self._known: frozenset[str] = frozenset(known_splits)
        self._default_split = default_split
        self._max_member_bytes = int(max_member_bytes)

    @property
    def kind(self) -> str:
        """``"tar"``."""
        return "tar"

    @property
    def archive(self) -> Path:
        """The archive this source streams."""
        return self._archive

    def source_label(self, split: str) -> str:
        """``tar:<archive>!<split>`` — the archive plus the sub-tree the split came from."""
        return f"tar:{self._archive}!{split}"

    def count_splits(
        self, splits: Sequence[str], *, cap: int = DEFAULT_COUNT_CAP
    ) -> dict[str, SplitCount]:
        """Count members per split by reading tar headers only (no payload is decoded)."""
        wanted = tuple(dict.fromkeys(splits))
        totals = dict.fromkeys(wanted, 0)
        capped = dict.fromkeys(wanted, False)
        if wanted:
            with self._open() as bundle:
                for member in bundle:
                    split = self._split_of(member)
                    if split is None or split not in totals or capped[split]:
                        continue
                    totals[split] += 1
                    if totals[split] >= cap:
                        capped[split] = True
                        if all(capped.values()):
                            break
        return {
            split: SplitCount(
                split=split,
                source=self.source_label(split),
                n_files=totals[split],
                capped=capped[split],
                available=totals[split] > 0,
            )
            for split in wanted
        }

    def iter_records(
        self, splits: Sequence[str], *, limit: int | None = None
    ) -> Iterator[tuple[str, InkmlRecord]]:
        """Stream the archive once and yield the members of every requested split."""
        wanted = tuple(dict.fromkeys(splits))
        if not wanted:
            return
        remaining: dict[str, int | None] = dict.fromkeys(wanted, limit)
        with self._open() as bundle:
            for member in bundle:
                split = self._split_of(member)
                if split is None or split not in remaining:
                    continue
                left = remaining[split]
                if left is not None and left <= 0:
                    if self._exhausted(remaining):
                        break
                    continue
                payload = self._read_member(bundle, member)
                if payload is None:  # pragma: no cover - a member that vanished mid-stream
                    continue
                yield (
                    split,
                    InkmlRecord(
                        sample_id=PurePosixPath(member.name).stem,
                        payload=payload,
                        member=member.name,
                    ),
                )
                if left is not None:
                    remaining[split] = left - 1
                    if self._exhausted(remaining):
                        break

    # ------------------------------------------------------------------ internals
    def _open(self) -> tarfile.TarFile:
        """Open the archive in streaming mode (single forward pass, no seeking)."""
        try:
            return tarfile.open(self._archive, mode="r|*")
        except tarfile.TarError as exc:
            raise invalid_data(
                "inkml_archive_unreadable", path=str(self._archive), error=str(exc)
            ) from exc

    def _split_of(self, member: tarfile.TarInfo) -> str | None:
        """Split of an InkML member, or ``None`` when the member is not one of ours."""
        if not member.isfile():
            return None
        name = member.name.replace("\\", "/")
        if not name.lower().endswith(INKML_SUFFIX):
            return None
        for part in reversed(PurePosixPath(name).parts[:-1]):
            if part in self._known:
                return part
        return self._default_split

    def _read_member(self, bundle: tarfile.TarFile, member: tarfile.TarInfo) -> bytes | None:
        """Read one member with a hard size cap (never allocates more than the cap plus a byte)."""
        handle = bundle.extractfile(member)
        if handle is None:  # pragma: no cover - only for non-regular members, filtered above
            return None
        with handle:
            payload = handle.read(self._max_member_bytes + 1)
        if len(payload) > self._max_member_bytes:
            _log.warning(
                "data.sources.member_too_large",
                archive=str(self._archive),
                member=member.name,
                declared_size=int(member.size),
                limit=self._max_member_bytes,
            )
        return payload

    @staticmethod
    def _exhausted(remaining: dict[str, int | None]) -> bool:
        """Whether every split has reached its limit, so the pass can stop early."""
        return all(left is not None and left <= 0 for left in remaining.values())


def looks_like_tar(path: Path) -> bool:
    """Whether ``path``'s name ends in one of :data:`TAR_SUFFIXES` (case-insensitive)."""
    name = Path(path).name.lower()
    return any(name.endswith(suffix) for suffix in TAR_SUFFIXES)


def open_inkml_source(
    path: Path,
    *,
    known_splits: Sequence[str] = MATHWRITING_SPLITS,
    default_split: str | None = None,
) -> InkmlSource:
    """Open ``path`` as a directory source or a tar source, whichever it is.

    Args:
        path: An extracted corpus root **or** a ``.tgz``/``.tar`` archive.
        known_splits: Split directory names inside an archive.
        default_split: Split for archive members outside any split directory.

    Returns:
        A :class:`DirectoryInkmlSource` for a directory, a :class:`TarInkmlSource` for a file.

    Raises:
        NotFoundError: If ``path`` is neither an existing directory nor an existing file.
    """
    path = Path(path)
    if path.is_dir():
        return DirectoryInkmlSource(path)
    if path.is_file():
        return TarInkmlSource(path, known_splits=known_splits, default_split=default_split)
    raise missing_path(path, "inkml_source_missing")


__all__ = [
    "DEFAULT_COUNT_CAP",
    "INKML_SUFFIX",
    "TAR_SUFFIXES",
    "DirectoryInkmlSource",
    "InkmlRecord",
    "InkmlSource",
    "SplitCount",
    "TarInkmlSource",
    "looks_like_tar",
    "open_inkml_source",
]
