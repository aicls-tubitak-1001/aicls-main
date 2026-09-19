"""Download the MathWriting corpus — **GATED: DATASET_DOWNLOAD** (ADR-0012, egitim-altyapisi §9).

This is one of exactly two files in the repository that are allowed to contain network code
(``tools/check_gates.py`` allow-lists ``training/scripts/download_*.py``). The rule this file
implements is the project owner's standing instruction: *no dataset is fetched without explicit,
auditable approval*.

Shape of the script, and why:

1. it reads the **descriptive** catalog entry (``aicls.data.catalog``) — url, size, licence,
   sha256, target directory — and prints it together with the free disk space, so the operator
   can see the whole cost of the action *before* deciding;
2. ``--dry-run`` stops there and never even asks for the gate, so preparing an approval request
   is itself a gate-free action;
3. otherwise :func:`aicls.core.gates.require_gate` demands all three conditions together
   (``--i-have-approval`` **and** ``AICLS_GATE_DATASET_DOWNLOAD=approved`` **and** an ``approved``
   row in ``docs/gelistirme/onay-defteri.md``). A refusal exits **3** and lists every missing
   condition in Turkish;
4. only after the gate passes does control reach :func:`download_archive`, the single function
   in this file that touches the network. Nothing above it creates a directory or writes a byte.

Three things about the archive itself, verified against the source on 2026-09-08:

* it is **``mathwriting-2024.tgz``** (a gzip-compressed tar, 3 096 141 721 B). The ``.zip`` URL
  the catalogue used to carry answers HTTP 403 — it does not exist;
* the licence is **CC BY-NC-SA 4.0**, not Apache-2.0. NonCommercial and ShareAlike travel with
  anything trained on it; the plan prints the licence on every invocation for exactly that
  reason;
* ``--excerpt`` fetches the official 1.5 MB sample archive instead. It has the same split layout
  and the same licence, so the entire pipeline — download, shard, tokenizer, evaluate — can be
  proven end to end before 2.9 GiB and hours are spent.

The transfer is **resumable and atomic**: bytes go into ``<archive>.part``, an interrupted run
continues with an HTTP ``Range`` request instead of starting a 3 GB download over, and the file
only appears under its real name after the announced length has actually arrived.

Extraction is **off by default**: ``prepare_mathwriting.py --archive`` streams InkML straight out
of the ``.tgz``, which avoids ~20 GB of extracted tree and 630 000 small-file writes that a
``drvfs`` data root cannot absorb. Pass ``--extract`` when a real directory tree is wanted.

Usage (from the repository root, project venv)::

    python training/scripts/download_mathwriting.py --dry-run
    python training/scripts/download_mathwriting.py --excerpt --i-have-approval  # 1.5 MB first
    python training/scripts/download_mathwriting.py --i-have-approval            # needs env+ledger

Exit codes: ``0`` success or dry run, ``1`` a typed AI-CLS error (bad checksum, no disk space),
``3`` the approval gate refused.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
import tarfile
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:  # plain checkout without an editable install
    sys.path.insert(0, str(_SRC))

from aicls.core.errors import AiclsError, InputValidationError, NotFoundError  # noqa: E402
from aicls.core.gates import (  # noqa: E402
    CLI_FLAG,
    GateError,
    default_ledger_path,
    is_ci_environment,
    is_gate_approved_in_ledger,
    read_gate_ledger,
    require_gate,
)
from aicls.core.hashing import sha256_file  # noqa: E402
from aicls.core.logging import get_logger  # noqa: E402
from aicls.core.messages_tr import msg  # noqa: E402
from aicls.core.settings import GateName, get_settings  # noqa: E402
from aicls.data import CatalogEntry, DataManifest, catalog_entry  # noqa: E402

_log = get_logger(__name__)

#: Approval gate this script sits behind.
GATE: GateName = "DATASET_DOWNLOAD"

#: Catalog key of the full corpus.
CATALOG_NAME = "mathwriting"

#: Catalog key of the official 1.5 MB sample archive (``--excerpt``).
EXCERPT_CATALOG_NAME = "mathwriting_excerpt"

#: Directory of the corpus under ``AICLS_DATA_ROOT`` (both archives share it).
DEST_SUBDIR = "mathwriting"

#: Only these hosts may be fetched from, whatever the catalog says (defence in depth).
ALLOWED_HOSTS: frozenset[str] = frozenset({"storage.googleapis.com"})

#: Extracted-to-compressed size ratio of the MathWriting archive (~20 GB from ~2.9 GiB).
DEFAULT_EXTRACT_FACTOR = 7.0

#: Read size of one network chunk.
CHUNK_BYTES = 1 << 20

#: Socket timeout of the download, in seconds.
TIMEOUT_S = 60.0

#: Suffix of the in-progress file; a resumed transfer appends to it.
PARTIAL_SUFFIX = ".part"

#: HTTP status of a successful ``Range`` request.
HTTP_PARTIAL_CONTENT = 206

#: User agent sent with the request (identifies the project in the bucket's logs).
USER_AGENT = "ai-cls/1.0 (+125E748)"

GIB = float(1 << 30)
MIB = float(1 << 20)


def human_size(n: int) -> str:
    """Format a byte count in the largest unit that still reads sensibly.

    The excerpt archive is 1.5 MB and the full corpus is 2.9 GiB; printing both in GiB makes the
    first one read ``0.00 GiB``, which is exactly the number an operator must be able to trust.
    """
    if n >= 1 << 30:
        return f"{n / GIB:.2f} GiB"
    if n >= 1 << 20:
        return f"{n / MIB:.2f} MiB"
    return f"{n} B"


@dataclass(frozen=True, slots=True)
class DownloadPlan:
    """Everything the operator needs in order to approve (or refuse) the download.

    Attributes:
        entry: The catalog row being acted on.
        dest: Directory the corpus would be written to.
        archive: Path the compressed archive would be written to.
        resume_bytes: Bytes already present in ``<archive>.part`` from an interrupted run.
        free_bytes: Free space on the volume that holds ``dest``.
        needed_bytes: Archive plus, when ``extract`` is set, the estimated extracted size.
        extract: Whether the archive would also be unpacked into ``<dest>/raw``.
        gate: Name of the approval gate.
        gate_open: Whether all three conditions currently hold.
        missing: Which conditions are missing (``ci``/``cli_flag``/``env``/``ledger``).
    """

    entry: CatalogEntry
    dest: Path
    archive: Path
    resume_bytes: int
    free_bytes: int
    needed_bytes: int
    extract: bool
    gate: str
    gate_open: bool
    missing: tuple[str, ...]

    @property
    def enough_disk(self) -> bool:
        """Whether the volume holding ``dest`` can hold archive plus extracted corpus."""
        return self.free_bytes >= self.needed_bytes

    def to_dict(self) -> dict[str, Any]:
        """JSON document of the plan (``--json``), for an approval-request PR."""
        return {
            "action": "download",
            "dataset": self.entry.name,
            "version": self.entry.version,
            "license": self.entry.license,
            "url": self.entry.url,
            "expected_bytes": self.entry.expected_bytes,
            "sha256": self.entry.sha256,
            "dest": str(self.dest),
            "archive": str(self.archive),
            "resume_bytes": self.resume_bytes,
            "extract": self.extract,
            "free_bytes": self.free_bytes,
            "needed_bytes": self.needed_bytes,
            "enough_disk": self.enough_disk,
            "gate": self.gate,
            "gate_open": self.gate_open,
            "missing_conditions": list(self.missing),
        }

    def render(self) -> str:
        """Human-readable plan; printed by every invocation, dry run or not."""
        sha = self.entry.sha256 or "(unknown until the first approved download)"
        unpack = f"yes -> {self.dest / 'raw'}" if self.extract else "no"
        lines = [
            f"plan: download {self.entry.name} ({self.entry.kind})",
            f"  version        : {self.entry.version}",
            f"  licence        : {self.entry.license}  "
            "(NonCommercial + ShareAlike — see catalog.yaml notes_tr)",
            f"  url            : {self.entry.url or 'n/a'}",
            f"  archive size   : {self.entry.expected_bytes} B "
            f"({human_size(self.entry.expected_bytes)})",
            f"  sha256         : {sha}",
            f"  target dir     : {self.dest}",
            f"  archive path   : {self.archive}",
            f"  extract        : {unpack} "
            "(prepare_mathwriting.py --archive streams the .tgz instead)",
            f"  resume from    : {self.resume_bytes} B already in {PARTIAL_SUFFIX} file",
            f"  free disk      : {human_size(self.free_bytes)}",
            f"  needed disk    : {human_size(self.needed_bytes)} "
            f"({'ok' if self.enough_disk else 'NOT ENOUGH'})",
            f"  gate           : {self.gate}",
            f"  gate status    : {'open' if self.gate_open else 'closed'}"
            + (f" — missing: {', '.join(self.missing)}" if self.missing else ""),
        ]
        return "\n".join(lines)


def free_bytes_for(path: Path) -> int:
    """Free bytes on the volume that would hold ``path`` (walks up to an existing parent)."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return int(shutil.disk_usage(probe).free)
    except OSError:  # pragma: no cover - unreadable volume
        return 0


def gate_status(name: str, *, cli_flag: bool, ledger_path: Path | None = None) -> list[str]:
    """Read-only preview of the triple lock: which conditions are currently missing.

    This *reports*; it never approves, logs an invocation or unlocks anything. A closed gate is
    still closed after calling it. Used by the plan so the operator sees, in one place, what an
    approval would require.

    Args:
        name: Gate name.
        cli_flag: Whether ``--i-have-approval`` was given.
        ledger_path: Ledger override (tests); the repository ledger by default.

    Returns:
        The missing conditions, in the order :func:`aicls.core.gates.require_gate` lists them.
    """
    missing: list[str] = []
    if is_ci_environment():
        missing.append("ci")
    if not cli_flag:
        missing.append("cli_flag")
    try:
        approved_env = get_settings().is_gate_env_approved(name)
    except (AiclsError, ValueError):  # pragma: no cover - a broken .env
        approved_env = False
    if not approved_env:
        missing.append("env")
    ledger = ledger_path or default_ledger_path()
    try:
        entries = read_gate_ledger(ledger) if ledger.is_file() else []
    except AiclsError:  # a malformed ledger is a closed gate, never an open one
        entries = []
    if not is_gate_approved_in_ledger(name, entries):
        missing.append("ledger")
    return missing


def build_plan(
    dest: Path | None = None,
    *,
    cli_flag: bool = False,
    ledger_path: Path | None = None,
    extract: bool = False,
    extract_factor: float = DEFAULT_EXTRACT_FACTOR,
    excerpt: bool = False,
) -> DownloadPlan:
    """Describe the download without performing any part of it.

    Args:
        dest: Target directory; ``AICLS_DATA_ROOT/mathwriting`` when omitted.
        cli_flag: Whether ``--i-have-approval`` was given (affects the reported gate status only).
        ledger_path: Ledger override (tests).
        extract: Whether the archive would also be unpacked (drives the disk estimate).
        extract_factor: Extracted-size estimate as a multiple of the archive size.
        excerpt: Plan the official 1.5 MB sample archive instead of the full corpus.

    Returns:
        The plan; nothing on disk is touched.
    """
    entry = catalog_entry(EXCERPT_CATALOG_NAME if excerpt else CATALOG_NAME)
    target = Path(dest) if dest is not None else get_settings().data_root / DEST_SUBDIR
    archive = target / entry.archive_name
    partial = archive.with_name(archive.name + PARTIAL_SUFFIX)
    resume = partial.stat().st_size if partial.is_file() else 0
    factor = max(0.0, extract_factor) if extract else 0.0
    needed = int(entry.expected_bytes * (1.0 + factor)) - resume
    missing = gate_status(GATE, cli_flag=cli_flag, ledger_path=ledger_path)
    return DownloadPlan(
        entry=entry,
        dest=target,
        archive=archive,
        resume_bytes=resume,
        free_bytes=free_bytes_for(target),
        needed_bytes=max(0, needed),
        extract=extract,
        gate=GATE,
        gate_open=not missing,
        missing=tuple(missing),
    )


def validate_url(url: str | None) -> str:
    """Return ``url`` after checking it is an ``https`` URL on an allow-listed host.

    Raises:
        InputValidationError: If the catalog has no url, the scheme is not https, or the host is
            not in :data:`ALLOWED_HOSTS`.
    """
    if not url:
        raise InputValidationError(
            msg("VALIDATION_ERROR"), details={"reason": "catalog_url_missing", "gate": GATE}
        )
    parts = urlparse(url)
    if parts.scheme != "https":
        raise InputValidationError(
            msg("VALIDATION_ERROR"), details={"reason": "url_not_https", "url": url}
        )
    if parts.hostname not in ALLOWED_HOSTS:
        raise InputValidationError(
            msg("VALIDATION_ERROR"),
            details={
                "reason": "url_host_not_allowed",
                "host": parts.hostname,
                "allowed": sorted(ALLOWED_HOSTS),
            },
        )
    return url


def announced_total(response: Any, start: int) -> int:
    """Total resource size the server announces, in bytes, or ``0`` when it announces none.

    Args:
        response: The open HTTP response.
        start: Byte offset the request resumed from (a ``Range`` response only reports the rest).

    Returns:
        The total size of the whole resource, not of this transfer.
    """
    headers = getattr(response, "headers", None)
    if headers is None:  # pragma: no cover - defensive
        return 0
    content_range = str(headers.get("Content-Range") or "")
    if "/" in content_range:
        tail = content_range.rsplit("/", 1)[-1].strip()
        if tail.isdigit():
            return int(tail)
    length = str(headers.get("Content-Length") or "").strip()
    return start + int(length) if length.isdigit() else 0


# ----------------------------------------------------------------------------------------------
# THE ONLY NETWORK CODE IN THIS FILE
# ----------------------------------------------------------------------------------------------
def download_archive(
    url: str, target: Path, *, expected_bytes: int = 0, resume: bool = True
) -> Path:
    """Stream ``url`` into ``target``. **Never call this before ``require_gate`` has returned.**

    Kept deliberately small and in one place: every other function in this module is inert, so a
    reviewer only has to read this one to know what the script can do to the network.

    Resumable and atomic. Bytes land in ``<target>.part``; when that file already holds data the
    request carries a ``Range`` header and the transfer continues from there, so a 3 GB download
    that dropped at 90 % costs the remaining 10 %. The size the server announces is compared with
    the catalogue **before the first byte is written**, and ``<target>`` only comes into existence
    through ``os.replace`` once the whole archive has arrived.

    Args:
        url: Already validated by :func:`validate_url`.
        target: Destination file.
        expected_bytes: Size announced by the catalog; a mismatch aborts before writing.
        resume: Continue an interrupted transfer from ``<target>.part`` when one exists.

    Returns:
        ``target``.

    Raises:
        InputValidationError: If the server announces a different size than the catalogue, or the
            transfer ends short. The ``.part`` file is kept in the short case so the next run
            resumes instead of restarting.
    """
    import urllib.request  # imported here so the module is inert until the gate has passed

    validate_url(url)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + PARTIAL_SUFFIX)
    start = partial.stat().st_size if resume and partial.is_file() else 0
    if expected_bytes and start > expected_bytes:
        # a leftover from a different version: appending to it would produce rubbish
        _log.warning(
            "training.download_mathwriting.stale_partial", size=start, expected=expected_bytes
        )
        partial.unlink(missing_ok=True)
        start = 0

    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(url, headers=headers)
    written = start
    # url validated above: https scheme, allow-listed host
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        if start and int(getattr(response, "status", 200) or 200) != HTTP_PARTIAL_CONTENT:
            _log.warning("training.download_mathwriting.range_ignored", offset=start)
            start = 0
            written = 0
        total = announced_total(response, start)
        if expected_bytes and total and total != expected_bytes:
            raise InputValidationError(
                msg("VALIDATION_ERROR"),
                details={
                    "reason": "content_length_mismatch",
                    "announced": total,
                    "expected": expected_bytes,
                    "url": url,
                },
            )
        goal = expected_bytes or total
        with partial.open("ab" if start else "wb") as handle:
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                if goal:
                    done = 100.0 * written / goal
                    print(
                        f"\r  {human_size(written)} / {human_size(goal)} ({done:5.1f} %)   ",
                        end="",
                    )
    print()
    if expected_bytes and written != expected_bytes:
        raise InputValidationError(
            msg("VALIDATION_ERROR"),
            details={
                "reason": "short_download",
                "written": written,
                "expected": expected_bytes,
                "partial": str(partial),
                "hint": "re-run the command; the transfer resumes from the .part file",
            },
        )
    partial.replace(target)
    _log.info(
        "training.download_mathwriting.fetched",
        url=url,
        bytes=written,
        resumed_from=start,
        path=str(target),
    )
    return target


def safe_member_target(root: Path, name: str) -> Path:
    """Resolve a tar member name under ``root``, refusing anything that escapes it.

    Args:
        root: The resolved extraction root.
        name: The member name as recorded in the archive.

    Returns:
        The absolute path the member would be written to.

    Raises:
        InputValidationError: On an absolute path, a ``..`` traversal or any other escape.
    """
    candidate = (root / name).resolve()
    if candidate != root and not candidate.is_relative_to(root):
        raise InputValidationError(
            msg("VALIDATION_ERROR"),
            details={"reason": "tar_member_escapes_root", "member": name},
        )
    return candidate


def extract_archive(archive: Path, dest: Path) -> int:
    """Extract the ``.tgz`` into ``dest``, one streaming pass, regular files only.

    ``tarfile.extractall`` is deliberately not used: members are resolved and checked one by one
    (tar-slip), and anything that is not a plain file or directory — symlink, hard link, device,
    fifo — is refused rather than written, because none of them belongs in a corpus of InkML.

    Args:
        archive: The downloaded ``.tgz``/``.tar``.
        dest: Directory to extract into (created when missing).

    Returns:
        Number of extracted files.

    Raises:
        InputValidationError: On a member that would escape ``dest`` or is not a regular file.
    """
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    count = 0
    with tarfile.open(archive, "r|*") as bundle:
        for member in bundle:
            resolved = safe_member_target(root, member.name)
            if member.isdir():
                resolved.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise InputValidationError(
                    msg("VALIDATION_ERROR"),
                    details={"reason": "tar_member_not_a_regular_file", "member": member.name},
                )
            source = bundle.extractfile(member)
            if source is None:  # pragma: no cover - only for non-regular members, refused above
                continue
            resolved.parent.mkdir(parents=True, exist_ok=True)
            with source, resolved.open("wb") as sink:
                shutil.copyfileobj(source, sink, CHUNK_BYTES)
            count += 1
    _log.info("training.download_mathwriting.extracted", members=count, dest=str(dest))
    return count


def write_manifest(entry: CatalogEntry, dest: Path) -> Path:
    """Hash the downloaded root into ``manifest.json`` (the fingerprint reports cite)."""
    manifest = DataManifest.build(
        dest,
        entry.name,
        license=entry.license,
        source_url=entry.url or "",
        version=entry.version,
    )
    return manifest.save(dest)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the script."""
    parser = argparse.ArgumentParser(
        prog="download_mathwriting.py",
        description="Download the MathWriting corpus (GATED: DATASET_DOWNLOAD, ADR-0012).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="target directory (default AICLS_DATA_ROOT/mathwriting)",
    )
    parser.add_argument(
        "--excerpt",
        action="store_true",
        help="fetch the official 1.5 MB sample archive instead of the 2.9 GiB corpus",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and exit 0 without asking for the gate",
    )
    parser.add_argument("--json", action="store_true", help="print the plan as JSON")
    parser.add_argument(
        CLI_FLAG,
        dest="i_have_approval",
        action="store_true",
        help="one of the three approval conditions (ADR-0012); never set it for someone else",
    )
    parser.add_argument(
        "--extract-factor",
        type=float,
        default=DEFAULT_EXTRACT_FACTOR,
        help="extracted-size estimate as a multiple of the archive size (disk check)",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="also unpack the archive into <dest>/raw (~20 GB; the preparation script does not "
        "need it, it streams the .tgz)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="explicit form of the default: download only, never unpack",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore any leftover .part file and start the transfer from zero",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="keep the archive after --extract (it is always kept without --extract)",
    )
    return parser


def main(argv: list[str] | None = None, *, ledger_path: Path | None = None) -> int:
    """Entry point.

    Args:
        argv: Command line (``sys.argv[1:]`` when omitted).
        ledger_path: Approval-ledger override. Deliberately **not** a command-line flag: the
            ledger is the auditable half of the lock and must stay the file in git; only tests
            point it elsewhere.

    Returns:
        Process exit code: 0 ok/dry run, 1 typed error, 3 gate refused.
    """
    args = build_parser().parse_args(argv)
    extract = bool(args.extract) and not bool(args.no_extract)
    try:
        plan = build_plan(
            args.dest,
            cli_flag=bool(args.i_have_approval),
            ledger_path=ledger_path,
            extract=extract,
            extract_factor=float(args.extract_factor),
            excerpt=bool(args.excerpt),
        )
    except AiclsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return int(exc.exit_code)

    print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False) if args.json else plan.render())
    if args.dry_run:
        print("dry run: nothing was downloaded and no gate was asked for.")
        return 0

    try:
        require_gate(GATE, cli_flag=bool(args.i_have_approval), ledger_path=ledger_path)
    except GateError as exc:
        print(f"\n{exc}", file=sys.stderr)
        print(
            f"hint: {CLI_FLAG} + AICLS_GATE_{GATE}=approved + a row in "
            "docs/gelistirme/onay-defteri.md (ADR-0012).",
            file=sys.stderr,
        )
        return int(exc.exit_code)

    try:
        return _run_approved(
            plan,
            extract=extract,
            keep=bool(args.keep_archive),
            resume=not bool(args.no_resume),
        )
    except AiclsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(f"details: {exc.details}", file=sys.stderr)
        return int(exc.exit_code)


def _run_approved(plan: DownloadPlan, *, extract: bool, keep: bool, resume: bool) -> int:
    """Everything that happens **after** the gate accepted; never reached otherwise.

    Raises:
        InputValidationError: On insufficient disk space or a checksum mismatch.
        NotFoundError: If the archive vanished between download and hashing.
    """
    if not plan.enough_disk:
        raise InputValidationError(
            msg("VALIDATION_ERROR"),
            details={
                "reason": "insufficient_disk_space",
                "free_bytes": plan.free_bytes,
                "needed_bytes": plan.needed_bytes,
                "dest": str(plan.dest),
            },
        )
    url = validate_url(plan.entry.url)
    plan.dest.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {plan.archive}")
    if plan.resume_bytes:
        print(f"resuming from {plan.resume_bytes} B already transferred")
    archive = download_archive(
        url, plan.archive, expected_bytes=plan.entry.expected_bytes, resume=resume
    )
    if not archive.is_file():  # pragma: no cover - defensive
        raise NotFoundError(
            msg("FILE_NOT_FOUND", path=str(archive)), details={"path": str(archive)}
        )

    digest = sha256_file(archive)
    print(f"sha256: {digest}")
    if plan.entry.sha256 and digest != plan.entry.sha256:
        raise InputValidationError(
            msg("VALIDATION_ERROR"),
            details={"reason": "sha256_mismatch", "expected": plan.entry.sha256, "got": digest},
        )
    if not plan.entry.sha256:
        print(
            "note: catalog.yaml carries no sha256 for this entry yet — record the digest above "
            "in src/aicls/data/catalog.yaml so later verifications are exact."
        )

    if extract:
        members = extract_archive(archive, plan.dest / "raw")
        print(f"extracted {members} members into {plan.dest / 'raw'}")
        if not keep:
            archive.unlink(missing_ok=True)

    manifest_path = write_manifest(plan.entry, plan.dest)
    print(f"manifest: {manifest_path}")
    print(f"licence: {plan.entry.license} — NonCommercial and ShareAlike apply to derived models")
    if extract and not keep:
        print(
            "next: python training/scripts/prepare_mathwriting.py "
            f"--src {plan.dest / 'raw'} --out {plan.dest / 'shards'}"
        )
    else:
        print(
            "next: python training/scripts/prepare_mathwriting.py "
            f"--archive {archive} --out {plan.dest / 'shards'}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
