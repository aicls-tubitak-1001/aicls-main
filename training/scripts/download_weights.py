"""Download pre-trained weights — **GATED: WEIGHTS_DOWNLOAD** (ADR-0012, egitim-altyapisi §9).

The second (and last) file in the repository allowed to contain network code. Everything the
rest of the tree does with a checkpoint goes through a **local directory** under
``AICLS_MODELS_ROOT/pretrained/<name>/``; no ``from_pretrained("org/name")`` exists anywhere
else, and ``tools/check_gates.py`` fails the build if one appears.

Same four-step shape as ``download_mathwriting.py``:

1. print the catalog entry (repo id, licence, size, sha256, target directory, free disk);
2. ``--dry-run`` stops there without asking for the gate;
3. :func:`aicls.core.gates.require_gate` demands flag **and** environment variable **and** an
   ``approved`` ledger row, else exit **3** with every missing condition in Turkish;
4. only then :func:`download_snapshot` runs — the single network function of this file.

Usage::

    python training/scripts/download_weights.py --list
    python training/scripts/download_weights.py --model whisper-small --dry-run
    python training/scripts/download_weights.py --model openai/whisper-small --i-have-approval

``--model`` accepts either the catalog key (``whisper-small``) or the full repository id
(``openai/whisper-small``); the id is *derived from the catalog*, never typed into the code.

Exit codes: ``0`` success or dry run, ``1`` a typed AI-CLS error, ``3`` the gate refused.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:  # plain checkout without an editable install
    sys.path.insert(0, str(_SRC))

from aicls.core.errors import AiclsError, InputValidationError, InternalError  # noqa: E402
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
from aicls.data import CatalogEntry, DataManifest, load_catalog  # noqa: E402

_log = get_logger(__name__)

#: Approval gate this script sits behind.
GATE: GateName = "WEIGHTS_DOWNLOAD"

#: Sub-directory of ``AICLS_MODELS_ROOT`` that holds untouched upstream checkpoints.
PRETRAINED_DIRNAME = "pretrained"

#: Files a checkpoint directory is expected to gain (used for the post-download report only).
WEIGHT_FILENAMES: tuple[str, ...] = ("model.safetensors", "pytorch_model.bin")

GIB = float(1 << 30)


@dataclass(frozen=True, slots=True)
class WeightsPlan:
    """What ``--model`` would fetch, where to and at what cost.

    Attributes:
        entry: Catalog row of the checkpoint.
        repo_id: Upstream repository id, derived from the catalog (``source/version``).
        dest: Local directory the files would land in.
        free_bytes: Free space on that volume.
        needed_bytes: Announced download size.
        gate: Approval gate name.
        gate_open: Whether all three conditions currently hold.
        missing: Which conditions are missing.
    """

    entry: CatalogEntry
    repo_id: str
    dest: Path
    free_bytes: int
    needed_bytes: int
    gate: str
    gate_open: bool
    missing: tuple[str, ...]

    @property
    def enough_disk(self) -> bool:
        """Whether the volume can hold the announced download."""
        return self.free_bytes >= self.needed_bytes

    def to_dict(self) -> dict[str, Any]:
        """JSON document of the plan (``--json``), for an approval-request PR."""
        return {
            "action": "download",
            "weights": self.entry.name,
            "repo_id": self.repo_id,
            "version": self.entry.version,
            "license": self.entry.license,
            "expected_bytes": self.entry.expected_bytes,
            "sha256": self.entry.sha256,
            "dest": str(self.dest),
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
        return "\n".join(
            [
                f"plan: download {self.entry.name} ({self.entry.kind})",
                f"  title          : {self.entry.title}",
                f"  repo id        : {self.repo_id}",
                f"  version        : {self.entry.version}",
                f"  licence        : {self.entry.license}",
                f"  download size  : {self.entry.expected_bytes} B "
                f"({self.entry.expected_bytes / GIB:.2f} GiB)",
                f"  sha256         : {sha}",
                f"  target dir     : {self.dest}",
                f"  free disk      : {self.free_bytes / GIB:.2f} GiB "
                f"({'ok' if self.enough_disk else 'NOT ENOUGH'})",
                f"  gate           : {self.gate}",
                f"  gate status    : {'open' if self.gate_open else 'closed'}"
                + (f" — missing: {', '.join(self.missing)}" if self.missing else ""),
            ]
        )


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

    Reports only; it never approves, never writes the invocation log and never unlocks anything.

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


def weights_entries() -> dict[str, CatalogEntry]:
    """Every ``weights`` row of ``catalog.yaml``, keyed by catalog name."""
    return dict(load_catalog().weights)


def repo_id_of(entry: CatalogEntry) -> str:
    """Upstream repository id of a catalog row (``<source>/<version>``).

    The id is *derived*, so the only place a hub coordinate is written down stays
    ``src/aicls/data/catalog.yaml`` — where it is reviewable data, not code.

    Raises:
        InputValidationError: If the row has no ``source``.
    """
    if not entry.source:
        raise InputValidationError(
            msg("VALIDATION_ERROR"),
            details={"reason": "catalog_source_missing", "entry": entry.name},
        )
    return f"{entry.source}/{entry.version}"


def resolve_entry(model: str) -> CatalogEntry:
    """Look ``model`` up as a catalog key or as a full repository id.

    Args:
        model: ``"whisper-small"`` or ``"openai/whisper-small"``.

    Returns:
        The catalog row.

    Raises:
        InputValidationError: If nothing in the catalog matches. The message names every known
            key, because guessing a hub id is exactly what this project forbids.
    """
    rows = weights_entries()
    if model in rows:
        return rows[model]
    for entry in rows.values():
        if repo_id_of(entry) == model:
            return entry
    raise InputValidationError(
        msg("VALIDATION_ERROR"),
        details={
            "reason": "unknown_weights_entry",
            "requested": model,
            "available": sorted(rows),
            "hint": "add the checkpoint to src/aicls/data/catalog.yaml first (ADR-0012)",
        },
    )


def build_plan(
    model: str,
    *,
    dest: Path | None = None,
    cli_flag: bool = False,
    ledger_path: Path | None = None,
) -> WeightsPlan:
    """Describe the download without performing any part of it.

    Args:
        model: Catalog key or repository id.
        dest: Target directory; ``AICLS_MODELS_ROOT/pretrained/<name>`` when omitted.
        cli_flag: Whether ``--i-have-approval`` was given (affects the reported status only).
        ledger_path: Ledger override (tests).

    Returns:
        The plan; nothing on disk is touched.
    """
    entry = resolve_entry(model)
    settings = get_settings()
    root = settings.models_root or (settings.data_root / "models")
    target = Path(dest) if dest is not None else root / PRETRAINED_DIRNAME / entry.name
    missing = gate_status(GATE, cli_flag=cli_flag, ledger_path=ledger_path)
    return WeightsPlan(
        entry=entry,
        repo_id=repo_id_of(entry),
        dest=target,
        free_bytes=free_bytes_for(target),
        needed_bytes=int(entry.expected_bytes),
        gate=GATE,
        gate_open=not missing,
        missing=tuple(missing),
    )


# ----------------------------------------------------------------------------------------------
# THE ONLY NETWORK CODE IN THIS FILE
# ----------------------------------------------------------------------------------------------
def download_snapshot(repo_id: str, dest: Path, *, allow_patterns: list[str] | None = None) -> Path:
    """Fetch ``repo_id`` into ``dest``. **Never call this before ``require_gate`` returned.**

    ``huggingface_hub`` is imported inside the function so that importing this module cannot
    reach the network even indirectly, and so that a missing dependency is a typed error rather
    than an ``ImportError`` traceback.

    Args:
        repo_id: Upstream repository id derived from the catalog.
        dest: Local directory to populate (created here, never before the gate).
        allow_patterns: Restrict the snapshot to these file patterns.

    Returns:
        ``dest``.

    Raises:
        InternalError: If ``huggingface_hub`` is not installed.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on the venv
        raise InternalError(
            msg("INTERNAL"),
            details={
                "reason": "huggingface_hub_not_installed",
                "hint": "pip install huggingface_hub (behind the CUDA_INSTALL / ml extra)",
            },
        ) from exc

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        allow_patterns=allow_patterns,
    )
    _log.info("training.download_weights.fetched", repo_id=repo_id, dest=str(dest))
    return dest


def digest_report(dest: Path) -> dict[str, str]:
    """sha256 of every weight file found in ``dest`` (recorded in the approval trail)."""
    return {name: sha256_file(dest / name) for name in WEIGHT_FILENAMES if (dest / name).is_file()}


def write_manifest(entry: CatalogEntry, repo_id: str, dest: Path) -> Path:
    """Hash the downloaded directory into ``manifest.json``."""
    manifest = DataManifest.build(
        dest,
        entry.name,
        license=entry.license,
        source_url=f"hf://{repo_id}",
        version=entry.version,
    )
    return manifest.save(dest)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the script."""
    parser = argparse.ArgumentParser(
        prog="download_weights.py",
        description="Download pre-trained weights (GATED: WEIGHTS_DOWNLOAD, ADR-0012).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="catalog key (whisper-small) or repository id (openai/whisper-small)",
    )
    parser.add_argument("--list", action="store_true", help="list the catalog weights and exit")
    parser.add_argument("--dest", type=Path, default=None, help="target directory override")
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
        "--allow-patterns",
        nargs="*",
        default=None,
        help="restrict the snapshot to these file patterns",
    )
    return parser


def main(argv: list[str] | None = None, *, ledger_path: Path | None = None) -> int:
    """Entry point.

    Args:
        argv: Command line (``sys.argv[1:]`` when omitted).
        ledger_path: Approval-ledger override. Deliberately **not** a command-line flag: the
            ledger is the auditable half of the lock; only tests point it elsewhere.

    Returns:
        Process exit code: 0 ok/dry run, 1 typed error, 3 gate refused.
    """
    args = build_parser().parse_args(argv)
    if args.list or not args.model:
        # --list means "show the catalog and stop"; no --model at all is a usage error.
        for name, entry in sorted(weights_entries().items()):
            print(f"{name:26s} {repo_id_of(entry):34s} {entry.describe()}")
        if args.list:
            return 0
        print("\nnothing to do: pass --model <catalog key|repo id>", file=sys.stderr)
        return 1

    try:
        plan = build_plan(
            args.model,
            dest=args.dest,
            cli_flag=bool(args.i_have_approval),
            ledger_path=ledger_path,
        )
    except AiclsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(f"details: {exc.details}", file=sys.stderr)
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
        return _run_approved(plan, allow_patterns=args.allow_patterns)
    except AiclsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return int(exc.exit_code)


def _run_approved(plan: WeightsPlan, *, allow_patterns: list[str] | None) -> int:
    """Everything that happens **after** the gate accepted; never reached otherwise.

    Raises:
        InputValidationError: On insufficient disk space.
        InternalError: If ``huggingface_hub`` is missing.
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
    print(f"downloading {plan.repo_id} -> {plan.dest}")
    dest = download_snapshot(plan.repo_id, plan.dest, allow_patterns=allow_patterns)
    for name, digest in digest_report(dest).items():
        print(f"sha256 {name}: {digest}")
    manifest_path = write_manifest(plan.entry, plan.repo_id, dest)
    print(f"manifest: {manifest_path}")
    print(
        "next: point a config's model.base_dir at this directory "
        f"(model: {{base_dir: {dest}}}) — never a hub id."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
