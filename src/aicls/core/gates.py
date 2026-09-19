"""Approval gates (ADR-0012): triple lock for dataset/weight downloads, CUDA install, training.

A gated action runs only when **all three** hold:

1. the caller passed ``--i-have-approval`` (``cli_flag=True``);
2. ``AICLS_GATE_<NAME>=approved`` is set (environment or ``.env`` via :class:`Settings`);
3. ``docs/gelistirme/onay-defteri.md`` has an ``approved`` row for the gate whose latest row is
   not ``revoked``.

Any other outcome raises :class:`GateError` (exit code 3, HTTP 503 ``GATE_BLOCKED``) whose Turkish
message lists the missing conditions. CI (``CI``/``GITHUB_ACTIONS`` set) always refuses. Every
successful check is appended to ``AICLS_DATA_ROOT/gate_invocations.log`` as one JSON line.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
import datetime as dt
import getpass
import json
import os
from pathlib import Path
import re
from typing import Any, Literal, get_args

from aicls.core import messages_tr
from aicls.core.errors import AiclsError, ErrorCode, InputValidationError, NotFoundError
from aicls.core.hashing import sha256_json
from aicls.core.logging import get_logger
from aicls.core.paths import ensure_dir, find_repo_root
from aicls.core.settings import GateName, Settings, get_settings

GATE_NAMES: tuple[GateName, ...] = get_args(GateName)
GateStatus = Literal["approved", "revoked"]
LEDGER_RELATIVE_PATH = Path("docs") / "gelistirme" / "onay-defteri.md"
LEDGER_HEADER = "| KAPI | durum | tarih | kim | kapsam |"
INVOCATION_LOG_NAME = "gate_invocations.log"
CLI_FLAG = "--i-have-approval"

_SEPARATOR_RE = re.compile(r"^\|(\s*:?-+:?\s*\|){5}$")
_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class GateEntry:
    """One row of the approval ledger table."""

    gate: str
    status: str
    date: str
    who: str
    scope: str


class GateError(AiclsError):
    """A gated action was refused; ``exit_code`` 3 (ADR-0012)."""

    exit_code = 3
    default_code = ErrorCode.GATE_BLOCKED

    def __init__(
        self,
        message_tr: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(self.default_code, message_tr, details, None)


class GateLedgerError(InputValidationError):
    """The ledger file is malformed (header, separator or a row)."""


def default_ledger_path() -> Path:
    """``<repo>/docs/gelistirme/onay-defteri.md``."""
    return find_repo_root() / LEDGER_RELATIVE_PATH


def _ledger_error(path: Path, line_no: int, reason: str) -> GateLedgerError:
    return GateLedgerError(
        messages_tr.msg("GATE_LEDGER_INVALID", path=str(path), line=line_no, reason=reason),
        details={"path": str(path), "line": line_no, "reason": reason},
    )


def read_gate_ledger(path: Path | None = None) -> list[GateEntry]:
    """Parse the approval ledger table (``| KAPI | durum | tarih | kim | kapsam |``).

    Rows are read until the first non-table line. Format is strict: five cells, a known gate
    name, ``approved``/``revoked``, an ISO date and non-empty ``kim``/``kapsam``.

    Raises:
        NotFoundError: If the file does not exist.
        GateLedgerError: On a malformed header, separator or row.
    """
    ledger = path or default_ledger_path()
    if not ledger.is_file():
        raise NotFoundError(
            messages_tr.msg("FILE_NOT_FOUND", path=str(ledger)), details={"path": str(ledger)}
        )
    lines = ledger.read_text(encoding="utf-8").splitlines()
    header_idx = next((i for i, ln in enumerate(lines) if ln.strip() == LEDGER_HEADER), None)
    if header_idx is None:
        raise _ledger_error(ledger, 0, f"header {LEDGER_HEADER!r} not found")
    if header_idx + 1 >= len(lines) or not _SEPARATOR_RE.match(lines[header_idx + 1].strip()):
        raise _ledger_error(ledger, header_idx + 2, "5-column separator row expected")
    entries: list[GateEntry] = []
    for offset, raw in enumerate(lines[header_idx + 2 :], start=header_idx + 3):
        text = raw.strip()
        if not text.startswith("|"):
            break
        cells = [c.strip() for c in text.strip("|").split("|")]
        if len(cells) != 5:
            raise _ledger_error(ledger, offset, f"row must have 5 cells, got {len(cells)}")
        gate, status, date, who, scope = cells
        if gate not in GATE_NAMES:
            raise _ledger_error(ledger, offset, f"unknown gate {gate!r}")
        if status not in get_args(GateStatus):
            raise _ledger_error(ledger, offset, f"status must be approved|revoked: {status!r}")
        try:
            dt.date.fromisoformat(date)
        except ValueError as exc:
            raise _ledger_error(ledger, offset, f"date must be YYYY-MM-DD: {date!r}") from exc
        if not who:
            raise _ledger_error(ledger, offset, "kim column must not be empty")
        if not scope:
            raise _ledger_error(ledger, offset, "kapsam column must not be empty")
        entries.append(GateEntry(gate=gate, status=status, date=date, who=who, scope=scope))
    return entries


def latest_ledger_entry(name: str, entries: Iterable[GateEntry]) -> GateEntry | None:
    """Last ledger row for ``name`` (the newest row is authoritative), or None."""
    result: GateEntry | None = None
    for entry in entries:
        if entry.gate == name:
            result = entry
    return result


def is_gate_approved_in_ledger(name: str, entries: Iterable[GateEntry]) -> bool:
    """True when the latest ledger row for ``name`` has status ``approved``."""
    latest = latest_ledger_entry(name, entries)
    return latest is not None and latest.status == "approved"


def is_ci_environment(environ: Mapping[str, str] | None = None) -> bool:
    """True under CI (``CI`` or ``GITHUB_ACTIONS`` truthy); gates always refuse there."""
    env = os.environ if environ is None else environ
    return any(env.get(key, "").strip().lower() in _TRUTHY for key in ("CI", "GITHUB_ACTIONS"))


def _git_sha(repo: Path) -> str | None:
    """Best-effort HEAD commit from ``.git`` files (no subprocess, no git dependency)."""
    head = repo / ".git" / "HEAD"
    try:
        content = head.read_text(encoding="utf-8").strip()
        if not content.startswith("ref:"):
            return content or None
        ref = content.removeprefix("ref:").strip()
        ref_file = repo / ".git" / ref
        if ref_file.is_file():
            return ref_file.read_text(encoding="utf-8").strip() or None
        packed = repo / ".git" / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    except OSError:
        return None
    return None


def _config_hash(settings: Settings) -> str:
    dump = settings.model_dump(mode="json")  # SecretStr fields are masked in json mode
    return sha256_json(dump)[:16]


def _append_invocation_log(
    settings: Settings, name: str, ledger: Path, entry: GateEntry | None
) -> Path:
    record: dict[str, Any] = {
        "gate": name,
        "at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "who": _current_user(),
        "git_sha": _git_sha(_safe_repo_root(ledger)),
        "config_hash": _config_hash(settings),
        "ledger": str(ledger),
        "ledger_entry": asdict(entry) if entry is not None else None,
        "cwd": str(Path.cwd()),
    }
    log_path = ensure_dir(settings.data_root) / INVOCATION_LOG_NAME
    with log_path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return log_path


def _current_user() -> str:
    try:
        return getpass.getuser()
    except (OSError, ImportError):  # pragma: no cover - depends on host account setup
        return "unknown"


def _safe_repo_root(ledger: Path) -> Path:
    try:
        return find_repo_root(ledger.parent)
    except AiclsError:
        return ledger.parent


def gate_status(
    name: GateName,
    *,
    cli_flag: bool = False,
    ledger_path: Path | None = None,
    settings: Settings | None = None,
) -> list[str]:
    """Read-only preview of the triple lock: which conditions are currently missing.

    This **reports**; it never approves anything, never writes the invocation log and never
    raises on a closed gate. It exists so that preparing an approval request is itself a
    gate-free job (docs/gelistirme/onay-defteri.md: run ``--dry-run --json`` and attach the plan
    to the PR) — a ``--dry-run`` that first demanded the approval it is being used to request
    would be a circular procedure.

    Args:
        name: One of :data:`GATE_NAMES`.
        cli_flag: Whether ``--i-have-approval`` was given.
        ledger_path: Ledger file override (tests); defaults to the repository ledger.
        settings: Settings override; defaults to :func:`get_settings`.

    Returns:
        The missing conditions, in the order :func:`require_gate` reports them: ``ci``,
        ``cli_flag``, ``env``, ``ledger``. Empty means the lock would currently open.

    Raises:
        InputValidationError: If ``name`` is not a known gate.
    """
    if name not in GATE_NAMES:
        raise InputValidationError(
            messages_tr.msg("GATE_UNKNOWN", gate=name), details={"gate": name}
        )
    missing: list[str] = []
    if is_ci_environment():
        missing.append("ci")
    if not cli_flag:
        missing.append("cli_flag")
    cfg = settings if settings is not None else get_settings()
    try:
        env_approved = cfg.is_gate_env_approved(name)
    except (AiclsError, ValueError):  # pragma: no cover - a broken .env is a closed gate
        env_approved = False
    if not env_approved:
        missing.append("env")
    ledger = ledger_path or default_ledger_path()
    try:
        entries = read_gate_ledger(ledger) if ledger.is_file() else []
    except AiclsError:  # a malformed ledger closes every gate, it never opens one
        entries = []
    if not is_gate_approved_in_ledger(name, entries):
        missing.append("ledger")
    return missing


def require_gate(
    name: GateName,
    *,
    cli_flag: bool,
    ledger_path: Path | None = None,
    settings: Settings | None = None,
) -> None:
    """Enforce the triple lock for ``name``; return silently when every condition holds.

    Args:
        name: ``DATASET_DOWNLOAD``, ``WEIGHTS_DOWNLOAD``, ``CUDA_INSTALL`` or ``TRAINING``.
        cli_flag: Whether ``--i-have-approval`` was given.
        ledger_path: Ledger file override (tests); defaults to the repository ledger.
        settings: Settings override; defaults to :func:`get_settings`.

    Raises:
        InputValidationError: If ``name`` is not a known gate.
        GateError: If any condition is missing (``details["missing"]`` lists ``cli_flag``,
            ``env``, ``ledger`` and/or ``ci``), exit code 3.
    """
    if name not in GATE_NAMES:
        raise InputValidationError(
            messages_tr.msg("GATE_UNKNOWN", gate=name), details={"gate": name}
        )
    cfg = settings if settings is not None else get_settings()
    ledger = ledger_path or default_ledger_path()

    missing: list[str] = []
    reasons: list[str] = []
    if is_ci_environment():
        missing.append("ci")
        reasons.append(messages_tr.msg("GATE_CI_REFUSED"))
    if not cli_flag:
        missing.append("cli_flag")
        reasons.append(messages_tr.msg("GATE_MISSING_FLAG"))
    if not cfg.is_gate_env_approved(name):
        missing.append("env")
        reasons.append(messages_tr.msg("GATE_MISSING_ENV", gate=name))
    entries = read_gate_ledger(ledger) if ledger.is_file() else []
    entry = latest_ledger_entry(name, entries)
    if not is_gate_approved_in_ledger(name, entries):
        missing.append("ledger")
        reasons.append(messages_tr.msg("GATE_MISSING_LEDGER", gate=name))

    if missing:
        get_logger(__name__).warning("gate_refused", gate=name, missing=missing, ledger=str(ledger))
        raise GateError(
            messages_tr.msg("GATE_REFUSED", gate=name, missing="; ".join(reasons)),
            details={
                "gate": name,
                "missing": missing,
                "cli_flag": CLI_FLAG,
                "env_var": f"AICLS_GATE_{name}",
                "ledger": str(ledger),
            },
        )

    log_path = _append_invocation_log(cfg, name, ledger, entry)
    get_logger(__name__).info(
        "gate_passed",
        gate=name,
        ledger=str(ledger),
        approved_by=entry.who if entry else None,
        approved_on=entry.date if entry else None,
        invocation_log=str(log_path),
    )


__all__ = [
    "CLI_FLAG",
    "GATE_NAMES",
    "INVOCATION_LOG_NAME",
    "LEDGER_HEADER",
    "LEDGER_RELATIVE_PATH",
    "GateEntry",
    "GateError",
    "GateLedgerError",
    "GateName",
    "GateStatus",
    "default_ledger_path",
    "gate_status",
    "is_ci_environment",
    "is_gate_approved_in_ledger",
    "latest_ledger_entry",
    "read_gate_ledger",
    "require_gate",
]
