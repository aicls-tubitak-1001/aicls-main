"""Zero-dependency foundations: settings, paths, logging, errors, Turkish messages, gates.

Public contract (modules/core.json ``public_api``, docs/mimari/moduller.md#aiclscore). Every other
``aicls`` package imports from here; this package imports nothing from ``aicls`` (ADR-0002) and
never imports torch/transformers.
"""

from __future__ import annotations

from aicls.core.errors import (
    ERROR_TYPE_BASE,
    HTTP_STATUS_BY_CODE,
    AiclsError,
    ErrorCode,
    InputValidationError,
    InternalError,
    NotFoundError,
)
from aicls.core.gates import (
    GATE_NAMES,
    GateEntry,
    GateError,
    GateLedgerError,
    GateName,
    gate_status,
    is_gate_approved_in_ledger,
    read_gate_ledger,
    require_gate,
)
from aicls.core.hashing import (
    canonical_json,
    fingerprint_jsonl,
    sha256_bytes,
    sha256_file,
    sha256_json,
)
from aicls.core.ids import is_ulid, short_hash, ulid, ulid_timestamp_ms
from aicls.core.latex_safety import (
    ALLOWED_COMMANDS,
    ALLOWED_ENVIRONMENTS,
    FORBIDDEN_COMMANDS,
    MAX_LATEX_LENGTH,
    find_disallowed_commands,
    sanitize_latex,
)
from aicls.core.logging import (
    bind_request_id,
    clear_request_context,
    configure_logging,
    get_logger,
    log_error,
    request_context,
)
from aicls.core.messages_tr import MESSAGES, MSG_UNSUPPORTED_PROBLEM, TITLES, msg
from aicls.core.paths import (
    data_root,
    ensure_dir,
    find_repo_root,
    models_root,
    package_resource,
    repo_root,
    runs_root,
)
from aicls.core.settings import Settings, get_settings, settings_override
from aicls.core.timing import StageTimer
from aicls.core.turkish import TR_SUFFIXES, strip_diacritics, strip_suffix, tr_casefold, tr_lower

__all__ = [
    "ALLOWED_COMMANDS",
    "ALLOWED_ENVIRONMENTS",
    "ERROR_TYPE_BASE",
    "FORBIDDEN_COMMANDS",
    "GATE_NAMES",
    "HTTP_STATUS_BY_CODE",
    "MAX_LATEX_LENGTH",
    "MESSAGES",
    "MSG_UNSUPPORTED_PROBLEM",
    "TITLES",
    "TR_SUFFIXES",
    "AiclsError",
    "ErrorCode",
    "GateEntry",
    "GateError",
    "GateLedgerError",
    "GateName",
    "InputValidationError",
    "InternalError",
    "NotFoundError",
    "Settings",
    "StageTimer",
    "bind_request_id",
    "canonical_json",
    "clear_request_context",
    "configure_logging",
    "data_root",
    "ensure_dir",
    "find_disallowed_commands",
    "find_repo_root",
    "fingerprint_jsonl",
    "gate_status",
    "get_logger",
    "get_settings",
    "is_gate_approved_in_ledger",
    "is_ulid",
    "log_error",
    "models_root",
    "msg",
    "package_resource",
    "read_gate_ledger",
    "repo_root",
    "request_context",
    "require_gate",
    "runs_root",
    "sanitize_latex",
    "settings_override",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "short_hash",
    "strip_diacritics",
    "strip_suffix",
    "tr_casefold",
    "tr_lower",
    "ulid",
    "ulid_timestamp_ms",
]
