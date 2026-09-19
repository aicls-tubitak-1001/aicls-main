"""SHA-256 helpers for files, JSON documents and JSONL datasets.

:func:`fingerprint_jsonl` is the dataset identity written into ``EvalReport.dataset_fingerprint``
(docs/ml/degerlendirme.md). It is **order-sensitive**: reordering rows changes the fingerprint,
because row order is part of a frozen test set. Line-ending style and a trailing newline are
ignored so the same data checked out with CRLF or LF fingerprints identically.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from aicls.core import messages_tr
from aicls.core.errors import InputValidationError, NotFoundError

_CHUNK = 1 << 20
FINGERPRINT_PREFIX = "sha256:"


def _require_file(p: Path) -> Path:
    if not p.is_file():
        raise NotFoundError(
            messages_tr.msg("FILE_NOT_FOUND", path=str(p)), details={"path": str(p)}
        )
    return p


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(p: Path) -> str:
    """Hex SHA-256 of a file's bytes, streamed in 1 MiB chunks.

    Raises:
        NotFoundError: If ``p`` is not an existing file.
    """
    digest = hashlib.sha256()
    with _require_file(p).open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, set | frozenset):
        return sorted(obj, key=repr)
    if isinstance(obj, bytes):
        return obj.hex()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    """Canonical JSON text: sorted keys, compact separators, UTF-8 (no ASCII escaping).

    Paths become POSIX strings, sets become sorted lists, bytes become hex.

    Raises:
        InputValidationError: If ``obj`` contains a value that cannot be serialised.
    """
    try:
        return json.dumps(
            obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default
        )
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            messages_tr.msg("JSON_NOT_SERIALISABLE", reason=str(exc)), details={"reason": str(exc)}
        ) from exc


def sha256_json(obj: Any) -> str:
    """Hex SHA-256 of the canonical JSON form of ``obj`` (key order independent)."""
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def fingerprint_jsonl(p: Path) -> str:
    """Order-sensitive dataset fingerprint ``"sha256:<hex>"`` of a JSONL file.

    Each line is hashed as ``line + "\\n"`` after stripping ``\\r``; a missing final newline does
    not matter. Rows are *not* parsed or canonicalised: byte-level content is the identity.

    Raises:
        NotFoundError: If ``p`` is not an existing file.
    """
    digest = hashlib.sha256()
    with _require_file(p).open("rb") as fh:
        for raw in fh:
            line = raw.rstrip(b"\r\n")
            digest.update(line)
            digest.update(b"\n")
    return FINGERPRINT_PREFIX + digest.hexdigest()


__all__ = [
    "FINGERPRINT_PREFIX",
    "canonical_json",
    "fingerprint_jsonl",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
]
