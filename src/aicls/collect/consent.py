"""Versioned consent texts loaded from a package resource (KVKK audit trail).

The Turkish wording a participant sees is *data*, not a user-interface string: it is versioned,
hashed and copied into ``consent_versions`` so that every acceptance points at exactly the words
that were shown (docs/ml/veri-yonetimi.md §4). It therefore lives in
``aicls/collect/resources/consent_texts.yaml`` rather than in :mod:`aicls.core.messages_tr`,
which holds the interchangeable interface strings.

A published version is immutable: a wording change means a new entry with a new ``version`` and
``effective_from`` date, never an edit, because the SHA-256 digest is the audit evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import lru_cache
from typing import Any

import yaml  # type: ignore[import-untyped]  # PyYAML ships no stubs and types-PyYAML is absent

from aicls.core.errors import InputValidationError, NotFoundError
from aicls.core.hashing import sha256_bytes
from aicls.core.messages_tr import msg
from aicls.core.paths import package_resource

#: Package and resource name of the consent text catalogue.
CONSENT_PACKAGE = "aicls.collect"
CONSENT_RESOURCE = "resources/consent_texts.yaml"

#: Maximum length of a consent version label (``consent_versions.version`` is ``VARCHAR(16)``).
MAX_VERSION_LENGTH = 16


@dataclass(frozen=True, slots=True)
class ConsentText:
    """One published version of the Turkish consent text.

    Attributes:
        version: Short label such as ``"1.0"``; unique across the catalogue.
        text_tr: The full Turkish wording shown to the participant.
        sha256: Hex SHA-256 of ``text_tr`` encoded as UTF-8 — the audit digest stored with every
            acceptance.
        effective_from: First day on which this version may be accepted.
    """

    version: str
    text_tr: str
    sha256: str
    effective_from: date

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable form matching ``ConsentTextOut`` (docs/mimari/api.md §3.8)."""
        return {
            "version": self.version,
            "text_tr": self.text_tr,
            "sha256": self.sha256,
            "effective_from": self.effective_from.isoformat(),
        }


def _coerce_date(value: Any, *, version: str) -> date:
    """Return ``value`` as a :class:`datetime.date` (YAML gives ``date`` or ``str``)."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            msg("VALIDATION_ERROR"),
            details={"field": "effective_from", "version": version, "value": str(value)},
        ) from exc


def _build(entry: Any, *, index: int) -> ConsentText:
    """Validate one catalogue entry and turn it into a :class:`ConsentText`."""
    if not isinstance(entry, dict):
        raise InputValidationError(
            msg("VALIDATION_ERROR"), details={"resource": CONSENT_RESOURCE, "index": index}
        )
    version = str(entry.get("version", "")).strip()
    text_tr = str(entry.get("text_tr", "")).strip()
    if not version or len(version) > MAX_VERSION_LENGTH or not text_tr:
        raise InputValidationError(
            msg("VALIDATION_ERROR"),
            details={"resource": CONSENT_RESOURCE, "index": index, "version": version},
        )
    return ConsentText(
        version=version,
        text_tr=text_tr,
        sha256=sha256_bytes(text_tr.encode("utf-8")),
        effective_from=_coerce_date(entry.get("effective_from"), version=version),
    )


@lru_cache(maxsize=1)
def consent_texts() -> tuple[ConsentText, ...]:
    """Return every published consent version, oldest ``effective_from`` first.

    The catalogue is parsed once per process (the resource never changes at runtime).

    Raises:
        NotFoundError: If the package resource is missing.
        InputValidationError: If the catalogue is malformed, empty or has duplicate versions.
    """
    path = package_resource(CONSENT_PACKAGE, CONSENT_RESOURCE)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = document.get("consents") if isinstance(document, dict) else None
    if not isinstance(entries, list) or not entries:
        raise InputValidationError(
            msg("VALIDATION_ERROR"), details={"resource": CONSENT_RESOURCE, "reason": "empty"}
        )
    texts = tuple(
        sorted(
            (_build(entry, index=index) for index, entry in enumerate(entries)),
            key=lambda text: (text.effective_from, text.version),
        )
    )
    versions = [text.version for text in texts]
    if len(set(versions)) != len(versions):
        raise InputValidationError(
            msg("VALIDATION_ERROR"),
            details={"resource": CONSENT_RESOURCE, "reason": "duplicate_version"},
        )
    return texts


def current_consent(*, on: date | None = None) -> ConsentText:
    """Return the consent version in force on ``on`` (default: today, UTC).

    Args:
        on: Day to evaluate; the newest version whose ``effective_from`` is not in the future
            wins, so a future wording can be staged in the catalogue ahead of time.

    Raises:
        NotFoundError: If no version is effective yet on that day.
    """
    day = on if on is not None else datetime.now(UTC).date()
    effective = [text for text in consent_texts() if text.effective_from <= day]
    if not effective:
        raise NotFoundError(
            msg("NOT_FOUND"), details={"resource": CONSENT_RESOURCE, "on": day.isoformat()}
        )
    return effective[-1]


def get_consent_text(version: str) -> ConsentText:
    """Return the published consent text labelled ``version``.

    Raises:
        NotFoundError: If no such version exists.
    """
    for text in consent_texts():
        if text.version == version:
            return text
    raise NotFoundError(
        msg("NOT_FOUND"),
        details={
            "resource": CONSENT_RESOURCE,
            "version": version,
            "known": [text.version for text in consent_texts()],
        },
    )


__all__ = [
    "CONSENT_PACKAGE",
    "CONSENT_RESOURCE",
    "MAX_VERSION_LENGTH",
    "ConsentText",
    "consent_texts",
    "current_consent",
    "get_consent_text",
]
