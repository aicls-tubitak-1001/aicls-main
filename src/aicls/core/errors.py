"""Error hierarchy: :class:`ErrorCode` and :class:`AiclsError` with RFC 9457 rendering.

Every error raised by the platform is an :class:`AiclsError` (or subclass) carrying a stable
:class:`ErrorCode`, a Turkish user message, structured ``details`` and an HTTP status. The API
layer converts it to ``application/problem+json`` via :meth:`AiclsError.to_problem`
(docs/mimari/api.md §2); the CLI maps :attr:`AiclsError.exit_code` to the process exit code.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, ClassVar

from aicls.core import messages_tr

#: Base of the ``type`` URI in problem details (docs/mimari/api.md §2).
ERROR_TYPE_BASE = "https://ai-cls.example/errors"

#: Keys of a problem-details document that ``details`` must not override.
_RESERVED_PROBLEM_KEYS = frozenset({"type", "title", "status", "detail_tr", "code", "instance"})


class ErrorCode(StrEnum):
    """Stable machine-readable error codes (docs/mimari/api.md §2)."""

    PARSE_ERROR = "PARSE_ERROR"
    UNSUPPORTED_CONSTRUCT = "UNSUPPORTED_CONSTRUCT"
    UNSUPPORTED_PROBLEM = "UNSUPPORTED_PROBLEM"
    CAS_TIMEOUT = "CAS_TIMEOUT"
    NLU_NO_INTENT = "NLU_NO_INTENT"
    OCR_INPUT_INVALID = "OCR_INPUT_INVALID"
    ASR_INPUT_INVALID = "ASR_INPUT_INVALID"
    UPLOAD_INVALID = "UPLOAD_INVALID"
    CONSENT_REQUIRED = "CONSENT_REQUIRED"
    AUTH_INVALID = "AUTH_INVALID"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    RATE_LIMITED = "RATE_LIMITED"
    TRAINING_BUSY = "TRAINING_BUSY"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    GATE_BLOCKED = "GATE_BLOCKED"
    INTERNAL = "INTERNAL"


#: Default HTTP status per code (docs/mimari/api.md §2 table).
HTTP_STATUS_BY_CODE: dict[ErrorCode, int] = {
    ErrorCode.PARSE_ERROR: 422,
    ErrorCode.UNSUPPORTED_CONSTRUCT: 422,
    ErrorCode.UNSUPPORTED_PROBLEM: 422,
    ErrorCode.CAS_TIMEOUT: 504,
    ErrorCode.NLU_NO_INTENT: 422,
    ErrorCode.OCR_INPUT_INVALID: 400,
    ErrorCode.ASR_INPUT_INVALID: 400,
    ErrorCode.UPLOAD_INVALID: 400,
    ErrorCode.CONSENT_REQUIRED: 403,
    ErrorCode.AUTH_INVALID: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.TRAINING_BUSY: 503,
    ErrorCode.VALIDATION_ERROR: 422,
    ErrorCode.GATE_BLOCKED: 503,
    ErrorCode.INTERNAL: 500,
}


class AiclsError(Exception):
    """Base class for every error raised by AI-CLS code.

    Args:
        code: An :class:`ErrorCode` or its string value. Unknown strings degrade to
            :attr:`ErrorCode.INTERNAL` and the original value is kept in ``details``.
        message_tr: Turkish user-facing detail text; defaults to the registered message for the
            code (:mod:`aicls.core.messages_tr`).
        details: Extra structured fields (``position``, ``errors``, ...) merged into the
            problem-details document. Copied, never shared with the caller.
        http_status: HTTP status override; defaults to :data:`HTTP_STATUS_BY_CODE`.

    Attributes:
        exit_code: Process exit code the CLI uses when this error escapes (1 by default,
            3 for :class:`aicls.core.gates.GateError`).
    """

    exit_code: ClassVar[int] = 1
    default_code: ClassVar[ErrorCode] = ErrorCode.INTERNAL

    code: ErrorCode
    message_tr: str
    details: dict[str, Any]
    http_status: int

    def __init__(
        self,
        code: ErrorCode | str | None = None,
        message_tr: str | None = None,
        details: Mapping[str, Any] | None = None,
        http_status: int | None = None,
    ) -> None:
        resolved_details: dict[str, Any] = dict(details) if details else {}
        if code is None:
            resolved_code = self.default_code
        elif isinstance(code, ErrorCode):
            resolved_code = code
        else:
            try:
                resolved_code = ErrorCode(code)
            except ValueError:
                resolved_code = ErrorCode.INTERNAL
                resolved_details.setdefault("original_code", code)
        self.code = resolved_code
        self.message_tr = message_tr if message_tr is not None else messages_tr.msg(resolved_code)
        self.details = resolved_details
        self.http_status = (
            http_status if http_status is not None else HTTP_STATUS_BY_CODE[resolved_code]
        )
        super().__init__(f"{self.code.value}: {self.message_tr}")

    def __str__(self) -> str:
        return f"{self.code.value}: {self.message_tr}"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(code={self.code.value!r}, message_tr={self.message_tr!r}, "
            f"details={self.details!r}, http_status={self.http_status})"
        )

    @property
    def title_tr(self) -> str:
        """Short Turkish title for the code (RFC 9457 ``title``)."""
        return messages_tr.TITLES[self.code.value]

    def to_problem(self, *, instance: str | None = None) -> dict[str, Any]:
        """Render as an RFC 9457 problem-details dictionary (docs/mimari/api.md §2).

        Args:
            instance: Request or operation id (ULID) to place in ``instance``; omitted if None.

        Returns:
            A JSON-serialisable dict with ``type``, ``title``, ``status``, ``detail_tr``,
            ``code`` and optionally ``instance``, followed by the ``details`` fields. Detail keys
            colliding with the reserved fields are nested under ``details`` instead.
        """
        problem: dict[str, Any] = {
            "type": f"{ERROR_TYPE_BASE}/{self.code.value}",
            "title": self.title_tr,
            "status": self.http_status,
            "detail_tr": self.message_tr,
            "code": self.code.value,
        }
        if instance is not None:
            problem["instance"] = instance
        collisions: dict[str, Any] = {}
        for key, value in self.details.items():
            if key in _RESERVED_PROBLEM_KEYS or key == "details":
                collisions[key] = value
            else:
                problem[key] = value
        if collisions:
            problem["details"] = collisions
        return problem


class InputValidationError(AiclsError):
    """Input failed validation (``VALIDATION_ERROR``, 422)."""

    default_code = ErrorCode.VALIDATION_ERROR

    def __init__(
        self,
        message_tr: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(self.default_code, message_tr, details, http_status)


class NotFoundError(AiclsError):
    """A record, file or resource does not exist (``NOT_FOUND``, 404)."""

    default_code = ErrorCode.NOT_FOUND

    def __init__(
        self,
        message_tr: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(self.default_code, message_tr, details, http_status)


class InternalError(AiclsError):
    """Unexpected failure (``INTERNAL``, 500)."""

    default_code = ErrorCode.INTERNAL

    def __init__(
        self,
        message_tr: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(self.default_code, message_tr, details, http_status)


__all__ = [
    "ERROR_TYPE_BASE",
    "HTTP_STATUS_BY_CODE",
    "AiclsError",
    "ErrorCode",
    "InputValidationError",
    "InternalError",
    "NotFoundError",
]
