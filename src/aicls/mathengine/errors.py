"""CAS errors and their process-boundary encoding (ADR-0016).

Two errors are specific to the math engine:

* :class:`UnsupportedProblem` -- the İP-4 risk-plan policy. SymPy could not produce a usable
  answer (an integral that stays unevaluated, a ``ConditionSet``, an internal SymPy failure),
  so the user gets the *contract* sentence
  :data:`aicls.core.messages_tr.MSG_UNSUPPORTED_PROBLEM` and nothing else. The technical
  reason travels in ``details["reason"]``, never in the Turkish text.
* :class:`CasTimeout` -- the computation exceeded ``settings.compute_timeout_s``; the sandbox
  worker was killed and respawned.

Sandbox workers run in a separate process and cannot pickle these exceptions reliably (their
``__init__`` signatures differ from ``Exception.__reduce__``'s assumption), so
:func:`error_to_payload` / :func:`payload_to_error` move them across the pipe as plain JSON-safe
dictionaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from aicls.core import messages_tr
from aicls.core.errors import (
    AiclsError,
    ErrorCode,
    InputValidationError,
    InternalError,
    NotFoundError,
)

#: Maximum characters kept from a foreign exception message placed in ``details``.
MAX_DETAIL_CHARS = 200


class UnsupportedProblem(AiclsError):  # noqa: N818 - name fixed by the module contract
    """SymPy cannot answer this problem (``UNSUPPORTED_PROBLEM``, HTTP 422).

    Args:
        reason: Stable machine-readable slug (``"unevaluated_integral"``,
            ``"condition_set"``, ``"sympy_error"``); mirrored into ``details["reason"]``.
        details: Extra structured fields merged into the problem document.
        message_tr: Override for the Turkish text; defaults to the frozen contract sentence.

    Attributes:
        message_tr: :data:`aicls.core.messages_tr.MSG_UNSUPPORTED_PROBLEM` unless overridden.
        reason: The slug passed in (``"unsupported"`` by default).
    """

    default_code = ErrorCode.UNSUPPORTED_PROBLEM

    reason: str

    def __init__(
        self,
        reason: str = "unsupported",
        *,
        details: Mapping[str, Any] | None = None,
        message_tr: str | None = None,
    ) -> None:
        self.reason = reason
        merged: dict[str, Any] = {"reason": reason}
        if details:
            merged.update(details)
        super().__init__(
            self.default_code,
            message_tr if message_tr is not None else messages_tr.MSG_UNSUPPORTED_PROBLEM,
            merged,
        )


class CasTimeout(AiclsError):  # noqa: N818 - name fixed by the module contract
    """The computation exceeded its budget (``CAS_TIMEOUT``, HTTP 504).

    Args:
        timeout_s: The budget that was exceeded; when given it appears in the Turkish text
            (``MSG_CAS_TIMEOUT_SECONDS``) and in ``details["timeout_s"]``.
        details: Extra structured fields (``op``, ``restarts``, ...).
    """

    default_code = ErrorCode.CAS_TIMEOUT

    timeout_s: float | None

    def __init__(
        self,
        timeout_s: float | None = None,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.timeout_s = timeout_s
        merged: dict[str, Any] = {}
        if timeout_s is not None:
            merged["timeout_s"] = timeout_s
        if details:
            merged.update(details)
        text = (
            messages_tr.MSG_CAS_TIMEOUT
            if timeout_s is None
            else messages_tr.msg("CAS_TIMEOUT_SECONDS", timeout_s=timeout_s)
        )
        super().__init__(self.default_code, text, merged)


class _ErrorFactory(Protocol):
    """The ``(message_tr, *, details, http_status)`` constructor the core errors share."""

    def __call__(
        self,
        message_tr: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
        http_status: int | None = None,
    ) -> AiclsError:
        """Build the error."""
        ...


#: Errors with the ``UnsupportedProblem``/``CasTimeout`` constructors, handled one by one.
ERROR_CLASSES: dict[ErrorCode, type[AiclsError]] = {
    ErrorCode.UNSUPPORTED_PROBLEM: UnsupportedProblem,
    ErrorCode.CAS_TIMEOUT: CasTimeout,
}

#: Core errors that share the ``(message_tr, *, details, http_status)`` constructor, so a
#: worker's failure arrives in the parent as the same class the worker raised.
SIMPLE_ERROR_CLASSES: dict[ErrorCode, _ErrorFactory] = {
    ErrorCode.VALIDATION_ERROR: InputValidationError,
    ErrorCode.NOT_FOUND: NotFoundError,
    ErrorCode.INTERNAL: InternalError,
}


def _json_safe(value: Any) -> Any:
    """Reduce ``value`` to something ``json``/``pickle`` and pydantic can both carry."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_json_safe(item) for item in value]
    return str(value)[:MAX_DETAIL_CHARS]


def error_to_payload(error: AiclsError) -> dict[str, Any]:
    """Encode ``error`` as a JSON-safe dictionary for the sandbox pipe.

    Args:
        error: Any :class:`~aicls.core.errors.AiclsError`.

    Returns:
        ``{"code": ..., "message_tr": ..., "details": {...}, "http_status": ...}``.
    """
    return {
        "code": error.code.value,
        "message_tr": error.message_tr,
        "details": _json_safe(error.details),
        "http_status": error.http_status,
    }


def payload_to_error(payload: Mapping[str, Any]) -> AiclsError:
    """Rebuild the error encoded by :func:`error_to_payload`.

    Args:
        payload: The dictionary produced in the worker process.

    Returns:
        :class:`UnsupportedProblem`, :class:`CasTimeout` or a plain
        :class:`~aicls.core.errors.AiclsError`; an unknown code degrades to
        :attr:`~aicls.core.errors.ErrorCode.INTERNAL` (the base class does that for us).
    """
    raw_code = str(payload.get("code", ErrorCode.INTERNAL.value))
    message_tr = str(payload.get("message_tr", ""))
    details = dict(payload.get("details") or {})
    http_status = payload.get("http_status")
    try:
        code = ErrorCode(raw_code)
    except ValueError:
        return AiclsError(raw_code, message_tr, details, http_status)
    factory = ERROR_CLASSES.get(code)
    if factory is UnsupportedProblem:
        reason = str(details.pop("reason", "unsupported"))
        return UnsupportedProblem(reason, details=details, message_tr=message_tr)
    if factory is CasTimeout:
        timeout_s = details.pop("timeout_s", None)
        return CasTimeout(
            float(timeout_s) if timeout_s is not None else None,
            details=details,
        )
    simple = SIMPLE_ERROR_CLASSES.get(code)
    if simple is not None:
        return simple(message_tr, details=details, http_status=http_status)
    return AiclsError(code, message_tr, details, http_status)


__all__ = [
    "ERROR_CLASSES",
    "MAX_DETAIL_CHARS",
    "SIMPLE_ERROR_CLASSES",
    "CasTimeout",
    "UnsupportedProblem",
    "error_to_payload",
    "payload_to_error",
]
