"""Errors raised by the speech edge (:mod:`aicls.asr`).

Two families are enough for this package:

* :class:`AsrInputError` — the caller sent something we refuse to decode (not a RIFF/WAVE
  container, wrong sample width, unsupported sample rate, too long, too large). It maps to
  :attr:`~aicls.core.errors.ErrorCode.ASR_INPUT_INVALID` (HTTP 400, ADR-0017).
* :class:`AsrModelError` — a transcriber cannot be brought up (missing local model directory,
  ``transformers``/``torch`` not installed). It maps to
  :attr:`~aicls.core.errors.ErrorCode.INTERNAL` and carries the Turkish "model not ready" text.

The Turkish user text always comes from :mod:`aicls.core.messages_tr`; the machine-readable
``reason`` lives in ``details`` (English), so the API can log it without leaking it to the user.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aicls.core import messages_tr
from aicls.core.errors import AiclsError, ErrorCode


class AsrInputError(AiclsError):
    """Audio input failed validation (``ASR_INPUT_INVALID``, 400)."""

    default_code = ErrorCode.ASR_INPUT_INVALID

    def __init__(
        self,
        message_tr: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(self.default_code, message_tr, details, http_status)


class AsrModelError(AiclsError):
    """A transcriber backend could not be loaded or run (``INTERNAL``, 500)."""

    default_code = ErrorCode.INTERNAL

    def __init__(
        self,
        message_tr: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(self.default_code, message_tr, details, http_status)


def invalid_audio(reason: str, **details: Any) -> AsrInputError:
    """Build an :class:`AsrInputError` with the standard Turkish text and an English ``reason``.

    Args:
        reason: Stable English slug (``"not_riff_wave"``, ``"sample_rate_unsupported"``, ...)
            recorded in ``details`` for logs, metrics and tests.
        **details: Extra structured fields merged into the problem-details document.

    Returns:
        The error, ready to ``raise``.
    """
    return AsrInputError(messages_tr.MSG_ASR_INPUT_INVALID, details={"reason": reason, **details})


def model_not_ready(reason: str, **details: Any) -> AsrModelError:
    """Build an :class:`AsrModelError` with the Turkish "model not ready" text.

    Args:
        reason: Stable English slug (``"transformers_missing"``, ``"model_dir_missing"``, ...).
        **details: Extra structured fields merged into the problem-details document.

    Returns:
        The error, ready to ``raise``.
    """
    return AsrModelError(messages_tr.MSG_MODEL_NOT_READY, details={"reason": reason, **details})


__all__ = [
    "AsrInputError",
    "AsrModelError",
    "invalid_audio",
    "model_not_ready",
]
