"""Errors raised by the voice data-collection flow.

Both map onto the problem-details contract of docs/mimari/api.md §2: a participant who has not
accepted the current consent text gets ``403 CONSENT_REQUIRED``, and the whole tab is
``503 GATE_BLOCKED`` until an ethics approval reference is recorded in ``AICLS_ETHICS_APPROVAL_REF``
(docs/ml/veri-yonetimi.md §4).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aicls.core.errors import AiclsError, ErrorCode
from aicls.core.messages_tr import msg


class ConsentRequiredError(AiclsError):
    """The speaker has not accepted the current consent text (``CONSENT_REQUIRED``, 403)."""

    default_code = ErrorCode.CONSENT_REQUIRED

    def __init__(
        self,
        message_tr: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(self.default_code, message_tr, details)


class ContributionGateError(AiclsError):
    """The collection tab is closed because no ethics approval is recorded (503)."""

    default_code = ErrorCode.GATE_BLOCKED

    def __init__(
        self,
        message_tr: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(self.default_code, message_tr or msg("ETHICS_APPROVAL_MISSING"), details)


__all__ = ["ConsentRequiredError", "ContributionGateError"]
