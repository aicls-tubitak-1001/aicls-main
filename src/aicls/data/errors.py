"""Errors raised by the dataset layer (:mod:`aicls.data`).

The package reuses the core hierarchy instead of inventing codes: everything a caller got wrong
(bad ratios, an unknown split, a manifest that does not parse, a batch that cannot be padded) is a
:class:`~aicls.core.errors.InputValidationError`, and everything that is simply not on disk is a
:class:`~aicls.core.errors.NotFoundError`. The Turkish user text always comes from
:mod:`aicls.core.messages_tr`; the machine-readable ``reason`` slug stays English and lives in
``details`` so logs, metrics and tests can key on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aicls.core import messages_tr
from aicls.core.errors import InputValidationError, NotFoundError


def invalid_data(reason: str, **details: Any) -> InputValidationError:
    """Build an :class:`~aicls.core.errors.InputValidationError` with an English ``reason``.

    Args:
        reason: Stable slug (``"ratios_sum"``, ``"unknown_split"``, ...) recorded in ``details``.
        **details: Extra structured fields merged into the problem-details document.

    Returns:
        The error, ready to ``raise``.
    """
    return InputValidationError(details={"reason": reason, **details})


def missing_path(path: Path, reason: str = "path_missing", **details: Any) -> NotFoundError:
    """Build a :class:`~aicls.core.errors.NotFoundError` for a file or directory.

    Args:
        path: The path that does not exist.
        reason: Stable English slug describing what was expected there.
        **details: Extra structured fields merged into the problem-details document.

    Returns:
        The error, ready to ``raise``.
    """
    return NotFoundError(
        messages_tr.msg("FILE_NOT_FOUND", path=str(path)),
        details={"reason": reason, "path": str(path), **details},
    )


__all__ = ["invalid_data", "missing_path"]
