"""Lazy torch access.

Importing :mod:`aicls.ocr` must stay cheap and torch-free (ADR-0002 layering; the API process
loads the package even when it serves the stub recogniser). Every torch-using module therefore
calls :func:`require_torch` inside a function body, and a missing installation surfaces as an
:class:`~aicls.core.errors.AiclsError` with a Turkish message instead of a bare ``ImportError``.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

from aicls.core import messages_tr
from aicls.core.errors import AiclsError, ErrorCode

#: Imported modules, cached so repeated calls stay cheap (and patchable in tests).
_MODULES: dict[str, ModuleType | None] = {"torch": None, "torchvision.models": None}


class TorchUnavailableError(AiclsError):
    """Torch (or torchvision) is not installed in this environment (``INTERNAL``, HTTP 500)."""

    default_code = ErrorCode.INTERNAL

    def __init__(self, package: str, error: str) -> None:
        super().__init__(
            ErrorCode.INTERNAL,
            messages_tr.MSG_MODEL_NOT_READY,
            {"missing_package": package, "error": error, "hint": "pip install .[dev] (CPU torch)"},
        )


def require_torch() -> Any:
    """Import and cache :mod:`torch`.

    Returns:
        The ``torch`` module.

    Raises:
        TorchUnavailableError: If torch is not installed.
    """
    cached = _MODULES["torch"]
    if cached is None:
        try:
            import torch
        except ImportError as exc:
            raise TorchUnavailableError("torch", str(exc)) from exc
        _MODULES["torch"] = cached = torch
    return cached


def require_torchvision() -> Any:
    """Import and cache :mod:`torchvision.models`.

    Returns:
        The ``torchvision.models`` module.

    Raises:
        TorchUnavailableError: If torchvision is not installed (it ships with the gated ML
            extra; a minimal CPU environment may run without it).
    """
    cached = _MODULES["torchvision.models"]
    if cached is None:
        try:
            from torchvision import models
        except ImportError as exc:
            raise TorchUnavailableError("torchvision", str(exc)) from exc
        _MODULES["torchvision.models"] = cached = models
    return cached


def reset_module_cache() -> None:
    """Forget the imported modules (used by tests that simulate a torch-free environment)."""
    for key in _MODULES:
        _MODULES[key] = None


__all__ = ["TorchUnavailableError", "require_torch", "require_torchvision", "reset_module_cache"]
