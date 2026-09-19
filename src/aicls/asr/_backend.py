"""Shared plumbing for the two torch-backed transcribers (lazy imports, local-only checkpoints).

Nothing in this module imports ``torch`` or ``transformers`` at module level: :mod:`aicls.asr` is
below ``aicls.api`` in the layering (ADR-0002) and must stay importable on a machine with no ML
stack at all. Both helpers raise :class:`aicls.asr.errors.AsrModelError` with a Turkish
"model not ready" text rather than letting an ``ImportError`` escape.

**Weights are never downloaded.** :func:`require_model_dir` insists on an existing local
directory, and both call sites pass that directory (never a Hugging Face hub id) to
``from_pretrained`` together with ``local_files_only=True``; ``tools/check_gates.py`` enforces
this statically (ADR-0012).
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

from aicls.asr.errors import model_not_ready
from aicls.core import messages_tr
from aicls.core.errors import NotFoundError
from aicls.core.logging import get_logger

_log = get_logger("aicls.asr")

#: Files that must be present for a directory to look like a saved checkpoint.
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")


def require_model_dir(model_dir: Path | str) -> Path:
    """Validate that ``model_dir`` is an existing local checkpoint directory.

    Args:
        model_dir: Path under ``AICLS_MODELS_ROOT`` holding ``config.json`` and the weights.

    Returns:
        The resolved directory.

    Raises:
        NotFoundError: If the directory does not exist.
        AsrModelError: If it exists but holds no ``config.json`` or no weight file — that is
            almost always a half-finished ``aicls models import``.
    """
    path = Path(model_dir).expanduser()
    if not path.is_dir():
        raise NotFoundError(
            messages_tr.msg("FILE_NOT_FOUND", path=str(path)),
            details={"path": str(path), "reason": "model_dir_missing"},
        )
    if not (path / "config.json").is_file():
        raise model_not_ready("model_config_missing", path=str(path))
    if not any((path / name).is_file() for name in _WEIGHT_FILES):
        raise model_not_ready("model_weights_missing", path=str(path), expected=list(_WEIGHT_FILES))
    return path


def import_torch() -> ModuleType:
    """Import ``torch`` lazily.

    Raises:
        AsrModelError: If torch is not installed (CPU-only hosts without the gated ML stack).
    """
    try:
        import torch
    except ImportError as exc:
        raise model_not_ready("torch_missing", error=str(exc)) from exc
    return torch


def import_transformers() -> ModuleType:
    """Import ``transformers`` lazily.

    Raises:
        AsrModelError: If transformers is not installed.
    """
    try:
        import transformers
    except ImportError as exc:
        raise model_not_ready("transformers_missing", error=str(exc)) from exc
    return transformers


def select_device(device: str) -> str:
    """Resolve a device string, logging the fallback when CUDA was asked for but is absent."""
    from aicls.asr.interfaces import resolve_device

    resolved = resolve_device(device)
    if resolved != device.strip().lower():
        _log.warning("asr_device_fallback", requested=device, resolved=resolved)
    return resolved


def mean_probability(torch: ModuleType, scores: Any, token_ids: Any) -> float:
    """Mean probability of the selected tokens across decoding steps.

    Args:
        torch: The imported torch module.
        scores: Per-step logits as returned by ``generate(output_scores=True)``.
        token_ids: The generated ids, aligned with the last ``len(scores)`` steps.

    Returns:
        A confidence in ``[0, 1]``; 1.0 when scores are unavailable.
    """
    if not scores:
        return 1.0
    generated = token_ids[0][-len(scores) :]
    probabilities = []
    for step, logits in enumerate(scores):
        distribution = torch.softmax(logits[0].float(), dim=-1)
        probabilities.append(float(distribution[int(generated[step])]))
    return float(sum(probabilities) / len(probabilities))


__all__ = [
    "import_torch",
    "import_transformers",
    "mean_probability",
    "require_model_dir",
    "select_device",
]
