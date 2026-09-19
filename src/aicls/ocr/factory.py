"""Recogniser factory: one entry point for every OCR backend (docs/mimari/moduller.md#aiclsocr).

``get_recognizer`` is what the API's model registry and the evaluation harness call. The name
comes from configuration (``AICLS_MODELS_ENABLED`` / ``model_versions.ocr``), so switching from
the stub to a trained checkpoint never touches calling code — and the torch backend is imported
only when it is actually requested.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from aicls.core.logging import get_logger
from aicls.ocr.stub import StubRecognizer
from aicls.ocr.types import OcrInputError

if TYPE_CHECKING:
    from aicls.ocr.types import Recognizer

_log = get_logger(__name__)

#: Backend names accepted by :func:`get_recognizer`.
RECOGNIZER_NAMES: tuple[str, ...] = ("stub", "torch")


def available_recognizers() -> tuple[str, ...]:
    """Return the recogniser names this build supports."""
    return RECOGNIZER_NAMES


def get_recognizer(
    name: str = "stub",
    *,
    model_dir: Path | str | None = None,
    device: str = "cpu",
) -> Recognizer:
    """Instantiate a recogniser backend.

    Args:
        name: ``"stub"`` (deterministic, torch-free) or ``"torch"`` (a local checkpoint).
        model_dir: Checkpoint directory; required for ``"torch"``, ignored by the stub.
        device: Torch device string for the ``"torch"`` backend.

    Returns:
        A ready :class:`~aicls.ocr.types.Recognizer`.

    Raises:
        OcrInputError: If ``name`` is unknown or ``model_dir`` is missing for ``"torch"``.
        NotFoundError: If the checkpoint directory does not exist.
        TorchUnavailableError: If ``"torch"`` is requested without torch installed.
    """
    if name == "stub":
        return StubRecognizer()
    if name == "torch":
        if model_dir is None:
            raise OcrInputError(
                details={"reason": "model_dir_required", "name": name, "allowed": RECOGNIZER_NAMES}
            )
        from aicls.ocr.torch_recognizer import TorchRecognizer

        _log.info("ocr.recognizer_requested", name=name, device=device)
        return TorchRecognizer.load(Path(model_dir), device=device)
    raise OcrInputError(
        details={"reason": "unknown_recognizer", "name": name, "allowed": RECOGNIZER_NAMES}
    )


__all__ = ["RECOGNIZER_NAMES", "available_recognizers", "get_recognizer"]
