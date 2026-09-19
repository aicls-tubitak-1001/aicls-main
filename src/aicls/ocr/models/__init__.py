"""OCR model zoo (ADR-0006 §3, docs/ml/egitim-altyapisi.md §3).

Importing this package is free: it pulls in no torch. The submodules
(:mod:`~aicls.ocr.models.encoders`, :mod:`~aicls.ocr.models.decoders`,
:mod:`~aicls.ocr.models.seq2seq`) do import torch at module level, and
:func:`build_model` is the single door that opens them — which is what keeps
``import aicls.ocr`` torch-free for an API process that only serves the stub
(``test_importing_aicls_ocr_does_not_import_torch``).
"""

from __future__ import annotations

from typing import Any

from aicls.ocr.config import OcrModelConfig
from aicls.ocr.torch_utils import require_torch


def build_model(cfg: OcrModelConfig, vocab_size: int) -> Any:
    """Build the encoder-decoder model described by ``cfg``.

    Args:
        cfg: Architecture description (family, encoder, decoder, sizes).
        vocab_size: Tokenizer vocabulary size, specials included.

    Returns:
        A ``torch.nn.Module`` whose ``forward(inputs, tgt_ids)`` returns ``(B, T, V)`` logits.

    Raises:
        aicls.ocr.torch_utils.TorchUnavailableError: If torch is not installed.
        aicls.ocr.types.OcrInputError: If the configuration is inconsistent.
    """
    require_torch()  # fail with a Turkish, typed error instead of a bare ImportError
    from aicls.ocr.models.seq2seq import build_seq2seq

    return build_seq2seq(cfg, vocab_size)


__all__ = ["OcrModelConfig", "build_model"]
