"""Torch-backed recogniser loading a *local* checkpoint directory (ADR-0006, ADR-0012).

A checkpoint directory follows the layout of docs/ml/egitim-altyapisi.md §5::

    <AICLS_MODELS_ROOT>/ocr-0.3.0/
        model.safetensors | model.pt      # weights
        config.json | config.yaml         # resolved OcrModelConfig (``model:`` block)
        vocab.json                        # LatexTokenizer vocabulary
        model_version.txt                 # optional; defaults to the directory name

Weights are **always** read from this directory. No hub identifier is ever passed to any loader
(``tools/check_gates.py`` scans for that); downloading a checkpoint is a gated, explicit action of
``training/scripts/download_weights.py``.

Torch is imported inside :meth:`TorchRecognizer.load`, so importing this module — and with it
:mod:`aicls.ocr` — stays torch-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from aicls.core import messages_tr
from aicls.core.errors import NotFoundError
from aicls.core.logging import get_logger
from aicls.ocr.config import OcrModelConfig
from aicls.ocr.features import normalize_strokes
from aicls.ocr.postprocess import postprocess_latex
from aicls.ocr.preprocess import PreprocessConfig, preprocess_image
from aicls.ocr.render import render_ink
from aicls.ocr.tokenizer import LatexTokenizer, default_tokenizer
from aicls.ocr.torch_utils import require_torch
from aicls.ocr.types import Image, Ink, OcrInputError, Recognition

if TYPE_CHECKING:
    from collections.abc import Sequence

_log = get_logger(__name__)

#: Weight file names looked up in a checkpoint directory, in order of preference.
WEIGHT_FILES: tuple[str, ...] = ("model.safetensors", "model.pt", "model.pth")
#: Configuration file names looked up in a checkpoint directory, in order of preference.
CONFIG_FILES: tuple[str, ...] = ("config.json", "config.yaml", "config.yml")
#: Preprocessing applied to a raster before it reaches an image model.
DEFAULT_INFERENCE_PREPROCESS = PreprocessConfig(steps=["grayscale", "normalize01", "resize_pad"])


def _first_existing(model_dir: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    return None


def _load_config(path: Path | None) -> OcrModelConfig:
    """Read ``config.json``/``config.yaml``; fall back to the default architecture."""
    if path is None:
        return OcrModelConfig()
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix == ".json":
            payload = json.loads(text)
        else:
            # types-PyYAML is not installed in this environment (see deps_needed)
            import yaml  # type: ignore[import-untyped]

            payload = yaml.safe_load(text) or {}
    except (json.JSONDecodeError, ValueError) as exc:
        raise OcrInputError(
            details={"reason": "model_config_unreadable", "path": str(path), "error": str(exc)}
        ) from exc
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        payload = payload["model"]
    if not isinstance(payload, dict):
        raise OcrInputError(details={"reason": "model_config_shape", "path": str(path)})
    known = set(OcrModelConfig.model_fields)
    # A ``null`` in a serialised config means "not specified", not "the value is None". The
    # trainer writes its whole schema out, optional keys included, so a checkpoint's config.yaml
    # carries `model.max_len: null` -- and passing that through replaced this model's default of
    # 160 with a value the field cannot hold. Every checkpoint the training loop has ever written
    # was unreadable here because of it; `aicls.ml.config` resolves the same key the same way
    # (``self.max_len or DEFAULT_MAX_LEN``), so this makes the reader agree with the writer.
    return OcrModelConfig(**{k: v for k, v in payload.items() if k in known and v is not None})


class TorchRecognizer:
    """Encoder-decoder recogniser running a locally stored checkpoint on CPU or GPU.

    Use :meth:`load`; the constructor is for tests and for the training loop, which already holds
    a live model.

    Args:
        model: The built ``torch.nn.Module``.
        tokenizer: Vocabulary matching the checkpoint.
        config: Architecture description of ``model``.
        device: Torch device string.
        model_version: Version reported in the :class:`~aicls.ocr.types.Recognition`.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: LatexTokenizer,
        config: OcrModelConfig,
        *,
        device: str = "cpu",
        model_version: str = "ocr-dev",
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        self.device = device
        self.model_version = model_version
        # A recogniser is an inference object; a module handed to it must not be in training
        # mode. Only `load()` used to call this, so a recogniser built directly from a module —
        # what the tests and the API replay path do — ran **dropout at inference**, and since
        # the batched decoders landed it also silently fell back to full recomputation, because
        # `seq2seq._supports_incremental` refuses the key/value cache for a training-mode module
        # (a one-position step cannot reproduce a whole-prefix pass's dropout draws). `eval()` is
        # idempotent, so `load()` calling it too costs nothing.
        eval_mode = getattr(model, "eval", None)
        if callable(eval_mode):
            eval_mode()

    @classmethod
    def load(cls, model_dir: Path | str, device: str = "cpu") -> TorchRecognizer:
        """Load a checkpoint directory.

        Args:
            model_dir: Directory holding the weights, config and vocabulary.
            device: Torch device string (``"cpu"``, ``"cuda"``, ...).

        Returns:
            A ready recogniser in ``eval()`` mode.

        Raises:
            NotFoundError: If the directory or the weight file is missing.
            TorchUnavailableError: If torch is not installed.
        """
        torch = require_torch()
        directory = Path(model_dir)
        if not directory.is_dir():
            raise NotFoundError(
                messages_tr.msg("FILE_NOT_FOUND", path=str(directory)),
                details={"path": str(directory), "reason": "model_dir_missing"},
            )
        weights_path = _first_existing(directory, WEIGHT_FILES)
        if weights_path is None:
            raise NotFoundError(
                messages_tr.msg("FILE_NOT_FOUND", path=str(directory / WEIGHT_FILES[0])),
                details={
                    "path": str(directory),
                    "reason": "weights_missing",
                    "looked_for": WEIGHT_FILES,
                },
            )
        config = _load_config(_first_existing(directory, CONFIG_FILES))
        vocab_path = directory / "vocab.json"
        tokenizer = (
            LatexTokenizer.from_file(vocab_path) if vocab_path.is_file() else default_tokenizer()
        )

        from aicls.ocr.models import build_model

        model = build_model(config, tokenizer.vocab_size)
        if weights_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(weights_path), device=device)
        else:
            state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        version_file = directory / "model_version.txt"
        version = (
            version_file.read_text(encoding="utf-8").strip()
            if version_file.is_file()
            else directory.name
        )
        _log.info("ocr.checkpoint_loaded", path=str(directory), version=version, device=device)
        return cls(model, tokenizer, config, device=device, model_version=version)

    def warmup(self) -> None:
        """Run one tiny forward pass so the first request does not pay for lazy kernels."""
        torch = require_torch()
        with torch.no_grad():
            self._model(
                self._example_input(),
                torch.zeros(1, 1, dtype=torch.long, device=self.device),
            )

    def _example_input(self) -> Any:
        torch = require_torch()
        if self._config.is_stroke_family:
            return torch.zeros(1, 4, self._config.stroke_feature_dim, device=self.device)
        height, width = self._config.image_size
        return torch.ones(1, 1, height, width, device=self.device)

    def _to_tensor(self, inp: Ink | Image) -> Any:
        """Turn a sample into the input tensor of the configured family.

        The tensor is created **on** :attr:`device`. :meth:`load` moved the weights there, and a
        CPU input against CUDA weights is a ``RuntimeError`` on the first forward pass, not a
        silent transfer (reproduced on the RTX 5090, 2026-09-08).
        """
        torch = require_torch()
        if self._config.is_stroke_family:
            if not isinstance(inp, Ink):
                raise OcrInputError(details={"reason": "stroke_model_needs_ink"})
            features = normalize_strokes(inp)
            if features.shape[0] == 0:
                raise OcrInputError(details={"reason": "empty_ink"})
            stroke = torch.from_numpy(np.ascontiguousarray(features)).unsqueeze(0)
            return stroke.to(self.device)
        if isinstance(inp, Ink):
            raster = render_ink(inp, self._config.image_size)
        else:
            raster = preprocess_image(
                inp,
                PreprocessConfig(
                    steps=DEFAULT_INFERENCE_PREPROCESS.steps, size=self._config.image_size
                ),
            )
        array = np.ascontiguousarray(raster.astype(np.float32))
        return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(self.device)

    def recognize(self, inp: Ink | Image, *, beam: int = 4) -> Recognition:
        """Recognise one sample with greedy (``beam == 1``) or beam decoding.

        Args:
            inp: The handwriting sample.
            beam: Beam width.

        Returns:
            The postprocessed recognition with up to ``beam`` alternatives.

        Raises:
            OcrInputError: If the sample does not match the model family or is empty.
        """
        started = perf_counter()
        if beam < 1:
            raise OcrInputError(details={"reason": "beam_width", "beam": int(beam)})
        if isinstance(inp, np.ndarray):
            inp = Image(inp)
        if isinstance(inp, Ink) and inp.is_empty:
            raise OcrInputError(details={"reason": "empty_ink"})

        from aicls.ocr.models.seq2seq import beam_search

        tensor = self._to_tensor(inp)
        hypotheses = beam_search(self._model, tensor, beam=beam, max_len=self._config.max_len)
        decoded = [
            (postprocess_latex(self._tokenizer.decode(ids)), float(np.exp(score)))
            for ids, score in hypotheses
        ]
        best_latex, best_score = decoded[0]
        return Recognition(
            latex=best_latex,
            tokens=self._tokenizer.tokenize(best_latex),
            confidence=best_score,
            alternatives=[(text, round(score, 6)) for text, score in decoded],
            elapsed_ms=(perf_counter() - started) * 1000.0,
            model_version=self.model_version,
        )


__all__ = ["CONFIG_FILES", "DEFAULT_INFERENCE_PREPROCESS", "WEIGHT_FILES", "TorchRecognizer"]
