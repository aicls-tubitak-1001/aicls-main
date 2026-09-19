"""Deterministic stub recogniser — the torch-free default of the whole platform.

Every layer above OCR (the API, the WebSocket ink stream, the evaluation harness, the e2e tests)
must run before a single model has been trained and without torch in the process. The stub
provides exactly that: a :class:`~aicls.ocr.types.Recognizer` whose answer is a pure function of
its input.

Rules:

1. If the sample carries a ground-truth ``label`` (dataset fixtures, replayed recordings), that
   label is returned — postprocessed like a real output.
2. Otherwise the SHA-256 of the (rounded) geometry selects an entry of the curriculum table
   :data:`aicls.ocr.curriculum.CURRICULUM_LATEX`.

The answer never depends on wall-clock time, dict ordering or process state
(``test_stub_recognizer_deterministic``).
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from aicls.core.hashing import sha256_bytes
from aicls.ocr.curriculum import CURRICULUM_LATEX
from aicls.ocr.postprocess import postprocess_latex
from aicls.ocr.tokenizer import default_tokenizer
from aicls.ocr.types import Image, Ink, OcrInputError, Recognition

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Version string reported by the stub (models registry convention ``<kind>-<semver>``).
STUB_MODEL_VERSION = "stub-0.1.0"


def input_fingerprint(inp: Ink | Image) -> str:
    """Stable ``sha256:`` fingerprint of a handwriting sample's geometry.

    Coordinates are rounded to two decimals first, so a re-encoded payload (float32 -> JSON ->
    float32) keeps its fingerprint.

    Args:
        inp: The sample to fingerprint.

    Returns:
        The hex digest returned by :func:`aicls.core.hashing.sha256_bytes`.

    Raises:
        OcrInputError: If the input is neither an :class:`~aicls.ocr.types.Ink` nor an
            :class:`~aicls.ocr.types.Image`.
    """
    if isinstance(inp, Ink):
        points = np.round(inp.all_points(), 2).astype(np.float32)
        payload = b"ink:" + str(inp.n_strokes).encode("ascii") + b":" + points.tobytes()
    elif isinstance(inp, Image):
        payload = b"img:" + str(inp.shape).encode("ascii") + b":" + inp.array.tobytes()
    else:
        raise OcrInputError(details={"reason": "unsupported_input", "type": type(inp).__name__})
    return sha256_bytes(payload)


class StubRecognizer:
    """Hash-driven recogniser used whenever no trained checkpoint is active.

    Args:
        table: LaTeX candidates to choose from; defaults to the curriculum table.
        model_version: Version string reported in :class:`~aicls.ocr.types.Recognition`.

    Raises:
        OcrInputError: If ``table`` is empty.
    """

    def __init__(
        self,
        table: Sequence[str] = CURRICULUM_LATEX,
        *,
        model_version: str = STUB_MODEL_VERSION,
    ) -> None:
        entries = tuple(table)
        if not entries:
            raise OcrInputError(details={"reason": "empty_stub_table"})
        self._table = entries
        self.model_version = model_version

    def warmup(self) -> None:
        """Build the tokenizer cache; there is nothing else to load."""
        default_tokenizer()

    def recognize(self, inp: Ink | Image, *, beam: int = 4) -> Recognition:
        """Return the deterministic answer for ``inp``.

        Args:
            inp: An :class:`~aicls.ocr.types.Ink`, an :class:`~aicls.ocr.types.Image` or a raw
                numpy array (wrapped as an image).
            beam: Number of alternatives to report (>= 1).

        Returns:
            The recognition, whose first alternative is always the returned LaTeX.

        Raises:
            OcrInputError: If ``beam`` is not positive, the input type is unsupported or the ink
                carries no point at all.
        """
        started = perf_counter()
        if beam < 1:
            raise OcrInputError(details={"reason": "beam_width", "beam": int(beam)})
        sample = Image(inp) if isinstance(inp, np.ndarray) else inp
        if isinstance(sample, Ink) and sample.is_empty:
            raise OcrInputError(details={"reason": "empty_ink"})

        digest = input_fingerprint(sample).rsplit(":", 1)[-1]
        value = int(digest[:16], 16)
        index = value % len(self._table)

        label = sample.label if isinstance(sample, Ink) else None
        if label:
            latex = postprocess_latex(label)
            confidence = 1.0
        else:
            latex = postprocess_latex(self._table[index])
            confidence = 0.50 + (value % 46) / 100.0

        alternatives: list[tuple[str, float]] = [(latex, round(confidence, 4))]
        for offset in range(1, min(beam, len(self._table) + 1)):
            candidate = postprocess_latex(self._table[(index + offset) % len(self._table)])
            score = round(max(confidence - 0.1 * offset, 0.0), 4)
            alternatives.append((candidate, score))
        alternatives = alternatives[:beam]

        tokenizer = default_tokenizer()
        return Recognition(
            latex=latex,
            tokens=tokenizer.tokenize(latex),
            confidence=confidence,
            alternatives=alternatives,
            elapsed_ms=(perf_counter() - started) * 1000.0,
            model_version=self.model_version,
        )


__all__ = ["STUB_MODEL_VERSION", "StubRecognizer", "input_fingerprint"]
