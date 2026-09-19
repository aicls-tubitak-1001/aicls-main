"""Handwriting (InkML/Canvas JSON/PNG) -> LaTeX: types, IO, features, models, recognisers.

Public contract: ``modules/ocr.json`` and docs/mimari/moduller.md#aiclsocr. Both input families of
ADR-0013 live behind one :class:`Recognizer` interface — online traces (:class:`Ink`) and offline
rasters (:class:`Image`) — so the pending online/offline decision is a configuration key, not a
rewrite.

Importing this package pulls in numpy, Pillow and defusedxml, but **never torch**: the model zoo
(:mod:`aicls.ocr.models`) and :class:`TorchRecognizer` import it lazily inside their entry points
(``test_importing_aicls_ocr_does_not_import_torch``).
"""

from __future__ import annotations

from aicls.ocr.canvas import ink_from_canvas_json, ink_to_canvas_json
from aicls.ocr.config import OcrModelConfig
from aicls.ocr.curriculum import CURRICULUM_LATEX
from aicls.ocr.factory import RECOGNIZER_NAMES, available_recognizers, get_recognizer
from aicls.ocr.features import (
    STROKE_FEATURE_DIM,
    STROKE_FEATURE_NAMES,
    fit_cubic_bezier,
    normalize_strokes,
    resample_polyline,
)
from aicls.ocr.inkml import MAX_INKML_BYTES, read_inkml, write_inkml
from aicls.ocr.models import build_model
from aicls.ocr.postprocess import (
    PrecheckReport,
    canonicalize_whitespace,
    grammar_precheck,
    postprocess_latex,
    repair_braces,
)
from aicls.ocr.preprocess import PREPROCESS_STEPS, PreprocessConfig, preprocess_image
from aicls.ocr.render import DEFAULT_RENDER_SIZE, render_ink, render_ink_to_image
from aicls.ocr.stub import STUB_MODEL_VERSION, StubRecognizer
from aicls.ocr.tokenizer import LatexTokenizer, default_tokenizer
from aicls.ocr.torch_recognizer import TorchRecognizer
from aicls.ocr.types import (
    MAX_POINTS_PER_INK,
    MAX_STROKES_PER_INK,
    Image,
    Ink,
    OcrInputError,
    Recognition,
    Recognizer,
    Stroke,
)

__all__ = [
    "CURRICULUM_LATEX",
    "DEFAULT_RENDER_SIZE",
    "MAX_INKML_BYTES",
    "MAX_POINTS_PER_INK",
    "MAX_STROKES_PER_INK",
    "PREPROCESS_STEPS",
    "RECOGNIZER_NAMES",
    "STROKE_FEATURE_DIM",
    "STROKE_FEATURE_NAMES",
    "STUB_MODEL_VERSION",
    "Image",
    "Ink",
    "LatexTokenizer",
    "OcrInputError",
    "OcrModelConfig",
    "PrecheckReport",
    "PreprocessConfig",
    "Recognition",
    "Recognizer",
    "Stroke",
    "StubRecognizer",
    "TorchRecognizer",
    "available_recognizers",
    "build_model",
    "canonicalize_whitespace",
    "default_tokenizer",
    "fit_cubic_bezier",
    "get_recognizer",
    "grammar_precheck",
    "ink_from_canvas_json",
    "ink_to_canvas_json",
    "normalize_strokes",
    "postprocess_latex",
    "preprocess_image",
    "read_inkml",
    "render_ink",
    "render_ink_to_image",
    "repair_braces",
    "resample_polyline",
    "write_inkml",
]
