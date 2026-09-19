"""StubRecognizer and `get_recognizer` tests (the torch-free path used by the API and eval)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aicls.core.errors import AiclsError
from aicls.ocr import (
    Image,
    Ink,
    OcrInputError,
    Recognition,
    Stroke,
    StubRecognizer,
    get_recognizer,
)
from aicls.ocr.curriculum import CURRICULUM_LATEX
from aicls.ocr.stub import STUB_MODEL_VERSION


def ink_a() -> Ink:
    return Ink(strokes=[Stroke.from_xyt([0, 10, 20], [0, 15, 0], [0, 8, 16])])


def ink_b() -> Ink:
    return Ink(strokes=[Stroke.from_xyt([0, 5, 30], [0, 25, 5], [0, 9, 21])])


class TestStubRecognizer:
    def test_stub_recognizer_deterministic(self) -> None:
        first = StubRecognizer().recognize(ink_a())
        second = StubRecognizer().recognize(ink_a())

        assert first.latex == second.latex
        assert first.tokens == second.tokens
        assert first.confidence == second.confidence
        assert first.alternatives == second.alternatives
        assert first.model_version == second.model_version == STUB_MODEL_VERSION
        assert first.latex in CURRICULUM_LATEX

    def test_different_inputs_give_different_outputs(self) -> None:
        recognizer = StubRecognizer()
        assert recognizer.recognize(ink_a()).latex != recognizer.recognize(ink_b()).latex

    def test_label_wins_when_present(self) -> None:
        ink = Ink(strokes=ink_a().strokes, label="x^{2}+1")
        recognition = StubRecognizer().recognize(ink)
        assert recognition.latex == "x^{2}+1"
        assert recognition.confidence == pytest.approx(1.0)

    def test_label_is_postprocessed(self) -> None:
        ink = Ink(strokes=ink_a().strokes, label="  \\frac { a } { b ")
        assert StubRecognizer().recognize(ink).latex == "\\frac{a}{b}"

    def test_tokens_match_the_latex(self) -> None:
        recognition = StubRecognizer().recognize(ink_a())
        from aicls.ocr.tokenizer import default_tokenizer

        assert recognition.tokens == default_tokenizer().tokenize(recognition.latex)

    def test_alternatives_are_sorted_and_bounded(self) -> None:
        recognition = StubRecognizer().recognize(ink_a(), beam=3)
        scores = [score for _, score in recognition.alternatives]
        assert len(recognition.alternatives) == 3
        assert scores == sorted(scores, reverse=True)
        assert recognition.alternatives[0][0] == recognition.latex
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_beam_one_returns_a_single_alternative(self) -> None:
        assert len(StubRecognizer().recognize(ink_a(), beam=1).alternatives) == 1

    def test_invalid_beam_rejected(self) -> None:
        with pytest.raises(OcrInputError):
            StubRecognizer().recognize(ink_a(), beam=0)

    def test_images_are_recognised_too(self) -> None:
        img = Image(np.full((16, 32), 255, dtype=np.uint8))
        recognition = StubRecognizer().recognize(img)
        assert isinstance(recognition, Recognition)
        assert recognition.latex in CURRICULUM_LATEX

    def test_image_and_ink_hashes_differ(self) -> None:
        img = Image(np.zeros((8, 8), dtype=np.uint8))
        assert StubRecognizer().recognize(img).latex != ""

    def test_raw_numpy_image_is_accepted(self) -> None:
        recognition = StubRecognizer().recognize(np.zeros((8, 8), dtype=np.uint8))  # type: ignore[arg-type]
        assert recognition.latex in CURRICULUM_LATEX

    def test_unsupported_input_type_rejected(self) -> None:
        with pytest.raises(OcrInputError):
            StubRecognizer().recognize("x^{2}")  # type: ignore[arg-type]

    def test_empty_ink_is_rejected(self) -> None:
        with pytest.raises(OcrInputError):
            StubRecognizer().recognize(Ink(strokes=[]))

    def test_warmup_is_a_cheap_noop(self) -> None:
        recognizer = StubRecognizer()
        recognizer.warmup()
        assert recognizer.recognize(ink_a()).elapsed_ms >= 0.0

    def test_custom_table_is_used(self) -> None:
        recognizer = StubRecognizer(table=["a", "b"])
        assert recognizer.recognize(ink_a()).latex in {"a", "b"}

    def test_empty_table_rejected(self) -> None:
        with pytest.raises(OcrInputError):
            StubRecognizer(table=[])


class TestGetRecognizer:
    def test_default_is_the_stub(self) -> None:
        assert isinstance(get_recognizer(), StubRecognizer)

    def test_unknown_name_is_rejected(self) -> None:
        with pytest.raises(OcrInputError) as exc:
            get_recognizer("magic")
        assert "magic" in str(exc.value.details)

    def test_torch_recognizer_requires_a_model_dir(self) -> None:
        with pytest.raises(OcrInputError):
            get_recognizer("torch")

    def test_torch_recognizer_reports_a_missing_model_dir(self, tmp_path: Path) -> None:
        with pytest.raises(AiclsError):
            get_recognizer("torch", model_dir=tmp_path / "absent")

    def test_available_names_are_listed(self) -> None:
        from aicls.ocr import available_recognizers

        assert set(available_recognizers()) == {"stub", "torch"}
