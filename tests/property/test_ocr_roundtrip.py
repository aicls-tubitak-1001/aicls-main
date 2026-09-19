"""Property-based tests for the OCR edge: tokenizer roundtrip, brace repair, preprocessing."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st
import numpy as np
import pytest

from aicls.ocr import Ink, PreprocessConfig, Stroke, preprocess_image, render_ink
from aicls.ocr.features import resample_polyline
from aicls.ocr.postprocess import repair_braces
from aicls.ocr.tokenizer import SPECIAL_TOKENS, default_tokenizer

pytestmark = pytest.mark.property

TOKENIZER = default_tokenizer()
CONTENT_TOKENS = [t for t in TOKENIZER.vocab if t not in SPECIAL_TOKENS]

token_sequences = st.lists(st.sampled_from(CONTENT_TOKENS), min_size=1, max_size=24)
latex_text = st.text(alphabet="\\{}^_ abxy012+-=", min_size=0, max_size=40)


@given(tokens=token_sequences)
def test_decode_encode_is_the_identity_on_token_sequences(tokens: list[str]) -> None:
    ids = [TOKENIZER.token_to_id(t) for t in tokens]
    assert TOKENIZER.encode(TOKENIZER.decode(ids)) == ids


@given(tokens=token_sequences)
def test_decode_output_normalises_to_itself(tokens: list[str]) -> None:
    latex = TOKENIZER.decode([TOKENIZER.token_to_id(t) for t in tokens])
    assert TOKENIZER.normalize(latex) == latex


@given(text=latex_text)
def test_tokenize_never_loses_non_whitespace_characters(text: str) -> None:
    joined = "".join(TOKENIZER.tokenize(text))
    assert joined == "".join(text.split())


@given(text=latex_text)
def test_repair_braces_always_balances(text: str) -> None:
    repaired = repair_braces(text)
    depth = 0
    index = 0
    while index < len(repaired):
        char = repaired[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            assert depth >= 0
        index += 1
    assert depth == 0


@given(
    values=st.lists(st.integers(min_value=0, max_value=255), min_size=6, max_size=6),
)
def test_normalize01_stays_in_unit_range(values: list[int]) -> None:
    img = np.array(values, dtype=np.uint8).reshape(2, 3)
    out = preprocess_image(img, PreprocessConfig(steps=["normalize01"]))
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


@given(
    xs=st.lists(
        st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=12,
    ),
    spacing=st.floats(min_value=0.5, max_value=20.0),
)
def test_resample_preserves_endpoints(xs: list[float], spacing: float) -> None:
    points = np.stack([np.array(xs, dtype=np.float32), np.zeros(len(xs), np.float32)], axis=1)
    out = resample_polyline(points, spacing)
    assert np.allclose(out[0], points[0], atol=1e-3)
    if not np.allclose(points[0], points[-1], atol=1e-6):
        assert np.allclose(out[-1], points[-1], atol=1e-3)


@given(
    coords=st.lists(
        st.tuples(
            st.floats(min_value=0, max_value=400, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0, max_value=200, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=8,
    )
)
def test_render_ink_always_lands_in_unit_range(coords: list[tuple[float, float]]) -> None:
    stroke = Stroke.from_xyt(
        [x for x, _ in coords],
        [y for _, y in coords],
        list(range(len(coords))),
    )
    arr = render_ink(Ink(strokes=[stroke]), size=(24, 48))
    assert arr.shape == (24, 48)
    assert float(arr.min()) >= 0.0
    assert float(arr.max()) <= 1.0
