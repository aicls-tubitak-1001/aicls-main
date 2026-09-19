"""InkML reader/writer tests (ADR-0013): roundtrip, XXE defence and input limits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aicls.core.errors import ErrorCode
from aicls.ocr import Ink, OcrInputError, Stroke, read_inkml, write_inkml

XXE_DOC = b"""<?xml version="1.0"?>
<!DOCTYPE ink [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<ink xmlns="http://www.w3.org/2003/InkML">
  <annotation type="label">&xxe;</annotation>
  <trace>0 0 0, 1 1 8</trace>
</ink>
"""

BILLION_LAUGHS = b"""<?xml version="1.0"?>
<!DOCTYPE lolz [
  <!ENTITY lol "lol">
  <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
]>
<ink xmlns="http://www.w3.org/2003/InkML"><trace>0 0 0, 1 1 8</trace></ink>
"""


def synthetic_ink() -> Ink:
    """Two deterministic strokes with sub-pixel coordinates and millisecond stamps."""
    first = Stroke(
        np.array(
            [[10.5, 20.25, 0.0], [12.75, 22.5, 8.0], [15.0, 25.125, 16.0]],
            dtype=np.float32,
        )
    )
    second = Stroke(
        np.array([[30.0, 10.0, 40.0], [33.5, 12.5, 48.0]], dtype=np.float32),
    )
    return Ink(
        strokes=[first, second],
        label="x^{2}",
        writer_id="w-0001",
        meta={"t0_ms": 1_700_000_000_000.0, "source": "canvas"},
    )


class TestRoundtrip:
    def test_inkml_read_write_roundtrip_synthetic(self) -> None:
        ink = synthetic_ink()
        blob = write_inkml(ink)
        assert isinstance(blob, bytes)
        assert b"\r\n" not in blob
        back = read_inkml(blob)

        assert back.n_strokes == ink.n_strokes
        assert back.label == ink.label
        assert back.writer_id == ink.writer_id
        assert back.meta["source"] == "canvas"
        assert back.t0_ms == pytest.approx(1_700_000_000_000.0)
        for original, restored in zip(ink.strokes, back.strokes, strict=True):
            assert np.array_equal(original.points, restored.points)

    def test_roundtrip_is_byte_stable(self) -> None:
        ink = synthetic_ink()
        assert write_inkml(read_inkml(write_inkml(ink))) == write_inkml(ink)

    def test_roundtrip_through_a_file(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.inkml"
        path.write_bytes(write_inkml(synthetic_ink()))
        back = read_inkml(path)
        assert back.label == "x^{2}"
        assert back.n_points == 5

    def test_empty_ink_roundtrip(self) -> None:
        back = read_inkml(write_inkml(Ink(strokes=[])))
        assert back.n_strokes == 0

    def test_reader_accepts_str_and_namespaceless_documents(self) -> None:
        doc = "<ink><trace>0 0 0, 1 2 8</trace></ink>"
        ink = read_inkml(doc)
        assert ink.n_strokes == 1
        assert ink.strokes[0].points[1].tolist() == [1.0, 2.0, 8.0]

    def test_two_column_traces_get_synthetic_timestamps(self) -> None:
        ink = read_inkml("<ink><trace>0 0, 1 2, 3 4</trace></ink>")
        assert ink.strokes[0].points.shape == (3, 3)
        assert ink.strokes[0].points[:, 2].tolist() == [0.0, 1.0, 2.0]


class TestSecurity:
    def test_inkml_parser_rejects_external_entities(self) -> None:
        with pytest.raises(OcrInputError) as exc:
            read_inkml(XXE_DOC)
        assert exc.value.code is ErrorCode.OCR_INPUT_INVALID
        assert exc.value.details.get("reason") == "xml_dtd_forbidden"

    def test_inkml_parser_rejects_entity_expansion(self) -> None:
        with pytest.raises(OcrInputError):
            read_inkml(BILLION_LAUGHS)

    def test_reader_rejects_oversized_payload(self) -> None:
        payload = b"<ink>" + b" " * (1_048_577) + b"</ink>"
        with pytest.raises(OcrInputError) as exc:
            read_inkml(payload)
        assert exc.value.details.get("reason") == "too_large"

    def test_reader_rejects_malformed_xml(self) -> None:
        with pytest.raises(OcrInputError):
            read_inkml(b"<ink><trace>0 0 0")

    def test_reader_rejects_wrong_root_element(self) -> None:
        with pytest.raises(OcrInputError):
            read_inkml(b"<svg><trace>0 0 0</trace></svg>")

    def test_reader_rejects_non_numeric_trace(self) -> None:
        with pytest.raises(OcrInputError):
            read_inkml("<ink><trace>a b c</trace></ink>")

    def test_reader_rejects_missing_file(self, tmp_path: Path) -> None:
        from aicls.core.errors import NotFoundError

        with pytest.raises(NotFoundError):
            read_inkml(tmp_path / "absent.inkml")

    def test_reader_rejects_point_budget(self) -> None:
        points = ", ".join(f"{i} {i} {i}" for i in range(20_001))
        with pytest.raises(OcrInputError):
            read_inkml(f"<ink><trace>{points}</trace></ink>")
