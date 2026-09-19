"""Contract tests between the browser client (``web/``) and the server-side codecs.

The front-end and the recognition packages were written in parallel against ``docs/mimari/api.md``,
so nothing in either test suite proves that the bytes one produces are the bytes the other accepts.
These tests close that gap for the two payloads that cross the wire:

* the Canvas JSON that ``web/src/lib/inkml.ts::strokesToCanvasJson`` emits, read by
  :func:`aicls.ocr.ink_from_canvas_json`;
* the 44-byte RIFF/WAVE PCM16 container that ``web/src/lib/wavEncoder.ts::encodeWavBytes`` writes,
  read by :func:`aicls.asr.read_wav_pcm16`.

The payloads here are literal mirrors of the TypeScript sources named above. Changing either side's
shape must change this file too — that is the point of it.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np
import pytest

from aicls.asr import ACCEPTED_SAMPLE_RATES, MAX_SECONDS, MAX_WAV_BYTES, read_wav_pcm16
from aicls.ocr import (
    MAX_POINTS_PER_INK,
    OcrInputError,
    ink_from_canvas_json,
    ink_to_canvas_json,
)

pytestmark = pytest.mark.integration

#: ``TARGET_SAMPLE_RATE`` in ``web/src/lib/wavEncoder.ts``.
WEB_TARGET_SAMPLE_RATE = 16_000

#: ``MAX_DURATION_S`` in ``web/src/lib/wavEncoder.ts``.
WEB_MAX_DURATION_S = 30

#: ``MAX_INK_POINTS`` in ``web/src/lib/inkml.ts``.
WEB_MAX_INK_POINTS = 20_000


def web_canvas_json(strokes: list[list[tuple[float, float, int]]]) -> dict[str, Any]:
    """Build the exact payload shape of ``strokesToCanvasJson`` (parallel coordinate arrays)."""
    return {
        "version": 1,
        "canvas": {"w": 800, "h": 300},
        "strokes": [
            {
                "x": [round(x, 2) for x, _, _ in stroke],
                "y": [round(y, 2) for _, y, _ in stroke],
                "t": [t for _, _, t in stroke],
            }
            for stroke in strokes
        ],
    }


def web_wav_bytes(samples: np.ndarray, sample_rate: int = WEB_TARGET_SAMPLE_RATE) -> bytes:
    """Build the exact 44-byte canonical container of ``encodeWavBytes`` (mono PCM16, LE)."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.where(clipped < 0, clipped * 32768.0, clipped * 32767.0).astype(np.int16)
    payload = pcm.tobytes()
    bytes_per_sample = 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(payload),
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        1,  # mono
        sample_rate,
        sample_rate * bytes_per_sample,
        bytes_per_sample,
        8 * bytes_per_sample,
        b"data",
        len(payload),
    )
    assert len(header) == 44
    return header + payload


class TestCanvasJsonContract:
    """``web`` ink payload -> :func:`aicls.ocr.ink_from_canvas_json`."""

    def test_the_web_canvas_payload_is_parsed_into_an_ink(self) -> None:
        payload = web_canvas_json([[(10.5, 20.25, 0), (12.0, 21.0, 16)], [(30.0, 40.0, 120)]])

        ink = ink_from_canvas_json(payload)

        assert len(ink.strokes) == 2
        assert ink.strokes[0].points.shape == (2, 3)
        assert ink.strokes[0].points[0][0] == pytest.approx(10.5)
        assert ink.strokes[1].points[0][1] == pytest.approx(40.0)

    def test_the_nested_canvas_size_survives_the_conversion(self) -> None:
        """Regression: the client nests the surface size, so flat width/height would drop it."""
        payload = web_canvas_json([[(1.0, 2.0, 0)]])

        ink = ink_from_canvas_json(payload)

        assert ink.meta["canvas_width"] == 800.0
        assert ink.meta["canvas_height"] == 300.0

    def test_timestamps_stay_relative_with_the_origin_in_meta(self) -> None:
        epoch = 1_700_000_000_000
        payload = web_canvas_json([[(1.0, 1.0, epoch), (2.0, 2.0, epoch + 16)]])

        ink = ink_from_canvas_json(payload)

        assert ink.meta["t0_ms"] == float(epoch)
        assert ink.strokes[0].points[1][2] == pytest.approx(16.0)

    def test_a_replayed_payload_carries_the_nested_canvas_object_back(self) -> None:
        """``ink_to_canvas_json`` must emit the shape the client sent, so a replay round-trips."""
        ink = ink_from_canvas_json(web_canvas_json([[(1.0, 2.0, 0), (3.0, 4.0, 8)]]))

        replayed = ink_to_canvas_json(ink)

        assert replayed["canvas"] == {"w": 800.0, "h": 300.0}
        assert ink_from_canvas_json(replayed).meta["canvas_width"] == 800.0

    def test_a_payload_without_a_canvas_object_keeps_the_size_out_of_meta(self) -> None:
        ink = ink_from_canvas_json({"strokes": [{"x": [1.0], "y": [2.0], "t": [0]}]})

        assert "canvas_width" not in ink.meta
        assert "canvas" not in ink_to_canvas_json(ink)

    @pytest.mark.parametrize(
        "canvas",
        [
            pytest.param("800x300", id="not_an_object"),
            pytest.param({"w": "800", "h": None}, id="unusable_values"),
            pytest.param({"w": True, "h": False}, id="booleans_are_not_sizes"),
        ],
    )
    def test_an_unusable_canvas_field_is_ignored_rather_than_rejected(self, canvas: Any) -> None:
        """Surface size is provenance only; a malformed one must not fail a recognition."""
        payload = {"strokes": [{"x": [1.0], "y": [2.0], "t": [0]}], "canvas": canvas}

        ink = ink_from_canvas_json(payload)

        assert len(ink.strokes) == 1
        assert "canvas_width" not in ink.meta

    @pytest.mark.parametrize(
        ("strokes", "reason"),
        [
            pytest.param([{"x": [1.0], "y": [2.0], "t": [0, 1]}], "stroke_arrays_t", id="ragged_t"),
            pytest.param([{"x": [1.0], "z": [2.0]}], "stroke_shape", id="no_y_axis"),
            pytest.param(["1,2"], "stroke_type", id="stroke_is_a_string"),
            pytest.param([[[1.0]]], "point_shape", id="point_without_y"),
        ],
    )
    def test_a_malformed_stroke_is_refused_with_a_named_reason(
        self, strokes: list[Any], reason: str
    ) -> None:
        """A hostile or buggy client must hit a typed 400, never an unhandled exception."""
        with pytest.raises(OcrInputError) as excinfo:
            ink_from_canvas_json({"strokes": strokes})

        assert excinfo.value.details["reason"] == reason

    def test_the_client_and_server_point_budgets_agree(self) -> None:
        assert WEB_MAX_INK_POINTS == MAX_POINTS_PER_INK

    def test_the_server_refuses_a_payload_over_the_budget_the_client_enforces(self) -> None:
        """The client cap is a convenience; the server must refuse a hand-crafted payload too."""
        size = MAX_POINTS_PER_INK + 1
        payload = web_canvas_json([[(float(i), 0.0, i) for i in range(size)]])

        with pytest.raises(OcrInputError) as excinfo:
            ink_from_canvas_json(payload)

        assert excinfo.value.details["reason"] == "too_many_points"


class TestWavContract:
    """``web`` recorder output -> :func:`aicls.asr.read_wav_pcm16`."""

    def test_the_web_encoder_container_is_accepted(self) -> None:
        samples = (0.4 * np.sin(np.linspace(0.0, 40.0, 16_000, dtype=np.float64))).astype(
            np.float32
        )

        audio = read_wav_pcm16(web_wav_bytes(samples))

        assert audio.sr == WEB_TARGET_SAMPLE_RATE
        assert audio.n_samples == samples.size
        assert audio.duration_s == pytest.approx(1.0)
        assert np.abs(audio.samples - samples).max() <= 2.0 / 32767.0

    def test_the_client_rate_and_limits_are_the_ones_the_server_accepts(self) -> None:
        assert WEB_TARGET_SAMPLE_RATE in ACCEPTED_SAMPLE_RATES
        assert float(WEB_MAX_DURATION_S) == MAX_SECONDS
        # 30 s of 16 kHz mono PCM16 plus the header is far below the upload ceiling.
        assert WEB_MAX_DURATION_S * WEB_TARGET_SAMPLE_RATE * 2 + 44 < MAX_WAV_BYTES
