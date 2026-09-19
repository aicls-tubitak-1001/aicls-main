"""Canvas JSON -> Ink tests (`/ws/v1/ink` and `POST /ocr/recognize` payloads, ADR-0013)."""

from __future__ import annotations

from typing import Any

import pytest

from aicls.ocr import OcrInputError, ink_from_canvas_json, ink_to_canvas_json

T0 = 1_700_000_000_000


def canvas_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "width": 800,
        "height": 200,
        "label": "x^{2}",
        "writer_id": "w-42",
        "strokes": [
            {
                "points": [
                    {"x": 10.0, "y": 20.0, "t": T0},
                    {"x": 11.5, "y": 21.0, "t": T0 + 16},
                    {"x": 13.0, "y": 22.5, "t": T0 + 33},
                ]
            },
            {"points": [{"x": 40.0, "y": 20.0, "t": T0 + 250}]},
        ],
    }


class TestCanvasJson:
    def test_ink_from_canvas_json_preserves_timestamps(self) -> None:
        ink = ink_from_canvas_json(canvas_payload())

        assert ink.t0_ms == float(T0)
        absolute = [
            [float(ink.t0_ms + t) for t in stroke.points[:, 2].tolist()] for stroke in ink.strokes
        ]
        assert absolute == [
            [float(T0), float(T0 + 16), float(T0 + 33)],
            [float(T0 + 250)],
        ]
        # relative timestamps are exact in float32 and monotonically increasing
        assert ink.strokes[0].points[:, 2].tolist() == [0.0, 16.0, 33.0]
        assert ink.strokes[1].points[0, 2] == 250.0
        assert ink.duration_ms == pytest.approx(250.0)

    def test_metadata_is_carried(self) -> None:
        ink = ink_from_canvas_json(canvas_payload())
        assert ink.label == "x^{2}"
        assert ink.writer_id == "w-42"
        assert ink.meta["canvas_width"] == 800.0
        assert ink.meta["canvas_height"] == 200.0

    def test_relative_timestamps_stay_relative(self) -> None:
        obj = {"strokes": [{"points": [{"x": 0, "y": 0, "t": 0}, {"x": 1, "y": 1, "t": 8}]}]}
        ink = ink_from_canvas_json(obj)
        assert ink.t0_ms == 0.0
        assert ink.strokes[0].points[:, 2].tolist() == [0.0, 8.0]

    def test_parallel_array_form_is_accepted(self) -> None:
        obj = {"strokes": [{"x": [0, 1], "y": [2, 3], "t": [0, 8]}]}
        ink = ink_from_canvas_json(obj)
        assert ink.strokes[0].points.tolist() == [[0.0, 2.0, 0.0], [1.0, 3.0, 8.0]]

    def test_triplet_form_is_accepted(self) -> None:
        obj = {"strokes": [[[0, 2, 0], [1, 3, 8]]]}
        ink = ink_from_canvas_json(obj)
        assert ink.strokes[0].points.tolist() == [[0.0, 2.0, 0.0], [1.0, 3.0, 8.0]]

    def test_points_without_timestamps_are_indexed(self) -> None:
        obj = {"strokes": [{"points": [{"x": 0, "y": 0}, {"x": 1, "y": 1}]}]}
        ink = ink_from_canvas_json(obj)
        assert ink.strokes[0].points[:, 2].tolist() == [0.0, 1.0]

    def test_empty_strokes_are_dropped(self) -> None:
        obj = {"strokes": [{"points": []}, {"points": [{"x": 1, "y": 1, "t": 0}]}]}
        assert ink_from_canvas_json(obj).n_strokes == 1

    @pytest.mark.parametrize(
        "obj",
        [
            [],
            {"strokes": "nope"},
            {"strokes": [{"points": "nope"}]},
            {"strokes": [{"points": [{"x": "a", "y": 1, "t": 0}]}]},
            {"strokes": [{"x": [0, 1], "y": [2]}]},
            {"strokes": [{"points": [{"y": 1, "t": 0}]}]},
            {"strokes": [{"points": [{"x": float("nan"), "y": 1, "t": 0}]}]},
        ],
    )
    def test_invalid_payloads_rejected(self, obj: Any) -> None:
        with pytest.raises(OcrInputError):
            ink_from_canvas_json(obj)

    def test_stroke_budget_enforced(self) -> None:
        obj = {"strokes": [{"points": [{"x": 0, "y": 0, "t": 0}]} for _ in range(2_001)]}
        with pytest.raises(OcrInputError):
            ink_from_canvas_json(obj)

    def test_canvas_json_roundtrip(self) -> None:
        ink = ink_from_canvas_json(canvas_payload())
        again = ink_from_canvas_json(ink_to_canvas_json(ink))
        assert again.label == ink.label
        assert again.writer_id == ink.writer_id
        assert again.t0_ms == ink.t0_ms
        assert again.strokes[0].points.tolist() == ink.strokes[0].points.tolist()
