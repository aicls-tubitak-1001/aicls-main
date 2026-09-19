"""Stroke resampling, normalisation and delta+Bezier feature tests."""

from __future__ import annotations

import numpy as np
import pytest

from aicls.ocr import Ink, Stroke, normalize_strokes
from aicls.ocr.features import (
    STROKE_FEATURE_DIM,
    STROKE_FEATURE_NAMES,
    fit_cubic_bezier,
    resample_polyline,
)


def line_ink(n: int = 21, *, length: float = 100.0) -> Ink:
    xs = np.linspace(0.0, length, n)
    ys = np.zeros(n)
    ts = np.arange(n, dtype=np.float64) * 10.0
    return Ink(strokes=[Stroke.from_xyt(xs, ys, ts)])


class TestResample:
    def test_resample_polyline_uses_constant_arc_length(self) -> None:
        pts = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        out = resample_polyline(pts, 2.0)
        assert out.shape == (6, 2)
        steps = np.linalg.norm(np.diff(out, axis=0), axis=1)
        assert np.allclose(steps, 2.0, atol=1e-4)

    def test_resample_keeps_endpoints(self) -> None:
        pts = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        out = resample_polyline(pts, 1.0)
        assert np.allclose(out[0], pts[0])
        assert np.allclose(out[-1], pts[-1])

    def test_degenerate_polyline_is_returned_unchanged(self) -> None:
        pts = np.zeros((4, 2), dtype=np.float32)
        out = resample_polyline(pts, 2.0)
        assert out.shape == (1, 2)

    def test_single_point_polyline(self) -> None:
        pts = np.array([[1.0, 2.0]], dtype=np.float32)
        assert resample_polyline(pts, 2.0).tolist() == [[1.0, 2.0]]

    def test_invalid_spacing_rejected(self) -> None:
        from aicls.ocr import OcrInputError

        with pytest.raises(OcrInputError):
            resample_polyline(np.zeros((2, 2), dtype=np.float32), 0.0)


class TestBezier:
    def test_fit_cubic_bezier_on_a_straight_line(self) -> None:
        pts = np.stack([np.linspace(0, 1, 10), np.zeros(10)], axis=1).astype(np.float32)
        ctrl = fit_cubic_bezier(pts)
        assert ctrl.shape == (4, 2)
        assert np.allclose(ctrl[0], pts[0], atol=1e-5)
        assert np.allclose(ctrl[-1], pts[-1], atol=1e-5)
        # control points of a straight line lie on the line (y stays 0)
        assert np.allclose(ctrl[:, 1], 0.0, atol=1e-5)

    def test_fit_cubic_bezier_on_a_curve_bulges(self) -> None:
        ts = np.linspace(0.0, np.pi, 24)
        pts = np.stack([ts, np.sin(ts)], axis=1).astype(np.float32)
        ctrl = fit_cubic_bezier(pts)
        assert ctrl[1, 1] > 0.5
        assert ctrl[2, 1] > 0.5

    def test_fit_cubic_bezier_short_stroke(self) -> None:
        pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        ctrl = fit_cubic_bezier(pts)
        assert ctrl.shape == (4, 2)
        assert np.allclose(ctrl[0], [0.0, 0.0])
        assert np.allclose(ctrl[3], [1.0, 1.0])


class TestNormalizeStrokes:
    def test_shape_and_dtype(self) -> None:
        feats = normalize_strokes(line_ink(), resample_px=10.0)
        assert feats.ndim == 2
        assert feats.shape[1] == STROKE_FEATURE_DIM == len(STROKE_FEATURE_NAMES)
        assert feats.dtype == np.float32

    def test_resample_px_controls_sequence_length(self) -> None:
        dense = normalize_strokes(line_ink(), resample_px=2.0)
        sparse = normalize_strokes(line_ink(), resample_px=20.0)
        assert dense.shape[0] > sparse.shape[0]

    def test_scale_invariance_of_normalisation(self) -> None:
        small = normalize_strokes(line_ink(length=100.0), resample_px=10.0)
        big = normalize_strokes(line_ink(length=1000.0), resample_px=100.0)
        assert small.shape == big.shape
        assert np.allclose(small[:, :2], big[:, :2], atol=1e-4)

    def test_pen_flags_mark_stroke_boundaries(self) -> None:
        ink = Ink(
            strokes=[
                Stroke.from_xyt([0, 10], [0, 0], [0, 10]),
                Stroke.from_xyt([20, 30], [5, 5], [30, 40]),
            ]
        )
        feats = normalize_strokes(ink, resample_px=5.0)
        down = STROKE_FEATURE_NAMES.index("pen_down")
        up = STROKE_FEATURE_NAMES.index("pen_up")
        assert feats[0, down] == 1.0
        assert feats[-1, up] == 1.0
        assert feats[:, down].sum() == 2.0
        assert feats[:, up].sum() == 2.0

    def test_first_delta_is_zero(self) -> None:
        feats = normalize_strokes(line_ink(), resample_px=10.0)
        dx = STROKE_FEATURE_NAMES.index("dx")
        dy = STROKE_FEATURE_NAMES.index("dy")
        assert feats[0, dx] == 0.0
        assert feats[0, dy] == 0.0

    def test_empty_ink_gives_empty_feature_matrix(self) -> None:
        feats = normalize_strokes(Ink(strokes=[]))
        assert feats.shape == (0, STROKE_FEATURE_DIM)

    def test_single_point_ink_is_handled(self) -> None:
        feats = normalize_strokes(Ink(strokes=[Stroke.from_xyt([5.0], [5.0], [0.0])]))
        assert feats.shape == (1, STROKE_FEATURE_DIM)
        assert np.isfinite(feats).all()

    def test_features_are_finite_and_deterministic(self) -> None:
        ink = line_ink()
        first = normalize_strokes(ink, resample_px=3.0)
        second = normalize_strokes(ink, resample_px=3.0)
        assert np.array_equal(first, second)
        assert np.isfinite(first).all()

    def test_bezier_columns_are_constant_within_a_stroke(self) -> None:
        feats = normalize_strokes(line_ink(), resample_px=10.0)
        c1x = STROKE_FEATURE_NAMES.index("bezier_c1x")
        assert np.allclose(feats[:, c1x], feats[0, c1x])
