"""StageTimer: nested stages, accumulation, Server-Timing header (RFC 9211)."""

from __future__ import annotations

import pytest

from aicls.core.errors import AiclsError
from aicls.core.timing import StageTimer


class FakeClock:
    """Deterministic perf-counter replacement (seconds)."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, ms: float) -> None:
        self.now += ms / 1000.0


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def test_stage_records_milliseconds(clock: FakeClock) -> None:
    timer = StageTimer(clock=clock)
    with timer.stage("nlu"):
        clock.advance(12.3)
    assert timer.stages == {"nlu": pytest.approx(12.3)}


def test_nested_stages_each_recorded_and_outer_includes_inner(clock: FakeClock) -> None:
    timer = StageTimer(clock=clock)
    with timer.stage("pipeline"):
        clock.advance(1.0)
        with timer.stage("nlu"):
            clock.advance(10.0)
        with timer.stage("cas"):
            clock.advance(40.0)
            with timer.stage("simplify"):
                clock.advance(5.0)
        clock.advance(2.0)
    stages = timer.stages
    assert list(stages) == ["pipeline", "nlu", "cas", "simplify"]
    assert stages["nlu"] == pytest.approx(10.0)
    assert stages["cas"] == pytest.approx(45.0)
    assert stages["simplify"] == pytest.approx(5.0)
    assert stages["pipeline"] == pytest.approx(58.0)
    assert timer.depth == 0


def test_repeated_stage_name_accumulates(clock: FakeClock) -> None:
    timer = StageTimer(clock=clock)
    for _ in range(3):
        with timer.stage("cas"):
            clock.advance(2.0)
    assert timer.stages["cas"] == pytest.approx(6.0)
    assert timer.counts["cas"] == 3


def test_stage_recorded_even_when_body_raises(clock: FakeClock) -> None:
    timer = StageTimer(clock=clock)
    with pytest.raises(RuntimeError), timer.stage("parse"):
        clock.advance(3.0)
        raise RuntimeError("boom")
    assert timer.stages["parse"] == pytest.approx(3.0)
    assert timer.depth == 0


def test_current_stage_and_depth_during_nesting(clock: FakeClock) -> None:
    timer = StageTimer(clock=clock)
    assert timer.current_stage is None
    with timer.stage("a"):
        assert timer.current_stage == "a"
        assert timer.depth == 1
        with timer.stage("b"):
            assert timer.current_stage == "b"
            assert timer.depth == 2
        assert timer.current_stage == "a"


def test_add_external_timing_and_total(clock: FakeClock) -> None:
    timer = StageTimer(clock=clock)
    with timer.stage("nlu"):
        clock.advance(4.0)
    timer.add("render", 0.4)
    clock.advance(1.0)
    assert timer.stages == {"nlu": pytest.approx(4.0), "render": pytest.approx(0.4)}
    assert timer.total_ms == pytest.approx(5.0)  # wall time since construction


def test_server_timing_header_format(clock: FakeClock) -> None:
    timer = StageTimer(clock=clock)
    with timer.stage("nlu"):
        clock.advance(12.3)
    with timer.stage("parse"):
        clock.advance(1.1)
    with timer.stage("cas"):
        clock.advance(41.0)
    with timer.stage("render"):
        clock.advance(0.4)
    clock.advance(3.4)
    header = timer.server_timing_header()
    assert header == "nlu;dur=12.3, parse;dur=1.1, cas;dur=41.0, render;dur=0.4, app;dur=58.2"
    assert timer.server_timing_header(total_name=None) == (
        "nlu;dur=12.3, parse;dur=1.1, cas;dur=41.0, render;dur=0.4"
    )


def test_server_timing_header_empty_timer(clock: FakeClock) -> None:
    timer = StageTimer(clock=clock)
    clock.advance(0.05)
    assert timer.server_timing_header() == "app;dur=0.1"
    assert timer.server_timing_header(total_name=None) == ""


def test_invalid_stage_name_rejected(clock: FakeClock) -> None:
    timer = StageTimer(clock=clock)
    with pytest.raises(AiclsError), timer.stage("bad name;"):
        pass
    with pytest.raises(AiclsError):
        timer.add("", 1.0)
    with pytest.raises(AiclsError):
        timer.add("x", -1.0)


def test_as_dict_and_reset(clock: FakeClock) -> None:
    timer = StageTimer(clock=clock)
    with timer.stage("nlu"):
        clock.advance(1.0)
    snapshot = timer.as_dict()
    assert snapshot == {"nlu": pytest.approx(1.0)}
    snapshot["nlu"] = 99.0
    assert timer.stages["nlu"] == pytest.approx(1.0)  # copies, not views
    timer.reset()
    assert timer.stages == {}
    assert timer.total_ms == pytest.approx(0.0)


def test_real_clock_smoke() -> None:
    timer = StageTimer()
    with timer.stage("x"):
        pass
    assert timer.stages["x"] >= 0.0
    assert timer.total_ms >= timer.stages["x"]
