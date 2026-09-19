"""Unit tests for the API's pure plumbing: rate limiter, uploads, envelopes, buffers, export."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from aicls.api.errors import (
    MAX_VALIDATION_ERRORS,
    PROBLEM_MEDIA_TYPE,
    RateLimitedError,
    problem_response,
)
from aicls.api.export import operation_to_pdf, operation_to_tex
from aicls.api.middleware import SECURITY_HEADERS
from aicls.api.model_registry import ModelRegistry
from aicls.api.nlu_bridge import command_for_intent, command_payload, primary_latex
from aicls.api.pipeline import MAX_TEXT_CHARS, PipelineResult, _check_text
from aicls.api.ratelimit import DEFAULT_RULES, RateLimiter, RateRule, rules_for
from aicls.api.uploads import (
    MAX_INK_BYTES,
    check_image_bytes,
    check_ink_size,
    check_wav_bytes,
    upload_too_large,
)
from aicls.api.ws.audio import SAMPLE_WIDTH, AudioBuffer, is_endpoint
from aicls.api.ws.common import MessageRate, WsTimings
from aicls.api.ws.ink import InkBuffer
from aicls.api.ws.protocol import (
    MAX_FRAME_CHARS,
    WsEnvelope,
    envelope,
    error_envelope,
    slow_down,
)
from aicls.asr.audio_io import write_wav_pcm16
from aicls.asr.synth import make_speech_like
from aicls.core.errors import AiclsError, ErrorCode, InputValidationError
from aicls.core.settings import Settings
from aicls.db.enums import InputKind, ResultKind
from aicls.mathengine import Diff
from aicls.ocr.types import MAX_POINTS_PER_INK, OcrInputError


# ---------------------------------------------------------------------------------------------
# rate limiter
# ---------------------------------------------------------------------------------------------
def test_rate_limiter_admits_up_to_the_limit_then_reports_retry_after() -> None:
    """A sliding window admits ``limit`` hits and then names the wait."""
    clock = {"now": 0.0}
    limiter = RateLimiter(clock=lambda: clock["now"])
    rule = RateRule("auth_ip", 3, 60.0, "ip")
    assert [limiter.check(rule, "1.2.3.4") for _ in range(3)] == [None, None, None]
    retry = limiter.check(rule, "1.2.3.4")
    assert retry is not None
    assert 59.0 <= retry <= 60.0
    assert limiter.check(rule, "5.6.7.8") is None  # a different identity has its own bucket

    clock["now"] = 60.5
    assert limiter.check(rule, "1.2.3.4") is None


def test_rate_limiter_window_slides_rather_than_resetting() -> None:
    """A fixed window would allow ``2 * limit`` across a boundary; a sliding one does not."""
    clock = {"now": 0.0}
    limiter = RateLimiter(clock=lambda: clock["now"])
    rule = RateRule("compute", 2, 10.0, "user")
    assert limiter.check(rule, "u") is None
    clock["now"] = 9.9
    assert limiter.check(rule, "u") is None
    assert limiter.check(rule, "u") is not None


def test_rate_limiter_reset_and_bounded_table() -> None:
    """The counter table is resettable and does not grow without bound."""
    limiter = RateLimiter()
    rule = RateRule("compute", 5, 60.0, "user")
    for index in range(50):
        limiter.check(rule, f"user-{index}")
    assert limiter.tracked_keys == 50
    limiter.reset()
    assert limiter.tracked_keys == 0


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/auth/login", ("auth_ip", "auth_account")),
        ("/compute/text", ("compute",)),
        ("/ocr/recognize", ("upload",)),
        ("/asr/transcribe", ("upload",)),
        ("/contribute/recordings", ("recordings",)),
        ("/health/live", ()),
        ("/book/search", ()),
    ],
)
def test_rules_for_matches_the_documented_table(path: str, expected: tuple[str, ...]) -> None:
    """api.md §1 rate limits are matched by longest prefix."""
    assert tuple(rule.name for rule in rules_for(path)) == expected


def test_recordings_rule_wins_over_the_contribute_prefix() -> None:
    """``/contribute/recordings`` is 60/hour, not the generic contribute budget."""
    assert "/contribute/recordings" in DEFAULT_RULES
    rule = rules_for("/contribute/recordings")[0]
    assert (rule.limit, rule.window_s) == (60, 3600.0)


# ---------------------------------------------------------------------------------------------
# problem documents
# ---------------------------------------------------------------------------------------------
def test_problem_response_shape_and_headers() -> None:
    """A problem response carries the media type, the request id and ``Retry-After``."""
    response = problem_response(RateLimitedError(5), request_id="01J")
    assert response.status_code == 429
    assert response.media_type == PROBLEM_MEDIA_TYPE
    assert response.headers["Retry-After"] == "5"
    assert response.headers["X-Request-Id"] == "01J"

    unauthorised = problem_response(AiclsError(ErrorCode.AUTH_INVALID))
    assert unauthorised.headers["WWW-Authenticate"] == "Bearer"
    assert "X-Request-Id" not in unauthorised.headers


def test_validation_error_cap_is_declared() -> None:
    """A pathological body cannot produce an unbounded problem document."""
    assert 1 <= MAX_VALIDATION_ERRORS <= 20


def test_security_headers_are_the_documented_set() -> None:
    """docs/guvenlik-gizlilik.md §4 lists exactly these five headers."""
    assert set(SECURITY_HEADERS) == {
        "Content-Security-Policy",
        "X-Content-Type-Options",
        "X-Frame-Options",
        "Referrer-Policy",
        "Permissions-Policy",
    }
    assert "'unsafe-inline'" in SECURITY_HEADERS["Content-Security-Policy"]  # KaTeX
    assert SECURITY_HEADERS["Permissions-Policy"] == "microphone=(self)"


# ---------------------------------------------------------------------------------------------
# upload guards
# ---------------------------------------------------------------------------------------------
def test_check_image_bytes_accepts_png_and_jpeg() -> None:
    """Only the two documented rasters pass the magic-byte check."""
    assert check_image_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32) == "png"
    assert check_image_bytes(b"\xff\xd8\xff" + b"0" * 32) == "jpeg"
    with pytest.raises(OcrInputError):
        check_image_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32, max_bytes=8)


def test_check_wav_bytes_requires_a_riff_wave_container() -> None:
    """ADR-0017: RIFF/WAVE only, size checked first."""
    payload = write_wav_pcm16(make_speech_like(duration_s=0.5))
    check_wav_bytes(payload)
    with pytest.raises(AiclsError):
        check_wav_bytes(payload, max_bytes=16)
    with pytest.raises(AiclsError):
        check_wav_bytes(b"RIFF" + b"0" * 20)  # RIFF but not WAVE


def test_check_ink_size_bounds_the_canvas_payload() -> None:
    """A Canvas JSON payload larger than 1 MB is refused before it is parsed."""
    check_ink_size({"strokes": []})
    with pytest.raises(OcrInputError) as failure:
        check_ink_size({"strokes": [{"points": [{"x": 1, "y": 2, "t": 3}] * 60000}]})
    assert failure.value.details["reason"] == "too_large"
    assert failure.value.details["limit"] == MAX_INK_BYTES


def test_upload_too_large_is_an_upload_invalid() -> None:
    """The oversized-upload error is the documented code and status."""
    error = upload_too_large(99, 10, "audio")
    assert error.code is ErrorCode.UPLOAD_INVALID
    assert error.http_status == 400
    assert error.details["field"] == "audio"


# ---------------------------------------------------------------------------------------------
# WebSocket protocol and buffers
# ---------------------------------------------------------------------------------------------
def test_envelope_round_trip_and_reply_keeps_seq() -> None:
    """``{type, seq, payload}`` in, the same ``seq`` back."""
    parsed = WsEnvelope.parse('{"type": "stroke", "seq": 7, "payload": {"points": []}}')
    assert parsed.type == "stroke"
    assert parsed.seq == 7
    assert parsed.reply("partial", {"latex": "x"}) == {
        "type": "partial",
        "seq": 7,
        "payload": {"latex": "x"},
    }
    assert envelope("pong")["seq"] == 0
    assert slow_down(250)["payload"] == {"retry_after_ms": 250}
    assert error_envelope(AiclsError(ErrorCode.NOT_FOUND), 3)["payload"]["problem"]["status"] == 404


@pytest.mark.parametrize(
    "raw", ["not json", '{"seq": 1}', '{"type": "", "seq": 1}', '{"type": "x", "extra": 1}']
)
def test_envelope_refuses_a_malformed_frame(raw: str) -> None:
    """A bad frame is a ``VALIDATION_ERROR``, never an unhandled exception."""
    with pytest.raises(AiclsError) as failure:
        WsEnvelope.parse(raw)
    assert failure.value.code is ErrorCode.VALIDATION_ERROR


def test_envelope_refuses_an_oversized_frame() -> None:
    """A frame larger than the cap is refused before JSON parsing."""
    with pytest.raises(AiclsError) as failure:
        WsEnvelope.parse("x" * (MAX_FRAME_CHARS + 1))
    assert failure.value.details["reason"] == "frame_too_large"


def test_ink_buffer_enforces_the_point_budget() -> None:
    """The stream applies the same 20 000-point budget as the REST endpoint."""
    buffer = InkBuffer()
    assert buffer.empty
    buffer.add({"points": [{"x": 1, "y": 2, "t": 3}], "canvas": {"w": 800, "h": 300}})
    assert not buffer.empty
    assert buffer.payload()["canvas"] == {"w": 800, "h": 300}

    with pytest.raises(OcrInputError):
        buffer.add({"points": []})
    with pytest.raises(OcrInputError) as flood:
        buffer.add({"points": [{"x": 1, "y": 2, "t": 3}] * (MAX_POINTS_PER_INK + 1)})
    assert flood.value.details["reason"] == "too_many_points"

    buffer.clear()
    assert buffer.empty


def test_audio_buffer_caps_the_utterance_and_measures_energy() -> None:
    """A 30 s cap, and a trailing-RMS reading the endpointer can use."""
    buffer = AudioBuffer(sample_rate=16000, max_seconds=0.5)
    loud = (np.full(4000, 0.4, dtype=np.float64) * 32767).astype("<i2").tobytes()
    buffer.add(loud)
    assert buffer.duration_s == pytest.approx(0.25)
    assert buffer.size == 4000 * SAMPLE_WIDTH
    assert buffer.trailing_rms() > 0.3

    buffer.add(loud)
    with pytest.raises(AiclsError) as overflow:
        buffer.add(loud)
    assert overflow.value.details["reason"] == "buffer_overflow"
    buffer.clear()
    assert buffer.empty


def test_is_endpoint_needs_speech_then_silence() -> None:
    """Silence alone is not an endpoint; speech followed by silence is."""
    buffer = AudioBuffer(sample_rate=16000, max_seconds=5.0)
    silence = np.zeros(8000, dtype="<i2").tobytes()
    buffer.add(silence)
    assert is_endpoint(buffer, 250) is False

    buffer.had_speech = True
    assert is_endpoint(buffer, 250) is True

    loud = (np.full(8000, 0.4, dtype=np.float64) * 32767).astype("<i2").tobytes()
    buffer.add(loud)
    assert is_endpoint(buffer, 250) is False


def test_message_rate_strikes_before_closing() -> None:
    """``slow_down`` first, disconnect only after repeated offences."""
    clock = {"now": 0.0}
    rate = MessageRate(limit=2, window_s=1.0, clock=lambda: clock["now"])
    assert rate.allow() is True
    assert rate.allow() is True
    assert rate.allow() is False
    assert rate.strikes == 1
    assert rate.exhausted is False
    for _ in range(4):
        rate.allow()
    assert rate.exhausted is True
    clock["now"] = 2.0
    assert rate.allow() is True
    assert rate.strikes == 0


def test_ws_timings_are_overridable() -> None:
    """The stream budgets are data, so a deployment can tune them without a code change."""
    fast = WsTimings(pen_up_debounce_ms=10)
    assert fast.pen_up_debounce_ms == 10
    assert fast.idle_final_ms == 1500


# ---------------------------------------------------------------------------------------------
# nlu bridge and pipeline helpers
# ---------------------------------------------------------------------------------------------
def test_primary_latex_and_command_payload() -> None:
    """``input_latex`` is the command's own LaTeX field, and the payload is JSON-safe."""
    command = Diff(op="diff", expr="x^{2}", var="x")
    assert primary_latex(command) == "x^{2}"
    assert primary_latex(None) is None
    payload = command_payload(command)
    assert payload == {"op": "diff", "expr": "x^{2}", "var": "x", "order": 1}
    assert command_payload(None) is None


def test_command_for_intent_uses_the_rule_based_resolver() -> None:
    """A named intent still has to fill its slots from the text; it is never invented."""
    command, slots = command_for_intent("x karenin türevini al", "differentiate")
    assert command is not None
    assert command.op == "diff"
    assert slots["expr"] == "x^{2}"

    missing, _ = command_for_intent("türevini al", "differentiate")
    assert missing is None

    unknown, empty = command_for_intent("x karenin türevini al", "bilinmeyen_intent")
    assert unknown is None
    assert empty == {}


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_check_text_rejects_empty_input(text: str) -> None:
    """An empty command is a validation error, not an NLU refusal."""
    with pytest.raises(InputValidationError) as failure:
        _check_text(text)
    assert failure.value.details["field"] == "text"


def test_check_text_bounds_the_length() -> None:
    """The 500-character bound of docs/guvenlik-gizlilik.md §6."""
    assert _check_text("  x karenin türevini al  ") == "x karenin türevini al"
    with pytest.raises(InputValidationError) as failure:
        _check_text("a" * (MAX_TEXT_CHARS + 1))
    assert failure.value.details["max_len"] == MAX_TEXT_CHARS


def test_pipeline_result_defaults_are_empty_not_none() -> None:
    """A result the caller does not fill in is still safe to serialise."""
    result = PipelineResult(kind="ui")
    assert result.book_units == ()
    assert result.timings == {}
    assert result.model_versions == {}
    assert result.operation_id is None


# ---------------------------------------------------------------------------------------------
# model registry
# ---------------------------------------------------------------------------------------------
def test_model_registry_defaults_to_the_stubs() -> None:
    """Without a database and with models disabled, both components are the stubs."""
    registry = ModelRegistry(Settings(env="test", models_enabled=False))
    assert registry.state("ocr") == "stub"
    assert registry.state("asr") == "stub"
    assert registry.versions() == {"ocr": "stub-0.1.0", "asr": "asr-stub-1"}
    assert registry.states() == {"ocr": "stub", "asr": "stub"}
    first = registry.get_recognizer()
    assert registry.get_recognizer() is first  # cached
    registry.reload()
    assert registry.get_recognizer() is not first


def test_model_registry_accepts_an_injected_backend() -> None:
    """A warm swap replaces the instance and reports ``loaded``."""

    class Dummy:
        model_version = "dummy"

        def warmup(self) -> None:
            """Nothing to load."""

        def recognize(self, inp: Any, *, beam: int = 4) -> Any:
            """Never called in this test."""
            raise NotImplementedError

    registry = ModelRegistry(Settings(env="test", models_enabled=False))
    dummy = Dummy()
    registry.set_recognizer(dummy, version="ocr-9.9.9")
    assert registry.get_recognizer() is dummy
    assert registry.state("ocr") == "loaded"
    assert registry.version("ocr") == "ocr-9.9.9"


# ---------------------------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------------------------
class _Row:
    """A minimal stand-in for an ``operations`` row (the export reads five fields)."""

    def __init__(self, **fields: Any) -> None:
        self.source_text = fields.get("source_text")
        self.input_latex = fields.get("input_latex")
        self.result_latex = fields.get("result_latex")
        self.error_code = fields.get("error_code")
        self.input_kind = InputKind.TEXT
        self.result_kind = ResultKind.EXPR


def test_operation_to_tex_is_a_standalone_document() -> None:
    """The export compiles as a document and carries both formulas."""
    source = operation_to_tex(
        _Row(source_text="x karenin türevini al", input_latex="x^{2}", result_latex=r"2 \cdot x")
    )
    assert source.startswith(r"\documentclass")
    assert r"\begin{document}" in source
    assert r"\end{document}" in source
    assert "x^{2}" in source
    assert r"2 \cdot x" in source


def test_operation_to_tex_reports_a_failed_operation() -> None:
    """A refused command still exports, with its error code."""
    source = operation_to_tex(_Row(source_text="?", error_code="UNSUPPORTED_PROBLEM"))
    assert "UNSUPPORTED_PROBLEM" in source


def test_operation_to_pdf_is_a_single_page_pdf() -> None:
    """The PDF is produced with Matplotlib mathtext; no TeX distribution is required."""
    payload = operation_to_pdf(
        _Row(source_text="x karenin türevini al", input_latex="x^{2}", result_latex=r"2 \cdot x")
    )
    assert payload.startswith(b"%PDF")
    assert len(payload) > 500
