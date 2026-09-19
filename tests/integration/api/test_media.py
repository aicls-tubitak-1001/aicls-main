"""Handwriting and speech endpoints, and the upload guards in front of them (api.md §3.5–§3.6)."""

from __future__ import annotations

import io
from typing import Any

from fastapi import FastAPI
import httpx
import numpy as np
import pytest

from aicls.asr.audio_io import write_wav_pcm16
from aicls.asr.synth import make_speech_like
from aicls.asr.types import Audio, Transcript
from aicls.ocr.types import MAX_POINTS_PER_INK
from tests.integration.api.conftest import account, auth_header

pytestmark = pytest.mark.integration


def canvas_json(n_points: int = 40, strokes: int = 2) -> dict[str, Any]:
    """A deterministic Canvas JSON payload (the shape the web client sends, ADR-0013)."""
    return {
        "version": 1,
        "canvas": {"w": 800, "h": 300},
        "strokes": [
            {
                "points": [
                    {"x": 10.0 + index, "y": 40.0 + (index % 7), "t": 1700000000000 + index * 8}
                    for index in range(n_points)
                ]
            }
            for _stroke in range(strokes)
        ],
    }


def png_bytes(width: int = 32, height: int = 16) -> bytes:
    """A tiny grayscale PNG with a diagonal stroke on it."""
    from PIL import Image as PILImage

    array = np.full((height, width), 255, dtype=np.uint8)
    for index in range(min(width, height)):
        array[index, index] = 0
    buffer = io.BytesIO()
    PILImage.fromarray(array, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def wav_bytes(duration_s: float = 1.6) -> bytes:
    """A synthetic PCM16 WAV that passes the quality checks."""
    return write_wav_pcm16(make_speech_like(duration_s=duration_s))


class FixedTranscriber:
    """A transcriber that always answers the same Turkish command (test double)."""

    model_version = "test-fixed"

    def __init__(self, text: str = "x karenin türevini al") -> None:
        self.text = text

    def warmup(self) -> None:
        """Nothing to load."""

    def transcribe(self, audio: Audio) -> Transcript:
        """Return the fixed sentence with a high confidence."""
        del audio
        return Transcript(
            text=self.text, confidence=0.93, elapsed_ms=1.0, model_version=self.model_version
        )


@pytest.fixture
async def student(client: httpx.AsyncClient, app: FastAPI) -> dict[str, str]:
    """A signed-in student."""
    _user, token = await account(client, app, "medya@ornek.edu.tr")
    return auth_header(token)


async def test_api_ocr_recognize_canvas_json(
    client: httpx.AsyncClient, student: dict[str, str]
) -> None:
    """Canvas JSON is recognised by the stub and the recognition is stored."""
    response = await client.post("/ocr/recognize", json={"ink": canvas_json()}, headers=student)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["latex"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["alternatives"]
    assert body["model_version"] == "stub-0.1.0"
    assert body["recognition_id"]
    assert body["timings"]["total"] >= 0


async def test_api_ocr_upload_rejects_non_png_magic(
    client: httpx.AsyncClient, student: dict[str, str]
) -> None:
    """A file claiming to be a PNG but starting with anything else is refused at the door."""
    response = await client.post(
        "/ocr/recognize",
        files={"image": ("sahte.png", b"bu bir PNG degil", "image/png")},
        headers=student,
    )
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "OCR_INPUT_INVALID"
    assert body["reason"] == "bad_magic"


async def test_api_ocr_upload_accepts_a_real_png(
    client: httpx.AsyncClient, student: dict[str, str]
) -> None:
    """The offline path decodes a genuine PNG and recognises it."""
    response = await client.post(
        "/ocr/recognize",
        files={"image": ("el-yazisi.png", png_bytes(), "image/png")},
        headers=student,
    )
    assert response.status_code == 200, response.text
    assert response.json()["latex"]


async def test_api_ocr_ink_json_over_20000_points_rejected(
    client: httpx.AsyncClient, student: dict[str, str]
) -> None:
    """The 20 000-point budget of api.md §1 is enforced before any model runs."""
    payload = canvas_json(n_points=MAX_POINTS_PER_INK + 1, strokes=1)
    response = await client.post("/ocr/recognize", json={"ink": payload}, headers=student)
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "OCR_INPUT_INVALID"
    assert body["reason"] == "too_many_points"
    assert body["limit"] == MAX_POINTS_PER_INK


async def test_api_ocr_feedback_records_the_correction(
    client: httpx.AsyncClient, student: dict[str, str]
) -> None:
    """A corrected recognition is stored for İP-3 retraining."""
    recognised = await client.post("/ocr/recognize", json={"ink": canvas_json()}, headers=student)
    recognition_id = recognised.json()["recognition_id"]
    response = await client.post(
        "/ocr/feedback",
        json={"recognition_id": recognition_id, "corrected_latex": "x^{2}"},
        headers=student,
    )
    assert response.status_code == 204


async def test_api_asr_transcribe_wav_only(
    client: httpx.AsyncClient, student: dict[str, str]
) -> None:
    """A WAV is transcribed; anything else is an ``ASR_INPUT_INVALID`` (ADR-0017)."""
    ok = await client.post(
        "/asr/transcribe",
        files={"audio": ("komut.wav", wav_bytes(), "audio/wav")},
        headers=student,
    )
    assert ok.status_code == 200, ok.text
    body = ok.json()
    assert body["text"]
    assert body["model_version"] == "asr-stub-1"
    assert body["transcript_id"]

    mp3 = await client.post(
        "/asr/transcribe",
        files={"audio": ("komut.mp3", b"ID3\x04\x00\x00\x00\x00\x00\x00mp3 verisi", "audio/mpeg")},
        headers=student,
    )
    assert mp3.status_code == 400
    assert mp3.json()["code"] == "ASR_INPUT_INVALID"
    assert mp3.json()["reason"] == "not_riff_wave"


async def test_api_asr_command_returns_compute_out(
    client: httpx.AsyncClient, app: FastAPI, student: dict[str, str]
) -> None:
    """Voice → transcript → NLU → CAS in one call, with ``nlu.source == "asr+rules"``."""
    app.state.registry.set_transcriber(FixedTranscriber())
    response = await client.post(
        "/asr/command",
        files={"audio": ("komut.wav", wav_bytes(), "audio/wav")},
        headers=student,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "result"
    assert body["result"]["latex"] == r"2 \cdot x"
    assert body["nlu"]["source"] == "asr+rules"
    assert body["transcript"] == "x karenin türevini al"
    assert body["confidence"] == pytest.approx(0.93)


async def test_api_asr_feedback_records_the_correction(
    client: httpx.AsyncClient, student: dict[str, str]
) -> None:
    """A corrected transcript is stored for İP-3 retraining."""
    transcribed = await client.post(
        "/asr/transcribe",
        files={"audio": ("komut.wav", wav_bytes(), "audio/wav")},
        headers=student,
    )
    response = await client.post(
        "/asr/feedback",
        json={
            "transcript_id": transcribed.json()["transcript_id"],
            "corrected_text": "x karenin türevini al",
        },
        headers=student,
    )
    assert response.status_code == 204


async def test_api_upload_rejects_an_empty_part(
    client: httpx.AsyncClient, student: dict[str, str]
) -> None:
    """An empty upload is an ``UPLOAD_INVALID``, not a crash inside a decoder."""
    response = await client.post(
        "/asr/transcribe", files={"audio": ("bos.wav", b"", "audio/wav")}, headers=student
    )
    assert response.status_code == 400
    assert response.json()["code"] == "UPLOAD_INVALID"


async def test_api_ocr_rejects_an_unsupported_content_type(
    client: httpx.AsyncClient, student: dict[str, str]
) -> None:
    """``POST /ocr/recognize`` accepts JSON or multipart and says so when it gets neither."""
    response = await client.post(
        "/ocr/recognize",
        content=b"x",
        headers={**student, "Content-Type": "text/plain"},
    )
    assert response.status_code == 400
    assert response.json()["reason"] == "unsupported_content_type"
