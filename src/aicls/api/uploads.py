"""Upload guards: size, magic bytes and format, before a model touches anything.

docs/guvenlik-gizlilik.md §6 fixes the limits; the point of this module is that they are checked
*first*, on the raw bytes, so a PNG renamed to ``.wav`` (or a 40 MB "image") is refused before
Pillow, ``wave`` or a recogniser sees it. The Turkish text always comes from the owning package's
error type, so the same upload produces the same message whether it arrived over REST or over a
WebSocket.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aicls.asr.audio_io import MAX_WAV_BYTES
from aicls.asr.errors import invalid_audio
from aicls.core import messages_tr
from aicls.core.errors import AiclsError, ErrorCode
from aicls.ocr.types import MAX_IMAGE_BYTES, OcrInputError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from starlette.datastructures import UploadFile

#: PNG signature (docs/guvenlik-gizlilik.md §6).
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
#: JPEG SOI marker.
JPEG_MAGIC = b"\xff\xd8\xff"
#: RIFF container marker and its WAVE form tag.
RIFF_MAGIC = b"RIFF"
WAVE_MAGIC = b"WAVE"

#: Largest accepted Canvas/InkML JSON payload (api.md §1).
MAX_INK_BYTES = 1024 * 1024

#: Bytes read per chunk while an upload is being size-checked.
_CHUNK = 64 * 1024

#: Content types accepted for a handwriting raster.
IMAGE_CONTENT_TYPES: frozenset[str] = frozenset({"image/png", "image/jpeg", "image/jpg"})

#: Content types accepted for a recording (ADR-0017: WAV only, no container conversion).
AUDIO_CONTENT_TYPES: frozenset[str] = frozenset(
    {"audio/wav", "audio/wave", "audio/x-wav", "application/octet-stream", ""}
)


def upload_too_large(size: int, limit: int, field: str) -> AiclsError:
    """The ``UPLOAD_INVALID`` (400) error for an oversized payload."""
    return AiclsError(
        ErrorCode.UPLOAD_INVALID,
        messages_tr.MSG_UPLOAD_INVALID,
        {"reason": "too_large", "field": field, "size": size, "limit": limit},
    )


async def read_upload(file: UploadFile, *, max_bytes: int, field: str = "file") -> bytes:
    """Read an upload, refusing it as soon as it passes ``max_bytes``.

    Args:
        file: The multipart part.
        max_bytes: Hard ceiling in bytes.
        field: Field name reported in the problem document.

    Returns:
        The complete payload.

    Raises:
        aicls.core.errors.AiclsError: ``UPLOAD_INVALID`` when the payload is empty or too large.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise upload_too_large(total, max_bytes, field)
        chunks.append(chunk)
    if total == 0:
        raise AiclsError(
            ErrorCode.UPLOAD_INVALID,
            messages_tr.MSG_UPLOAD_INVALID,
            {"reason": "empty", "field": field},
        )
    return b"".join(chunks)


def check_image_bytes(data: bytes, *, max_bytes: int = MAX_IMAGE_BYTES) -> str:
    """Verify an uploaded raster's size and magic bytes.

    Args:
        data: The raw upload.
        max_bytes: Size ceiling (2 MB by default, api.md §1).

    Returns:
        ``"png"`` or ``"jpeg"``.

    Raises:
        OcrInputError: If the payload is too large or is not a PNG/JPEG.
    """
    if len(data) > max_bytes:
        raise OcrInputError(details={"reason": "too_large", "size": len(data), "limit": max_bytes})
    if data.startswith(PNG_MAGIC):
        return "png"
    if data.startswith(JPEG_MAGIC):
        return "jpeg"
    raise OcrInputError(details={"reason": "bad_magic", "expected": ["png", "jpeg"]})


def check_wav_bytes(data: bytes, *, max_bytes: int = MAX_WAV_BYTES) -> None:
    """Verify an uploaded recording's size and RIFF/WAVE container.

    Raises:
        AsrInputError: If the payload is too large or is not a RIFF/WAVE container.
    """
    if len(data) > max_bytes:
        raise invalid_audio("too_large", size=len(data), limit=max_bytes)
    if len(data) < 12 or not data.startswith(RIFF_MAGIC) or data[8:12] != WAVE_MAGIC:
        raise invalid_audio("not_riff_wave")


def check_ink_size(payload: object, *, max_bytes: int = MAX_INK_BYTES) -> None:
    """Refuse a Canvas JSON payload larger than :data:`MAX_INK_BYTES`.

    Raises:
        OcrInputError: If the serialised payload exceeds the limit.
    """
    import json

    try:
        size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError) as exc:  # pragma: no cover - a dict of JSON scalars
        raise OcrInputError(details={"reason": "ink_not_serialisable"}) from exc
    if size > max_bytes:
        raise OcrInputError(details={"reason": "too_large", "size": size, "limit": max_bytes})


__all__ = [
    "AUDIO_CONTENT_TYPES",
    "IMAGE_CONTENT_TYPES",
    "JPEG_MAGIC",
    "MAX_INK_BYTES",
    "PNG_MAGIC",
    "RIFF_MAGIC",
    "WAVE_MAGIC",
    "check_image_bytes",
    "check_ink_size",
    "check_wav_bytes",
    "read_upload",
    "upload_too_large",
]
