"""Handwriting recognition endpoints (İP-3, api.md §3.5).

``POST /ocr/recognize`` is one path with two bodies, exactly as api.md §3.5 and the web client
(``web/src/api/paths.ts``) describe it: the browser's Canvas JSON (the online path of ADR-0013,
what ``/ws/v1/ink`` streams) **or** a PNG/JPEG upload (the offline path). FastAPI cannot express
"either body" through parameter annotations, so the request is dispatched on its content type and
the two request bodies are declared for OpenAPI with ``openapi_extra``.

Both bodies land on the same :class:`~aicls.ocr.types.Recognizer` through the model registry, so
the online/offline decision stays a configuration key rather than a rewrite.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request, status
from starlette.datastructures import UploadFile

from aicls.api.deps import CurrentUser, DbSession, Pipeline, Timer
from aicls.api.responses import recognize_out
from aicls.api.schemas import OcrFeedbackIn, RecognizeOut
from aicls.api.uploads import check_image_bytes, check_ink_size, read_upload
from aicls.core import messages_tr
from aicls.core.errors import NotFoundError
from aicls.core.latex_safety import sanitize_latex
from aicls.db.repositories.compute import RecognitionRepository
from aicls.mathengine.models import MAX_INPUT_CHARS
from aicls.ocr.canvas import ink_from_canvas_json
from aicls.ocr.types import MAX_IMAGE_BYTES, Image, Ink, OcrInputError

router = APIRouter(prefix="/ocr", tags=["ocr"])

#: OpenAPI description of the two accepted request bodies of ``POST /ocr/recognize``.
RECOGNIZE_REQUEST_BODY: dict[str, Any] = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "required": ["ink"],
                    "properties": {
                        "ink": {"type": "object", "description": "Canvas JSON (ADR-0013)"},
                        "workspace_id": {"type": "string", "maxLength": 26},
                    },
                }
            },
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["image"],
                    "properties": {
                        "image": {
                            "type": "string",
                            "format": "binary",
                            "description": "PNG or JPEG, at most 2 MB",
                        },
                        "workspace_id": {"type": "string", "maxLength": 26},
                    },
                }
            },
        },
    }
}


def canvas_payload(body: dict[str, Any]) -> dict[str, Any]:
    """Return the Canvas JSON object, whether it arrived nested under ``ink`` or at top level."""
    inner = body.get("ink")
    return dict(inner) if isinstance(inner, dict) else dict(body)


async def _sample_from_request(request: Request, timer: Timer) -> Ink | Image:
    """Decode the request body into an :class:`Ink` (JSON) or an :class:`Image` (multipart)."""
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type == "application/json":
        raw = await request.body()
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise OcrInputError(details={"reason": "body_not_json"}) from exc
        if not isinstance(body, dict):
            raise OcrInputError(details={"reason": "payload_type"})
        payload = canvas_payload(body)
        check_ink_size(payload)
        with timer.stage("parse"):
            return ink_from_canvas_json(payload)
    if content_type == "multipart/form-data":
        form = await request.form()
        upload = form.get("image")
        if not isinstance(upload, UploadFile):
            raise OcrInputError(details={"reason": "image_part_missing"})
        data = await read_upload(upload, max_bytes=MAX_IMAGE_BYTES, field="image")
        check_image_bytes(data)
        with timer.stage("parse"):
            return Image.from_bytes(data)
    raise OcrInputError(
        details={
            "reason": "unsupported_content_type",
            "content_type": content_type,
            "accepted": ["application/json", "multipart/form-data"],
        }
    )


@router.post("/recognize", response_model=RecognizeOut, openapi_extra=RECOGNIZE_REQUEST_BODY)
async def recognize(
    request: Request,
    user: CurrentUser,
    session: DbSession,
    pipeline: Pipeline,
    timer: Timer,
) -> RecognizeOut:
    """Recognise one handwriting sample and return its LaTeX with alternatives."""
    sample = await _sample_from_request(request, timer)
    outcome = pipeline.run_ink(sample, user, session=session, timer=timer)
    return recognize_out(outcome, timer)


@router.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
def ocr_feedback(body: OcrFeedbackIn, user: CurrentUser, session: DbSession) -> None:
    """Record the user's corrected LaTeX for a recognition (İP-3 risk plan: extra samples)."""
    repo = RecognitionRepository(session)
    row = repo.get(body.recognition_id)
    if row is None or (row.user_id is not None and row.user_id != user.id):
        raise NotFoundError(
            messages_tr.MSG_NOT_FOUND,
            details={"table": "recognitions", "id": body.recognition_id},
        )
    repo.set_correction(row.id, sanitize_latex(body.corrected_latex, max_len=MAX_INPUT_CHARS))


__all__ = ["RECOGNIZE_REQUEST_BODY", "canvas_payload", "router"]
