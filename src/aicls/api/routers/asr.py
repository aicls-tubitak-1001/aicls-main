"""Speech endpoints: transcript only, or transcript → NLU → CAS (İP-3, api.md §3.6).

ADR-0017 fixes the wire format: 16 kHz mono PCM16 WAV, at most 30 s and 10 MB, no container
conversion on the server (no ffmpeg). 44.1/48 kHz uploads are resampled in numpy by
:func:`aicls.asr.read_wav_pcm16`; anything else is an ``ASR_INPUT_INVALID`` (400).

``POST /asr/command`` returns the same ``ComputeOut`` the text path returns, with
``nlu.source = "asr+rules"`` and the transcript attached, because the voice path always asks the
user to confirm before the answer counts (api.md §3.6).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile, status

from aicls.api.deps import CurrentUser, DbSession, Pipeline, Timer
from aicls.api.responses import compute_out, transcribe_out
from aicls.api.schemas import AsrFeedbackIn, ComputeOut, TranscribeOut
from aicls.api.uploads import check_wav_bytes, read_upload
from aicls.asr.audio_io import MAX_WAV_BYTES, read_wav_pcm16
from aicls.core import messages_tr
from aicls.core.errors import NotFoundError
from aicls.db.repositories.compute import RecognitionRepository
from aicls.nlu.types import ParseContext

router = APIRouter(prefix="/asr", tags=["asr"])


@router.post("/transcribe", response_model=TranscribeOut)
async def transcribe(
    user: CurrentUser,
    session: DbSession,
    pipeline: Pipeline,
    timer: Timer,
    audio: Annotated[UploadFile, File(description="PCM16 WAV, mono, <= 30 s, <= 10 MB")],
) -> TranscribeOut:
    """Transcribe one recording and return the Turkish text."""
    payload = await read_upload(audio, max_bytes=MAX_WAV_BYTES, field="audio")
    check_wav_bytes(payload)
    with timer.stage("decode"):
        waveform = read_wav_pcm16(payload)
    outcome = pipeline.run_audio(
        waveform, user, None, None, session=session, timer=timer, to_command=False
    )
    return transcribe_out(outcome, timer)


@router.post("/command", response_model=ComputeOut)
async def command(
    user: CurrentUser,
    session: DbSession,
    pipeline: Pipeline,
    timer: Timer,
    audio: Annotated[UploadFile, File(description="PCM16 WAV, mono, <= 30 s, <= 10 MB")],
    workspace_id: Annotated[str | None, Form(max_length=26)] = None,
    last_latex: Annotated[str | None, Form(max_length=2048)] = None,
    active_var: Annotated[str | None, Form(max_length=24)] = None,
) -> ComputeOut:
    """Transcribe, understand and compute in one call (the voice path of İP-4)."""
    payload = await read_upload(audio, max_bytes=MAX_WAV_BYTES, field="audio")
    check_wav_bytes(payload)
    with timer.stage("decode"):
        waveform = read_wav_pcm16(payload)
    context = ParseContext(last_latex=last_latex, active_var=active_var, modality="voice")
    outcome = pipeline.run_audio(
        waveform, user, context, workspace_id, session=session, timer=timer
    )
    return compute_out(outcome, timer)


@router.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
def asr_feedback(body: AsrFeedbackIn, user: CurrentUser, session: DbSession) -> None:
    """Record the user's corrected transcript (İP-3 correction loop)."""
    from aicls.db.models import Transcript

    row = session.get(Transcript, body.transcript_id)
    if row is None or (row.user_id is not None and row.user_id != user.id):
        raise NotFoundError(
            messages_tr.MSG_NOT_FOUND,
            details={"table": "transcripts", "id": body.transcript_id},
        )
    RecognitionRepository(session).set_transcript_correction(row.id, body.corrected_text)


__all__ = ["router"]
