"""The "Model Gelişimine Katkıda Bulun" tab (İP-3, api.md §3.8, profiles ``full`` and ``collect``).

Two rules from docs/guvenlik-gizlilik.md §2 are visible in the routing itself:

* **the ethics gate** — while ``AICLS_ETHICS_APPROVAL_REF`` is empty every endpoint of this
  router answers ``503 GATE_BLOCKED``. The one exception is ``POST /contribute/withdraw``:
  closing the tab must never prevent a participant from having their data deleted.
* **quality control is not an error** — a rejected recording is a ``200`` payload with
  ``accepted:false`` and the Turkish reasons, so the participant simply reads the sentence again
  (api.md §2). Nothing is written to disk and no row is created.
"""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Header, Query, Request, UploadFile, status

from aicls.api.deps import DbSession, OptionalUser, StaffUser
from aicls.api.participants import PARTICIPANT_TOKEN_HEADER, ParticipantRepository
from aicls.api.schemas import (
    ConsentIn,
    ConsentTextOut,
    ContribStatsOut,
    ProgressOut,
    PromptOut,
    QcCheckOut,
    QcOut,
    RecordingOut,
    SpeakerOut,
)
from aicls.api.security import hash_token
from aicls.api.uploads import check_wav_bytes, read_upload
from aicls.asr.audio_io import MAX_WAV_BYTES
from aicls.collect import (
    Demographics,
    accept_consent,
    accept_recording,
    current_consent,
    next_prompts,
    require_contribution_enabled,
    withdraw,
)
from aicls.collect.errors import ConsentRequiredError
from aicls.core import messages_tr
from aicls.core.errors import NotFoundError
from aicls.core.logging import get_logger
from aicls.db.models import Speaker
from aicls.db.repositories.audio import RecordingRepository
from aicls.db.repositories.registry import AuditRepository

#: Target contribution per participant, in seconds (İP-3: ~50 h over ~25 000 utterances).
TARGET_SECONDS = 1800.0

_logger = get_logger(__name__)


def contribution_gate() -> None:
    """Refuse every collection endpoint while no ethics approval is recorded.

    Raises:
        ContributionGateError: 503 ``GATE_BLOCKED`` when ``AICLS_ETHICS_APPROVAL_REF`` is empty.
    """
    require_contribution_enabled()


router = APIRouter(
    prefix="/contribute", tags=["contribute"], dependencies=[Depends(contribution_gate)]
)

#: Withdrawal is deliberately outside the gate (KVKK); mounted under the same prefix.
withdraw_router = APIRouter(prefix="/contribute", tags=["contribute"])


def _ip_hashes(request: Request) -> tuple[str | None, str | None]:
    """Salted hashes of the client address and user agent (raw values never stored)."""
    from aicls.collect.service import hash_ip

    host = request.client.host if request.client else ""
    agent = request.headers.get("user-agent", "")
    return (hash_ip(host) if host else None, hash_ip(agent) if agent else None)


def current_speaker(
    session: DbSession,
    user: OptionalUser,
    token: Annotated[str | None, Header(alias=PARTICIPANT_TOKEN_HEADER)] = None,
) -> Speaker:
    """Resolve the participant from the token header, or from the signed-in account.

    Raises:
        ConsentRequiredError: When neither identifies a speaker who accepted the consent. The
            same answer is given for an unknown token and for a signed-in user who never
            consented, so a token cannot be probed.
    """
    repo = ParticipantRepository(session)
    speaker = repo.by_token_hash(hash_token(token)) if token else None
    if speaker is None and user is not None:
        speaker = RecordingRepository(session).speaker_for_user(user.id)
    if speaker is None or speaker.withdrawn_at is not None:
        raise ConsentRequiredError(
            messages_tr.MSG_CONSENT_REQUIRED, details={"reason": "no_consent"}
        )
    return speaker


Participant = Annotated[Speaker, Depends(current_speaker)]


@router.get("/consent/current", response_model=ConsentTextOut)
def consent_current() -> ConsentTextOut:
    """The consent text a participant must accept (versioned and hashed)."""
    consent = current_consent()
    return ConsentTextOut(
        version=consent.version,
        text_tr=consent.text_tr,
        sha256=consent.sha256,
        effective_from=consent.effective_from.isoformat(),
    )


@router.post("/consent", response_model=SpeakerOut, status_code=status.HTTP_201_CREATED)
def post_consent(
    request: Request, body: ConsentIn, session: DbSession, user: OptionalUser
) -> SpeakerOut:
    """Record an acceptance and return the pseudonym plus a participant token."""
    ip_hash, agent_hash = _ip_hashes(request)
    speaker = accept_consent(
        RecordingRepository(session),
        user_id=user.id if user is not None else None,
        version=body.version,
        demographics=Demographics(**body.demographics.model_dump()),
        ip_hash=ip_hash,
        user_agent_hash=agent_hash,
    )
    token = secrets.token_urlsafe(24)
    ParticipantRepository(session).set_token(speaker, hash_token(token))
    AuditRepository(session).log(
        user.id if user is not None else None,
        "consent_accept",
        ("speakers", speaker.id),
        {"version": body.version},
        ip_hash=ip_hash or "",
    )
    return SpeakerOut(speaker_pseudonym=speaker.pseudonym, token=token)


@router.get("/prompts/next", response_model=list[PromptOut])
def prompts_next(
    speaker: Participant,
    session: DbSession,
    n: Annotated[int, Query(ge=1, le=20)] = 5,
) -> list[PromptOut]:
    """The next sentences this participant should read (least-recorded, scenario-balanced)."""
    rows = next_prompts(RecordingRepository(session), speaker.id, n)
    return [
        PromptOut(
            id=row.id,
            text_tr=row.text_tr,
            scenario_tag=row.scenario_tag,
            difficulty=row.difficulty,
        )
        for row in rows
    ]


@router.post("/recordings", response_model=RecordingOut)
async def post_recording(
    speaker: Participant,
    session: DbSession,
    audio: Annotated[UploadFile, File(description="PCM16 WAV, <= 30 s, <= 10 MB")],
    prompt_id: Annotated[str, Form(min_length=1, max_length=26)],
) -> RecordingOut:
    """Upload one utterance; the quality verdict is part of the 200 payload, never an error."""
    payload = await read_upload(audio, max_bytes=MAX_WAV_BYTES, field="audio")
    check_wav_bytes(payload)
    decision = accept_recording(
        RecordingRepository(session),
        speaker_id=speaker.id,
        prompt_id=prompt_id,
        wav=payload,
    )
    return RecordingOut(
        id=decision.recording_id,
        accepted=decision.accepted,
        qc=QcOut(
            passed=decision.qc.passed,
            checks=[QcCheckOut(**check.as_dict()) for check in decision.qc.checks],
            reasons_tr=list(decision.qc.reasons_tr),
        ),
        duration_s=round(decision.duration_s, 3),
    )


@router.get("/recordings/{recording_id}", response_model=RecordingOut)
def get_recording(recording_id: str, speaker: Participant, session: DbSession) -> RecordingOut:
    """One of the participant's own recordings."""
    row = ParticipantRepository(session).owned_recording(recording_id, speaker.id)
    if row is None:
        raise NotFoundError(
            messages_tr.MSG_NOT_FOUND, details={"table": "recordings", "id": recording_id}
        )
    document = dict(row.qc_json or {})
    return RecordingOut(
        id=row.id,
        accepted=row.qc_passed,
        qc=QcOut(
            passed=bool(document.get("passed", row.qc_passed)),
            checks=[QcCheckOut(**check) for check in document.get("checks", [])],
            reasons_tr=list(document.get("reasons_tr", [])),
        ),
        duration_s=round(row.duration_s, 3),
    )


@router.delete("/recordings/{recording_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_recording(recording_id: str, speaker: Participant, session: DbSession) -> None:
    """Soft-delete one of the participant's own recordings."""
    if not ParticipantRepository(session).soft_delete(recording_id, speaker.id):
        raise NotFoundError(
            messages_tr.MSG_NOT_FOUND, details={"table": "recordings", "id": recording_id}
        )


@router.get("/me/progress", response_model=ProgressOut)
def my_progress(speaker: Participant, session: DbSession) -> ProgressOut:
    """How much this participant has contributed so far."""
    progress = ParticipantRepository(session).progress(speaker.id)
    return ProgressOut(
        n_accepted=progress.n_accepted,
        seconds_accepted=round(progress.seconds_accepted, 3),
        target_seconds=TARGET_SECONDS,
    )


@router.get("/stats", response_model=ContribStatsOut)
def contribution_stats(_user: StaffUser, session: DbSession) -> ContribStatsOut:
    """The İP-3 collection dashboard (instructor/admin)."""
    stats = RecordingRepository(session).stats()
    return ContribStatsOut(
        hours_accepted=round(stats.hours_accepted, 4),
        utterances=stats.utterances,
        speakers=stats.speakers,
        by_university=dict(stats.by_university),
        by_gender=dict(stats.by_gender),
        by_age_band=dict(stats.by_age_band),
        qc_pass_rate=round(stats.qc_pass_rate, 4),
    )


@withdraw_router.post("/withdraw", status_code=status.HTTP_204_NO_CONTENT)
def withdraw_consent(speaker: Participant, session: DbSession) -> None:
    """KVKK withdrawal: the audio goes, the proof of consent stays. Never gated."""
    report = withdraw(RecordingRepository(session), speaker.id)
    AuditRepository(session).log(
        None,
        "consent_withdraw",
        ("speakers", speaker.id),
        {"recordings_deleted": report.recordings_deleted, "files": report.files_deleted},
    )
    _logger.info("api.contribute.withdrawn", pseudonym=speaker.pseudonym)


__all__ = ["TARGET_SECONDS", "contribution_gate", "current_speaker", "router", "withdraw_router"]
