"""The "Model Gelişimine Katkıda Bulun" flow: consent, prompts, recordings, withdrawal.

The order of the gates is the whole point of this module and is enforced in exactly one place:

1. **Ethics approval** — no ``AICLS_ETHICS_APPROVAL_REF``, no collection at all (503). Withdrawal
   is deliberately *not* gated: a participant must always be able to take their data back.
2. **Consent** — the speaker must exist, must not have withdrawn, and must have accepted the
   consent version that is currently in force (403 ``CONSENT_REQUIRED``).
3. **Input validation** — the prompt must exist and be active, and the upload must be a PCM16
   WAV the server can decode (``aicls.asr.audio_io``).
4. **Quality control** — :func:`aicls.asr.qc.run_qc` decides; a rejected upload leaves no file
   and no row behind (docs/mimari/api.md §3.8).

Only when all four pass is a file written, and the database row is inserted *first* so that a
failed write is rolled back with the transaction rather than leaving an orphaned recording.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import ipaddress
from pathlib import Path
import re
from typing import Any

from aicls.asr.audio_io import read_wav_pcm16, write_wav_pcm16
from aicls.asr.qc import QcReport, QcThresholds, run_qc
from aicls.collect.consent import current_consent
from aicls.collect.errors import ConsentRequiredError, ContributionGateError
from aicls.collect.models import Demographics, RecordingDecision, WithdrawReport
from aicls.collect.storage import delete_speaker_files, new_recording_path, write_bytes_atomic
from aicls.core.errors import InputValidationError, NotFoundError
from aicls.core.hashing import sha256_bytes
from aicls.core.logging import get_logger
from aicls.core.messages_tr import msg
from aicls.core.settings import get_settings
from aicls.db.base import utcnow
from aicls.db.models import ConsentVersion, PromptSentence, Speaker
from aicls.db.repositories.audio import RecordingRepository

#: Largest number of sentences one ``GET /contribute/prompts/next`` may ask for.
MAX_PROMPTS = 50

#: How many candidates are fetched per requested prompt before scenario balancing.
_POOL_FACTOR = 4

#: Hard ceiling on the candidate pool (matches ``aicls.db`` page size).
_MAX_POOL = 200

#: Longest ethics approval reference the ``consent_versions`` column can hold.
MAX_ETHICS_REF_LENGTH = 80

#: Accepted shape of a stored hash (hex digest, possibly truncated).
_DIGEST_RE = re.compile(r"^[0-9a-fA-F]{16,80}$")

_log = get_logger(__name__)


# ---------------------------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------------------------
def contribution_enabled() -> bool:
    """True when an ethics approval reference is recorded (docs/ml/veri-yonetimi.md §4)."""
    return bool(get_settings().ethics_approval_ref.strip())


def require_contribution_enabled() -> None:
    """Refuse the collection flow while no ethics approval is recorded.

    Raises:
        ContributionGateError: If ``AICLS_ETHICS_APPROVAL_REF`` is empty (503 ``GATE_BLOCKED``).
        InputValidationError: If the reference is longer than the database column
            (:data:`MAX_ETHICS_REF_LENGTH`).
    """
    reference = get_settings().ethics_approval_ref.strip()
    if not reference:
        raise ContributionGateError(details={"setting": "AICLS_ETHICS_APPROVAL_REF"})
    if len(reference) > MAX_ETHICS_REF_LENGTH:
        # The reference is copied into consent_versions.ethics_approval_ref (VARCHAR(80));
        # a longer value would be truncated by SQLite and rejected by strict MySQL.
        raise InputValidationError(
            msg("VALIDATION_ERROR"),
            details={
                "field": "AICLS_ETHICS_APPROVAL_REF",
                "max_length": MAX_ETHICS_REF_LENGTH,
                "length": len(reference),
            },
        )


def hash_ip(value: str, *, salt: str | None = None) -> str | None:
    """Return a salted SHA-256 of a network address, or ``None`` for an empty value.

    The raw address never reaches the database (docs/ml/veri-yonetimi.md §4: "IP/tarayıcı —
    tuzlu hash"). The default salt is the server's JWT secret, so digests cannot be reversed with
    a rainbow table of the IPv4 space by anyone without that secret.

    Args:
        value: The address (or user-agent string) to hash.
        salt: Explicit salt; defaults to ``AICLS_JWT_SECRET``.
    """
    text = (value or "").strip()
    if not text:
        return None
    pepper = salt if salt is not None else get_settings().jwt_secret.get_secret_value()
    return hashlib.sha256(f"{pepper}:{text}".encode()).hexdigest()


def _check_digest(value: str | None, *, field: str) -> str | None:
    """Validate that ``value`` is a hash and not the personal datum it should have replaced.

    Raises:
        InputValidationError: If it is an IP address or not a hex digest.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        ipaddress.ip_address(text)
    except ValueError:
        pass
    else:
        raise InputValidationError(
            msg("VALIDATION_ERROR"), details={"field": field, "reason": "raw_ip_address"}
        )
    if not _DIGEST_RE.match(text):
        raise InputValidationError(
            msg("VALIDATION_ERROR"), details={"field": field, "reason": "not_a_digest"}
        )
    return text.lower()


# ---------------------------------------------------------------------------------------------
# lookups
# ---------------------------------------------------------------------------------------------
def _load_speaker(repo: RecordingRepository, speaker_id: str) -> Speaker:
    """Return the speaker addressed by ULID or pseudonym.

    Raises:
        NotFoundError: If no such speaker exists.
    """
    speaker = repo.get_speaker(speaker_id)
    if speaker is None:
        raise NotFoundError(msg("NOT_FOUND"), details={"table": "speakers", "id": speaker_id})
    return speaker


def _load_prompt(repo: RecordingRepository, prompt_id: str) -> PromptSentence:
    """Return an *active* prompt sentence by id.

    ``session.get`` is a primary-key lookup rather than a query, so this stays within the
    "repositories own the queries" rule (docs/mimari/veri-modeli.md §1 rule 8).

    Raises:
        NotFoundError: If the prompt does not exist or was retired.
    """
    prompt = repo.session.get(PromptSentence, prompt_id)
    if prompt is None or not prompt.is_active:
        raise NotFoundError(
            msg("NOT_FOUND"), details={"table": "prompt_sentences", "id": prompt_id}
        )
    return prompt


def require_consent(repo: RecordingRepository, speaker_id: str) -> Speaker:
    """Return the speaker only if they may contribute right now.

    A participant may contribute when they exist, have not withdrawn, and accepted the consent
    version currently in force. A new consent wording therefore requires a fresh acceptance
    before the next upload — that is what "sürümlü rıza" means in docs/ml/veri-yonetimi.md §4.

    Raises:
        ConsentRequiredError: In every other case (403; the answer is deliberately the same for
            an unknown pseudonym so the endpoint does not confirm who is registered).
    """
    speaker = repo.get_speaker(speaker_id)
    if speaker is None:
        raise ConsentRequiredError(
            msg("CONSENT_REQUIRED"), details={"reason": "no_consent_on_record"}
        )
    if speaker.withdrawn_at is not None:
        raise ConsentRequiredError(msg("CONSENT_REQUIRED"), details={"reason": "withdrawn"})
    consent = current_consent()
    stored = repo.get_consent_version(consent.version)
    if stored is None or stored.id != speaker.consent_version_id:
        accepted = repo.session.get(ConsentVersion, speaker.consent_version_id)
        raise ConsentRequiredError(
            msg("CONSENT_REQUIRED"),
            details={
                "reason": "consent_version_outdated",
                "expected_version": consent.version,
                "accepted_version": None if accepted is None else accepted.version,
            },
        )
    return speaker


def qc_document(report: QcReport) -> dict[str, Any]:
    """Render a quality report the way ``recordings.qc_json`` expects it.

    ``RecordingRepository.add`` lifts ``passed``, ``snr_db``, ``clipping_ratio`` and
    ``silence_ratio`` out of the **top level** of this mapping into their own columns, while
    :meth:`aicls.asr.qc.QcReport.as_dict` nests every measurement inside ``checks`` for the API
    response. Flattening the measurements next to the nested document is what keeps both
    contracts satisfied — without it the ``snr_db`` column would silently stay ``0.0`` and the
    QC dashboard would report a corpus with no signal-to-noise ratio at all
    (``tests/integration/test_collect_contracts.py``).
    """
    document: dict[str, Any] = report.as_dict()
    document.update({check.name: check.value for check in report.checks})
    return document


def _demographics(value: Demographics | Mapping[str, Any]) -> Demographics:
    """Accept either a :class:`Demographics` or the mapping an API schema dumps."""
    if isinstance(value, Demographics):
        return value
    return Demographics.model_validate(dict(value))


# ---------------------------------------------------------------------------------------------
# public flow
# ---------------------------------------------------------------------------------------------
def accept_consent(
    repo: RecordingRepository,
    *,
    user_id: str | None = None,
    version: str,
    demographics: Demographics | Mapping[str, Any],
    ip_hash: str | None = None,
    user_agent_hash: str | None = None,
) -> Speaker:
    """Record an acceptance of the current consent text and return the speaker.

    The consent text itself is stored once (``consent_versions``) and shared by every speaker who
    accepted it, so the exact wording and its digest remain available for a KVKK audit even after
    a later version is published.

    Args:
        repo: The voice data-collection repository.
        user_id: Optional link to a signed-in account, kept only for progress tracking and never
            exported. Accepting twice with the same account re-uses the existing pseudonym.
        version: The consent version the participant accepted; must be the current one.
        demographics: Band-only demographics (:class:`~aicls.collect.models.Demographics` or the
            equivalent mapping).
        ip_hash: Salted hash of the client address — see :func:`hash_ip`. A raw address is
            rejected.
        user_agent_hash: Salted hash of the user agent, same rule.

    Returns:
        The persisted, pseudonymous :class:`~aicls.db.models.Speaker` (``P00001`` and up).

    Raises:
        ContributionGateError: If no ethics approval is recorded.
        InputValidationError: If the version is not the current one, the demographics are not
            band-only, or a hash field carries a raw identifier.
    """
    require_contribution_enabled()
    settings = get_settings()
    consent = current_consent()
    if version != consent.version:
        raise InputValidationError(
            msg("VALIDATION_ERROR"),
            details={"field": "version", "expected": consent.version, "given": version},
        )
    bands = _demographics(demographics)
    checked_ip = _check_digest(ip_hash, field="ip_hash")
    checked_agent = _check_digest(user_agent_hash, field="user_agent_hash")

    stored = repo.upsert_consent_version(
        consent.version,
        consent.text_tr,
        sha256=consent.sha256,
        effective_from=consent.effective_from,
        ethics_approval_ref=settings.ethics_approval_ref,
    )

    if user_id:
        existing = repo.speaker_for_user(user_id)
        if existing is not None:
            if existing.withdrawn_at is not None:
                raise InputValidationError(
                    msg("VALIDATION_ERROR"),
                    details={"field": "user_id", "reason": "speaker_withdrawn"},
                )
            existing.consent_version_id = stored.id
            existing.consented_at = utcnow()
            repo.flush()
            return existing

    speaker = repo.create_speaker(
        stored.id,
        user_id=user_id,
        age_band=bands.age_band,
        gender=bands.gender,
        university=bands.university,
        accent_region=bands.accent_region,
        ip_hash=checked_ip,
        user_agent_hash=checked_agent,
        consented_at=utcnow(),
    )
    _log.info(
        "collect_consent_accepted",
        pseudonym=speaker.pseudonym,
        consent_version=consent.version,
        linked_account=bool(user_id),
    )
    return speaker


def _balance_by_scenario(candidates: Sequence[PromptSentence], n: int) -> list[PromptSentence]:
    """Round-robin the least-recorded candidates over their reading scenarios.

    ``candidates`` arrives ordered by "fewest recordings first"; grouping preserves that order
    inside each scenario and the groups themselves keep the order of their best candidate, so the
    first suggestion is always the globally least-recorded sentence while a single scenario can
    no longer fill the whole batch (docs/mimari/moduller.md: "en az kaydedilmiş, senaryo
    dengeli").
    """
    groups: dict[str, list[PromptSentence]] = {}
    for prompt in candidates:
        groups.setdefault(prompt.scenario_tag, []).append(prompt)
    balanced: list[PromptSentence] = []
    queues = list(groups.values())
    while len(balanced) < n and queues:
        for queue in list(queues):
            if len(balanced) >= n:
                break
            balanced.append(queue.pop(0))
            if not queue:
                queues.remove(queue)
    return balanced


def next_prompts(repo: RecordingRepository, speaker_id: str, n: int = 5) -> list[PromptSentence]:
    """Return the next ``n`` sentences this speaker should read.

    Balancing has two levels: the repository orders active, not-yet-read prompts by how many
    recordings exist for them, and :func:`_balance_by_scenario` then spreads the batch over the
    reading scenarios so a session never consists of five derivative commands.

    Args:
        repo: The voice data-collection repository.
        speaker_id: Speaker ULID or pseudonym.
        n: How many sentences to return (1..:data:`MAX_PROMPTS`).

    Raises:
        NotFoundError: If the speaker does not exist.
        ConsentRequiredError: If the speaker has withdrawn their consent.
        InputValidationError: If ``n`` is out of range.
    """
    if not 1 <= int(n) <= MAX_PROMPTS:
        raise InputValidationError(
            msg("VALIDATION_ERROR"), details={"field": "n", "value": n, "max": MAX_PROMPTS}
        )
    speaker = _load_speaker(repo, speaker_id)
    if speaker.withdrawn_at is not None:
        raise ConsentRequiredError(msg("CONSENT_REQUIRED"), details={"reason": "withdrawn"})
    pool = repo.next_prompts(speaker.id, min(int(n) * _POOL_FACTOR, _MAX_POOL))
    return _balance_by_scenario(pool, int(n))


def accept_recording(
    repo: RecordingRepository,
    *,
    speaker_id: str,
    prompt_id: str,
    wav: bytes,
    thresholds: QcThresholds | None = None,
    data_root: Path | None = None,
) -> RecordingDecision:
    """Apply the consent and quality gates to one upload and store it if it passes.

    A rejected upload is never written to disk and never becomes a row: the caller answers 200
    with ``accepted=false`` and the Turkish reasons, so the participant can simply try again
    (docs/mimari/api.md §3.8).

    Args:
        repo: The voice data-collection repository.
        speaker_id: Speaker ULID or pseudonym.
        prompt_id: The sentence that was read.
        wav: Raw uploaded bytes; must be PCM16 RIFF/WAVE (ADR-0017).
        thresholds: Quality bounds; :class:`~aicls.asr.qc.QcThresholds` defaults when omitted.
        data_root: Root to write under; defaults to ``AICLS_DATA_ROOT``.

    Returns:
        A :class:`~aicls.collect.models.RecordingDecision`.

    Raises:
        ContributionGateError: If no ethics approval is recorded.
        ConsentRequiredError: If the speaker may not contribute.
        NotFoundError: If the prompt does not exist or is inactive.
        AsrInputError: If the payload is not a decodable PCM16 WAV.
    """
    require_contribution_enabled()
    speaker = require_consent(repo, speaker_id)
    prompt = _load_prompt(repo, prompt_id)

    audio = read_wav_pcm16(wav)
    report = run_qc(audio, thresholds or QcThresholds())
    if not report.passed:
        _log.info(
            "collect_recording_rejected",
            pseudonym=speaker.pseudonym,
            failed=[check.name for check in report.failed_checks],
        )
        return RecordingDecision(accepted=False, qc=report, duration_s=audio.duration_s)

    payload = write_wav_pcm16(audio)
    digest = sha256_bytes(payload)
    path, rel_path = new_recording_path(speaker.pseudonym, data_root=data_root)
    recording = repo.add(
        speaker.id,
        prompt.id,
        rel_path,
        qc_document(report),
        audio.duration_s,
        digest,
        sample_rate=audio.sr,
        size_bytes=len(payload),
    )
    write_bytes_atomic(path, payload)
    _log.info(
        "collect_recording_accepted",
        pseudonym=speaker.pseudonym,
        recording_id=recording.id,
        duration_s=round(audio.duration_s, 3),
    )
    return RecordingDecision(
        accepted=True,
        qc=report,
        recording_id=recording.id,
        path=path,
        rel_path=rel_path,
        sha256=digest,
        duration_s=audio.duration_s,
    )


def withdraw(
    repo: RecordingRepository, speaker_id: str, *, data_root: Path | None = None
) -> WithdrawReport:
    """Withdraw a participant's consent: delete the data, keep the proof (KVKK).

    Deliberately **not** gated by the ethics approval reference — closing the collection tab must
    never block a withdrawal. The database side soft-deletes the recordings and clears the
    network hashes while keeping the speaker and consent rows (the evidence that consent was
    given and later withdrawn); this function removes the audio and the derived features from
    disk. Calling it twice is safe and preserves the original withdrawal timestamp.

    Args:
        repo: The voice data-collection repository.
        speaker_id: Speaker ULID or pseudonym (``P00012``).
        data_root: Root to clean; defaults to ``AICLS_DATA_ROOT``.

    Returns:
        A :class:`~aicls.collect.models.WithdrawReport`.

    Raises:
        NotFoundError: If no such speaker exists.
    """
    speaker = _load_speaker(repo, speaker_id)
    already_withdrawn = speaker.withdrawn_at is not None
    recordings = 0 if already_withdrawn else repo.mark_withdrawn(speaker.id)
    files, size = delete_speaker_files(speaker.pseudonym, data_root=data_root)
    _log.info(
        "collect_withdrawn",
        pseudonym=speaker.pseudonym,
        recordings=recordings,
        files=files,
        repeated=already_withdrawn,
    )
    return WithdrawReport(
        speaker_pseudonym=speaker.pseudonym,
        recordings_deleted=recordings,
        files_deleted=files,
        bytes_deleted=size,
        consent_kept=True,
        withdrawn_at=speaker.withdrawn_at,
    )


__all__ = [
    "MAX_ETHICS_REF_LENGTH",
    "MAX_PROMPTS",
    "accept_consent",
    "accept_recording",
    "contribution_enabled",
    "hash_ip",
    "next_prompts",
    "qc_document",
    "require_consent",
    "require_contribution_enabled",
    "withdraw",
]
