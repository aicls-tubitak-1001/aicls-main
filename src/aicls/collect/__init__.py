"""Voice data-collection flow: consent, prompts, recordings, withdrawal and export (İP-3).

``aicls.collect`` is the domain logic behind the "Model Gelişimine Katkıda Bulun" tab
(docs/mimari/moduller.md#aiclscollect, docs/ml/veri-yonetimi.md §4). It owns the rules that make
the corpus lawful and usable:

* :mod:`~aicls.collect.consent` — the versioned, hashed Turkish consent text shipped as a package
  resource; :func:`current_consent` is what ``GET /contribute/consent/current`` returns.
* :mod:`~aicls.collect.service` — :func:`accept_consent` (pseudonym ``P00001``, band-only
  demographics, salted hashes), :func:`next_prompts` (least-recorded, scenario-balanced),
  :func:`accept_recording` (consent gate → quality gate → file and row) and :func:`withdraw`
  (KVKK: the audio goes, the proof of consent stays).
* :mod:`~aicls.collect.manifest` — :func:`export_manifest`, which writes a manifest with no
  identity fields and a speaker-disjoint split that freezes previously assigned test speakers.
* :mod:`~aicls.collect.storage` — where a recording lives under ``AICLS_DATA_ROOT``.

The package depends only on :mod:`aicls.core`, :mod:`aicls.db` and :mod:`aicls.asr` (ADR-0002)
and never imports torch or transformers.

Typical use::

    from aicls.collect import accept_consent, accept_recording, current_consent, next_prompts

    speaker = accept_consent(
        repo,
        user_id=None,
        version=current_consent().version,
        demographics=Demographics(age_band="18-24", gender="f", university="mus"),
        ip_hash=hash_ip(request.client.host),
    )
    prompt = next_prompts(repo, speaker.id, n=1)[0]
    decision = accept_recording(repo, speaker_id=speaker.id, prompt_id=prompt.id, wav=payload)
"""

from __future__ import annotations

from aicls.collect.consent import (
    CONSENT_PACKAGE,
    CONSENT_RESOURCE,
    ConsentText,
    consent_texts,
    current_consent,
    get_consent_text,
)
from aicls.collect.errors import ConsentRequiredError, ContributionGateError
from aicls.collect.manifest import (
    DEFAULT_RATIOS,
    DEFAULT_SPLIT_SEED,
    MANIFEST_FIELDS,
    SOURCE_LABELS,
    SPLITS_FILENAME,
    SplitFile,
    assign_splits,
    export_manifest,
    load_splits,
    save_splits,
)
from aicls.collect.models import (
    MAX_ACCENT_REGION_LENGTH,
    SPLIT_NAMES,
    Demographics,
    ManifestReport,
    RecordingDecision,
    WithdrawReport,
)
from aicls.collect.service import (
    MAX_PROMPTS,
    accept_consent,
    accept_recording,
    contribution_enabled,
    hash_ip,
    next_prompts,
    require_consent,
    require_contribution_enabled,
    withdraw,
)
from aicls.collect.storage import (
    AUDIO_RAW_DIR,
    audio_raw_root,
    new_recording_path,
    speaker_audio_dir,
    speaker_feature_dir,
)

__all__ = [
    "AUDIO_RAW_DIR",
    "CONSENT_PACKAGE",
    "CONSENT_RESOURCE",
    "DEFAULT_RATIOS",
    "DEFAULT_SPLIT_SEED",
    "MANIFEST_FIELDS",
    "MAX_ACCENT_REGION_LENGTH",
    "MAX_PROMPTS",
    "SOURCE_LABELS",
    "SPLITS_FILENAME",
    "SPLIT_NAMES",
    "ConsentRequiredError",
    "ConsentText",
    "ContributionGateError",
    "Demographics",
    "ManifestReport",
    "RecordingDecision",
    "SplitFile",
    "WithdrawReport",
    "accept_consent",
    "accept_recording",
    "assign_splits",
    "audio_raw_root",
    "consent_texts",
    "contribution_enabled",
    "current_consent",
    "export_manifest",
    "get_consent_text",
    "hash_ip",
    "load_splits",
    "new_recording_path",
    "next_prompts",
    "require_consent",
    "require_contribution_enabled",
    "save_splits",
    "speaker_audio_dir",
    "speaker_feature_dir",
    "withdraw",
]
