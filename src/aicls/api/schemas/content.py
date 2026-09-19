"""Book, contribution, feedback and telemetry payloads (api.md §3.7–§3.10)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from aicls.api.schemas.common import ApiModel, RequestModel
from aicls.api.schemas.compute import ContentUnitOut

#: Longest free-text search query (docs/guvenlik-gizlilik.md §5).
MAX_QUERY_CHARS = 200
#: Longest feedback text.
MAX_FEEDBACK_CHARS = 4000
#: Number of answers in a SUS questionnaire.
SUS_ITEMS = 10


class HitOut(ApiModel):
    """One textbook search hit (api.md §3.7)."""

    unit: ContentUnitOut
    score: float
    matched_alias: str | None = None
    stage: str = "alias"


class ResolveIn(RequestModel):
    """``POST /book/resolve`` body — a Turkish fetch command."""

    text: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)


class ChapterOut(ApiModel):
    """One chapter of the book tree."""

    id: str
    n_units: int = 0


class LoadReportOut(ApiModel):
    """``POST /admin/book/load`` response (İP-8 100 % tagging criterion)."""

    units: int
    triples: int
    aliases: int
    tagged_ratio: float


class ConsentTextOut(ApiModel):
    """``GET /contribute/consent/current`` response."""

    version: str
    text_tr: str
    sha256: str
    effective_from: str = ""


class DemographicsIn(RequestModel):
    """Band-only demographics (KVKK data minimisation; unknown fields are rejected)."""

    age_band: Literal["18-24", "25-34", "35-44", "45+"] = "25-34"
    gender: Literal["f", "m", "na"] = "na"
    university: Literal["mus", "mersin", "amasya", "other"] = "other"
    accent_region: str | None = Field(default=None, max_length=40)


class ConsentIn(RequestModel):
    """``POST /contribute/consent`` body."""

    version: str = Field(min_length=1, max_length=16)
    accepted: Literal[True]
    demographics: DemographicsIn = DemographicsIn()


class SpeakerOut(ApiModel):
    """``POST /contribute/consent`` response: the pseudonym and a participant token."""

    speaker_pseudonym: str
    token: str


class PromptOut(ApiModel):
    """One sentence to read aloud."""

    id: str
    text_tr: str
    scenario_tag: str = "genel"
    difficulty: int = 1


class QcCheckOut(ApiModel):
    """One automatic quality measurement and its verdict."""

    name: str
    value: float
    threshold: float
    passed: bool


class QcOut(ApiModel):
    """The quality-control verdict of one recording."""

    passed: bool
    checks: list[QcCheckOut] = Field(default_factory=list)
    reasons_tr: list[str] = Field(default_factory=list)


class RecordingOut(ApiModel):
    """``POST /contribute/recordings`` response — always HTTP 200 (api.md §2)."""

    id: str | None = None
    accepted: bool
    qc: QcOut
    duration_s: float = 0.0


class ProgressOut(ApiModel):
    """``GET /contribute/me/progress`` response."""

    n_accepted: int = 0
    seconds_accepted: float = 0.0
    target_seconds: float = 0.0


class ContribStatsOut(ApiModel):
    """``GET /contribute/stats`` response (the İP-3 dashboard)."""

    hours_accepted: float = 0.0
    utterances: int = 0
    speakers: int = 0
    by_university: dict[str, int] = Field(default_factory=dict)
    by_gender: dict[str, int] = Field(default_factory=dict)
    by_age_band: dict[str, int] = Field(default_factory=dict)
    qc_pass_rate: float = 0.0


class FeedbackIn(RequestModel):
    """``POST /feedback`` body (İP-9)."""

    category: Literal["critical", "usability", "suggestion"]
    text: str = Field(min_length=1, max_length=MAX_FEEDBACK_CHARS)
    page: str = Field(default="", max_length=120)
    screenshot: str | None = Field(default=None, max_length=26)


class FeedbackOut(ApiModel):
    """One feedback entry with its triage state."""

    id: str
    category: str
    text: str
    page: str = ""
    status: str = "open"
    created_at: str = ""


class FeedbackPatch(RequestModel):
    """``PATCH /admin/feedback/{id}`` body."""

    status: Literal["open", "triaged", "fixed", "deferred"]


class SusIn(RequestModel):
    """``POST /feedback/sus`` body — ten answers in 1..5."""

    answers: list[int] = Field(min_length=SUS_ITEMS, max_length=SUS_ITEMS)
    cohort: str = Field(default="pilot", max_length=40)
    collected_week: int | None = Field(default=None, ge=1, le=520)


class SusOut(ApiModel):
    """``POST /feedback/sus`` response."""

    score: float


class LatencySampleIn(RequestModel):
    """``POST /telemetry/latency`` body (ADR-0014, client-measured end-to-end time)."""

    operation_id: str | None = Field(default=None, max_length=26)
    modality: Literal["text", "latex", "ink", "audio"] = "text"
    client_submit_ts: float = Field(ge=0)
    client_render_ts: float = Field(ge=0)
    render_ms: float = Field(default=0.0, ge=0)


class WithdrawOut(ApiModel):
    """``POST /contribute/withdraw`` summary (returned in the audit log, not to the client)."""

    speaker_pseudonym: str
    recordings_deleted: int = 0
    files_deleted: int = 0
    consent_kept: bool = True


def unit_payload(unit: Any) -> ContentUnitOut:
    """Convert an :class:`aicls.book.schema.ContentUnit` into its API shape."""
    from aicls.api.schemas.compute import AssetOut, RenderedOut

    return ContentUnitOut(
        id=unit.id,
        type=str(unit.type.value),
        chapter=unit.chapter,
        number=unit.number,
        title_tr=unit.title_tr,
        aliases=list(unit.aliases),
        jsonld=dict(unit.jsonld),
        rendered=RenderedOut(
            body_html=unit.body_md,
            latex_blocks=list(unit.latex_blocks),
            assets=[
                AssetOut(
                    id=ref.id,
                    url=f"/api/v1/book/assets/{ref.id}",
                    alt_tr=ref.alt_tr,
                    caption_tr=ref.caption_tr,
                )
                for ref in unit.assets
            ],
        ),
    )


__all__ = [
    "MAX_FEEDBACK_CHARS",
    "MAX_QUERY_CHARS",
    "SUS_ITEMS",
    "ChapterOut",
    "ConsentIn",
    "ConsentTextOut",
    "ContribStatsOut",
    "DemographicsIn",
    "FeedbackIn",
    "FeedbackOut",
    "FeedbackPatch",
    "HitOut",
    "LatencySampleIn",
    "LoadReportOut",
    "ProgressOut",
    "PromptOut",
    "QcCheckOut",
    "QcOut",
    "RecordingOut",
    "ResolveIn",
    "SpeakerOut",
    "SusIn",
    "SusOut",
    "WithdrawOut",
    "unit_payload",
]
