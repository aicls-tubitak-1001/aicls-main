"""Compute, handwriting and speech payloads (api.md §3.4, §3.5, §3.6).

``ComputeOut`` is the single response shape of the whole command path: the same document
answers a computed result, a clarification question, a textbook lookup and a client-side UI
action, discriminated by ``kind``. That is what lets the web client render one component tree
regardless of whether the user typed, wrote or spoke.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from aicls.api.pipeline import MAX_TEXT_CHARS
from aicls.api.schemas.common import (
    ApiModel,
    ComputeKindName,
    NluSourceName,
    RequestModel,
    ResultKindName,
    TimingsOut,
)
from aicls.mathengine.models import MAX_INPUT_CHARS


class ContextIn(RequestModel):
    """``ComputeTextIn.context`` — what the session already has on the board."""

    last_latex: str | None = Field(default=None, max_length=MAX_INPUT_CHARS)
    active_var: str | None = Field(default=None, max_length=24)


class ComputeTextIn(RequestModel):
    """``POST /compute/text`` body."""

    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    workspace_id: str | None = Field(default=None, max_length=26)
    context: ContextIn | None = None
    confirm_intent: str | None = Field(default=None, max_length=64)


class ComputeLatexIn(RequestModel):
    """``POST /compute/latex`` body: a typed CAS command, validated by the math engine."""

    command: dict[str, Any]
    workspace_id: str | None = Field(default=None, max_length=26)


class ParseIn(RequestModel):
    """``POST /compute/parse`` body."""

    latex: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)


class ParseOut(ApiModel):
    """``POST /compute/parse`` response — a diagnostic, never a computation."""

    ok: bool
    ast: dict[str, Any] | None = None
    srepr: str | None = None
    canonical_latex: str | None = None
    repairs: list[str] = Field(default_factory=list)
    error: dict[str, Any] | None = None


class StepOut(ApiModel):
    """One narrated solution step."""

    title_tr: str
    latex: str
    rule: str = ""
    detail_tr: str = ""


class ResultOut(ApiModel):
    """``ComputeOut.result`` — what SymPy computed."""

    kind: ResultKindName
    latex: str
    plain: str
    srepr: str = ""
    steps: list[StepOut] = Field(default_factory=list)
    plot_url: str | None = None
    cache_hit: bool = False


class ClarificationCandidateOut(ApiModel):
    """One option of a clarification question."""

    intent: str
    command: dict[str, Any] | None = None
    latex_preview: str = ""
    score: float = 0.0
    label_tr: str = ""


class ClarificationOut(ApiModel):
    """``ComputeOut.clarification`` — the Turkish question and its candidates."""

    question_tr: str
    missing_slot: str | None = None
    candidates: list[ClarificationCandidateOut] = Field(default_factory=list)


class NluOut(ApiModel):
    """``ComputeOut.nlu`` — the "how I understood you" panel."""

    intent: str
    score: float = 0.0
    trace: list[str] = Field(default_factory=list)
    source: NluSourceName = "rules"


class AssetOut(ApiModel):
    """A figure attached to a textbook unit."""

    id: str
    url: str
    alt_tr: str = ""
    caption_tr: str = ""


class RenderedOut(ApiModel):
    """Ready-to-render form of a textbook unit."""

    body_html: str = ""
    latex_blocks: list[str] = Field(default_factory=list)
    assets: list[AssetOut] = Field(default_factory=list)


class ContentUnitOut(ApiModel):
    """``ContentUnitOut`` — the JSON-LD document plus its rendered form (api.md §3.7)."""

    id: str
    type: str
    chapter: str = ""
    number: str = ""
    title_tr: str = ""
    aliases: list[str] = Field(default_factory=list)
    jsonld: dict[str, Any] = Field(default_factory=dict)
    rendered: RenderedOut = RenderedOut()


class ComputeOut(ApiModel):
    """The one response shape of the command path (api.md §3.4)."""

    kind: ComputeKindName
    operation_id: str | None = None
    input_latex: str | None = None
    command: dict[str, Any] | None = None
    result: ResultOut | None = None
    clarification: ClarificationOut | None = None
    book_units: list[ContentUnitOut] = Field(default_factory=list)
    nlu: NluOut | None = None
    timings: TimingsOut = TimingsOut()
    model_versions: dict[str, str | None] = Field(default_factory=dict)
    transcript: str | None = None
    confidence: float | None = None


class RecognizeOut(ApiModel):
    """``POST /ocr/recognize`` response (api.md §3.5)."""

    recognition_id: str | None = None
    latex: str
    confidence: float
    alternatives: list[tuple[str, float]] = Field(default_factory=list)
    timings: TimingsOut = TimingsOut()
    model_version: str | None = None


class OcrFeedbackIn(RequestModel):
    """``POST /ocr/feedback`` body (İP-3 correction loop)."""

    recognition_id: str = Field(min_length=1, max_length=26)
    corrected_latex: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)


class TranscribeOut(ApiModel):
    """``POST /asr/transcribe`` response (api.md §3.6)."""

    transcript_id: str | None = None
    text: str
    confidence: float
    timings: TimingsOut = TimingsOut()
    model_version: str | None = None


class AsrFeedbackIn(RequestModel):
    """``POST /asr/feedback`` body."""

    transcript_id: str = Field(min_length=1, max_length=26)
    corrected_text: str = Field(min_length=1, max_length=500)


class InkIn(RequestModel):
    """``POST /ocr/recognize`` JSON body (Canvas JSON, ADR-0013)."""

    ink: dict[str, Any]
    workspace_id: str | None = Field(default=None, max_length=26)


__all__ = [
    "AsrFeedbackIn",
    "AssetOut",
    "ClarificationCandidateOut",
    "ClarificationOut",
    "ComputeLatexIn",
    "ComputeOut",
    "ComputeTextIn",
    "ContentUnitOut",
    "ContextIn",
    "InkIn",
    "NluOut",
    "OcrFeedbackIn",
    "ParseIn",
    "ParseOut",
    "RecognizeOut",
    "RenderedOut",
    "ResultOut",
    "StepOut",
    "TranscribeOut",
]
