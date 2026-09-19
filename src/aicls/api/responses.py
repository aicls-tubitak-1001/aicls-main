"""Convert a :class:`~aicls.api.pipeline.PipelineResult` into the wire models (api.md §3.4).

Kept out of the routers so that the REST endpoints, the ``/ws/v1/audio`` ``command`` envelope and
the ASR command endpoint all serialise the *same* document — the web client parses one shape no
matter which transport delivered it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aicls.api.schemas import (
    ClarificationCandidateOut,
    ClarificationOut,
    ComputeOut,
    NluOut,
    RecognizeOut,
    ResultOut,
    StepOut,
    TranscribeOut,
    timings_out,
    unit_payload,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aicls.api.pipeline import PipelineResult
    from aicls.core.timing import StageTimer
    from aicls.mathengine.models import Result


def result_out(result: Result, plot_url: str | None) -> ResultOut:
    """Convert a CAS :class:`~aicls.mathengine.models.Result` (never the raw PNG bytes)."""
    return ResultOut(
        kind=result.kind,
        latex=result.latex,
        plain=result.plain,
        srepr=result.srepr,
        steps=[
            StepOut(
                title_tr=step.title_tr, latex=step.latex, rule=step.rule, detail_tr=step.detail_tr
            )
            for step in result.steps
        ],
        plot_url=plot_url,
        cache_hit=result.cache_hit,
    )


def compute_out(outcome: PipelineResult, timer: StageTimer | None = None) -> ComputeOut:
    """Build ``ComputeOut`` from a pipeline run.

    Args:
        outcome: What the pipeline produced.
        timer: The request timer, so ``timings.total`` is the whole request and not only the
            stages the pipeline measured.

    Returns:
        The response model; ``kind`` is one of ``result``, ``clarification``, ``book``, ``ui``.
    """
    kind = outcome.kind if outcome.kind in {"result", "clarification", "book", "ui"} else "result"
    total = timer.total_ms if timer is not None else outcome.timings.get("total", 0.0)
    clarification = None
    if outcome.clarification is not None:
        clarification = ClarificationOut(
            question_tr=outcome.clarification.question_tr,
            missing_slot=outcome.clarification.missing_slot,
            candidates=[
                ClarificationCandidateOut(
                    intent=candidate.intent,
                    command=candidate.command,
                    latex_preview=candidate.latex_preview,
                    score=candidate.score,
                    label_tr=candidate.label_tr,
                )
                for candidate in outcome.clarification.candidates
            ],
        )
    nlu = None
    if outcome.nlu is not None:
        nlu = NluOut(
            intent=outcome.nlu.intent,
            score=outcome.nlu.score,
            trace=list(outcome.nlu.trace),
            source=outcome.nlu.source,  # type: ignore[arg-type]  # constrained by NluSource
        )
    return ComputeOut(
        kind=kind,  # type: ignore[arg-type]  # narrowed above
        operation_id=outcome.operation_id,
        input_latex=outcome.input_latex,
        command=outcome.command,
        result=result_out(outcome.result, outcome.plot_url) if outcome.result else None,
        clarification=clarification,
        book_units=[unit_payload(unit) for unit in outcome.book_units],
        nlu=nlu,
        timings=timings_out(outcome.timings, total),
        model_versions=dict(outcome.model_versions),
        transcript=outcome.transcript.text if outcome.transcript is not None else None,
        confidence=(outcome.transcript.confidence if outcome.transcript is not None else None),
    )


def recognize_out(outcome: PipelineResult, timer: StageTimer | None = None) -> RecognizeOut:
    """Build ``RecognizeOut`` from a handwriting run."""
    recognition = outcome.recognition
    total = timer.total_ms if timer is not None else outcome.timings.get("total", 0.0)
    if recognition is None:  # pragma: no cover - run_ink always sets it
        return RecognizeOut(latex="", confidence=0.0, timings=timings_out(outcome.timings, total))
    return RecognizeOut(
        recognition_id=outcome.operation_id,
        latex=recognition.latex,
        confidence=recognition.confidence,
        alternatives=[(text, score) for text, score in recognition.alternatives],
        timings=timings_out(outcome.timings, total),
        model_version=recognition.model_version,
    )


def transcribe_out(outcome: PipelineResult, timer: StageTimer | None = None) -> TranscribeOut:
    """Build ``TranscribeOut`` from a speech run."""
    transcript = outcome.transcript
    total = timer.total_ms if timer is not None else outcome.timings.get("total", 0.0)
    if transcript is None:  # pragma: no cover - run_audio always sets it
        return TranscribeOut(text="", confidence=0.0, timings=timings_out(outcome.timings, total))
    return TranscribeOut(
        transcript_id=outcome.operation_id,
        text=transcript.text,
        confidence=transcript.confidence,
        timings=timings_out(outcome.timings, total),
        model_version=transcript.model_version,
    )


__all__ = ["compute_out", "recognize_out", "result_out", "transcribe_out"]
