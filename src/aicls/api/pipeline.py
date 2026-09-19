"""``PipelineService`` — the contractual pipeline behind every compute endpoint (İP-3, İP-4).

One object owns the four entry paths of the proposal's diagram::

    text   ──> aicls.nlu ──┐
    latex  ────────────────┤──> aicls.mathengine (SymPy, sandboxed) ──> LaTeX result
    ink    ──> aicls.ocr ──┤                                   └──> aicls.book (İP-8 lookups)
    audio  ──> aicls.asr ──┘

Three rules are implemented here rather than in the routers, because they are contract text and
must hold on every path:

1. **Only SymPy produces a result.** The NLU layer may name an intent and fill slots; a
   :class:`~aicls.nlu.types.Matched` outcome carrying a non-CAS command is routed to the book
   layer or back to the client, never guessed at.
2. **A question is not an error.** A :class:`~aicls.nlu.types.Clarification` becomes a 200
   payload with ``kind="clarification"`` (api.md §2, "Hata **olmayan** durumlar").
3. **Recognised but not computable is not a crash.** A construct that the grammar refuses
   (``\\forall``, ``\\binom`` — KNOWN_ISSUES item 2) reaches the answering endpoints as
   ``UNSUPPORTED_PROBLEM`` (422) carrying the frozen İP-4 sentence and the construct name, never
   as a 500. The diagnostic endpoint ``POST /compute/parse`` still reports the raw
   ``UNSUPPORTED_CONSTRUCT`` so the live preview can be precise.

Every stage is timed through the caller's :class:`~aicls.core.timing.StageTimer`, which is the
same object the ``Server-Timing`` header and ``operations.timings`` are built from (ADR-0014).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aicls.api.nlu_bridge import command_for_intent, command_payload, primary_latex
from aicls.core import messages_tr
from aicls.core.errors import AiclsError, ErrorCode, NotFoundError
from aicls.core.hashing import sha256_bytes
from aicls.core.logging import get_logger
from aicls.core.paths import ensure_dir
from aicls.core.settings import Settings
from aicls.core.timing import StageTimer
from aicls.db.enums import AssetKind, InputKind, NluSource
from aicls.db.repositories.compute import (
    AssetRepository,
    OperationRepository,
    RecognitionRepository,
)
from aicls.db.repositories.identity import WorkspaceRepository
from aicls.mathengine import CAS_COMMAND_TYPES, MathEngine, Result
from aicls.mathengine.errors import UnsupportedProblem
from aicls.nlu import (
    BookFetch,
    Clarification,
    Matched,
    NoMatch,
    ParseContext,
    UiAction,
    parse_command,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlalchemy.orm import Session

    from aicls.api.model_registry import ModelRegistry
    from aicls.asr.types import Audio, Transcript
    from aicls.book.schema import ContentUnit
    from aicls.db.models import User
    from aicls.mathengine.models import CasCommand
    from aicls.nlu.types import Command
    from aicls.ocr.types import Image, Ink, Recognition

#: Relative directory (under ``AICLS_DATA_ROOT``) plots are written to.
PLOT_DIR = "plots"

#: Longest text command accepted (docs/guvenlik-gizlilik.md §6).
MAX_TEXT_CHARS = 500

#: Number of book units returned for a ``book_fetch`` command.
BOOK_HITS = 3

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ClarificationCandidate:
    """One option offered by a clarification question (api.md §3.4)."""

    intent: str
    score: float
    command: dict[str, Any] | None = None
    latex_preview: str = ""
    label_tr: str = ""


@dataclass(frozen=True, slots=True)
class ClarificationPayload:
    """``ComputeOut.clarification`` — what to ask and what the answers would mean."""

    question_tr: str
    missing_slot: str | None = None
    candidates: tuple[ClarificationCandidate, ...] = ()


@dataclass(frozen=True, slots=True)
class NluTrace:
    """``ComputeOut.nlu`` — how the text was understood."""

    intent: str
    score: float
    trace: tuple[str, ...] = ()
    source: str = "rules"


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Everything one pipeline run produced.

    Attributes:
        kind: ``result`` | ``clarification`` | ``book`` | ``ui`` | ``recognition`` |
            ``transcript``. The first four are the ``ComputeOut.kind`` values of api.md §3.4;
            the last two are what the OCR and ASR endpoints return.
        input_latex: LaTeX the command was built from.
        command: The typed command as a JSON-safe mapping.
        result: The SymPy result, when one was computed.
        clarification: The question to ask, when the layer needs one.
        book_units: Textbook units for a ``book_fetch`` command (İP-8).
        timings: Milliseconds per stage, from the caller's :class:`StageTimer`.
        operation_id: ULID of the persisted ``operations`` row, when a workspace was given.
        nlu: The trace of the text-to-command layer.
        model_versions: Active OCR/ASR versions.
        recognition: Handwriting recognition, for :meth:`PipelineService.run_ink`.
        transcript: Speech transcript, for :meth:`PipelineService.run_audio`.
        plot_url: Relative URL of a rendered plot asset, when the command was ``plot``.
        cache_hit: Whether the CAS answered from its LRU cache.
    """

    kind: str
    input_latex: str | None = None
    command: dict[str, Any] | None = None
    result: Result | None = None
    clarification: ClarificationPayload | None = None
    book_units: tuple[ContentUnit, ...] = ()
    timings: dict[str, float] = field(default_factory=dict)
    operation_id: str | None = None
    nlu: NluTrace | None = None
    model_versions: dict[str, str | None] = field(default_factory=dict)
    recognition: Recognition | None = None
    transcript: Transcript | None = None
    plot_url: str | None = None
    cache_hit: bool = False


def as_unsupported(error: AiclsError) -> AiclsError:
    """Map a "recognised but not computable" failure onto ``UNSUPPORTED_PROBLEM``.

    ``aicls.latex2sympy`` refuses constructs outside the v1 grammar with
    ``UNSUPPORTED_CONSTRUCT``. On the answering endpoints the user's question is "can you do
    this?", and the contract answer is the frozen İP-4 sentence, with the construct named in the
    problem document so the interface can say *which* symbol to avoid (KNOWN_ISSUES item 2).

    Args:
        error: The failure raised by the parser or the CAS.

    Returns:
        An :class:`~aicls.mathengine.errors.UnsupportedProblem` for an unsupported construct,
        otherwise ``error`` unchanged.
    """
    if error.code is not ErrorCode.UNSUPPORTED_CONSTRUCT:
        return error
    details = {
        "construct": str(error.details.get("construct", "")),
        "original_code": ErrorCode.UNSUPPORTED_CONSTRUCT.value,
    }
    note = error.details.get("note")
    if note:
        details["note"] = str(note)
    return UnsupportedProblem("recognised_not_computable", details=details)


class PipelineService:
    """Run the pipeline and persist what it did.

    Args:
        engine: The CAS facade (validated commands, LRU cache, sandbox executor).
        registry: Resolves the active recogniser and transcriber.
        settings: Runtime settings (``data_root`` for plot assets).
    """

    def __init__(self, engine: MathEngine, registry: ModelRegistry, settings: Settings) -> None:
        self._engine = engine
        self._registry = registry
        self._settings = settings

    @property
    def engine(self) -> MathEngine:
        """The CAS facade this service computes with."""
        return self._engine

    @property
    def registry(self) -> ModelRegistry:
        """The model registry this service resolves models through."""
        return self._registry

    # -- text ---------------------------------------------------------------------------------
    def run_text(
        self,
        text: str,
        ctx: ParseContext | None = None,
        user: User | None = None,
        ws_id: str | None = None,
        *,
        session: Session | None = None,
        timer: StageTimer | None = None,
        confirm_intent: str | None = None,
        input_kind: InputKind = InputKind.TEXT,
        nlu_source: NluSource = NluSource.RULES,
    ) -> PipelineResult:
        """Turkish text → command → result (the İP-4 path).

        Args:
            text: The utterance, typed or straight out of the ASR decoder.
            ctx: Session context; ``last_latex`` is what makes "türevini al" work.
            user: The signed-in user, when the operation should be recorded.
            ws_id: Workspace to append the operation to; ``None`` skips persistence.
            session: Open database session; ``None`` skips persistence and book lookups.
            timer: Stage timer to record into; a fresh one is used when omitted.
            confirm_intent: Intent the user picked from a previous clarification.
            input_kind: ``text`` normally, ``audio`` when called from the voice path.
            nlu_source: ``rules`` normally, ``asr+rules`` on the voice path.

        Returns:
            The :class:`PipelineResult`.

        Raises:
            aicls.core.errors.InputValidationError: If the text is empty or too long.
            aicls.core.errors.AiclsError: ``NLU_NO_INTENT`` when nothing matched; the CAS errors
                otherwise.
        """
        clock = timer or StageTimer()
        cleaned = _check_text(text)
        with clock.stage("nlu"):
            outcome = (
                _forced(cleaned, confirm_intent, ctx)
                if confirm_intent
                else parse_command(cleaned, context=ctx)
            )
        if isinstance(outcome, NoMatch):
            raise AiclsError(
                ErrorCode.NLU_NO_INTENT,
                messages_tr.MSG_NLU_NO_INTENT,
                {"suggestions_tr": list(outcome.suggestions_tr), "reason": outcome.reason},
            )
        if isinstance(outcome, Clarification):
            return PipelineResult(
                kind="clarification",
                clarification=self._clarification(outcome, cleaned, ctx),
                timings=clock.as_dict(),
                nlu=NluTrace(outcome.intent, 0.0, tuple(outcome.trace), nlu_source.value),
                model_versions=self._registry.versions(),
            )
        trace = NluTrace(
            outcome.intent, round(outcome.score, 4), tuple(outcome.trace), nlu_source.value
        )
        return self._dispatch(
            outcome.command,
            trace,
            clock,
            user=user,
            ws_id=ws_id,
            session=session,
            source_text=cleaned,
            input_kind=input_kind,
            nlu_source=nlu_source,
        )

    # -- latex --------------------------------------------------------------------------------
    def run_latex(
        self,
        cmd: CasCommand | dict[str, Any],
        user: User | None = None,
        ws_id: str | None = None,
        *,
        session: Session | None = None,
        timer: StageTimer | None = None,
    ) -> PipelineResult:
        """Execute a typed CAS command directly (no NLU, the deterministic path)."""
        clock = timer or StageTimer()
        with clock.stage("parse"):
            command = self._engine.validate(cmd)
        return self._dispatch(
            command,
            None,
            clock,
            user=user,
            ws_id=ws_id,
            session=session,
            source_text=None,
            input_kind=InputKind.LATEX,
            nlu_source=None,
        )

    # -- ink ----------------------------------------------------------------------------------
    def run_ink(
        self,
        ink: Ink | Image,
        user: User | None = None,
        *,
        session: Session | None = None,
        timer: StageTimer | None = None,
        beam: int = 4,
    ) -> PipelineResult:
        """Recognise handwriting and return the LaTeX (İP-3, single shot)."""
        clock = timer or StageTimer()
        with clock.stage("ocr"):
            recognition = self._registry.get_recognizer().recognize(ink, beam=beam)
        recognition_id: str | None = None
        if session is not None:
            row = RecognitionRepository(session).add(
                recognition.latex,
                user_id=user.id if user is not None else None,
                confidence=recognition.confidence,
                alternatives={"alternatives": [list(a) for a in recognition.alternatives]},
                model_version=recognition.model_version,
                elapsed_ms=int(recognition.elapsed_ms),
            )
            recognition_id = row.id
        return PipelineResult(
            kind="recognition",
            input_latex=recognition.latex,
            recognition=recognition,
            operation_id=recognition_id,
            timings=clock.as_dict(),
            model_versions=self._registry.versions(),
        )

    # -- audio --------------------------------------------------------------------------------
    def run_audio(
        self,
        audio: Audio,
        user: User | None = None,
        ctx: ParseContext | None = None,
        ws_id: str | None = None,
        *,
        session: Session | None = None,
        timer: StageTimer | None = None,
        to_command: bool = True,
    ) -> PipelineResult:
        """Transcribe speech and (optionally) run the transcript through the text path."""
        clock = timer or StageTimer()
        with clock.stage("asr"):
            transcript = self._registry.get_transcriber().transcribe(audio)
        transcript_id: str | None = None
        if session is not None:
            row = RecognitionRepository(session).add_transcript(
                transcript.text,
                user_id=user.id if user is not None else None,
                confidence=transcript.confidence,
                model_version=transcript.model_version,
                elapsed_ms=int(transcript.elapsed_ms),
            )
            transcript_id = row.id
        if not to_command:
            return PipelineResult(
                kind="transcript",
                transcript=transcript,
                operation_id=transcript_id,
                timings=clock.as_dict(),
                model_versions=self._registry.versions(),
            )
        outcome = self.run_text(
            transcript.text,
            ctx if ctx is not None else ParseContext(modality="voice"),
            user,
            ws_id,
            session=session,
            timer=clock,
            input_kind=InputKind.AUDIO,
            nlu_source=NluSource.ASR_RULES,
        )
        return PipelineResult(
            kind=outcome.kind,
            input_latex=outcome.input_latex,
            command=outcome.command,
            result=outcome.result,
            clarification=outcome.clarification,
            book_units=outcome.book_units,
            timings=clock.as_dict(),
            operation_id=outcome.operation_id,
            nlu=outcome.nlu,
            model_versions=outcome.model_versions,
            transcript=transcript,
            plot_url=outcome.plot_url,
            cache_hit=outcome.cache_hit,
        )

    # -- internals ----------------------------------------------------------------------------
    def _dispatch(
        self,
        command: Command,
        trace: NluTrace | None,
        clock: StageTimer,
        *,
        user: User | None,
        ws_id: str | None,
        session: Session | None,
        source_text: str | None,
        input_kind: InputKind,
        nlu_source: NluSource | None,
    ) -> PipelineResult:
        """Route a typed command to the CAS, the book layer or the client."""
        if isinstance(command, UiAction):
            return PipelineResult(
                kind="ui",
                command=command_payload(command),
                timings=clock.as_dict(),
                nlu=trace,
                model_versions=self._registry.versions(),
            )
        if isinstance(command, BookFetch):
            units = self._book_units(command, session, clock)
            return PipelineResult(
                kind="book",
                command=command_payload(command),
                book_units=units,
                timings=clock.as_dict(),
                nlu=trace,
                model_versions=self._registry.versions(),
            )
        return self._compute(
            command,
            trace,
            clock,
            user=user,
            ws_id=ws_id,
            session=session,
            source_text=source_text,
            input_kind=input_kind,
            nlu_source=nlu_source,
        )

    def _compute(
        self,
        command: CasCommand,
        trace: NluTrace | None,
        clock: StageTimer,
        *,
        user: User | None,
        ws_id: str | None,
        session: Session | None,
        source_text: str | None,
        input_kind: InputKind,
        nlu_source: NluSource | None,
    ) -> PipelineResult:
        """Execute a CAS command, store the plot and append the operation."""
        try:
            with clock.stage("cas"):
                result = self._engine.execute(command)
        except AiclsError as exc:
            self._record_failure(
                exc,
                clock,
                user=user,
                ws_id=ws_id,
                session=session,
                source_text=source_text,
                command=command,
                input_kind=input_kind,
                nlu_source=nlu_source,
            )
            raise as_unsupported(exc) from exc
        with clock.stage("render"):
            plot_asset_id, plot_url = self._store_plot(result, user, session)
        operation_id = self._persist(
            session,
            ws_id,
            user,
            command=command,
            result=result,
            source_text=source_text,
            input_kind=input_kind,
            nlu_source=nlu_source,
            timings=clock.as_dict(),
            plot_asset_id=plot_asset_id,
        )
        return PipelineResult(
            kind="result",
            input_latex=primary_latex(command),
            command=command_payload(command),
            result=result,
            timings=clock.as_dict(),
            operation_id=operation_id,
            nlu=trace,
            model_versions=self._registry.versions(),
            plot_url=plot_url,
            cache_hit=result.cache_hit,
        )

    def _clarification(
        self, outcome: Clarification, text: str, ctx: ParseContext | None
    ) -> ClarificationPayload:
        """Attach a command preview to every candidate of a clarification question."""
        candidates: list[ClarificationCandidate] = []
        for alternative in outcome.candidates:
            command, _slots = command_for_intent(text, alternative.intent, context=ctx)
            candidates.append(
                ClarificationCandidate(
                    intent=alternative.intent,
                    score=round(alternative.score, 4),
                    command=command_payload(command),
                    latex_preview=primary_latex(command) or "",
                    label_tr=alternative.label_tr,
                )
            )
        return ClarificationPayload(
            question_tr=outcome.question_tr,
            missing_slot=outcome.missing_slot,
            candidates=tuple(candidates),
        )

    def _book_units(
        self, command: BookFetch, session: Session | None, clock: StageTimer
    ) -> tuple[ContentUnit, ...]:
        """Resolve a ``book_fetch`` command through :mod:`aicls.book` (İP-8)."""
        if session is None:
            return ()
        from aicls.book import BookResolver
        from aicls.book.schema import UnitType
        from aicls.db.repositories.book import BookRepository

        unit_type = UnitType(command.unit_type.value) if command.unit_type else None
        with clock.stage("book"):
            resolved = BookResolver(BookRepository(session)).resolve(
                command.query, unit_type=unit_type, k=BOOK_HITS
            )
        return tuple(item.unit for item in resolved)

    def _store_plot(
        self, result: Result, user: User | None, session: Session | None
    ) -> tuple[str | None, str | None]:
        """Persist ``result.plot_png`` as a content-addressed asset and return its URL."""
        if not result.plot_png or session is None:
            return None, None
        payload = result.plot_png
        digest = sha256_bytes(payload)
        relative = f"{PLOT_DIR}/{digest[:2]}/{digest}.png"
        target = self._settings.data_root / relative
        ensure_dir(target.parent)
        if not target.is_file():
            target.write_bytes(payload)
        asset = AssetRepository(session).add(
            AssetKind.PLOT_PNG,
            relative,
            sha256=digest,
            size_bytes=len(payload),
            mime="image/png",
            owner_id=user.id if user is not None else None,
        )
        return asset.id, f"/api/v1/compute/plots/{asset.id}.png"

    def _persist(
        self,
        session: Session | None,
        ws_id: str | None,
        user: User | None,
        *,
        command: CasCommand,
        result: Result,
        source_text: str | None,
        input_kind: InputKind,
        nlu_source: NluSource | None,
        timings: dict[str, float],
        plot_asset_id: str | None,
    ) -> str | None:
        """Append the finished operation to a workspace the user owns."""
        workspace = self._workspace(session, ws_id, user)
        if workspace is None or session is None or user is None:
            return None
        row = OperationRepository(session).add(
            workspace,
            user.id,
            {
                "input_kind": input_kind,
                "source_text": source_text,
                "input_latex": primary_latex(command),
                "command": command_payload(command) or {},
                "result_kind": result.kind,
                "result_latex": result.latex,
                "result_srepr": result.srepr,
                "steps": {"steps": [step.model_dump(mode="json") for step in result.steps]}
                if result.steps
                else None,
                "plot_asset_id": plot_asset_id,
                "timings": timings,
                "cache_hit": result.cache_hit,
                "model_versions": self._registry.versions(),
                "nlu_source": nlu_source,
            },
        )
        return row.id

    def _record_failure(
        self,
        error: AiclsError,
        clock: StageTimer,
        *,
        user: User | None,
        ws_id: str | None,
        session: Session | None,
        source_text: str | None,
        command: CasCommand,
        input_kind: InputKind,
        nlu_source: NluSource | None,
    ) -> None:
        """Record a failed operation so the history explains what the user asked for."""
        workspace = self._workspace(session, ws_id, user)
        if workspace is None or session is None or user is None:
            return
        OperationRepository(session).add(
            workspace,
            user.id,
            {
                "input_kind": input_kind,
                "source_text": source_text,
                "input_latex": primary_latex(command),
                "command": command_payload(command) or {},
                "error_code": as_unsupported(error).code.value,
                "timings": clock.as_dict(),
                "nlu_source": nlu_source,
            },
        )

    @staticmethod
    def _workspace(session: Session | None, ws_id: str | None, user: User | None) -> str | None:
        """Return ``ws_id`` when the user owns it.

        Raises:
            NotFoundError: If the workspace does not exist or belongs to somebody else — the
                same answer either way, so an id cannot be probed (IDOR).
        """
        if session is None or ws_id is None or user is None:
            return None
        owned = WorkspaceRepository(session).get_owned(ws_id, user.id)
        if owned is None:
            raise NotFoundError(
                messages_tr.MSG_NOT_FOUND, details={"table": "workspaces", "id": ws_id}
            )
        return owned.id


def _check_text(text: str) -> str:
    """Trim and length-check a text command (docs/guvenlik-gizlilik.md §6)."""
    from aicls.core.errors import InputValidationError

    cleaned = text.strip()
    if not cleaned:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR, details={"field": "text", "reason": "empty"}
        )
    if len(cleaned) > MAX_TEXT_CHARS:
        raise InputValidationError(
            messages_tr.msg("INPUT_TOO_LONG", max_len=MAX_TEXT_CHARS),
            details={"field": "text", "max_len": MAX_TEXT_CHARS, "length": len(cleaned)},
        )
    return cleaned


def _forced(text: str, intent_id: str, ctx: ParseContext | None) -> Matched | Clarification:
    """Re-resolve ``text`` against one confirmed intent (``confirm_intent``)."""
    command, slots = command_for_intent(text, intent_id, context=ctx)
    if command is None:
        return Clarification(
            question_tr=messages_tr.MSG_NLU_NO_INTENT,
            missing_slot=None,
            candidates=[],
            intent=intent_id,
            trace=[f"confirm_intent: {intent_id}", "slots incomplete"],
        )
    return Matched(
        command=command,
        intent=intent_id,
        slots=dict(slots),
        score=1.0,
        trace=[f"confirm_intent: {intent_id}"],
    )


def is_cas_command(command: Command) -> bool:
    """Whether ``command`` is one of the ten members SymPy can execute."""
    return isinstance(command, CAS_COMMAND_TYPES)


__all__ = [
    "BOOK_HITS",
    "MAX_TEXT_CHARS",
    "PLOT_DIR",
    "ClarificationCandidate",
    "ClarificationPayload",
    "NluTrace",
    "PipelineResult",
    "PipelineService",
    "as_unsupported",
    "is_cas_command",
]
