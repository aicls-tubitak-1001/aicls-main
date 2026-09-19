"""A torch-free, in-process pipeline client for the latency runner (ADR-0014).

The contractual latency figure ("komuttan sonuca ortalama < 3 s") is measured **in the browser**:
``performance.now()`` from the moment the command is sent to the moment KaTeX finishes rendering.
Nothing in CI can reproduce that. What CI *can* reproduce is the deterministic server-side part —
Turkish text -> command -> SymPy -> LaTeX — and that is what this client times, stage by stage,
with the same :class:`aicls.core.timing.StageTimer` the API uses, so the numbers line up with the
``Server-Timing`` header the browser will later report next to them.

A report produced from this client is therefore always marked ``stub``: it is a regression
instrument for the pipeline's own cost, not evidence for the contractual figure.
"""

from __future__ import annotations

from typing import Any

from aicls.core.errors import AiclsError
from aicls.core.logging import get_logger
from aicls.core.timing import StageTimer
from aicls.mathengine import MathEngine
from aicls.nlu import BookFetch, Matched, ParseContext, UiAction, parse_command

_logger = get_logger(__name__)


class LocalPipelineClient:
    """Times ``parse_command`` -> ``MathEngine.execute`` for one Turkish command.

    Args:
        engine: The engine to compute with; a fresh in-process one when omitted.
        inventory: NLU inventory; the packaged one when omitted.
        context: Parse context, so context-dependent commands ("türevini al") resolve.
        cached: Whether repeated commands may be served from the engine's LRU cache. The
            default is ``False`` — the cache is cleared before every command — because a
            latency figure that mostly measures dictionary lookups would be meaningless.

    Attributes:
        errors: How many commands the pipeline could not answer (they still contribute their
            measured time; a refusal is a response too).
        no_command: How many texts produced no CAS command at all (a book fetch, a UI action or
            a refusal).
    """

    def __init__(
        self,
        engine: MathEngine | None = None,
        *,
        inventory: Any = None,
        context: ParseContext | None = None,
        cached: bool = False,
    ) -> None:
        self._engine = engine if engine is not None else MathEngine()
        self._inventory = inventory
        self._context = context
        self._cached = cached
        self.errors = 0
        self.no_command = 0

    def run(self, text: str) -> dict[str, float]:
        """Execute one command and return ``{stage: milliseconds}`` including ``total``.

        Args:
            text: The Turkish command, as the ASR or the text box delivers it.

        Returns:
            The stage timings: ``nlu`` always, ``cas`` when a computation ran, and ``total``.
            Errors are counted, not raised: a corpus run must not stop at the first refusal.
        """
        if not self._cached:
            self._engine.clear_cache()
        timer = StageTimer()
        with timer.stage("nlu"):
            outcome = parse_command(text, context=self._context, inventory=self._inventory)
        # everything in the command union that is not a book lookup or a UI action is a
        # CAS command, i.e. the only kind that reaches SymPy
        if isinstance(outcome, Matched) and not isinstance(outcome.command, BookFetch | UiAction):
            try:
                with timer.stage("cas"):
                    self._engine.execute(outcome.command)
            except AiclsError as exc:
                self.errors += 1
                _logger.debug("eval.pipeline.command_failed", code=exc.code.value)
        else:
            self.no_command += 1
        stages = timer.stages
        stages["total"] = timer.total_ms
        return stages


__all__ = ["LocalPipelineClient"]
