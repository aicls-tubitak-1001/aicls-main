"""``aicls nlu try`` — parse one Turkish utterance into a typed command (İP-4, ADR-0005).

The text-to-command layer is the part of the pipeline a reviewer most often wants to poke at by
hand: ``aicls nlu try "x karenin türevini al"`` prints the intent that won, the resolved slots,
the typed command that would reach SymPy and the trace of *how* it was understood — the same
trace the UI shows next to a voice command.

Nothing is computed here. The command object is printed, never executed: only
:mod:`aicls.mathengine` produces results, which is the project's "no hallucination" claim.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer

from aicls.cli.common import emit, use_json

app = typer.Typer(
    name="nlu",
    help="Inspect the Turkish text-to-command layer.",
    no_args_is_help=True,
)


@app.command("try")
def try_command(
    text: Annotated[str, typer.Argument(help='The utterance, e.g. "x karenin türevini al".')],
    last_latex: Annotated[
        str,
        typer.Option("--last-latex", help="Expression already on the board (session context)."),
    ] = "",
    active_var: Annotated[
        str, typer.Option("--active-var", help="Variable the session is working with.")
    ] = "",
    modality: Annotated[
        str, typer.Option("--modality", help="How the text arrived: text, voice or ink.")
    ] = "text",
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Parse ``text`` and print the outcome (a command, a clarification or a refusal).

    Exit code is ``0`` for every outcome: "I did not understand" is a result of the layer
    (a :class:`aicls.nlu.NoMatch` with Turkish suggestions), not an error.
    """
    from aicls.nlu import Clarification, Matched, context_from_mapping, parse_command

    use_json(json_output)
    context = context_from_mapping(
        {
            key: value
            for key, value in (
                ("last_latex", last_latex),
                ("active_var", active_var),
                ("modality", modality),
            )
            if value
        }
    )
    outcome = parse_command(text, context=context)
    payload: dict[str, Any] = {"text": text, "trace": list(outcome.trace)}
    lines: list[str] = []
    if isinstance(outcome, Matched):
        payload.update(
            {
                "outcome": "matched",
                "intent": outcome.intent,
                "score": outcome.score,
                "source": outcome.source,
                "slots": outcome.slots,
                "command": outcome.command.model_dump(mode="json"),
                "alternatives": [
                    {"intent": alt.intent, "score": alt.score, "label_tr": alt.label_tr}
                    for alt in outcome.alternatives
                ],
            }
        )
        lines = [
            f"intent {outcome.intent} (score {outcome.score:.3f}, source {outcome.source})",
            f"slots {outcome.slots}",
            f"command {payload['command']}",
        ]
    elif isinstance(outcome, Clarification):
        payload.update(
            {
                "outcome": "clarification",
                "intent": outcome.intent,
                "missing_slot": outcome.missing_slot,
                "question_tr": outcome.question_tr,
                "candidates": [
                    {"intent": alt.intent, "score": alt.score, "label_tr": alt.label_tr}
                    for alt in outcome.candidates
                ],
            }
        )
        lines = [
            f"clarification needed (missing slot: {outcome.missing_slot})",
            outcome.question_tr,
        ]
    else:  # the outcome union has exactly three members, so this is NoMatch
        payload.update(
            {
                "outcome": "no_match",
                "reason": outcome.reason,
                "suggestions_tr": list(outcome.suggestions_tr),
            }
        )
        lines = [
            f"no match ({outcome.reason})",
            *(f"  try: {suggestion}" for suggestion in outcome.suggestions_tr),
        ]
    emit(payload, *lines, *(f"trace: {line}" for line in outcome.trace))


__all__ = ["app", "try_command"]
