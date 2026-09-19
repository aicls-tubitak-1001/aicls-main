"""Intent + slots -> a typed command (docs/mimari/metin-komut-katmani.md §6).

This is the layer's narrowest point and the reason the "no hallucination" claim holds: the only
things it can build are the ten :data:`aicls.mathengine.models.CasCommand` members,
:class:`~aicls.nlu.types.BookFetch` and :class:`~aicls.nlu.types.UiAction`. An intent id that is
not in the table below cannot be turned into an object at all, so nothing the layer does not
understand can reach SymPy.

Every CAS model validates its own fields (``VAR_PATTERN``, ``MAX_ORDER``, the KaTeX allow-list on
each LaTeX string). A rejection is converted into
:class:`aicls.nlu.errors.CommandBuildError` so a pydantic ``ValidationError`` never escapes the
package.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import ValidationError

from aicls.mathengine.models import (
    CasCommand,
    Diff,
    Evaluate,
    Expand,
    Factor,
    Integrate,
    Limit,
    Plot,
    Series,
    Simplify,
    Solve,
)
from aicls.nlu.errors import CommandBuildError
from aicls.nlu.inventory.schema import IntentSpec
from aicls.nlu.types import BookFetch, Command, UiAction, UnitType


def _diff(slots: Mapping[str, Any]) -> CasCommand:
    return Diff(expr=slots["expr"], var=slots.get("var", "x"), order=int(slots.get("order", 1)))


def _integrate(slots: Mapping[str, Any]) -> CasCommand:
    return Integrate(
        expr=slots["expr"],
        var=slots.get("var", "x"),
        lower=slots.get("lower"),
        upper=slots.get("upper"),
    )


def _limit(slots: Mapping[str, Any]) -> CasCommand:
    return Limit(
        expr=slots["expr"],
        var=slots.get("var", "x"),
        to=slots["point"],
        direction=slots.get("direction", "+-"),
    )


def _solve(slots: Mapping[str, Any]) -> CasCommand:
    return Solve(equation=slots["equation"], var=slots.get("var", "x"))


def _simplify(slots: Mapping[str, Any]) -> CasCommand:
    return Simplify(expr=slots["expr"])


def _expand(slots: Mapping[str, Any]) -> CasCommand:
    return Expand(expr=slots["expr"])


def _factor(slots: Mapping[str, Any]) -> CasCommand:
    return Factor(expr=slots["expr"])


def _series(slots: Mapping[str, Any]) -> CasCommand:
    return Series(
        expr=slots["expr"],
        var=slots.get("var", "x"),
        point=str(slots.get("point", "0")),
        order=int(slots.get("order", 6)),
    )


def _evaluate(slots: Mapping[str, Any]) -> CasCommand:
    return Evaluate(expr=slots["expr"], subs={slots.get("var", "x"): slots["value"]})


def _plot(slots: Mapping[str, Any]) -> CasCommand:
    return Plot(expr=slots["expr"], var=slots.get("var", "x"))


#: ``cas_op`` -> builder. The keys are exactly the ``op`` discriminators of
#: :data:`aicls.mathengine.models.CasCommand`.
CAS_BUILDERS: dict[str, Callable[[Mapping[str, Any]], CasCommand]] = {
    "diff": _diff,
    "integrate": _integrate,
    "limit": _limit,
    "solve": _solve,
    "simplify": _simplify,
    "expand": _expand,
    "factor": _factor,
    "series": _series,
    "evaluate": _evaluate,
    "plot": _plot,
}


def build_command(intent: IntentSpec, slots: Mapping[str, Any]) -> Command:
    """Build the typed command for ``intent`` from resolved ``slots``.

    Args:
        intent: The winning inventory intent.
        slots: Resolved slot values.

    Returns:
        A :data:`aicls.mathengine.models.CasCommand`, :class:`~aicls.nlu.types.BookFetch` or
        :class:`~aicls.nlu.types.UiAction`.

    Raises:
        aicls.nlu.errors.CommandBuildError: If the intent has no builder or the model rejects
            the values.
    """
    try:
        if intent.kind == "cas":
            builder = CAS_BUILDERS.get(intent.cas_op or "")
            if builder is None:
                raise CommandBuildError(intent.id, reason="unknown_cas_op")
            return builder(slots)
        if intent.kind == "book":
            unit_type = slots.get("unit_type")
            return BookFetch(
                unit_type=UnitType(unit_type) if unit_type else None,
                query=str(slots["query"]),
            )
        return UiAction(action=intent.ui_action)  # type: ignore[arg-type]  # validated by schema
    except (ValidationError, KeyError, ValueError) as exc:
        raise CommandBuildError(
            intent.id, reason="model_rejected", details={"error": str(exc)[:300]}
        ) from exc


__all__ = ["CAS_BUILDERS", "build_command"]
