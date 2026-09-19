"""Typed CAS command union, solution steps and the computation result.

The ten commands form a pydantic *discriminated union* on ``op``
(:data:`CasCommand`), exactly as documented in ``docs/mimari/moduller.md`` and
``docs/mimari/api.md`` §3.4. ``aicls.nlu`` produces them from Turkish text, ``aicls.api``
accepts them verbatim as the ``POST /compute/latex`` body, and :class:`aicls.mathengine.MathEngine`
executes them.

Every LaTeX-bearing field is a :data:`LatexStr`: it is passed through
:func:`aicls.core.latex_safety.sanitize_latex` (KaTeX allow-list) and limited to
:data:`MAX_INPUT_CHARS` characters *before* the model is constructed, so nothing outside the
allow-list ever reaches the parser, the sandbox pipe or the cache key.
"""

from __future__ import annotations

import json
import math
import re
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    model_validator,
)

from aicls.core import messages_tr
from aicls.core.errors import InputValidationError
from aicls.core.latex_safety import sanitize_latex

#: Contract limit for a single LaTeX field and for a command's total LaTeX payload (ADR-0016).
MAX_INPUT_CHARS = 2048

#: Upper bound on ``Evaluate.subs`` entries (a command is one classroom step, not a program).
MAX_SUBSTITUTIONS = 16

#: Highest derivative / series order accepted from the outside world.
MAX_ORDER = 12

#: Variable names the engine accepts: ``x``, ``t``, ``theta``, ``x_{1}`` (no braces to escape).
VAR_PATTERN = r"^[A-Za-z][A-Za-z0-9]{0,11}(?:_\{[A-Za-z0-9]{1,8}\})?$"

_VAR_RE = re.compile(VAR_PATTERN)


def _clean_latex(value: Any) -> Any:
    """Sanitise and length-check one LaTeX field (``BeforeValidator`` of :data:`LatexStr`)."""
    if not isinstance(value, str):
        return value
    cleaned = sanitize_latex(value, max_len=MAX_INPUT_CHARS)
    if not cleaned:
        raise InputValidationError(
            messages_tr.msg("VALIDATION_ERROR"), details={"field": "latex", "reason": "empty"}
        )
    return cleaned


#: A sanitised, non-empty LaTeX string of at most :data:`MAX_INPUT_CHARS` characters.
LatexStr: TypeAlias = Annotated[str, BeforeValidator(_clean_latex)]

#: A SymPy-safe variable name (see :data:`VAR_PATTERN`).
VarName: TypeAlias = Annotated[str, StringConstraints(pattern=VAR_PATTERN, max_length=24)]

#: What the result value *is*, mirrored by ``operations.result_kind`` (docs/mimari/veri-modeli.md).
ResultKind: TypeAlias = Literal["expr", "number", "set", "bool", "plot", "steps"]

#: Direction of a limit: from the right, from the left, or two-sided.
LimitDirection: TypeAlias = Literal["+", "-", "+-"]

#: Domain a solution set is computed over.
SolveDomain: TypeAlias = Literal["reals", "complexes"]


class _CommandBase(BaseModel):
    """Shared configuration: frozen, no unknown fields, no coercion surprises."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)


class Diff(_CommandBase):
    """``\\frac{d^{n}}{dvar^{n}} expr`` -- the proposal's ``diff(x**2, x)`` example."""

    op: Literal["diff"] = "diff"
    expr: LatexStr
    var: VarName = "x"
    order: int = Field(default=1, ge=1, le=MAX_ORDER)


class Integrate(_CommandBase):
    """``\\int expr\\,dvar``; definite when both bounds are given (``\\infty`` allowed)."""

    op: Literal["integrate"] = "integrate"
    expr: LatexStr
    var: VarName = "x"
    lower: LatexStr | None = None
    upper: LatexStr | None = None

    @model_validator(mode="after")
    def _bounds_come_in_pairs(self) -> Integrate:  # noqa: N804 - "after" validators are instance methods
        if (self.lower is None) != (self.upper is None):
            raise ValueError("lower and upper bounds must be given together")
        return self


class Limit(_CommandBase):
    """``\\lim_{var \\to to} expr`` with an optional one-sided direction."""

    op: Literal["limit"] = "limit"
    expr: LatexStr
    var: VarName = "x"
    to: LatexStr
    direction: LimitDirection = "+-"


class Solve(_CommandBase):
    """Solve ``equation`` for ``var`` over the reals (default) or the complexes."""

    op: Literal["solve"] = "solve"
    equation: LatexStr
    var: VarName = "x"
    domain: SolveDomain = "reals"


class Simplify(_CommandBase):
    """``sympy.simplify(expr)``."""

    op: Literal["simplify"] = "simplify"
    expr: LatexStr


class Expand(_CommandBase):
    """``sympy.expand(expr)``."""

    op: Literal["expand"] = "expand"
    expr: LatexStr


class Factor(_CommandBase):
    """``sympy.factor(expr)``."""

    op: Literal["factor"] = "factor"
    expr: LatexStr


class Series(_CommandBase):
    """Taylor/Laurent expansion of ``expr`` around ``point`` to ``order`` terms."""

    op: Literal["series"] = "series"
    expr: LatexStr
    var: VarName = "x"
    point: LatexStr = "0"
    order: int = Field(default=6, ge=1, le=MAX_ORDER * 2)


class Evaluate(_CommandBase):
    """Substitute ``subs`` into ``expr`` and evaluate the result."""

    op: Literal["evaluate"] = "evaluate"
    expr: LatexStr
    subs: Annotated[dict[VarName, LatexStr], Field(min_length=1, max_length=MAX_SUBSTITUTIONS)]


class Plot(_CommandBase):
    """Render ``expr`` over ``[xmin, xmax]`` as a deterministic Matplotlib Agg PNG."""

    op: Literal["plot"] = "plot"
    expr: LatexStr
    var: VarName = "x"
    xmin: float = -10.0
    xmax: float = 10.0

    @model_validator(mode="after")
    def _range_is_finite_and_increasing(self) -> Plot:  # noqa: N804 - "after" validators are instance methods
        if not (math.isfinite(self.xmin) and math.isfinite(self.xmax)):
            raise ValueError("xmin and xmax must be finite")
        if self.xmin >= self.xmax:
            raise ValueError("xmin must be strictly smaller than xmax")
        return self


#: The typed command union; ``op`` is the discriminator (docs/mimari/api.md §3.4).
CasCommand: TypeAlias = Annotated[
    Diff | Integrate | Limit | Solve | Simplify | Expand | Factor | Series | Evaluate | Plot,
    Field(discriminator="op"),
]

#: Reusable validator for raw ``{"op": ...}`` payloads coming from the API or the sandbox pipe.
COMMAND_ADAPTER: TypeAdapter[CasCommand] = TypeAdapter(CasCommand)

#: The concrete members of :data:`CasCommand`, for ``isinstance`` checks.
#:
#: :data:`aicls.nlu.types.Command` is a *wider* union than this one: it also carries
#: ``BookFetch`` and ``UiAction``, which must never reach SymPy. Callers that route a parsed
#: command therefore need a cheap way to ask "is this one of mine?", and
#: :meth:`aicls.mathengine.MathEngine.validate` uses it to refuse the others by name instead of
#: failing later with an ``AttributeError`` from inside the executor.
CAS_COMMAND_TYPES: tuple[type[BaseModel], ...] = (
    Diff,
    Integrate,
    Limit,
    Solve,
    Simplify,
    Expand,
    Factor,
    Series,
    Evaluate,
    Plot,
)

#: Names of the LaTeX-bearing fields, per command class (used by :func:`command_input_chars`).
_LATEX_FIELDS: tuple[str, ...] = ("expr", "equation", "lower", "upper", "to", "point")


class Step(BaseModel):
    """One narrated solution step.

    The web client (``web/src/components/StepList.tsx``) renders ``title_tr`` as the heading and
    ``latex`` through KaTeX; ``rule`` and ``detail_tr`` are optional extras it may ignore.

    Attributes:
        title_tr: Turkish heading (from :mod:`aicls.mathengine.messages`).
        latex: The step's mathematics, already KaTeX-safe.
        rule: Machine-readable slug (``"product_rule"``, ``"chain_rule"``, ``"result"``).
        detail_tr: Optional Turkish explanation under the heading.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    title_tr: str
    latex: str
    rule: str = ""
    detail_tr: str = ""


class Result(BaseModel):
    """What a command produced (``ComputeOut.result`` in docs/mimari/api.md §3.4).

    Attributes:
        kind: Shape of the value (``expr``, ``number``, ``set``, ``bool``, ``plot``, ``steps``).
        latex: KaTeX-safe rendering of the value.
        plain: ``sympy.sstr`` rendering, for the CLI and for copy-paste.
        srepr: ``sympy.srepr`` rendering -- the exact, re-constructible form used by
            ``aicls.eval`` to compare answers.
        steps: Narrated steps; empty when narration is not cheap for this command.
        plot_png: PNG bytes for ``op="plot"``, otherwise None.
        timings: Milliseconds per stage (``cas``, ``render``, ``total``).
        cache_hit: True when the engine served this result from its LRU cache.
        elapsed_ms: Wall-clock milliseconds the engine spent on this call.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ResultKind
    latex: str
    plain: str
    srepr: str
    steps: list[Step] = Field(default_factory=list)
    plot_png: bytes | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    cache_hit: bool = False
    elapsed_ms: float = 0.0


def command_input_chars(cmd: CasCommand) -> int:
    """Total number of LaTeX characters a command carries.

    ``MAX_INPUT_CHARS`` is enforced per field by :data:`LatexStr`; the engine also checks this
    sum so that a command with many substitutions cannot smuggle a large payload into the
    sandbox (ADR-0016, "girdi <= 2048 karakter").

    Args:
        cmd: Any command of the union.

    Returns:
        The number of characters across every LaTeX field, substitution values included.
    """
    total = 0
    for name in _LATEX_FIELDS:
        value = getattr(cmd, name, None)
        if isinstance(value, str):
            total += len(value)
    if isinstance(cmd, Evaluate):
        total += sum(len(value) for value in cmd.subs.values())
    return total


def check_input_size(cmd: CasCommand) -> None:
    """Raise when a command's total LaTeX payload exceeds :data:`MAX_INPUT_CHARS`.

    Args:
        cmd: The command about to be executed.

    Raises:
        aicls.core.errors.InputValidationError: If the sum of the LaTeX fields is too large.
    """
    size = command_input_chars(cmd)
    if size > MAX_INPUT_CHARS:
        raise InputValidationError(
            messages_tr.msg("INPUT_TOO_LONG", max_len=MAX_INPUT_CHARS),
            details={"max_len": MAX_INPUT_CHARS, "length": size, "op": cmd.op},
        )


def canonical_key(cmd: CasCommand) -> str:
    """Canonical cache key of a command: its JSON dump with sorted keys.

    Defaults are materialised by pydantic, so ``Diff(expr="x")`` and
    ``Diff(expr="x", var="x", order=1)`` share a key, and ``Evaluate`` substitutions are
    order-independent.

    Args:
        cmd: Any command of the union.

    Returns:
        A compact JSON string, at most a few kilobytes.
    """
    return json.dumps(
        cmd.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def is_valid_var(name: str) -> bool:
    """Whether ``name`` is an acceptable variable name (:data:`VAR_PATTERN`)."""
    return bool(_VAR_RE.match(name))


__all__ = [
    "CAS_COMMAND_TYPES",
    "COMMAND_ADAPTER",
    "MAX_INPUT_CHARS",
    "MAX_ORDER",
    "MAX_SUBSTITUTIONS",
    "VAR_PATTERN",
    "CasCommand",
    "Diff",
    "Evaluate",
    "Expand",
    "Factor",
    "Integrate",
    "LatexStr",
    "Limit",
    "LimitDirection",
    "Plot",
    "Result",
    "ResultKind",
    "Series",
    "Simplify",
    "Solve",
    "SolveDomain",
    "Step",
    "VarName",
    "canonical_key",
    "check_input_size",
    "command_input_chars",
    "is_valid_var",
]
