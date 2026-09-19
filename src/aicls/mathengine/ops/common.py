"""Shared plumbing for the individual CAS operations.

Three things every operation needs and must do identically:

1. **One symbol table per command.** ``sympy.Symbol("x")`` and ``sympy.Symbol("x", real=True)``
   are *different* symbols, so the variable an operation differentiates by has to be the very
   symbol the parser put into the expression. :func:`build_context` creates the table first and
   hands out both the table and the variable, which is also where the ``complexes`` domain
   drops the ``real=True`` assumption.
2. **One error policy.** SymPy signals "I cannot do this" in a dozen ways -- an unevaluated
   ``Integral``, a ``ConditionSet``, ``NotImplementedError``, ``PolynomialError``,
   ``RecursionError``. :func:`sympy_guard` and :func:`reject_unevaluated` funnel all of them
   into a single :class:`~aicls.mathengine.errors.UnsupportedProblem` carrying the frozen
   Turkish contract sentence, with the technical reason in ``details``.
3. **One rendering of the answer.** :func:`make_result` fills ``latex`` (KaTeX-safe), ``plain``
   and ``srepr`` from the same value, so the API, the CLI and ``aicls.eval`` never disagree
   about what was computed.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Any

import sympy
from sympy.logic.boolalg import BooleanAtom

from aicls.core.errors import AiclsError
from aicls.core.logging import get_logger
from aicls.latex2sympy import latex_to_sympy, sympy_to_latex
from aicls.latex2sympy.symbols import SymbolTable
from aicls.mathengine.errors import UnsupportedProblem
from aicls.mathengine.models import Result, ResultKind, Step

#: SymPy nodes whose presence in an answer means "SymPy gave up".
UNEVALUATED_NODES: tuple[type[Any], ...] = (
    sympy.Integral,
    sympy.Derivative,
    sympy.Limit,
    sympy.Sum,
    sympy.Product,
)

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Context:
    """The symbol table of one command and the variable it operates on.

    Attributes:
        table: Symbol table shared by every LaTeX field of the command.
        var: The operation's variable, taken from that table.
    """

    table: SymbolTable
    var: sympy.Symbol


def build_context(var_name: str, *, real: bool = True) -> Context:
    """Create a fresh symbol table and resolve ``var_name`` in it.

    Args:
        var_name: The command's variable (``"x"``, ``"theta"``, ``"x_{1}"``).
        real: When False the variable is declared without assumptions, which is what solving
            over the complexes needs (a ``real=True`` symbol has no complex roots).

    Returns:
        The :class:`Context` every ``parse_*`` call of this command must be given.
    """
    table = SymbolTable()
    if not real:
        table.declare(var_name)
    return Context(table=table, var=table.symbol(var_name))


def parse(latex: str, context: Context) -> sympy.Basic:
    """Parse one LaTeX field of a command into SymPy using the command's symbol table.

    Args:
        latex: A sanitised LaTeX string from a :data:`~aicls.mathengine.models.CasCommand`.
        context: The command's context.

    Returns:
        The SymPy object; calculus nodes stay unevaluated, as ``aicls.latex2sympy`` promises.

    Raises:
        aicls.latex2sympy.ParseError: If the input is not in the grammar.
        aicls.latex2sympy.UnsupportedConstruct: If it uses a construct outside v1 scope.
    """
    return latex_to_sympy(latex, symbols=context.table)


@contextmanager
def sympy_guard(op: str, *, reason: str = "sympy_error") -> Iterator[None]:
    """Translate any SymPy failure inside the block into :class:`UnsupportedProblem`.

    :class:`~aicls.core.errors.AiclsError` passes through untouched, so a parse error stays a
    parse error (HTTP 422 ``PARSE_ERROR``) instead of being flattened into "unsupported".

    Args:
        op: The command's ``op``, recorded in ``details``.
        reason: Slug for the ``details["reason"]`` field.

    Yields:
        None.

    Raises:
        UnsupportedProblem: For every non-``AiclsError`` exception raised in the block.
    """
    try:
        yield
    except AiclsError:
        raise
    except RecursionError as exc:
        raise UnsupportedProblem("recursion_limit", details={"op": op}) from exc
    except Exception as exc:
        # SymPy raises NotImplementedError, PolynomialError, CoercionFailed, ValueError and
        # friends; the module contract is that only AiclsError leaves this package.
        _logger.debug("mathengine.op.failed", op=op, error=type(exc).__name__)
        raise UnsupportedProblem(
            reason, details={"op": op, "error": type(exc).__name__, "message": str(exc)[:200]}
        ) from exc


def reject_unevaluated(
    value: Any,
    op: str,
    reason: str,
    nodes: Sequence[type[Any]] = UNEVALUATED_NODES,
) -> sympy.Basic:
    """Return ``value`` unless SymPy left an unevaluated node of ``nodes`` in it.

    Which node counts as a failure depends on the operation: an ``Integral`` left in the answer
    of ``\\int`` means SymPy gave up, while a ``Derivative`` left in the answer of ``d/dx f(x)``
    for an *undefined* ``f`` is the correct answer. Callers therefore pass the node types that
    mean failure for them.

    Args:
        value: Whatever the SymPy call returned.
        op: The command's ``op``, recorded in ``details``.
        reason: Slug for ``details["reason"]`` (``"unevaluated_integral"``, ...).
        nodes: Node types whose presence means SymPy could not finish.

    Returns:
        ``value``, typed as ``sympy.Basic``.

    Raises:
        UnsupportedProblem: If one of ``nodes`` survived, or ``value`` is not a SymPy object.
    """
    if not isinstance(value, sympy.Basic | sympy.MatrixBase):
        raise UnsupportedProblem(reason, details={"op": op, "type": type(value).__name__})
    if nodes and value.has(*nodes):
        raise UnsupportedProblem(reason, details={"op": op, "value": str(value)[:200]})
    return value


def classify(value: sympy.Basic) -> ResultKind:
    """Map a SymPy value to the ``result_kind`` enum of docs/mimari/veri-modeli.md."""
    if isinstance(value, BooleanAtom):
        return "bool"
    if isinstance(value, sympy.Set):
        return "set"
    if isinstance(value, sympy.MatrixBase):
        return "expr"
    if isinstance(value, sympy.Expr) and value.is_number:
        return "number"
    return "expr"


def make_result(
    value: sympy.Basic,
    *,
    kind: ResultKind | None = None,
    steps: Sequence[Step] = (),
    plot_png: bytes | None = None,
    cas_ms: float = 0.0,
) -> Result:
    """Render ``value`` into the :class:`~aicls.mathengine.models.Result` contract.

    Args:
        value: The computed SymPy object.
        kind: Override for :func:`classify` (``"plot"`` for graphs).
        steps: Narrated steps, if the operation produced any.
        plot_png: PNG bytes for a plot command.
        cas_ms: Milliseconds the SymPy call itself took.

    Returns:
        A frozen :class:`~aicls.mathengine.models.Result`.
    """
    return Result(
        kind=kind if kind is not None else classify(value),
        latex=sympy_to_latex(value),
        plain=sympy.sstr(value),
        srepr=sympy.srepr(value),
        steps=list(steps),
        plot_png=plot_png,
        timings={"cas": round(cas_ms, 3)},
        cache_hit=False,
        elapsed_ms=round(cas_ms, 3),
    )


@contextmanager
def measured() -> Iterator[list[float]]:
    """Time the enclosed block; the single-element list receives the elapsed milliseconds."""
    holder: list[float] = [0.0]
    started = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = (time.perf_counter() - started) * 1000.0


__all__ = [
    "UNEVALUATED_NODES",
    "Context",
    "build_context",
    "classify",
    "make_result",
    "measured",
    "parse",
    "reject_unevaluated",
    "sympy_guard",
]
