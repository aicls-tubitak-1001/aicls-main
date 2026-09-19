"""The parser front end: ``parse_latex`` and the cached :class:`LatexParser`.

Pipeline (docs/mimari/latex-ayristirici.md 3):

``src`` -> length guard -> :func:`~aicls.latex2sympy.normalize.normalize_latex` -> command
pre-check (:mod:`aicls.latex2sympy.commands`) -> Lark LALR + contextual lexer ->
:class:`~aicls.latex2sympy.transformer.LatexTransformer` -> typed AST.

The pre-check is what turns "this macro is not in the grammar" into a *named*
:class:`~aicls.latex2sympy.errors.UnsupportedConstruct` instead of an anonymous syntax error,
and it is the difference between ``strict=True`` (unknown macros are refused) and
``strict=False`` (decorative macros such as ``\\overline`` are stripped and parsing continues,
which is what OCR output needs).
"""

from __future__ import annotations

from functools import lru_cache
import re
from typing import Any

from lark import Lark, UnexpectedCharacters, UnexpectedInput, UnexpectedToken
from lark.exceptions import VisitError

from aicls.core import messages_tr
from aicls.core.errors import AiclsError, InputValidationError, InternalError
from aicls.core.logging import get_logger
from aicls.core.paths import package_resource
from aicls.latex2sympy import commands
from aicls.latex2sympy.ast import Node
from aicls.latex2sympy.errors import ParseError, UnsupportedConstruct
from aicls.latex2sympy.normalize import normalize_latex
from aicls.latex2sympy.transformer import LatexTransformer

#: Contract limit for a single LaTeX input (docs/mimari/latex-ayristirici.md 4).
MAX_INPUT_CHARS = 2048

#: Name reported for a mixed partial, ``\\frac{\\partial^{2} f}{\\partial x \\partial y}``.
#: The v1 :class:`~aicls.latex2sympy.ast.Derivative` carries a single differentiation variable, so
#: the shape is *recognised* and refused **by name** rather than as an anonymous syntax error -
#: the OCR -> CAS hand-off needs a named 422, not a bare ``ParseError`` (KNOWN_ISSUES parser 2).
MIXED_PARTIAL_CONSTRUCT = r"\frac{\partial^{2}}{\partial x \partial y}"

_RESOURCE_PACKAGE = "aicls.latex2sympy"
_GRAMMAR_NAME = "grammar/latex_math.lark"
_SNIPPET_RADIUS = 20

# Two ``\partial <var>`` groups in a row - the denominator of a mixed partial.  No sentence of the
# v1 grammar contains that sequence (a sum of two partials always has ``}`` and ``\frac{`` in
# between), so the match is exact and linear-time.
_PARTIAL_VAR = r"\\partial\s*(?:\\[a-zA-Z]+|[A-Za-z])\s*(?:\^\s*(?:\{\s*\d+\s*\}|\d+))?"
_MIXED_PARTIAL_RE = re.compile(_PARTIAL_VAR + r"\s*" + _PARTIAL_VAR)

_logger = get_logger(__name__)


@lru_cache(maxsize=1)
def grammar_source() -> str:
    """Return the text of ``grammar/latex_math.lark`` (cached)."""
    return package_resource(_RESOURCE_PACKAGE, _GRAMMAR_NAME).read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def lark_parser() -> Lark:
    """Build (once per process) the LALR parser with the contextual lexer."""
    return Lark(
        grammar_source(),
        start="start",
        parser="lalr",
        lexer="contextual",
        propagate_positions=True,
        maybe_placeholders=False,
    )


def _snippet(text: str, position: int) -> str:
    start = max(0, position - _SNIPPET_RADIUS)
    end = min(len(text), position + _SNIPPET_RADIUS)
    return text[start:end]


def _expected_names(error: UnexpectedInput) -> list[str]:
    raw: Any = getattr(error, "expected", None) or getattr(error, "allowed", None) or ()
    names = {str(item).lstrip("_") for item in raw}
    return sorted(names)


def _as_parse_error(error: UnexpectedInput, text: str) -> ParseError:
    line = int(getattr(error, "line", 1) or 1)
    column = int(getattr(error, "column", 1) or 1)
    position = int(getattr(error, "pos_in_stream", None) or max(column - 1, 0))
    details: dict[str, Any] = {"kind": type(error).__name__}
    if isinstance(error, UnexpectedToken):
        details["token"] = str(error.token)
    elif isinstance(error, UnexpectedCharacters):
        details["char"] = text[position : position + 1]
    return ParseError(
        line=line,
        col=column,
        expected=_expected_names(error),
        snippet=_snippet(text, position),
        details=details,
    )


def _drop_command(text: str, command: str) -> str:
    return re.sub(re.escape(command) + r"(?![A-Za-z])\s*", "", text)


def _check_commands(text: str, *, strict: bool) -> str:
    """Reject (or, when lenient, strip) commands the grammar does not implement."""
    result = text
    for command in commands.scan_commands(text):
        kind = commands.classify(command)
        if kind in {"supported", "decorative"}:
            continue
        if kind == "unsupported":
            if not strict and command in commands.LENIENT_DROP_COMMANDS:
                _logger.debug("latex.command_dropped", command=command, mode="lenient")
                result = _drop_command(result, command)
                continue
            raise UnsupportedConstruct(command, note=commands.UNSUPPORTED_CONSTRUCTS[command])
        if strict:
            raise UnsupportedConstruct(command)
        result = _drop_command(result, command)
    return result


def _check_mixed_partial(text: str) -> None:
    """Refuse a mixed partial derivative by name (see :data:`MIXED_PARTIAL_CONSTRUCT`).

    Args:
        text: The normalised input.

    Raises:
        UnsupportedConstruct: If the input differentiates with respect to two variables.
    """
    if _MIXED_PARTIAL_RE.search(text):
        raise UnsupportedConstruct(
            MIXED_PARTIAL_CONSTRUCT,
            note=commands.UNSUPPORTED_CONSTRUCTS.get(MIXED_PARTIAL_CONSTRUCT),
        )


class LatexParser:
    """Reusable front end; the module-level :func:`parse_latex` uses a shared instance."""

    def parse(self, src: str, *, strict: bool = True) -> Node:
        """Parse ``src`` into the typed AST (see :func:`parse_latex`)."""
        return self.parse_with_repairs(src, strict=strict)[0]

    def parse_with_repairs(self, src: str, *, strict: bool = True) -> tuple[Node, list[str]]:
        """Parse ``src`` and also return the repairs the normaliser applied.

        Returns:
            ``(node, repairs)`` - the API layer surfaces ``repairs`` in ``ParseOut``.
        """
        if len(src) > MAX_INPUT_CHARS:
            raise InputValidationError(
                messages_tr.msg("INPUT_TOO_LONG", max_len=MAX_INPUT_CHARS),
                details={"max_len": MAX_INPUT_CHARS, "length": len(src)},
            )
        normalised = normalize_latex(src)
        text = normalised.text
        if not text:
            raise ParseError(line=1, col=1, expected=["expression"], snippet="")
        text = _check_commands(text, strict=strict)
        _check_mixed_partial(text)
        if not text.strip():
            raise ParseError(line=1, col=1, expected=["expression"], snippet="")
        try:
            tree = lark_parser().parse(text)
        except UnexpectedInput as exc:
            raise _as_parse_error(exc, text) from exc
        try:
            node: Node = LatexTransformer(text).transform(tree)
        except VisitError as exc:
            # Lark wraps callback exceptions; the transformer only raises AiclsError.
            if isinstance(exc.orig_exc, AiclsError):
                raise exc.orig_exc from exc
            raise InternalError(details={"stage": "transform"}) from exc
        return node, list(normalised.repairs)


_PARSER = LatexParser()


def parse_latex(src: str, *, strict: bool = True) -> Node:
    """Parse a LaTeX math expression into the typed AST.

    Args:
        src: LaTeX source (at most :data:`MAX_INPUT_CHARS` characters). It is normalised
            first, so ``\\left(``, ``\\dfrac``, unicode operators and a single missing closing
            brace are handled transparently.
        strict: When True (default) any macro outside ``grammar/commands.yaml`` is refused
            with :class:`~aicls.latex2sympy.errors.UnsupportedConstruct`. When False,
            decorative macros (``\\overline``, ``\\vec``, ``\\mathbf`` ...) are stripped and
            parsing continues; structural constructs such as ``\\begin{cases}`` are still
            refused.

    Returns:
        The root :class:`~aicls.latex2sympy.ast.Node`; a top-level relation yields a
        :class:`~aicls.latex2sympy.ast.Relation`.

    Raises:
        aicls.core.errors.InputValidationError: If the input is longer than the contract limit.
        ParseError: If the (normalised) input is not a sentence of the grammar.
        UnsupportedConstruct: If it uses a construct outside the v1 scope.
        SemanticError: If it parses but has no meaning (integral without a differential, ...).
    """
    return _PARSER.parse(src, strict=strict)


__all__ = [
    "MAX_INPUT_CHARS",
    "MIXED_PARTIAL_CONSTRUCT",
    "LatexParser",
    "grammar_source",
    "lark_parser",
    "parse_latex",
]
