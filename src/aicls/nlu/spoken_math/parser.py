"""``spoken_to_latex``: Turkish spoken mathematics -> canonical LaTeX.

The pipeline is deliberately short and total:

``text`` -> :func:`aicls.nlu.normalize.normalize_text` -> Lark LALR over
``spoken_math/grammar.lark`` -> :class:`~aicls.nlu.spoken_math.transformer.SpokenMathTransformer`
-> :func:`aicls.latex2sympy.to_latex`.

Because the output comes from the LaTeX front end's own canonical printer, ``parse_latex`` is
guaranteed to accept it; :func:`spoken_to_latex` still re-parses in debug-free production code
only through :func:`aicls.nlu.slots`, where the round trip is the slot's validity criterion.

Nothing here is Turkish-string-bearing: failures raise
:class:`aicls.nlu.errors.SpokenMathError`, whose message comes from
:mod:`aicls.core.messages_tr`.
"""

from __future__ import annotations

from functools import lru_cache

from lark import Lark, UnexpectedInput
from lark.exceptions import VisitError

from aicls.core.paths import package_resource
from aicls.latex2sympy.ast import Node
from aicls.latex2sympy.to_latex import to_latex
from aicls.nlu.errors import SpokenMathError
from aicls.nlu.spoken_math.transformer import SpokenMathTransformer

#: Package resource holding the grammar.
GRAMMAR_PACKAGE = "aicls.nlu"
GRAMMAR_NAME = "spoken_math/grammar.lark"

#: Longest spoken fragment accepted (a classroom command, not a program).
MAX_SPOKEN_CHARS = 512


@lru_cache(maxsize=1)
def grammar_source() -> str:
    """Return the text of ``spoken_math/grammar.lark`` (cached)."""
    return package_resource(GRAMMAR_PACKAGE, GRAMMAR_NAME).read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def lark_parser() -> Lark:
    """Build (once per process) the LALR parser with the contextual lexer."""
    return Lark(
        grammar_source(),
        start="start",
        parser="lalr",
        lexer="contextual",
        propagate_positions=False,
        maybe_placeholders=False,
    )


def spoken_to_ast(text: str, *, normalized: bool = False) -> Node:
    """Parse spoken Turkish mathematics into a :mod:`aicls.latex2sympy.ast` node.

    Args:
        text: The fragment, raw or already normalised.
        normalized: ``True`` when ``text`` is already a canonical token stream with the suffix
            tokens removed (the slot extractor's case); skips a redundant normalisation pass.

    Returns:
        The typed AST node.

    Raises:
        aicls.nlu.errors.SpokenMathError: If the fragment is empty, too long, or not a sentence
            of the grammar.
    """
    if len(text) > MAX_SPOKEN_CHARS:
        raise SpokenMathError(text, reason="too_long", details={"max_len": MAX_SPOKEN_CHARS})
    if normalized:
        source = text.strip()
    else:
        from aicls.nlu.normalize import normalize_text

        analysed = normalize_text(text)
        source = analysed.slice_math(0, len(analysed.tokens))
    if not source:
        raise SpokenMathError(text, reason="empty")
    try:
        tree = lark_parser().parse(source)
    except UnexpectedInput as exc:
        position = int(getattr(exc, "column", 1) or 1)
        raise SpokenMathError(source, reason="parse_error", position=position) from exc
    try:
        node = SpokenMathTransformer().transform(tree)
    except VisitError as exc:
        # The transformer refuses readings the grammar cannot rule out on its own (two bare
        # numerals side by side); its own error carries the reason and must survive the wrapper.
        original = exc.orig_exc
        if isinstance(original, SpokenMathError):
            raise SpokenMathError(source, reason=original.reason) from exc
        raise SpokenMathError(source, reason="transform_error") from exc
    if not isinstance(node, Node):  # pragma: no cover - defensive
        raise SpokenMathError(source, reason="transform_error")
    return node


def spoken_to_latex(text: str, *, normalized: bool = False) -> str:
    """Convert spoken Turkish mathematics to canonical, KaTeX-safe LaTeX.

    ``"x kare çarpı sinüs x"`` becomes ``"x^{2} \\sin\\left(x\\right)"``.

    Args:
        text: The fragment, raw or already normalised.
        normalized: See :func:`spoken_to_ast`.

    Returns:
        The LaTeX string; ``parse_latex`` accepts it by construction.

    Raises:
        aicls.nlu.errors.SpokenMathError: If the fragment does not parse.
    """
    return to_latex(spoken_to_ast(text, normalized=normalized))


__all__ = [
    "GRAMMAR_NAME",
    "GRAMMAR_PACKAGE",
    "MAX_SPOKEN_CHARS",
    "grammar_source",
    "lark_parser",
    "spoken_to_ast",
    "spoken_to_latex",
]
