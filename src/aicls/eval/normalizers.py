"""Versioned normalisers ``latex_v1`` and ``tr_v1`` (ADR-0015, docs/ml/degerlendirme.md §1).

A metric is only comparable across two years if the text it is computed on is produced by a
*named, frozen* rule set. These two are that rule set:

* :func:`latex_v1` — for OCR hypotheses and expressions. Surface cleanup (reusing the parser's
  :func:`aicls.latex2sympy.normalize_latex`), tokenisation, canonical script braces, redundant
  group flattening, single-space join.
* :func:`tr_v1` — for ASR transcripts. NFC, Turkish casefold, punctuation removal, digits to
  Turkish number words, whitespace squeeze — **and nothing else**. There is deliberately no
  lexicon folding, no ASR-confusion table and no math-term joining here: those live in
  :mod:`aicls.nlu.normalize`, where they belong to the product, not to the measurement. Folding
  them into the metric would be tuning the metric to the test set, which the reviewers of the
  proposal explicitly warned against.

Both functions are idempotent (property tests), which is what lets a report say "these numbers
were produced with ``latex_v1``/``tr_v1``" without also having to say how many times it ran.

Changing a rule here means a **new version** (``latex_v2``), never an edit: the old function
stays so that a 2027 number can still be recomputed in 2028.
"""

from __future__ import annotations

import re
import unicodedata

from aicls.core import messages_tr
from aicls.core.errors import InputValidationError
from aicls.core.logging import get_logger
from aicls.core.turkish import tr_casefold
from aicls.latex2sympy import normalize_latex

_log = get_logger(__name__)

#: Normaliser versions written into every :class:`aicls.eval.report.EvalReport`.
NORMALIZER_VERSIONS: dict[str, str] = {"latex": "v1", "tr": "v1"}

#: How many times :func:`latex_v1` re-runs its own pipeline looking for a fixed point.
#:
#: Idempotence is the whole contract of a versioned normaliser: a CER is only comparable with
#: another CER if the strings behind both went through the *same* rule set, and a rule set whose
#: result depends on how many times it ran is not one rule set. One pass does not always land on
#: a fixed point, because :func:`aicls.latex2sympy.normalize_latex`'s surface repairs are not all
#: whitespace-insensitive: ``\sin{xx}^\alpha{x`` is left alone on the first pass and repaired on
#: the second, once the tokeniser has spaced and balanced it (found by
#: ``test_latex_v1_is_idempotent``, 2026-09-08). Iterating to a fixed point makes the property
#: structural instead of accidental. Two passes settle every input measured so far; the cap only
#: bounds a pathological one, and the fall-through is logged rather than silent.
MAX_LATEX_PASSES = 4

#: One LaTeX token: a command, a run of digits, or any other single character (rule 2).
_LATEX_TOKEN_RE = re.compile(r"\\[a-zA-Z]+|\d+|\S")

#: Script operators whose brace group is meaningful and must never be flattened (rule 4).
_SCRIPTS = frozenset({"^", "_"})

#: Turkish cardinal words, frozen with the metric. These are *metric vocabulary*, not user
#: interface text (they are never shown to a user), which is why they do not live in
#: :mod:`aicls.core.messages_tr` — the same reasoning as
#: :data:`aicls.asr.stub.STUB_SENTENCES`.
_UNITS: tuple[str, ...] = (
    "sıfır",
    "bir",
    "iki",
    "üç",
    "dört",
    "beş",
    "altı",
    "yedi",
    "sekiz",
    "dokuz",
)
_TENS: tuple[str, ...] = (
    "",
    "on",
    "yirmi",
    "otuz",
    "kırk",
    "elli",
    "altmış",
    "yetmiş",
    "seksen",
    "doksan",
)
_SCALES: tuple[tuple[int, str], ...] = ((10**9, "milyar"), (10**6, "milyon"), (1000, "bin"))

#: Digit runs at least this long are read digit by digit instead of as one number.
MAX_NUMBER_DIGITS = 12

_DIGIT_RUN_RE = re.compile(r"\d+|\D+")


# --------------------------------------------------------------------------------------------
# latex_v1
# --------------------------------------------------------------------------------------------
def latex_tokens(text: str) -> list[str]:
    """Split ``text`` into ``latex_v1`` tokens (rule 2), without normalising it first.

    Args:
        text: Already surface-normalised LaTeX.

    Returns:
        Commands (``\\frac``), digit runs (``12``) and single characters, whitespace dropped.

    Example:
        >>> latex_tokens("x^{12}")
        ['x', '^', '{', '12', '}']
    """
    return _LATEX_TOKEN_RE.findall(text)


def _is_operand(token: str) -> bool:
    """Whether ``token`` stands on its own — as a script's argument, or inside braces.

    A brace or another script operator does not: ``x^{2}`` is already braced, and ``^^`` is
    malformed input whose second ``^`` has no operand at all. Wrapping those would make the
    normaliser non-idempotent (``^^`` -> ``^{^}`` -> ``^{^{}}``), which the property test
    ``test_latex_v1_is_idempotent`` found.

    Rules 3 and 4 both ask this question, from opposite sides: rule 3 refuses to *add* braces
    around a non-operand, rule 4 refuses to *remove* them from one. The second half was missing
    at first, and the same property test found that too (``"{^}x"``).
    """
    return token not in {"{", "}"} and token not in _SCRIPTS


def _canonical_scripts(tokens: list[str]) -> list[str]:
    """Rule 3: brace a single-token superscript or subscript (``x^2`` -> ``x^{2}``)."""
    out: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        out.append(token)
        if token in _SCRIPTS and index + 1 < len(tokens) and _is_operand(tokens[index + 1]):
            out.extend(("{", tokens[index + 1], "}"))
            index += 2
            continue
        index += 1
    return out


def _flatten_once(tokens: list[str]) -> list[str]:
    """One flattening pass over ``tokens`` (see :func:`_flatten_groups`)."""
    out: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        # ``_is_operand`` is the same test rule 3 uses: only a group wrapping a single *operand*
        # is redundant. ``{^}`` is not — dropping its braces would let the operator adopt the
        # following token, so a second pass would produce ``^ { x }`` from ``^ x``.
        is_single_group = (
            token == "{"
            and index + 2 < len(tokens)
            and _is_operand(tokens[index + 1])
            and tokens[index + 2] == "}"
        )
        if is_single_group and (not out or out[-1] not in _SCRIPTS):
            out.append(tokens[index + 1])
            index += 3
            continue
        out.append(token)
        index += 1
    return out


def _flatten_groups(tokens: list[str]) -> list[str]:
    """Rule 4: drop a brace group holding exactly one operand, unless it is a script argument.

    Applied to a fixed point, because flattening an inner group can turn its parent into a
    single-token group: ``{{x}}`` -> ``{x}`` -> ``x``. One pass would leave ``{x}``, and running
    the normaliser again would then keep shrinking it — the non-idempotence the property test
    ``test_latex_v1_is_idempotent`` found. Termination is guaranteed: a pass that changes
    anything removes at least two tokens.
    """
    current = tokens
    while True:
        nxt = _flatten_once(current)
        if nxt == current:
            return current
        current = nxt


def _latex_v1_pass(text: str) -> str:
    """One ``latex_v1`` pass: surface repair, tokenise, canonical scripts, flatten, join."""
    surface = normalize_latex(text).text
    return " ".join(_flatten_groups(_canonical_scripts(latex_tokens(surface))))


def latex_v1(s: str) -> str:
    """Normalise LaTeX for metric comparison (frozen; docs/ml/degerlendirme.md §1).

    Args:
        s: Raw LaTeX (an OCR hypothesis, a dataset label, a printed answer).

    Returns:
        The tokens of the normalised expression joined by single spaces. The result is
        idempotent and is what :func:`aicls.eval.metrics.cer`, ``ter`` and ``exact_match``
        compare (CER drops the spaces again; TER uses them as the token boundary).

    Example:
        >>> latex_v1(r"\\dfrac{x^2}{\\left(y+1\\right)}")
        '\\\\frac { x ^ { 2 } } { ( y + 1 ) }'
    """
    if not s or not s.strip():
        return ""
    text = s
    for _ in range(MAX_LATEX_PASSES):
        result = _latex_v1_pass(text)
        if result == text:
            return result
        text = result
    _log.warning("eval.normalizers.latex_v1_no_fixed_point", passes=MAX_LATEX_PASSES)
    return text


# --------------------------------------------------------------------------------------------
# tr_v1
# --------------------------------------------------------------------------------------------
def turkish_number_words(value: int) -> list[str]:
    """Read a non-negative integer as Turkish words (frozen table).

    Args:
        value: The number to read (``>= 0``).

    Returns:
        The words, e.g. ``2026`` -> ``["iki", "bin", "yirmi", "altı"]``. ``1000`` is ``["bin"]``
        and ``100`` is ``["yüz"]``, following ordinary Turkish usage.

    Raises:
        InputValidationError: If ``value`` is negative.
    """
    if value < 0:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR, details={"reason": "negative_number", "value": value}
        )
    if value == 0:
        return [_UNITS[0]]
    words: list[str] = []
    remainder = value
    for scale, name in _SCALES:
        quotient, remainder = divmod(remainder, scale)
        if not quotient:
            continue
        if not (quotient == 1 and name == "bin"):
            words.extend(_below_thousand(quotient))
        words.append(name)
    words.extend(_below_thousand(remainder))
    return words


def _below_thousand(value: int) -> list[str]:
    """Words for ``0 <= value < 1000`` (``0`` contributes nothing)."""
    words: list[str] = []
    hundreds, rest = divmod(value, 100)
    if hundreds == 1:
        words.append("yüz")
    elif hundreds > 1:
        words.extend((_UNITS[hundreds], "yüz"))
    tens, units = divmod(rest, 10)
    if tens:
        words.append(_TENS[tens])
    if units:
        words.append(_UNITS[units])
    return words


def _expand_digits(token: str) -> list[str]:
    """Rule 4: a digit run becomes number words, the rest of the token stays a token.

    ``"0dan"`` -> ``["sıfır", "dan"]``; no suffix is attached to the number word, because
    attaching one would be a second, undocumented rule.
    """
    if not any(character.isdigit() for character in token):
        return [token]
    words: list[str] = []
    for part in _DIGIT_RUN_RE.findall(token):
        if not part[0].isdigit():
            words.append(part)
        elif len(part) > MAX_NUMBER_DIGITS:
            words.extend(_UNITS[int(digit)] for digit in part)
        else:
            words.extend(turkish_number_words(int(part)))
    return words


def _strip_punctuation(text: str) -> str:
    """Rule 3: delete every Unicode punctuation character (apostrophes and hyphens included)."""
    return "".join(
        character for character in text if not unicodedata.category(character).startswith("P")
    )


def tr_words(s: str) -> list[str]:
    """Return the ``tr_v1`` words of ``s`` (the unit WER counts)."""
    folded = _strip_punctuation(tr_casefold(unicodedata.normalize("NFC", s)))
    words: list[str] = []
    for token in folded.split():
        words.extend(word for word in _expand_digits(token) if word)
    return words


def tr_v1(s: str) -> str:
    """Normalise Turkish text for metric comparison (frozen; docs/ml/degerlendirme.md §1).

    Exactly five rules, in this order: NFC, Turkish casefold (``İ``->``i``, ``I``->``ı``),
    punctuation deletion, digit runs to Turkish number words, whitespace squeeze. **No lexicon
    folding** (ADR-0015).

    Args:
        s: Raw Turkish text (an ASR hypothesis or a reference transcript).

    Returns:
        The normalised text; ``" ".join(tr_words(s))``.

    Example:
        >>> tr_v1("0'dan 1'e KADAR")
        'sıfır dan bir e kadar'
    """
    return " ".join(tr_words(s))


# --------------------------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------------------------
def _identity(s: str) -> str:
    """The ``none`` normaliser: the raw column every report prints next to the normalised one."""
    return s


#: Normaliser name -> function. ``"none"`` is the raw column, not the absence of a rule.
NORMALIZERS = {"latex_v1": latex_v1, "tr_v1": tr_v1, "none": _identity}


def apply_norm(s: str, norm: str) -> str:
    """Apply the normaliser called ``norm``.

    Args:
        s: Text to normalise.
        norm: ``"latex_v1"``, ``"tr_v1"`` or ``"none"``.

    Returns:
        The normalised text.

    Raises:
        InputValidationError: If ``norm`` is not a known normaliser version. An unknown name is
            never silently treated as ``"none"``: that would publish an unlabelled number.
    """
    normaliser = NORMALIZERS.get(norm)
    if normaliser is None:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "unknown_normalizer", "norm": norm, "known": sorted(NORMALIZERS)},
        )
    return normaliser(s)


__all__ = [
    "MAX_LATEX_PASSES",
    "MAX_NUMBER_DIGITS",
    "NORMALIZERS",
    "NORMALIZER_VERSIONS",
    "apply_norm",
    "latex_tokens",
    "latex_v1",
    "tr_v1",
    "tr_words",
    "turkish_number_words",
]
