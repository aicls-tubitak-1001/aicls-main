"""Turkish normalisation with span tracking (docs/mimari/metin-komut-katmani.md §1).

``normalize_text`` is the single entry point every other part of the layer builds on. It turns
raw Turkish -- typed or straight out of the ASR decoder -- into a canonical token stream:

1. **NFC** and apostrophe splitting (``x'in`` → ``x`` ``in``);
2. **Turkish casefold** through :func:`aicls.core.turkish.tr_casefold` (``İ→i``, ``I→ı``; the
   default Unicode mapping would produce ``i`` + combining dot and break every later lookup);
3. **tokenisation** into words, numbers and operator symbols, each keeping the character span it
   occupied in the *original* string, so the UI can highlight what filled a slot;
4. **variant folding** (``türef`` → ``türev``, ``sinüs`` → ``sin``, ``+`` → ``artı``);
5. **suffix splitting** against the lexicon vocabulary (``türevini`` → ``türev`` ``ini``);
6. **phrase folding** (``mutlak değer`` → ``mutlakdeğer``, ``parantez aç`` → ``(``);
7. **filler removal** (``lütfen``, ``yani``, ...);
8. **number words to digits** (``yüz yirmi üç`` → ``123``, ``iki buçuk`` → ``2.5``).

The result is idempotent on its own rendering: ``normalize_text(n.text).text == n.text``
(``test_property_normalizer_idempotent``). Note that *token kinds* are not idempotent -- a
suffix token re-read from the flat string is an ordinary word again -- which is why the slot
extractor always slices the token list, never the string.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Literal, TypeAlias
import unicodedata

from aicls.core.errors import InputValidationError
from aicls.core.turkish import tr_casefold
from aicls.nlu.lexicon import Lexicon, load_lexicon
from aicls.nlu.numbers import NumberTable, format_number, load_numbers, to_number

#: What a token is: an ordinary word, an inflectional suffix split off a word, or a number.
TokenKind: TypeAlias = Literal["word", "suffix", "number"]

#: Apostrophe-like characters that separate a Turkish proper-noun suffix.
APOSTROPHES = "'’‘´`"

_TOKEN_RE = re.compile(
    r"(?P<apos>['’‘´`])"
    r"|(?P<num>\d+(?:[.,]\d+)?)"
    r"|(?P<word>[^\W\d_]+)"
    r"|(?P<sym>[+\-*/=^×÷()])"
)


@dataclass(frozen=True, slots=True)
class Token:
    """One normalised token with its span into the original text.

    Attributes:
        text: Canonical form (folded, split, digitised).
        kind: ``word``, ``suffix`` or ``number``.
        start: Start offset in the NFC-normalised original text.
        end: End offset (exclusive) in the NFC-normalised original text.
        raw: The original surface the token came from.
    """

    text: str
    kind: TokenKind
    start: int
    end: int
    raw: str


@dataclass(frozen=True, slots=True)
class NormalizedText:
    """The normalised form of one utterance.

    Attributes:
        original: The NFC-normalised input, unchanged otherwise.
        tokens: Canonical token strings, in order.
        spans: ``(start, end)`` character spans into :attr:`original`, parallel to
            :attr:`tokens`.
        kinds: Token kinds, parallel to :attr:`tokens`.
    """

    original: str
    tokens: tuple[str, ...]
    spans: tuple[tuple[int, int], ...]
    kinds: tuple[TokenKind, ...]

    @property
    def text(self) -> str:
        """The canonical tokens joined by single spaces -- what patterns match against."""
        return " ".join(self.tokens)

    @property
    def words(self) -> tuple[str, ...]:
        """Only the non-suffix tokens (what keyword scoring looks at)."""
        return tuple(
            token for token, kind in zip(self.tokens, self.kinds, strict=True) if kind != "suffix"
        )

    @property
    def items(self) -> tuple[Token, ...]:
        """The tokens as :class:`Token` records."""
        return tuple(
            Token(text=text, kind=kind, start=start, end=end, raw=self.original[start:end])
            for text, kind, (start, end) in zip(self.tokens, self.kinds, self.spans, strict=True)
        )

    @property
    def offsets(self) -> tuple[int, ...]:
        """Start offset of each token inside :attr:`text` (for regex-span → token mapping)."""
        out: list[int] = []
        cursor = 0
        for token in self.tokens:
            out.append(cursor)
            cursor += len(token) + 1
        return tuple(out)

    def token_range(self, start: int, end: int) -> tuple[int, int]:
        """Map a character range in :attr:`text` to a half-open token index range.

        Args:
            start: Start offset inside :attr:`text`.
            end: End offset (exclusive) inside :attr:`text`.

        Returns:
            ``(first, last_exclusive)`` token indices covering the range.
        """
        offsets = self.offsets
        first = len(self.tokens)
        last = 0
        for index, offset in enumerate(offsets):
            token_end = offset + len(self.tokens[index])
            if token_end <= start or offset >= end:
                continue
            first = min(first, index)
            last = max(last, index + 1)
        if first > last:
            return 0, 0
        return first, last

    def slice_math(self, first: int, last: int) -> str:
        """Join tokens ``[first, last)`` dropping suffix tokens (input for the math grammar)."""
        return " ".join(
            token
            for token, kind in zip(self.tokens[first:last], self.kinds[first:last], strict=True)
            if kind != "suffix"
        )

    def slice_text(self, first: int, last: int) -> str:
        """Join tokens ``[first, last)`` including suffix tokens."""
        return " ".join(self.tokens[first:last])

    def source_span(self, first: int, last: int) -> tuple[int, int]:
        """Character span in :attr:`original` covered by tokens ``[first, last)``."""
        if first >= last or not self.spans:
            return 0, 0
        return self.spans[first][0], self.spans[last - 1][1]


def _pre_clean(text: str) -> str:
    """NFC-normalise; apostrophes are kept and handled by the tokeniser."""
    return unicodedata.normalize("NFC", text)


def _raw_tokens(source: str) -> list[tuple[str, str, int, int]]:
    """Split ``source`` into ``(kind, surface, start, end)`` tuples."""
    out: list[tuple[str, str, int, int]] = []
    for match in _TOKEN_RE.finditer(source):
        kind = match.lastgroup or "word"
        out.append((kind, match.group(), match.start(), match.end()))
    return out


def _digit_token(surface: str) -> str:
    """Normalise a numeric literal: Turkish decimal comma becomes a dot."""
    return surface.replace(",", ".")


def _analyse_word(folded: str, lexicon: Lexicon) -> list[tuple[str, TokenKind]] | None:
    """Fold one already-casefolded word and split one suffix; ``None`` when unrecognised."""
    if lexicon.knows(folded):
        return [(folded, "word")]
    stem, suffix = lexicon.split_suffix(folded)
    if suffix is None:
        return None
    return [(lexicon.canonical(stem), "word"), (suffix, "suffix")]


def _analyse(surface: str, lexicon: Lexicon) -> list[tuple[str, TokenKind]]:
    """Fold one word through the variant table, splitting a suffix or a lost word boundary.

    An unrecognised token gets one last chance through :meth:`aicls.nlu.lexicon.Lexicon.resegment`
    -- ``"türevtanımını"`` is what a decoder writes when the speaker does not pause, and without
    this pass the whole utterance is lost (KNOWN_ISSUES PRIORITY 2).
    """
    folded = lexicon.canonical(tr_casefold(surface))
    analysed = _analyse_word(folded, lexicon)
    if analysed is not None:
        return analysed
    pieces = lexicon.resegment(folded)
    if pieces is None:
        return [(folded, "word")]
    out: list[tuple[str, TokenKind]] = []
    for piece in pieces:
        canonical = lexicon.canonical(piece)
        out.extend(_analyse_word(canonical, lexicon) or [(canonical, "word")])
    return out


#: Shortest detached suffix that may stand on its own. One-letter tokens are excluded because
#: "a" and "e" are a variable name and the constant e as often as they are case endings.
MIN_DETACHED_SUFFIX = 2


def _is_detached_suffix(
    token: str,
    suffixes: frozenset[str],
    after_apostrophe: bool,
    staged: Sequence[tuple[str, TokenKind, int, int]],
) -> bool:
    """Whether ``token`` is a case ending that was written as its own word.

    True right after an apostrophe (``x'in``), and true for a standalone ending of at least
    :data:`MIN_DETACHED_SUFFIX` characters that follows a content token (``x kare nin``). The
    İP-3 reading scenarios show both constantly: a slow dictation puts a pause before the ending
    and the decoder writes it separately.
    """
    if token not in suffixes:
        return False
    if after_apostrophe:
        return True
    if len(token) < MIN_DETACHED_SUFFIX:
        return False
    return bool(staged) and staged[-1][1] != "suffix"


def _fold_numbers(
    tokens: list[tuple[str, TokenKind, int, int]], numbers: NumberTable
) -> list[tuple[str, TokenKind, int, int]]:
    """Collapse runs of cardinal number words (and digits) into one numeric token.

    The run is shortened until it *is* a numeral (longest match first). ``"dört sıfır"`` is not a
    Turkish numeral, so it folds as ``4`` followed by ``0`` rather than being summed into a
    single ``4`` -- the silent token loss behind the wrong answer to
    ``"x kare eksi dört sıfır denklemini çöz"`` (KNOWN_ISSUES PRIORITY 1).
    """
    out: list[tuple[str, TokenKind, int, int]] = []
    index = 0
    while index < len(tokens):
        text, kind, start, end = tokens[index]
        if kind == "suffix" or not numbers.is_number_word(text):
            out.append((text, kind, start, end))
            index += 1
            continue
        run_end = index + 1
        while (
            run_end < len(tokens)
            and tokens[run_end][1] != "suffix"
            and numbers.is_number_word(tokens[run_end][0])
        ):
            run_end += 1
        folded = _fold_longest_numeral(tokens, index, run_end, numbers)
        if folded is None:
            # Not a numeral at all, not even on its own ("virgül"): leave it as written.
            out.append((text, kind, start, end))
            index += 1
            continue
        token, consumed = folded
        out.append(token)
        index = consumed
    return out


def _fold_longest_numeral(
    tokens: list[tuple[str, TokenKind, int, int]],
    start_index: int,
    run_end: int,
    numbers: NumberTable,
) -> tuple[tuple[str, TokenKind, int, int], int] | None:
    """Fold the longest prefix of ``tokens[start_index:run_end]`` that is a numeral.

    Returns:
        ``(token, next_index)`` for the folded numeral, or ``None`` when not even the single
        leading token reads as one.
    """
    for stop in range(run_end, start_index, -1):
        words = [item[0] for item in tokens[start_index:stop]]
        try:
            value = to_number(words, table=numbers)
        except InputValidationError:
            continue
        start = tokens[start_index][2]
        end = tokens[stop - 1][3]
        return (format_number(value), "number", start, end), stop
    return None


def normalize_text(
    text: str,
    *,
    lexicon: Lexicon | None = None,
    numbers: NumberTable | None = None,
) -> NormalizedText:
    """Normalise Turkish text into the canonical token stream.

    Args:
        text: Raw utterance (typed or ASR output).
        lexicon: Lexicon to fold with; the packaged one when omitted.
        numbers: Numeral table; the packaged one when omitted.

    Returns:
        The :class:`NormalizedText`; empty input yields an empty token tuple.
    """
    lex = lexicon or load_lexicon()
    num = numbers or load_numbers()
    source = _pre_clean(text)

    suffix_set = frozenset(lex.suffixes)
    staged: list[tuple[str, TokenKind, int, int]] = []
    after_apostrophe = False
    for group, surface, start, end in _raw_tokens(source):
        if group == "apos":
            after_apostrophe = True
            continue
        if group == "num":
            staged.append((_digit_token(surface), "number", start, end))
            after_apostrophe = False
            continue
        if group == "sym":
            staged.append((lex.canonical(surface), "word", start, end))
            after_apostrophe = False
            continue
        folded = tr_casefold(surface)
        if _is_detached_suffix(folded, suffix_set, after_apostrophe, staged):
            # "x'in", and "x kare nin" -- a slow dictation (or the ASR decoder) writes the case
            # ending as its own word. It must be recognised *before* the token is analysed,
            # otherwise "sini" would be split into the sine function plus an "i".
            staged.append((folded, "suffix", start, end))
            after_apostrophe = False
            continue
        after_apostrophe = False
        for token_text, token_kind in _analyse(surface, lex):
            staged.append((token_text, token_kind, start, end))

    staged = [item for item in staged if not (item[1] == "word" and item[0] in lex.fillers)]

    phrased: list[tuple[str, TokenKind, int, int]] = []
    for phrase, first, last in lex.fold_phrases([item[0] for item in staged]):
        phrase_kind: TokenKind = staged[first][1] if first == last else "word"
        phrased.append((phrase, phrase_kind, staged[first][2], staged[last][3]))

    final = _fold_numbers(phrased, num)
    return NormalizedText(
        original=source,
        tokens=tuple(item[0] for item in final),
        spans=tuple((item[2], item[3]) for item in final),
        kinds=tuple(item[1] for item in final),
    )


def normalized_words(text: str) -> tuple[str, ...]:
    """Convenience wrapper returning only the non-suffix tokens of ``text``."""
    return normalize_text(text).words


__all__ = [
    "APOSTROPHES",
    "NormalizedText",
    "Token",
    "TokenKind",
    "normalize_text",
    "normalized_words",
]
