"""The spoken-mathematics lexicon: variant folding, phrase folding and suffix splitting.

Turkish is agglutinative, so a command inventory that spelled out every inflected form would be
unmaintainable ("türev", "türevi", "türevini", "türevinin", "türevlerini", ...). Instead the
normaliser splits **one** inflectional suffix off a token *when the remaining stem is a word the
lexicon knows* -- ``"türevini"`` → ``türev`` + ``ini`` -- and the pattern compiler then tolerates
an optional suffix token after every literal (:mod:`aicls.nlu.patterns`).

The vocabulary guard is what keeps this safe: ``"kadar"`` is left alone because ``"ka"`` is not a
word, while ``"sonucunu"`` becomes ``sonuç`` + ``unu`` through the consonant-mutation table
(``c→ç``, ``ğ→k``, ``b→p``, ``d→t``), which is the regular Turkish alternation.

Suffix inventory: :data:`aicls.core.turkish.TR_SUFFIXES` (all vowel-harmony variants per family)
plus the longer sequences listed under ``extra_suffixes`` in ``inventory/lexicon.yaml``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

from aicls.core.turkish import TR_SUFFIXES, strip_diacritics, tr_casefold
from aicls.nlu.errors import InventoryError
from aicls.nlu.numbers import load_numbers
from aicls.nlu.resources import LEXICON_RESOURCE, load_yaml, resource_path

#: Regular Turkish final-consonant alternation applied when testing a candidate stem.
CONSONANT_MUTATION: dict[str, str] = {"ğ": "k", "c": "ç", "b": "p", "d": "t"}

#: A stem shorter than this is never produced by suffix splitting.
MIN_STEM = 2

#: Longest token :meth:`Lexicon.resegment` will try to re-cut. Longer tokens are decoder noise,
#: not two words run together, and the search cost grows with the square of the length.
MAX_RESEGMENT_LEN = 48

#: Most pieces a recovered segmentation may have. A decoder drops one or two boundaries in a
#: burst of fast speech; a five-way split is the search finding a coincidence, not a boundary.
MAX_RESEGMENT_PIECES = 4

#: A recovered piece must be at least this long, so that word-boundary recovery cannot degrade
#: into "every string is a product of single-letter variables".
MIN_RESEGMENT_PIECE = 2

#: How many single-letter pieces a segmentation may contain: exactly one, because a variable is
#: the only single-letter token spoken mathematics has ("xkarenin" -> "x" + "karenin").
MAX_SINGLE_LETTER_PIECES = 1

#: The letters a single-letter piece may be. Turkish-only letters are excluded: they are case
#: endings far more often than they are variable names.
_LETTER_RE = re.compile(r"^[a-z]$")


@dataclass(frozen=True, slots=True)
class ScanRule:
    """A lexical scan rule for a slot that is not captured by a pattern.

    Attributes:
        words: Consecutive canonical tokens that trigger the rule.
        value: The slot value the rule yields (``"+"``, ``"Definition"``).
    """

    words: tuple[str, ...]
    value: str


@dataclass(frozen=True, slots=True)
class Lexicon:
    """Everything ``inventory/lexicon.yaml`` declares, in lookup-ready form.

    Attributes:
        version: Lexicon version string.
        variants: Single-token surface → canonical token.
        phrases: Multi-word canonical sequence → single canonical token.
        vocabulary: Stems the suffix splitter may produce.
        fillers: Tokens dropped from the stream.
        suffixes: Suffix strings, longest first.
        scan: Scan rules per slot name (``direction``, ``unit_type``).
        ascii_index: Diacritic-free spelling -> the known word it unambiguously stands for.
        command_verbs: Imperative command verbs; word-boundary recovery never puts a
            single-letter piece (a variable) directly in front of one.
    """

    version: str
    variants: Mapping[str, str]
    phrases: Mapping[tuple[str, ...], str]
    vocabulary: frozenset[str]
    fillers: frozenset[str]
    suffixes: tuple[str, ...]
    scan: Mapping[str, tuple[ScanRule, ...]]
    ascii_index: Mapping[str, str] = field(default_factory=dict)
    command_verbs: frozenset[str] = frozenset()

    @property
    def max_phrase_len(self) -> int:
        """Length of the longest multi-word phrase (0 when there is none)."""
        return max((len(key) for key in self.phrases), default=0)

    def canonical(self, token: str) -> str:
        """Fold one token to its canonical spelling.

        Three steps, in order: the explicit variant table, the known-word check, and finally the
        diacritic-free index -- ASR decoders and phone keyboards routinely write ``sifir`` for
        ``sıfır`` and ``esittir`` for ``eşittir``, and :func:`aicls.core.turkish.strip_diacritics`
        recovers the intended word whenever exactly one known word matches.
        """
        folded = self.variants.get(token)
        if folded is not None:
            return folded
        if token in self.vocabulary:
            return token
        recovered = self.ascii_index.get(strip_diacritics(token))
        if recovered is not None and recovered != token:
            return self.variants.get(recovered, recovered)
        return token

    def knows(self, token: str) -> bool:
        """Whether ``token`` is a stem the splitter may leave standing."""
        return token in self.vocabulary

    def split_suffix(self, token: str) -> tuple[str, str | None]:
        """Split one inflectional suffix off ``token`` when the stem is a known word.

        Args:
            token: A canonicalised token.

        Returns:
            ``(stem, suffix)``; ``suffix`` is ``None`` when nothing was split off.
        """
        if token in self.vocabulary:
            return token, None
        for suffix in self.suffixes:
            if not token.endswith(suffix):
                continue
            stem = token[: -len(suffix)]
            if len(stem) < MIN_STEM:
                continue
            for candidate in self._stem_candidates(stem):
                if candidate in self.vocabulary:
                    return candidate, suffix
        return token, None

    def _stem_candidates(self, stem: str) -> tuple[str, ...]:
        """``stem``, its variant fold, its diacritic-free recovery and its mutated forms."""
        folded = self.variants.get(stem, stem)
        recovered = self.ascii_index.get(strip_diacritics(stem), stem)
        candidates = [stem, folded, recovered, self.variants.get(recovered, recovered)]
        for base in (stem, folded):
            replacement = CONSONANT_MUTATION.get(base[-1:])
            if replacement is not None:
                candidates.append(base[:-1] + replacement)
        seen: dict[str, None] = {}
        for candidate in candidates:
            seen.setdefault(candidate, None)
        return tuple(seen)

    def head_token(self, token: str) -> str:
        """The first canonical token ``token`` analyses into (``"karenin"`` → ``"kare"``)."""
        folded = self.canonical(token)
        if self.knows(folded):
            return folded
        stem, suffix = self.split_suffix(folded)
        return self.canonical(stem) if suffix is not None else folded

    def understands(self, token: str) -> bool:
        """Whether ``token`` is a known word, or a known stem plus one inflectional suffix.

        This is the recogniser word-boundary recovery is built on: a piece the normaliser would
        turn into real tokens anyway.
        """
        folded = self.canonical(token)
        if self.knows(folded):
            return True
        return self.split_suffix(folded)[1] is not None

    def resegment(self, token: str) -> tuple[str, ...] | None:
        """Recover a lost word boundary inside ``token`` (``türevtanımını`` → two words).

        Word-boundary loss is what a real ASR decoder produces on fast Turkish dictation, and it
        was this layer's only weak perturbation (``join_words`` accuracy 0.276, KNOWN_ISSUES
        PRIORITY 2). The search is a longest-match pass over the lexicon: every piece must be a
        word :meth:`understands` accepts, at most :data:`MAX_SINGLE_LETTER_PIECES` piece may be a
        single latin letter (a variable), and at most :data:`MAX_RESEGMENT_PIECES` pieces may be
        produced. A token that does not decompose that cleanly is left exactly as written --
        segmenting an out-of-scope word into command words would *invent* a command.

        Args:
            token: A canonically folded token the normaliser could not analyse.

        Returns:
            The recovered pieces (at least two), or ``None`` when the token is already understood
            or does not decompose.
        """
        if len(token) > MAX_RESEGMENT_LEN or len(token) < MIN_RESEGMENT_PIECE + 1:
            return None
        if self.understands(token):
            return None
        known: dict[str, bool] = {}

        def accepts(piece: str) -> bool:
            value = known.get(piece)
            if value is None:
                value = self.understands(piece)
                known[piece] = value
            return value

        return self._segment(token, 0, MAX_SINGLE_LETTER_PIECES, MAX_RESEGMENT_PIECES, accepts, "")

    def _segment(
        self,
        token: str,
        start: int,
        letters_left: int,
        pieces_left: int,
        accepts: Callable[[str], bool],
        previous: str,
    ) -> tuple[str, ...] | None:
        """Longest-match segmentation of ``token[start:]``; ``None`` when there is none.

        Args:
            token: The whole token being re-cut.
            start: Where this sub-problem begins.
            letters_left: How many single-letter pieces may still be produced.
            pieces_left: How many pieces may still be produced.
            accepts: Memoised :meth:`understands`.
            previous: Head token of the piece immediately to the left (``""`` at the start).
        """
        if start >= len(token):
            return ()
        if pieces_left <= 0:
            return None
        for stop in range(len(token), start, -1):
            piece = token[start:stop]
            single = len(piece) == 1
            if single:
                if letters_left <= 0 or not _LETTER_RE.match(piece):
                    continue
            elif not accepts(piece):
                continue
            rest = self._segment(
                token,
                stop,
                letters_left - int(single),
                pieces_left - 1,
                accepts,
                self.head_token(piece),
            )
            if rest is None:
                continue
            if single and not self._variable_fits(previous, rest):
                continue
            return (piece, *rest)
        return None

    def _variable_fits(self, previous: str, rest: tuple[str, ...]) -> bool:
        """Whether a single-letter piece really reads as a variable where it was cut.

        One rule, from how spoken mathematics is actually said: a variable is never adjacent to a
        bare imperative. "x integralini al" and "karekökx" are real; "kaç" is not "k" times the
        imperative "aç", and "tekraret" is not "tekrare" times ``t``.
        """
        neighbour = self.head_token(rest[0]) if rest else previous
        return bool(neighbour) and neighbour not in self.command_verbs

    def fold_phrases(self, tokens: Sequence[str]) -> list[tuple[str, int, int]]:
        """Fold multi-word phrases in ``tokens``.

        Args:
            tokens: Canonical tokens.

        Returns:
            Triples ``(token, first_index, last_index)``; a folded phrase keeps the index range
            it covered so the caller can rebuild the character span.
        """
        out: list[tuple[str, int, int]] = []
        index = 0
        width = self.max_phrase_len
        while index < len(tokens):
            matched = False
            for size in range(min(width, len(tokens) - index), 1, -1):
                key = tuple(tokens[index : index + size])
                replacement = self.phrases.get(key)
                if replacement is not None:
                    out.append((replacement, index, index + size - 1))
                    index += size
                    matched = True
                    break
            if not matched:
                out.append((tokens[index], index, index))
                index += 1
        return out


def _str_map(data: Mapping[str, Any], key: str) -> dict[str, str]:
    section = data.get(key, {}) or {}
    if not isinstance(section, dict):
        raise InventoryError("lexicon_section_invalid", details={"section": key})
    return {tr_casefold(str(k)): tr_casefold(str(v)) for k, v in section.items()}


def _str_list(data: Mapping[str, Any], key: str) -> list[str]:
    section = data.get(key, []) or []
    if not isinstance(section, list):
        raise InventoryError("lexicon_section_invalid", details={"section": key})
    return [tr_casefold(str(item)) for item in section]


def _scan_rules(data: Mapping[str, Any]) -> dict[str, tuple[ScanRule, ...]]:
    section = data.get("scan", {}) or {}
    if not isinstance(section, dict):
        raise InventoryError("lexicon_section_invalid", details={"section": "scan"})
    out: dict[str, tuple[ScanRule, ...]] = {}
    for slot, rules in section.items():
        if not isinstance(rules, list):
            raise InventoryError("lexicon_section_invalid", details={"section": f"scan.{slot}"})
        parsed: list[ScanRule] = []
        for rule in rules:
            if not isinstance(rule, dict) or "words" not in rule or "value" not in rule:
                raise InventoryError("scan_rule_invalid", details={"slot": str(slot)})
            words = rule["words"]
            if not isinstance(words, list) or not words:
                raise InventoryError("scan_rule_invalid", details={"slot": str(slot)})
            parsed.append(
                ScanRule(
                    words=tuple(tr_casefold(str(word)) for word in words),
                    value=str(rule["value"]),
                )
            )
        # Longest rule first so "sağ taraf" wins over a hypothetical single-word "sağ".
        parsed.sort(key=lambda rule: len(rule.words), reverse=True)
        out[str(slot)] = tuple(parsed)
    return out


def _ascii_index(words: Iterable[str]) -> dict[str, str]:
    """Map each *unambiguous* diacritic-free spelling to the word it stands for.

    ``sıfır`` and ``şekil`` both survive (``sifir``, ``sekil``); a spelling claimed by two
    different words is dropped, because guessing there would silently change the user's meaning.
    """
    owners: dict[str, set[str]] = {}
    for word in words:
        key = strip_diacritics(word)
        if key != word:
            owners.setdefault(key, set()).add(word)
    return {key: next(iter(names)) for key, names in owners.items() if len(names) == 1}


def _suffix_inventory(extra: Iterable[str]) -> tuple[str, ...]:
    """All suffix strings, longest first (core families + the lexicon's extras)."""
    pool = {suffix for family in TR_SUFFIXES.values() for suffix in family}
    pool.update(extra)
    return tuple(sorted(pool, key=lambda item: (-len(item), item)))


@lru_cache(maxsize=2)
def load_lexicon(path: str | None = None) -> Lexicon:
    """Load (and cache) the lexicon.

    The numeral vocabulary from ``inventory/numbers.yaml`` is merged into
    :attr:`Lexicon.vocabulary` so that ``"sıfırdan"`` splits into ``sıfır`` + ``dan``.

    Args:
        path: Optional override; the packaged ``inventory/lexicon.yaml`` when omitted.

    Returns:
        The parsed :class:`Lexicon`.

    Raises:
        aicls.nlu.errors.InventoryError: If a section is malformed.
    """
    resolved = Path(path) if path else resource_path(LEXICON_RESOURCE)
    data = load_yaml(resolved)
    numbers = load_numbers()
    variants = _str_map(data, "variants")
    phrase_map: dict[tuple[str, ...], str] = {}
    for key, value in _str_map(data, "phrases").items():
        parts = tuple(key.split())
        if len(parts) < 2:
            raise InventoryError("phrase_too_short", details={"phrase": key})
        phrase_map[parts] = value
    vocabulary = set(_str_list(data, "vocabulary"))
    vocabulary.update(numbers.words)
    vocabulary.update(numbers.ordinals)
    vocabulary.update(phrase_map.values())
    vocabulary.update(variants.values())
    return Lexicon(
        command_verbs=frozenset(_str_list(data, "command_verbs")),
        ascii_index=_ascii_index(vocabulary | set(variants)),
        version=str(data.get("version", "0")),
        variants=variants,
        phrases=phrase_map,
        vocabulary=frozenset(vocabulary),
        fillers=frozenset(_str_list(data, "fillers")),
        suffixes=_suffix_inventory(_str_list(data, "extra_suffixes")),
        scan=_scan_rules(data),
    )


__all__ = [
    "CONSONANT_MUTATION",
    "MAX_RESEGMENT_LEN",
    "MAX_RESEGMENT_PIECES",
    "MAX_SINGLE_LETTER_PIECES",
    "MIN_RESEGMENT_PIECE",
    "MIN_STEM",
    "Lexicon",
    "ScanRule",
    "load_lexicon",
]
