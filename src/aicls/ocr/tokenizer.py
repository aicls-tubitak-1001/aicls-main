"""LaTeX tokenizer: token inventory, vocabulary, encode/decode and canonical spacing.

Tokens are the units an encoder-decoder emits: one token per LaTeX command (``\\frac``), per
control symbol (``\\,``), per digit and per remaining non-space character. Whitespace is *not* a
token; :func:`join_tokens` re-inserts exactly the spaces LaTeX needs (only after a command word
followed by a letter), which makes ``decode(encode(s))`` a canonical normal form and the
roundtrip on canonical input the identity.

Vocabulary files are plain JSON (``vocab.json`` in every checkpoint directory, see
docs/ml/egitim-altyapisi.md §5) and always start with the four special tokens.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from aicls.core import messages_tr
from aicls.core.errors import NotFoundError
from aicls.ocr.curriculum import CURRICULUM_LATEX, DEFAULT_EXTRA_TOKENS
from aicls.ocr.types import OcrInputError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

#: Special tokens, always at ids 0..3 of every vocabulary.
SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<s>", "</s>", "<unk>")
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

#: Longest LaTeX string the tokenizer accepts (mirrors ``aicls.core.latex_safety``).
MAX_LATEX_LENGTH = 2048

# command word (\frac) | control symbol (\, \{ \\) | any single non-space character
_TOKEN_RE = re.compile(r"\\[a-zA-Z]+|\\[^a-zA-Z\s]|[^\s]")
_COMMAND_RE = re.compile(r"^\\[a-zA-Z]+$")


def tokenize_latex(latex: str) -> list[str]:
    """Split ``latex`` into tokens, dropping whitespace.

    Args:
        latex: Any LaTeX fragment (math mode).

    Returns:
        The token list; ``[]`` for empty or whitespace-only input.
    """
    return _TOKEN_RE.findall(latex)


def join_tokens(tokens: Sequence[str]) -> str:
    """Join tokens back into canonical LaTeX.

    A single space is inserted only where LaTeX needs one: between a command word and a following
    letter (``\\sin`` + ``x`` -> ``\\sin x``). Everything else is concatenated, which keeps the
    output stable under repeated tokenisation.

    Args:
        tokens: Token strings.

    Returns:
        The canonical LaTeX string.
    """
    parts: list[str] = []
    previous = ""
    for token in tokens:
        if previous and _COMMAND_RE.match(previous) and token[:1].isalpha():
            parts.append(" ")
        parts.append(token)
        previous = token
    return "".join(parts)


class LatexTokenizer:
    """Bidirectional token <-> id mapping for LaTeX labels.

    Args:
        tokens: Content tokens (without the specials); duplicates are dropped, order is kept.

    Raises:
        OcrInputError: If no content token is given.
    """

    def __init__(self, tokens: Iterable[str]) -> None:
        content: list[str] = []
        seen = set(SPECIAL_TOKENS)
        for token in tokens:
            if token not in seen:
                seen.add(token)
                content.append(token)
        if not content:
            raise OcrInputError(details={"reason": "empty_vocabulary"})
        self._vocab: tuple[str, ...] = (*SPECIAL_TOKENS, *content)
        self._ids: dict[str, int] = {token: index for index, token in enumerate(self._vocab)}

    # ------------------------------------------------------------------ construction
    @classmethod
    def build(
        cls,
        labels: Iterable[str],
        *,
        min_freq: int = 1,
        extra_tokens: Iterable[str] = (),
    ) -> LatexTokenizer:
        """Build a vocabulary from label strings.

        Args:
            labels: LaTeX labels (curriculum table, MathWriting shard labels, ...).
            min_freq: Drop tokens occurring fewer than this many times.
            extra_tokens: Tokens always kept (digits, letters, operators).

        Returns:
            A tokenizer whose content tokens are sorted (deterministic across runs and label
            orders, which keeps checkpoint vocabularies comparable).
        """
        counts: Counter[str] = Counter()
        for label in labels:
            counts.update(tokenize_latex(label))
        kept = {token for token, count in counts.items() if count >= min_freq}
        kept.update(extra_tokens)
        return cls(sorted(kept))

    @classmethod
    def from_file(cls, path: Path | str) -> LatexTokenizer:
        """Load a ``vocab.json`` written by :meth:`save` (or a plain JSON token list).

        Raises:
            NotFoundError: If ``path`` does not exist.
            OcrInputError: If the file is not valid JSON or has an unexpected shape.
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise NotFoundError(
                messages_tr.msg("FILE_NOT_FOUND", path=str(file_path)),
                details={"path": str(file_path)},
            )
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise OcrInputError(
                details={"reason": "vocab_not_json", "path": str(file_path), "error": str(exc)}
            ) from exc
        if isinstance(payload, list):
            tokens = [str(token) for token in payload]
        elif isinstance(payload, dict) and isinstance(payload.get("tokens"), list):
            tokens = [str(token) for token in payload["tokens"]]
        else:
            raise OcrInputError(details={"reason": "vocab_shape", "path": str(file_path)})
        return cls([token for token in tokens if token not in SPECIAL_TOKENS])

    def save(self, path: Path | str) -> None:
        """Write the vocabulary as ``vocab.json`` (UTF-8, LF, stable key order)."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "specials": list(SPECIAL_TOKENS),
            "tokens": list(self._vocab[len(SPECIAL_TOKENS) :]),
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        file_path.write_text(text, encoding="utf-8", newline="\n")

    def to_dict(self) -> dict[str, Any]:
        """Return the serialisable vocabulary payload."""
        return {
            "schema_version": 1,
            "specials": list(SPECIAL_TOKENS),
            "tokens": list(self._vocab[len(SPECIAL_TOKENS) :]),
        }

    # ------------------------------------------------------------------ inspection
    @property
    def vocab(self) -> tuple[str, ...]:
        """The full vocabulary, specials first."""
        return self._vocab

    @property
    def vocab_size(self) -> int:
        """Number of ids, specials included."""
        return len(self._vocab)

    def token_to_id(self, token: str) -> int:
        """Id of ``token``, :data:`UNK_ID` when unknown."""
        return self._ids.get(token, UNK_ID)

    def id_to_token(self, token_id: int) -> str:
        """Token string for ``token_id``.

        Raises:
            OcrInputError: If the id is outside the vocabulary.
        """
        if not 0 <= token_id < len(self._vocab):
            raise OcrInputError(
                details={
                    "reason": "token_id_out_of_range",
                    "id": int(token_id),
                    "vocab_size": len(self._vocab),
                }
            )
        return self._vocab[token_id]

    # ------------------------------------------------------------------ encode / decode
    def tokenize(self, latex: str) -> list[str]:
        """Tokenise without mapping to ids (see :func:`tokenize_latex`)."""
        return tokenize_latex(latex)

    def encode(
        self,
        latex: str,
        *,
        add_special: bool = False,
        max_len: int | None = None,
    ) -> list[int]:
        """Encode a LaTeX string into token ids.

        Args:
            latex: The string to encode.
            add_special: Wrap the sequence in ``<s> ... </s>``.
            max_len: Truncate the result (specials included) to this length.

        Raises:
            OcrInputError: If ``latex`` is longer than :data:`MAX_LATEX_LENGTH`.
        """
        if len(latex) > MAX_LATEX_LENGTH:
            raise OcrInputError(
                details={"reason": "too_long", "length": len(latex), "limit": MAX_LATEX_LENGTH}
            )
        ids = [self.token_to_id(token) for token in tokenize_latex(latex)]
        if add_special:
            ids = [BOS_ID, *ids, EOS_ID]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        """Decode token ids back to canonical LaTeX.

        Args:
            ids: Token ids; decoding stops at the first ``</s>`` when ``skip_special`` is set.
            skip_special: Drop ``<pad>``/``<s>``/``</s>``/``<unk>`` from the output.

        Raises:
            OcrInputError: If an id is outside the vocabulary.
        """
        tokens: list[str] = []
        for token_id in ids:
            token = self.id_to_token(int(token_id))
            if skip_special:
                if token == SPECIAL_TOKENS[EOS_ID]:
                    break
                if token in SPECIAL_TOKENS:
                    continue
            tokens.append(token)
        return join_tokens(tokens)

    def normalize(self, latex: str) -> str:
        """Canonical spacing of ``latex`` (vocabulary independent, keeps unknown tokens)."""
        return join_tokens(tokenize_latex(latex))


@lru_cache(maxsize=1)
def default_tokenizer() -> LatexTokenizer:
    """Process-wide tokenizer built from the curriculum table plus the ASCII fallback tokens."""
    return LatexTokenizer.build(CURRICULUM_LATEX, extra_tokens=DEFAULT_EXTRA_TOKENS)


__all__ = [
    "BOS_ID",
    "EOS_ID",
    "MAX_LATEX_LENGTH",
    "PAD_ID",
    "SPECIAL_TOKENS",
    "UNK_ID",
    "LatexTokenizer",
    "default_tokenizer",
    "join_tokens",
    "tokenize_latex",
]
