"""Sequence-to-sequence wiring plus greedy and beam decoding.

:func:`build_seq2seq` assembles the configured encoder and decoder into one ``nn.Module`` whose
``forward(inputs, tgt_ids)`` returns ``(B, T, V)`` logits — the shape the training loop
(:mod:`aicls.ml.trainer`) feeds to cross entropy.

Inference goes through :func:`greedy_decode_batch` (``beam=1``, the latency budget of ADR-0014) or
:func:`beam_search_batch` (the reported configuration, ``eval.beam: 4``). Both decode a **whole
batch** at once and both carry incremental decoder state, which is the difference between a
usable validation pass and an unusable one:

* On the RTX 5090, run ``ocr_excerpt_wiring`` (2026-09-08, real MathWriting excerpt shards,
  ``offline_resnet34_tfm``) spent **296 s decoding 16 validation samples — 18.5 s each** while
  20 *training* steps cost 1.4 s in total. Extrapolated to the shipped ``eval.max_n_val: 2000``
  that is a 10.3-hour validation pass against a 4-minute training interval.
* The two causes were structural, not incidental: every decoding step re-ran the decoder over the
  whole prefix (``O(T^2)``), and every sample — and, in beam search, every beam of every sample —
  was a separate ``batch == 1`` forward pass, so a 32 GB card ran hundreds of sequential tiny
  kernels and was latency-bound.

:func:`greedy_decode` and :func:`beam_search` keep their per-sample signatures for existing
callers and are thin wrappers over the batched entry points; new code should call the batched
functions with the collated evaluation batch it already has.

Both families are deterministic and score hypotheses with the length-normalised sum of log
probabilities, so ``exp(score)`` is a usable confidence. Every tensor they create is placed on
``memory.device``: a bare ``torch.tensor(...)`` lands on the CPU whatever the model does, and on a
GPU run the first decoding step then raises ``RuntimeError: Expected all tensors to be on the same
device`` (reproduced on the RTX 5090, 2026-09-08). ``_Stepper`` is the single place that builds a
decoder input, so the prefix cannot drift away from the memory again.

Importing this module imports torch; see :mod:`aicls.ocr.models` for the lazy entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn

from aicls.core.logging import get_logger
from aicls.ocr.config import OcrModelConfig
from aicls.ocr.models.decoders import DecoderState, IncrementalDecoder, build_decoder
from aicls.ocr.models.encoders import build_encoder
from aicls.ocr.tokenizer import BOS_ID, EOS_ID
from aicls.ocr.types import OcrInputError

#: Default length-normalisation exponent (1.0 = mean log probability).
DEFAULT_LENGTH_PENALTY = 1.0

_log = get_logger(__name__)


class Seq2Seq(nn.Module):
    """Encoder-decoder model for both OCR families.

    Attributes:
        config: The architecture description it was built from.
        vocab_size: Output vocabulary size.
    """

    def __init__(self, cfg: OcrModelConfig, vocab_size: int) -> None:
        super().__init__()
        self.config = cfg
        self.vocab_size = int(vocab_size)
        self.encoder = build_encoder(cfg)
        self.decoder = build_decoder(cfg, vocab_size)

    def encode(self, inputs: Tensor) -> Tensor:
        """Return the ``(B, S, d_model)`` encoder memory."""
        return self.encoder(inputs)

    def decode(self, memory: Tensor, tgt_ids: Tensor) -> Tensor:
        """Return ``(B, T, V)`` logits for a target prefix."""
        return self.decoder(memory, tgt_ids)

    def init_decode_state(self, memory: Tensor, *, max_len: int) -> DecoderState:
        """Open an incremental decoding state for ``memory`` (see :mod:`.decoders`)."""
        return cast("IncrementalDecoder", self.decoder).init_state(memory, max_len=max_len)

    def decode_step(self, state: DecoderState, token_ids: Tensor) -> Tensor:
        """Consume one token per row and return the ``(rows, V)`` next-token logits."""
        return cast("IncrementalDecoder", self.decoder).step(state, token_ids)

    def forward(self, inputs: Tensor, tgt_ids: Tensor) -> Tensor:
        """Teacher-forcing forward pass: inputs plus target prefix to logits."""
        return self.decode(self.encode(inputs), tgt_ids)


def build_seq2seq(cfg: OcrModelConfig, vocab_size: int) -> Seq2Seq:
    """Build the encoder-decoder model.

    Args:
        cfg: Architecture description.
        vocab_size: Tokenizer vocabulary size (specials included).

    Returns:
        The assembled :class:`Seq2Seq` module.

    Raises:
        OcrInputError: If ``vocab_size`` is not positive or the encoder does not match the family.
    """
    if vocab_size <= 0:
        raise OcrInputError(details={"reason": "vocab_size", "vocab_size": int(vocab_size)})
    return Seq2Seq(cfg, vocab_size)


# ------------------------------------------------------------------------------- validation
def _require_single_sample(inputs: Tensor) -> None:
    """Reject an input batch for the per-sample entry points."""
    if inputs.shape[0] != 1:
        raise OcrInputError(
            details={"reason": "decoding_batch_size", "batch": int(inputs.shape[0])}
        )


def _require_beam(beam: int) -> None:
    """Reject a non-positive beam width."""
    if beam < 1:
        raise OcrInputError(details={"reason": "beam_width", "beam": int(beam)})


def _normalised(score: float, tokens: list[int], length_penalty: float) -> float:
    """Length-normalised score of a hypothesis (``<s>`` excluded from the length)."""
    length = max(len(tokens) - 1, 1)
    return score / (length**length_penalty)


# ---------------------------------------------------------------------------------- steppers
class _Stepper:
    """Feeds one token per row and returns the logits of the position that follows it.

    Two implementations exist so that a model without incremental support still decodes — and
    still decodes *batched*, which is the larger of the two speed-ups:

    * :class:`_IncrementalStepper` keeps decoder state (key/value cache or RNN hidden state) and
      does ``O(1)`` decoder work per token;
    * :class:`_RecomputeStepper` re-runs ``model.decode(memory, prefix)`` over the growing prefix,
      which is what the pre-optimisation code did per sample and is kept as the reference
      behaviour for models that expose only ``encode``/``decode``.
    """

    def next_logits(self, tokens: Tensor) -> Tensor:
        """Return ``(rows, V)`` logits for the position after ``tokens``."""
        raise NotImplementedError  # pragma: no cover - abstract

    def reorder(self, index: Tensor) -> None:
        """Permute the rows so that row ``i`` continues the hypothesis at ``index[i]``."""
        raise NotImplementedError  # pragma: no cover - abstract


class _RecomputeStepper(_Stepper):
    """Batched full recomputation: correct everywhere, quadratic in the sequence length."""

    def __init__(self, model: Any, memory: Tensor) -> None:
        self.model = model
        self.memory = memory
        self.device = memory.device
        self.prefix = torch.empty((int(memory.shape[0]), 0), dtype=torch.long, device=self.device)

    def next_logits(self, tokens: Tensor) -> Tensor:
        """Append ``tokens`` to the prefix and re-decode it."""
        column = tokens.to(device=self.device, dtype=torch.long).unsqueeze(1)
        self.prefix = torch.cat([self.prefix, column], dim=1)
        logits: Tensor = self.model.decode(self.memory, self.prefix)[:, -1, :]
        return logits

    def reorder(self, index: Tensor) -> None:
        """Reorder the stored prefixes; the memory rows are beam-invariant."""
        self.prefix = self.prefix.index_select(0, index.to(self.device))


class _IncrementalStepper(_Stepper):
    """Key/value-cached decoding: linear in the sequence length."""

    def __init__(self, model: Any, memory: Tensor, max_len: int) -> None:
        self.model = model
        self.device = memory.device
        self.state: DecoderState = model.init_decode_state(memory, max_len=max_len)

    def next_logits(self, tokens: Tensor) -> Tensor:
        """Advance the cached state by one token."""
        logits: Tensor = self.model.decode_step(
            self.state, tokens.to(device=self.device, dtype=torch.long)
        )
        return logits

    def reorder(self, index: Tensor) -> None:
        """Reorder the cached state."""
        self.state.reorder_beams(index.to(self.device))


def _supports_incremental(model: Any) -> bool:
    """Whether ``model`` can decode incrementally *right now*.

    Two conditions, both load-bearing:

    * the model exposes ``init_decode_state``/``decode_step`` — a hand-written stand-in that only
      implements ``encode``/``decode`` must keep working;
    * the module is in eval mode. A one-position step cannot reproduce the dropout draws of a
      whole-prefix forward pass, so in training mode the cached path would silently decode
      something else. Decoding a training-mode module is a mistake anyway; falling back keeps it
      a slow mistake rather than a wrong one.
    """
    if not callable(getattr(model, "init_decode_state", None)):
        return False
    if not callable(getattr(model, "decode_step", None)):
        return False
    if bool(getattr(model, "training", False)):
        _log.debug("ocr.decode.incremental_disabled", reason="training_mode")
        return False
    return True


def _build_stepper(model: Any, memory: Tensor, *, max_len: int, incremental: bool) -> _Stepper:
    """Pick the fastest stepper this model supports."""
    if incremental and _supports_incremental(model):
        return _IncrementalStepper(model, memory, max_len)
    return _RecomputeStepper(model, memory)


# ------------------------------------------------------------------------------ batched greedy
def greedy_decode_batch(
    model: Seq2Seq,
    inputs: Tensor,
    *,
    max_len: int = 160,
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,
    incremental: bool = True,
) -> list[tuple[list[int], float]]:
    """Decode a whole batch greedily, one decoder step per token for the entire batch.

    Rows are independent throughout: attention never crosses the batch dimension, a row that has
    emitted ``eos_id`` stops accumulating score and stops appending tokens, and the tokens fed to
    a finished row afterwards are discarded. Decoding a batch therefore returns exactly what
    decoding each of its rows alone returns
    (``test_mixed_length_batch_decodes_like_single_samples``).

    Args:
        model: A model built by :func:`build_seq2seq` (or anything exposing ``encode``/``decode``).
        inputs: ``(B, ...)`` input tensor for the configured family, already collated.
        max_len: Maximum number of generated tokens per sample.
        bos_id: Start token id.
        eos_id: End token id; a row stops as soon as it is produced.
        incremental: Set ``False`` to force full recomputation (used by the equivalence tests).

    Returns:
        One ``(ids, score)`` pair per input row, in input order: the generated ids without
        ``<s>``/``</s>`` and the mean log probability of the sequence.
    """
    with torch.no_grad():
        memory = model.encode(inputs)
        count = int(memory.shape[0])
        stepper = _build_stepper(model, memory, max_len=max_len, incremental=incremental)
        emitted: list[list[int]] = [[] for _ in range(count)]
        totals = [0.0] * count
        finished = [False] * count
        last = torch.full((count,), bos_id, dtype=torch.long, device=memory.device)
        for _ in range(max_len):
            logits = stepper.next_logits(last)
            log_probs = torch.log_softmax(logits, dim=-1)
            best = torch.argmax(log_probs, dim=-1)
            chosen = best.tolist()
            step_scores = log_probs.gather(1, best.unsqueeze(1)).squeeze(1).tolist()
            for row in range(count):
                if finished[row]:
                    continue
                token = int(chosen[row])
                totals[row] += float(step_scores[row])
                emitted[row].append(token)
                finished[row] = token == eos_id
            if all(finished):
                break
            last = best
    return [
        (
            [token for token in emitted[row] if token != eos_id],
            totals[row] / (max(len(emitted[row]), 1) ** DEFAULT_LENGTH_PENALTY),
        )
        for row in range(count)
    ]


# -------------------------------------------------------------------------------- batched beam
@dataclass(slots=True)
class _Hypothesis:
    """One live beam: its tokens (``<s>`` included), its raw score and the row that carries it."""

    tokens: list[int]
    score: float
    finished: bool
    row: int


def beam_search_batch(
    model: Seq2Seq,
    inputs: Tensor,
    *,
    beam: int = 4,
    max_len: int = 160,
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,
    length_penalty: float = DEFAULT_LENGTH_PENALTY,
    incremental: bool = True,
) -> list[list[tuple[list[int], float]]]:
    """Decode a whole batch with beam search, all beams of all samples in one forward pass.

    The batch laid out for the decoder is ``memory.repeat_interleave(beam, dim=0)``: row
    ``sample * beam + b`` carries beam ``b`` of that sample. After each step the surviving
    hypotheses are re-seated onto their sample's own block of rows and the decoder state is
    permuted to follow them, so a beam can inherit from any parent beam *of its own sample* and
    from no other sample's.

    Hypothesis bookkeeping — the expansion order, the length-normalised comparison and the stable
    sort that breaks ties — is deliberately identical to the pre-optimisation per-sample code, so
    the returned token sequences are the same ones the wave-6 decoder produced
    (``test_batched_beam_matches_the_reference_token_for_token``).

    Args:
        model: A model built by :func:`build_seq2seq` (or anything exposing ``encode``/``decode``).
        inputs: ``(B, ...)`` input tensor for the configured family, already collated.
        beam: Beam width (``1`` reproduces :func:`greedy_decode_batch` up to argmax/top-1 ties).
        max_len: Maximum number of generated tokens per sample.
        bos_id: Start token id.
        eos_id: End token id.
        length_penalty: Exponent of the length normalisation (1.0 = mean log probability).
        incremental: Set ``False`` to force full recomputation (used by the equivalence tests).

    Returns:
        One list of up to ``beam`` ``(ids, score)`` hypotheses per input row, each sorted by
        descending score.

    Raises:
        OcrInputError: If ``beam`` is not positive.
    """
    _require_beam(beam)
    with torch.no_grad():
        memory = model.encode(inputs)
        count = int(memory.shape[0])
        width = min(beam, int(model.vocab_size))
        expanded = memory if width == 1 else memory.repeat_interleave(width, dim=0)
        device = expanded.device
        stepper = _build_stepper(model, expanded, max_len=max_len, incremental=incremental)
        groups: list[list[_Hypothesis]] = [
            [_Hypothesis([bos_id], 0.0, False, sample * width)] for sample in range(count)
        ]
        last = torch.full((count * width,), bos_id, dtype=torch.long, device=device)
        for _ in range(max_len):
            if all(hyp.finished for group in groups for hyp in group):
                break
            logits = stepper.next_logits(last)
            log_probs = torch.log_softmax(logits, dim=-1)
            top_scores, top_ids = torch.topk(log_probs, k=width, dim=-1)
            score_rows = top_scores.tolist()
            id_rows = top_ids.tolist()
            parents: list[int] = []
            next_tokens: list[int] = []
            for sample, group in enumerate(groups):
                survivors = _advance_group(
                    group, score_rows, id_rows, width, eos_id, length_penalty
                )
                base = sample * width
                for offset, hyp in enumerate(survivors):
                    parents.append(hyp.row)
                    hyp.row = base + offset
                    next_tokens.append(hyp.tokens[-1])
                groups[sample] = survivors
            stepper.reorder(torch.tensor(parents, dtype=torch.long, device=device))
            last = torch.tensor(next_tokens, dtype=torch.long, device=device)
    return [_rank(group, eos_id, length_penalty, width) for group in groups]


def _advance_group(
    group: list[_Hypothesis],
    score_rows: list[list[float]],
    id_rows: list[list[int]],
    width: int,
    eos_id: int,
    length_penalty: float,
) -> list[_Hypothesis]:
    """Expand one sample's beams by one token and keep the best ``width`` of them."""
    if all(hyp.finished for hyp in group):
        # The per-sample reference breaks out of its loop here, so the group is frozen: no
        # expansion, no re-sort. A finished sample must not drift because a slower sample in the
        # same batch is still decoding.
        return group
    candidates: list[_Hypothesis] = []
    for hyp in group:
        if hyp.finished:
            candidates.append(_Hypothesis(hyp.tokens, hyp.score, True, hyp.row))
            continue
        scores = score_rows[hyp.row]
        ids = id_rows[hyp.row]
        for rank in range(width):
            token = int(ids[rank])
            candidates.append(
                _Hypothesis(
                    [*hyp.tokens, token], hyp.score + float(scores[rank]), token == eos_id, hyp.row
                )
            )
    candidates.sort(
        key=lambda item: _normalised(item.score, item.tokens, length_penalty), reverse=True
    )
    return candidates[:width]


def _rank(
    group: list[_Hypothesis], eos_id: int, length_penalty: float, width: int
) -> list[tuple[list[int], float]]:
    """Strip the specials, normalise the scores and sort one sample's hypotheses."""
    scored = [
        (
            [token for token in hyp.tokens[1:] if token != eos_id],
            _normalised(hyp.score, hyp.tokens, length_penalty),
        )
        for hyp in group
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:width]


# ------------------------------------------------------------------- per-sample compatibility
def greedy_decode(
    model: Seq2Seq,
    inputs: Tensor,
    *,
    max_len: int = 160,
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,
) -> tuple[list[int], float]:
    """Decode one sample by always taking the most probable next token.

    Kept for existing callers; it is :func:`greedy_decode_batch` on a batch of one and therefore
    gets the incremental decoder but none of the batching win. Anything decoding more than one
    sample should call :func:`greedy_decode_batch` with the collated batch instead.

    Args:
        model: A model built by :func:`build_seq2seq`.
        inputs: ``(1, ...)`` input tensor for the configured family.
        max_len: Maximum number of generated tokens.
        bos_id: Start token id.
        eos_id: End token id; decoding stops as soon as it is produced.

    Returns:
        ``(ids, score)`` — the generated ids without ``<s>``/``</s>`` and the mean log probability
        of the sequence.

    Raises:
        OcrInputError: If ``inputs`` holds more than one sample.
    """
    _require_single_sample(inputs)
    return greedy_decode_batch(model, inputs, max_len=max_len, bos_id=bos_id, eos_id=eos_id)[0]


def beam_search(
    model: Seq2Seq,
    inputs: Tensor,
    *,
    beam: int = 4,
    max_len: int = 160,
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,
    length_penalty: float = DEFAULT_LENGTH_PENALTY,
) -> list[tuple[list[int], float]]:
    """Decode one sample with beam search.

    Kept for existing callers; it is :func:`beam_search_batch` on a batch of one. Anything
    decoding more than one sample should call :func:`beam_search_batch` instead — the beams of a
    single sample are already batched here, but the samples are not.

    Args:
        model: A model built by :func:`build_seq2seq`.
        inputs: ``(1, ...)`` input tensor for the configured family.
        beam: Beam width (``1`` reproduces :func:`greedy_decode`).
        max_len: Maximum number of generated tokens.
        bos_id: Start token id.
        eos_id: End token id.
        length_penalty: Exponent of the length normalisation (1.0 = mean log probability).

    Returns:
        Up to ``beam`` distinct hypotheses as ``(ids, score)``, sorted by descending score.

    Raises:
        OcrInputError: If ``beam`` is not positive or ``inputs`` holds more than one sample.
    """
    _require_beam(beam)
    _require_single_sample(inputs)
    return beam_search_batch(
        model,
        inputs,
        beam=beam,
        max_len=max_len,
        bos_id=bos_id,
        eos_id=eos_id,
        length_penalty=length_penalty,
    )[0]


__all__ = [
    "DEFAULT_LENGTH_PENALTY",
    "Seq2Seq",
    "beam_search",
    "beam_search_batch",
    "build_seq2seq",
    "greedy_decode",
    "greedy_decode_batch",
]
