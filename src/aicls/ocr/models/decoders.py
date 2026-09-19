"""Autoregressive decoders with attention (proposal A.2.1: LSTM / GRU / Transformer).

All three share one interface so the ablation table can switch between them with a config key::

    decoder(memory, tgt_ids) -> logits (B, T, V)

where ``memory`` is the ``(B, S, d_model)`` encoder output and ``tgt_ids`` are the teacher-forcing
(or, during decoding, so-far generated) token ids starting with ``<s>``. The Transformer decoder
uses a causal mask; the recurrent decoders use scaled dot-product (Luong) attention over the
memory.

**Incremental decoding.** ``forward()`` recomputes the whole prefix, which is what training needs
and what inference must not do: decoding ``T`` tokens that way costs ``O(T^2)`` decoder work and
was measured at 18.5 s *per validation sample* on an RTX 5090 (run
``ocr_excerpt_wiring``, 2026-09-08). Both decoders therefore also expose an incremental pair —
:meth:`TransformerDecoder.init_state` / :meth:`TransformerDecoder.step` and the recurrent
equivalents — that carries the state forward and costs ``O(1)`` decoder work per token:

* the Transformer caches the per-layer self-attention keys/values plus the cross-attention
  keys/values of the (constant) encoder memory;
* the recurrent decoders carry the RNN hidden state and the transposed memory.

``step()`` reproduces the arithmetic of ``forward()`` for the final position, using the very same
submodules and weights; :mod:`tests.unit.ocr.test_decoding_equivalence` pins that token for token
against the pre-optimisation implementation. It is valid **only in eval mode**: the dropout draws
of a one-position step cannot match those of a whole-prefix forward pass, so
:func:`aicls.ocr.models.seq2seq.greedy_decode_batch` falls back to full recomputation whenever the
module is training.

Importing this module imports torch; see :mod:`aicls.ocr.models` for the lazy entry point.
"""

from __future__ import annotations

from typing import Protocol, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812 - torch's own conventional alias

from aicls.ocr.config import OcrModelConfig
from aicls.ocr.models.encoders import sinusoidal_position_table
from aicls.ocr.tokenizer import PAD_ID
from aicls.ocr.types import OcrInputError


class DecoderState:
    """Opaque carrier for whatever a decoder must remember between two decoding steps.

    Beam search permutes its rows every step (a beam's continuation may come from any parent
    beam), so a state must be able to reorder itself.
    """

    def reorder_beams(self, index: Tensor) -> None:
        """Permute the batch dimension of every cached tensor with ``index``.

        Args:
            index: ``(rows,)`` long tensor of parent row indices.

        Note:
            Tensors derived from the encoder memory are deliberately **not** reordered. Beam
            search calls this only with permutations that keep every row inside its own sample's
            block of ``beam`` rows, and the memory rows inside one such block are identical by
            construction (``memory.repeat_interleave(beam, dim=0)``), so selecting them is a
            no-op that would copy megabytes per step.
        """
        raise NotImplementedError  # pragma: no cover - abstract


class IncrementalDecoder(Protocol):
    """The incremental half of the decoder interface, for callers that only need those two calls.

    :class:`TransformerDecoder` and :class:`RnnAttentionDecoder` both satisfy it with their own
    concrete :class:`DecoderState` subclass; the protocol exists so that
    :class:`aicls.ocr.models.seq2seq.Seq2Seq` can name what it expects of ``self.decoder``
    without importing both classes and without ``nn.Module``'s untyped attribute access.
    """

    def init_state(self, memory: Tensor, *, max_len: int) -> DecoderState:
        """Open a decoding state for ``memory``."""
        ...  # pragma: no cover - protocol

    def step(self, state: DecoderState, token_ids: Tensor) -> Tensor:
        """Consume one token per row and return the ``(rows, V)`` next-token logits."""
        ...  # pragma: no cover - protocol


def _decoder_layers(stack: nn.TransformerDecoder) -> list[nn.TransformerDecoderLayer]:
    """The stack's layers, typed — ``ModuleList`` iteration only promises ``nn.Module``."""
    return [cast(nn.TransformerDecoderLayer, layer) for layer in stack.layers]


def _split_heads(x: Tensor, num_heads: int) -> Tensor:
    """``(B, L, E)`` to ``(B, H, L, E/H)``, the layout ``F.multi_head_attention_forward`` uses."""
    batch, length, embed = x.shape
    return x.view(batch, length, num_heads, embed // num_heads).transpose(1, 2)


def _merge_heads(x: Tensor) -> Tensor:
    """``(B, H, L, E/H)`` back to ``(B, L, E)``."""
    batch, heads, length, head_dim = x.shape
    return x.transpose(1, 2).reshape(batch, length, heads * head_dim)


def _self_attention_qkv(attn: nn.MultiheadAttention, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Query/key/value projections of a self-attention input, one fused matmul.

    Mirrors ``torch.nn.functional._in_projection_packed``'s ``q is k is v`` branch: one
    ``linear`` against the packed ``in_proj_weight`` followed by a split into three ``E``-wide
    blocks in q, k, v order.
    """
    proj = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
    query, key, value = proj.chunk(3, dim=-1)
    heads = attn.num_heads
    return _split_heads(query, heads), _split_heads(key, heads), _split_heads(value, heads)


def _cross_attention_kv(attn: nn.MultiheadAttention, memory: Tensor) -> tuple[Tensor, Tensor]:
    """Key/value projections of the encoder memory for cross attention.

    Mirrors ``_in_projection_packed``'s ``k is v but q is not k`` branch, which projects the
    key/value pair with the *second* two thirds of ``in_proj_weight`` in one matmul.
    """
    embed = attn.embed_dim
    weight = attn.in_proj_weight[embed:]
    bias = None if attn.in_proj_bias is None else attn.in_proj_bias[embed:]
    key, value = F.linear(memory, weight, bias).chunk(2, dim=-1)
    return _split_heads(key, attn.num_heads), _split_heads(value, attn.num_heads)


def _cross_attention_q(attn: nn.MultiheadAttention, x: Tensor) -> Tensor:
    """Query projection for cross attention (first third of ``in_proj_weight``)."""
    embed = attn.embed_dim
    weight = attn.in_proj_weight[:embed]
    bias = None if attn.in_proj_bias is None else attn.in_proj_bias[:embed]
    return _split_heads(F.linear(x, weight, bias), attn.num_heads)


def _attend(
    attn: nn.MultiheadAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None,
) -> Tensor:
    """Scaled dot-product attention plus the output projection, in ``(B, L, E)`` layout."""
    out = F.scaled_dot_product_attention(query, key, value, attn_mask)
    return attn.out_proj(_merge_heads(out))


class TransformerDecoderState(DecoderState):
    """Per-layer key/value cache for :class:`TransformerDecoder`.

    Attributes:
        length: Number of tokens already written into the self-attention cache.
    """

    def __init__(
        self,
        *,
        cross_key: list[Tensor],
        cross_value: list[Tensor],
        positions: Tensor,
    ) -> None:
        self.cross_key = cross_key
        self.cross_value = cross_value
        self.positions = positions
        self.self_key: list[Tensor | None] = [None] * len(cross_key)
        self.self_value: list[Tensor | None] = [None] * len(cross_key)
        self.length = 0

    def reorder_beams(self, index: Tensor) -> None:
        """Reorder the self-attention cache; the memory-derived cache is row-invariant."""
        for layer in range(len(self.self_key)):
            cached_key = self.self_key[layer]
            cached_value = self.self_value[layer]
            if cached_key is not None:
                self.self_key[layer] = cached_key.index_select(0, index)
            if cached_value is not None:
                self.self_value[layer] = cached_value.index_select(0, index)

    def append(self, layer: int, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        """Append one position's key/value to layer ``layer`` and return the full cache."""
        cached_key = self.self_key[layer]
        cached_value = self.self_value[layer]
        full_key = key if cached_key is None else torch.cat([cached_key, key], dim=2)
        full_value = value if cached_value is None else torch.cat([cached_value, value], dim=2)
        self.self_key[layer] = full_key
        self.self_value[layer] = full_value
        return full_key, full_value


class RnnDecoderState(DecoderState):
    """Hidden state and transposed memory for :class:`RnnAttentionDecoder`.

    Attributes:
        length: Number of tokens already consumed.
    """

    def __init__(self, memory: Tensor) -> None:
        self.memory = memory
        self.memory_t = memory.transpose(1, 2)
        self.hidden: Tensor | tuple[Tensor, Tensor] | None = None
        self.length = 0

    def reorder_beams(self, index: Tensor) -> None:
        """Reorder the RNN hidden state; the memory is row-invariant (see the base class)."""
        if self.hidden is None:
            return
        if isinstance(self.hidden, tuple):
            short_term, long_term = self.hidden
            self.hidden = (short_term.index_select(1, index), long_term.index_select(1, index))
        else:
            self.hidden = self.hidden.index_select(1, index)


class TransformerDecoder(nn.Module):
    """Transformer decoder stack with causal self attention and cross attention."""

    def __init__(self, cfg: OcrModelConfig, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.d_model, padding_idx=PAD_ID)
        layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.heads,
            dim_feedforward=cfg.feedforward_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.layers)
        self.output = nn.Linear(cfg.d_model, vocab_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.d_model = cfg.d_model

    def forward(self, memory: Tensor, tgt_ids: Tensor) -> Tensor:
        """Return ``(B, T, V)`` logits for the given prefix."""
        embedded = self.embedding(tgt_ids)
        positions = sinusoidal_position_table(embedded.shape[1], self.d_model)
        embedded = self.dropout(embedded + positions.to(embedded.device).unsqueeze(0))
        length = tgt_ids.shape[1]
        mask = torch.triu(torch.full((length, length), float("-inf")), diagonal=1).to(
            embedded.device
        )
        hidden = self.decoder(embedded, memory, tgt_mask=mask)
        return self.output(hidden)

    def init_state(self, memory: Tensor, *, max_len: int) -> TransformerDecoderState:
        """Build the incremental state for a decode of at most ``max_len`` tokens.

        The cross-attention keys and values of every layer are projected **once** here: the
        encoder memory does not change while a sample is being decoded, so re-projecting it per
        token (as ``forward`` does) is pure waste.

        Args:
            memory: ``(rows, S, d_model)`` encoder memory, already expanded per beam.
            max_len: Upper bound on generated tokens; sizes the position table.

        Returns:
            A fresh :class:`TransformerDecoderState` positioned before the first token.
        """
        cross_key: list[Tensor] = []
        cross_value: list[Tensor] = []
        for layer in _decoder_layers(self.decoder):
            key, value = _cross_attention_kv(layer.multihead_attn, memory)
            cross_key.append(key)
            cross_value.append(value)
        # Row `t` of a length-L table is sin/cos of `t` whatever L is, so one table built here is
        # bit-for-bit the row `forward()` would have built at every single step.
        positions = sinusoidal_position_table(max(max_len, 1), self.d_model).to(memory.device)
        return TransformerDecoderState(
            cross_key=cross_key, cross_value=cross_value, positions=positions
        )

    def step(self, state: TransformerDecoderState, token_ids: Tensor) -> Tensor:
        """Consume one token per row and return the ``(rows, V)`` next-token logits.

        Args:
            state: The state returned by :meth:`init_state`, advanced in place.
            token_ids: ``(rows,)`` long tensor of the tokens at the current position.

        Returns:
            ``(rows, V)`` logits for the position that follows ``token_ids``.

        Raises:
            OcrInputError: If more tokens are requested than :meth:`init_state` was sized for.
        """
        if state.length >= state.positions.shape[0]:
            raise OcrInputError(
                details={"reason": "decode_state_exhausted", "max_len": int(state.length)}
            )
        hidden = self.embedding(token_ids).unsqueeze(1)
        position = state.positions[state.length : state.length + 1]
        hidden = self.dropout(hidden + position.to(hidden.device).unsqueeze(0))
        # A single query position attends to every cached position, so nothing is masked. The
        # all-zero additive mask is the last row of `forward()`'s causal mask and keeps
        # scaled_dot_product_attention on the same (masked) kernel the reference path uses.
        no_mask = torch.zeros(
            (1, 1, 1, state.length + 1), dtype=torch.float32, device=hidden.device
        )
        for index, layer in enumerate(_decoder_layers(self.decoder)):
            hidden = self._layer_step(layer, hidden, state, index, no_mask)
        if self.decoder.norm is not None:  # pragma: no cover - build_decoder passes no final norm
            hidden = self.decoder.norm(hidden)
        state.length += 1
        return self.output(hidden).squeeze(1)

    @staticmethod
    def _layer_step(
        layer: nn.TransformerDecoderLayer,
        hidden: Tensor,
        state: TransformerDecoderState,
        index: int,
        self_mask: Tensor,
    ) -> Tensor:
        """One ``norm_first`` decoder layer applied to the final position only.

        The three blocks are exactly ``TransformerDecoderLayer._sa_block`` / ``_mha_block`` /
        ``_ff_block`` with the key/value halves read from ``state`` instead of recomputed.
        """
        normed = layer.norm1(hidden)
        query, key, value = _self_attention_qkv(layer.self_attn, normed)
        full_key, full_value = state.append(index, key, value)
        attended = _attend(layer.self_attn, query, full_key, full_value, self_mask)
        hidden = hidden + layer.dropout1(attended)

        normed = layer.norm2(hidden)
        cross = _attend(
            layer.multihead_attn,
            _cross_attention_q(layer.multihead_attn, normed),
            state.cross_key[index],
            state.cross_value[index],
            None,
        )
        hidden = hidden + layer.dropout2(cross)

        normed = layer.norm3(hidden)
        feed_forward = layer.linear2(layer.dropout(layer.activation(layer.linear1(normed))))
        return hidden + layer.dropout3(feed_forward)


class RnnAttentionDecoder(nn.Module):
    """LSTM/GRU decoder with scaled dot-product attention over the encoder memory."""

    def __init__(self, cfg: OcrModelConfig, vocab_size: int, kind: str) -> None:
        super().__init__()
        rnn_class = nn.LSTM if kind == "lstm_attn" else nn.GRU
        self.embedding = nn.Embedding(vocab_size, cfg.d_model, padding_idx=PAD_ID)
        self.rnn = rnn_class(
            input_size=cfg.d_model,
            hidden_size=cfg.d_model,
            num_layers=cfg.layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.layers > 1 else 0.0,
        )
        self.attention_combine = nn.Linear(cfg.d_model * 2, cfg.d_model)
        self.output = nn.Linear(cfg.d_model, vocab_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.scale = cfg.d_model**0.5

    def forward(self, memory: Tensor, tgt_ids: Tensor) -> Tensor:
        """Return ``(B, T, V)`` logits for the given prefix."""
        embedded = self.dropout(self.embedding(tgt_ids))
        hidden, _ = self.rnn(embedded)
        return self._readout(hidden, memory.transpose(1, 2), memory)

    def _readout(self, hidden: Tensor, memory_t: Tensor, memory: Tensor) -> Tensor:
        """Attention over the memory plus the output projection, position by position."""
        scores = torch.bmm(hidden, memory_t) / self.scale
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, memory)
        combined = torch.tanh(self.attention_combine(torch.cat([hidden, context], dim=-1)))
        return self.output(self.dropout(combined))

    def init_state(self, memory: Tensor, *, max_len: int) -> RnnDecoderState:
        """Build the incremental state (RNN hidden state plus the transposed memory).

        Args:
            memory: ``(rows, S, d_model)`` encoder memory, already expanded per beam.
            max_len: Accepted for interface symmetry with the Transformer decoder; a recurrent
                decoder needs no position table and therefore no length bound.

        Returns:
            A fresh :class:`RnnDecoderState` with a zero (implicit) hidden state.
        """
        del max_len
        return RnnDecoderState(memory)

    def step(self, state: RnnDecoderState, token_ids: Tensor) -> Tensor:
        """Consume one token per row and return the ``(rows, V)`` next-token logits.

        Args:
            state: The state returned by :meth:`init_state`, advanced in place.
            token_ids: ``(rows,)`` long tensor of the tokens at the current position.

        Returns:
            ``(rows, V)`` logits for the position that follows ``token_ids``.
        """
        embedded = self.dropout(self.embedding(token_ids).unsqueeze(1))
        hidden, next_state = self.rnn(embedded, state.hidden)
        state.hidden = next_state
        state.length += 1
        return self._readout(hidden, state.memory_t, state.memory).squeeze(1)


def build_decoder(cfg: OcrModelConfig, vocab_size: int) -> nn.Module:
    """Build the decoder module described by ``cfg``.

    Args:
        cfg: Model configuration.
        vocab_size: Size of the tokenizer vocabulary (specials included).

    Returns:
        An ``nn.Module`` mapping ``(memory, tgt_ids)`` to ``(B, T, vocab_size)`` logits, and
        additionally exposing ``init_state`` / ``step`` for incremental decoding.

    Raises:
        OcrInputError: If ``vocab_size`` is not positive.
    """
    if vocab_size <= 0:
        raise OcrInputError(details={"reason": "vocab_size", "vocab_size": int(vocab_size)})
    if cfg.decoder == "transformer":
        return TransformerDecoder(cfg, vocab_size)
    return RnnAttentionDecoder(cfg, vocab_size, cfg.decoder)


__all__ = [
    "DecoderState",
    "IncrementalDecoder",
    "RnnAttentionDecoder",
    "RnnDecoderState",
    "TransformerDecoder",
    "TransformerDecoderState",
    "build_decoder",
]
