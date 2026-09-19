"""Correctness contract for the batched / incremental OCR decoders.

The optimisation this file guards changed *how* the decoder is run, never *what* it decodes. A
silent change of output would silently change every CER the project reports, so every test here
compares the new decoders against :mod:`tests.unit.ocr._reference_decoding` — a frozen copy of the
per-sample, full-recompute implementation that produced the ``ocr_excerpt_wiring`` run — or
against the same decoder driven one sample at a time.

Three failure modes are covered, in the order they are likely:

1. **masking / batching**: a batch of mixed-length inputs must decode exactly like its rows do
   alone, and a row that emits ``</s>`` early must not be disturbed by a row that runs to
   ``max_len``;
2. **incremental state**: the key/value cache (and the RNN hidden state) must reproduce full
   recomputation, including the beam reordering that re-seats a hypothesis onto another row;
3. **ranking**: the length-penalised beam ordering must be unchanged.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from aicls.ocr import OcrModelConfig, build_model
from aicls.ocr.tokenizer import BOS_ID, EOS_ID
from tests.unit.ocr._reference_decoding import reference_beam_search, reference_greedy_decode

torch = pytest.importorskip("torch", reason="CPU torch is a dev dependency")

VOCAB = 24
IMAGE_SIZE = (32, 64)


def tiny_config(**overrides: Any) -> OcrModelConfig:
    """A toy architecture (d_model 32, 2 layers) — small enough for a CPU unit test."""
    base: dict[str, Any] = {
        "family": "seq2seq_image",
        "encoder": "cnn_small",
        "decoder": "transformer",
        "d_model": 32,
        "layers": 2,
        "heads": 4,
        "dropout": 0.0,
        "image_size": IMAGE_SIZE,
        "max_len": 12,
    }
    base.update(overrides)
    return OcrModelConfig(**base)


def image_batch(batch: int = 4, seed: int = 1234) -> Any:
    """Deterministic ``(B, 1, H, W)`` rasters."""
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(batch, 1, *IMAGE_SIZE, generator=generator)


def stroke_batch(lengths: tuple[int, ...], seed: int = 1234) -> Any:
    """A collated ``(B, T_max, F)`` stroke batch: real lengths, zero padding — as `collate_ink`."""
    from aicls.ocr.features import STROKE_FEATURE_DIM

    generator = torch.Generator().manual_seed(seed)
    longest = max(lengths)
    batch = torch.zeros(len(lengths), longest, STROKE_FEATURE_DIM)
    for row, length in enumerate(lengths):
        batch[row, :length] = torch.rand(length, STROKE_FEATURE_DIM, generator=generator)
    return batch


def build_eval_model(**overrides: Any) -> Any:
    """A seeded toy model in eval mode (dropout off, incremental decoding legal)."""
    torch.manual_seed(1234)
    return build_model(tiny_config(**overrides), VOCAB).eval()


# --------------------------------------------------------------------------------------------
# a scripted model: full control over *when* each row emits </s>
# --------------------------------------------------------------------------------------------
class _ScriptedState:
    """Incremental state of :class:`_ScriptedModel`: position plus the running prefix sum."""

    def __init__(self, seeds: Any) -> None:
        self.seeds = seeds
        self.prefix_sum = torch.zeros_like(seeds)
        self.position = 0

    def reorder_beams(self, index: Any) -> None:
        """Follow the beams; exercises the reorder plumbing with observable values."""
        self.seeds = self.seeds.index_select(0, index)
        self.prefix_sum = self.prefix_sum.index_select(0, index)


class _ScriptedModel(torch.nn.Module):
    """Deterministic logits from ``(seed, position, prefix sum)`` with a forced ``</s>`` schedule.

    Random weights almost never emit ``</s>``, so a real toy model cannot exercise "one row
    finishes at step 2 while another runs to ``max_len``" — the exact situation in which a
    batched decoder leaks state between rows. This model can: ``eos_at[seed]`` is the position
    from which ``</s>`` becomes the most probable token for that sample.

    It implements both decoding interfaces, so the incremental path and the full-recompute path
    can be compared on identical, exactly representable arithmetic.
    """

    def __init__(self, eos_at: list[int], vocab_size: int = VOCAB) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.register_buffer("eos_at", torch.tensor(eos_at, dtype=torch.long))

    def encode(self, inputs: Any) -> Any:
        """``inputs`` is a ``(B, 1)`` tensor of sample seeds; memory is ``(B, 1, 1)``."""
        return inputs.reshape(inputs.shape[0], 1, 1).to(torch.float32)

    def _logits(self, seeds: Any, position: int, prefix_sum: Any) -> Any:
        """``(rows, V)`` logits — a smooth function of the row's history, plus the eos schedule."""
        vocab = torch.arange(self.vocab_size, dtype=torch.float32)
        values = torch.sin(
            seeds.unsqueeze(1) * 1.7
            + position * 0.37
            + vocab.unsqueeze(0) * 0.11
            + prefix_sum.unsqueeze(1) * 0.013
        )
        due = position >= self.eos_at[seeds.to(torch.long)]
        values[:, EOS_ID] = values[:, EOS_ID] + torch.where(due, 5.0, -5.0)
        return values

    def decode(self, memory: Any, tgt_ids: Any) -> Any:
        """Full-recompute interface: ``(B, T, V)`` logits for the whole prefix."""
        seeds = memory[:, 0, 0]
        steps = []
        for position in range(int(tgt_ids.shape[1])):
            prefix_sum = tgt_ids[:, : position + 1].sum(dim=1).to(torch.float32)
            steps.append(self._logits(seeds, position, prefix_sum))
        return torch.stack(steps, dim=1)

    def init_decode_state(self, memory: Any, *, max_len: int) -> _ScriptedState:
        """Incremental interface: open a state."""
        del max_len
        return _ScriptedState(memory[:, 0, 0])

    def decode_step(self, state: _ScriptedState, token_ids: Any) -> Any:
        """Incremental interface: consume one token per row."""
        state.prefix_sum = state.prefix_sum + token_ids.to(torch.float32)
        logits = self._logits(state.seeds, state.position, state.prefix_sum)
        state.position += 1
        return logits


def scripted_inputs(count: int) -> Any:
    """``(B, 1)`` seeds ``0..B-1`` for :class:`_ScriptedModel`."""
    return torch.arange(count, dtype=torch.float32).unsqueeze(1)


# --------------------------------------------------------------------------------------------
# 1. the batched decoders reproduce the frozen per-sample reference, token for token
# --------------------------------------------------------------------------------------------
class TestReferenceEquivalence:
    @pytest.mark.parametrize("decoder", ["transformer", "lstm_attn", "gru_attn"])
    def test_batched_greedy_matches_the_reference_token_for_token(self, decoder: str) -> None:
        from aicls.ocr.models.seq2seq import greedy_decode_batch

        model = build_eval_model(decoder=decoder)
        inputs = image_batch(4)
        expected = [reference_greedy_decode(model, inputs[i : i + 1], max_len=10) for i in range(4)]

        actual = greedy_decode_batch(model, inputs, max_len=10)

        assert [ids for ids, _ in actual] == [ids for ids, _ in expected]
        for (_, got), (_, want) in zip(actual, expected, strict=True):
            assert got == pytest.approx(want, abs=1e-5)

    @pytest.mark.parametrize("decoder", ["transformer", "lstm_attn", "gru_attn"])
    def test_batched_beam_matches_the_reference_token_for_token(self, decoder: str) -> None:
        from aicls.ocr.models.seq2seq import beam_search_batch

        model = build_eval_model(decoder=decoder)
        inputs = image_batch(3)
        expected = [
            reference_beam_search(model, inputs[i : i + 1], beam=4, max_len=8) for i in range(3)
        ]

        actual = beam_search_batch(model, inputs, beam=4, max_len=8)

        assert [[ids for ids, _ in group] for group in actual] == [
            [ids for ids, _ in group] for group in expected
        ]
        for got_group, want_group in zip(actual, expected, strict=True):
            for (_, got), (_, want) in zip(got_group, want_group, strict=True):
                assert got == pytest.approx(want, abs=1e-5)

    def test_beam_ranking_and_length_penalty_match_the_reference(self) -> None:
        """A non-default length penalty must reorder the batched hypotheses the same way."""
        from aicls.ocr.models.seq2seq import beam_search_batch

        model = build_eval_model()
        inputs = image_batch(2)
        expected = [
            reference_beam_search(model, inputs[i : i + 1], beam=3, max_len=7, length_penalty=0.6)
            for i in range(2)
        ]

        actual = beam_search_batch(model, inputs, beam=3, max_len=7, length_penalty=0.6)

        assert [[ids for ids, _ in group] for group in actual] == [
            [ids for ids, _ in group] for group in expected
        ]
        for group in actual:
            scores = [score for _, score in group]
            assert scores == sorted(scores, reverse=True)

    def test_public_per_sample_wrappers_still_match_the_reference(self) -> None:
        from aicls.ocr.models.seq2seq import beam_search, greedy_decode

        model = build_eval_model()
        sample = image_batch(1, seed=7)

        assert (
            greedy_decode(model, sample, max_len=9)[0]
            == (reference_greedy_decode(model, sample, max_len=9)[0])
        )
        assert [ids for ids, _ in beam_search(model, sample, beam=4, max_len=6)] == [
            ids for ids, _ in reference_beam_search(model, sample, beam=4, max_len=6)
        ]


# --------------------------------------------------------------------------------------------
# 2. incremental state reproduces full recomputation
# --------------------------------------------------------------------------------------------
class TestIncrementalState:
    @pytest.mark.parametrize("decoder", ["transformer", "lstm_attn", "gru_attn"])
    def test_incremental_greedy_matches_full_recompute(self, decoder: str) -> None:
        from aicls.ocr.models.seq2seq import greedy_decode_batch

        model = build_eval_model(decoder=decoder)
        inputs = image_batch(4)

        cached = greedy_decode_batch(model, inputs, max_len=10, incremental=True)
        plain = greedy_decode_batch(model, inputs, max_len=10, incremental=False)

        assert [ids for ids, _ in cached] == [ids for ids, _ in plain]

    @pytest.mark.parametrize("decoder", ["transformer", "lstm_attn", "gru_attn"])
    def test_incremental_beam_matches_full_recompute(self, decoder: str) -> None:
        from aicls.ocr.models.seq2seq import beam_search_batch

        model = build_eval_model(decoder=decoder)
        inputs = image_batch(3)

        cached = beam_search_batch(model, inputs, beam=4, max_len=8, incremental=True)
        plain = beam_search_batch(model, inputs, beam=4, max_len=8, incremental=False)

        assert [[ids for ids, _ in group] for group in cached] == [
            [ids for ids, _ in group] for group in plain
        ]

    @pytest.mark.parametrize("decoder", ["transformer", "lstm_attn", "gru_attn"])
    def test_reordering_a_state_before_the_first_step_is_a_no_op(self, decoder: str) -> None:
        """Defensive: a state with nothing cached yet has nothing to permute."""
        model = build_eval_model(decoder=decoder)
        with torch.no_grad():
            memory = model.encode(image_batch(2))
        state = model.init_decode_state(memory, max_len=4)

        state.reorder_beams(torch.tensor([1, 0], dtype=torch.long))

        assert state.length == 0

    def test_a_model_that_can_open_a_state_but_not_step_is_not_incremental(self) -> None:
        from aicls.ocr.models import seq2seq

        class _HalfWired:
            vocab_size = VOCAB

            def init_decode_state(self, memory: Any, *, max_len: int) -> Any:
                """Half of the interface is not the interface."""
                raise NotImplementedError(f"{memory}{max_len}")

        assert seq2seq._supports_incremental(_HalfWired()) is False

    def test_build_decoder_rejects_a_non_positive_vocabulary(self) -> None:
        from aicls.core.errors import AiclsError
        from aicls.ocr.models.decoders import build_decoder

        with pytest.raises(AiclsError):
            build_decoder(tiny_config(), 0)

    def test_beam_reordering_follows_the_parent_hypotheses(self) -> None:
        """A model whose logits depend on its own history exposes a wrong reorder immediately."""
        from aicls.ocr.models.seq2seq import beam_search_batch

        model = _ScriptedModel(eos_at=[9, 9, 9])
        inputs = scripted_inputs(3)

        cached = beam_search_batch(model, inputs, beam=3, max_len=8, incremental=True)
        plain = beam_search_batch(model, inputs, beam=3, max_len=8, incremental=False)
        alone = [
            beam_search_batch(model, inputs[i : i + 1], beam=3, max_len=8)[0] for i in range(3)
        ]

        assert cached == plain
        assert cached == alone

    def test_state_refuses_more_tokens_than_it_was_sized_for(self) -> None:
        from aicls.core.errors import AiclsError

        model = build_eval_model()
        with torch.no_grad():
            memory = model.encode(image_batch(1))
            state = model.init_decode_state(memory, max_len=2)
            tokens = torch.full((1,), BOS_ID, dtype=torch.long)
            model.decode_step(state, tokens)
            model.decode_step(state, tokens)
            with pytest.raises(AiclsError):
                model.decode_step(state, tokens)

    def test_training_mode_falls_back_to_full_recomputation(self) -> None:
        """Dropout draws cannot be reproduced one position at a time, so the cache is refused."""
        from aicls.ocr.models import seq2seq

        model = build_model(tiny_config(dropout=0.5), VOCAB).train()
        assert seq2seq._supports_incremental(model) is False
        model.eval()
        assert seq2seq._supports_incremental(model) is True

    def test_a_model_without_the_incremental_api_still_decodes(self) -> None:
        """`encode`/`decode` is the only hard requirement; batching alone still applies."""
        from aicls.ocr.models import seq2seq

        class _Minimal:
            vocab_size = VOCAB

            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def encode(self, inputs: Any) -> Any:
                return self._inner.encode(inputs)

            def decode(self, memory: Any, tgt_ids: Any) -> Any:
                return self._inner.decode(memory, tgt_ids)

        model = build_eval_model()
        minimal = _Minimal(model)
        assert seq2seq._supports_incremental(minimal) is False
        inputs = image_batch(2)

        assert [ids for ids, _ in seq2seq.greedy_decode_batch(minimal, inputs, max_len=6)] == [
            ids for ids, _ in seq2seq.greedy_decode_batch(model, inputs, max_len=6)
        ]


# --------------------------------------------------------------------------------------------
# 3. padding, mixed lengths and early finishers
# --------------------------------------------------------------------------------------------
class TestBatchIsolation:
    def test_mixed_length_batch_decodes_like_single_samples(self) -> None:
        """The stroke family: different real lengths, zero padding, one shared ``T_max``.

        This is the test that catches a mask bug. The rows share nothing but the batch
        dimension, so anything that lets one row's padded tail reach another row's attention —
        or that lets a shorter row inherit a longer row's state — shows up as a different token.
        """
        from aicls.ocr.models.seq2seq import beam_search_batch, greedy_decode_batch

        model = build_eval_model(family="seq2seq_stroke", encoder="stroke_transformer")
        inputs = stroke_batch((5, 12, 3, 9))

        batched_greedy = greedy_decode_batch(model, inputs, max_len=10)
        alone_greedy = [
            greedy_decode_batch(model, inputs[i : i + 1], max_len=10)[0] for i in range(4)
        ]
        assert [ids for ids, _ in batched_greedy] == [ids for ids, _ in alone_greedy]

        batched_beam = beam_search_batch(model, inputs, beam=3, max_len=8)
        alone_beam = [
            beam_search_batch(model, inputs[i : i + 1], beam=3, max_len=8)[0] for i in range(4)
        ]
        assert [[ids for ids, _ in g] for g in batched_beam] == [
            [ids for ids, _ in g] for g in alone_beam
        ]

    def test_a_row_that_finishes_early_is_not_disturbed_by_a_row_that_runs_long(self) -> None:
        from aicls.ocr.models.seq2seq import greedy_decode_batch

        model = _ScriptedModel(eos_at=[1, 4, 99, 0])
        inputs = scripted_inputs(4)

        batched = greedy_decode_batch(model, inputs, max_len=12)
        alone = [greedy_decode_batch(model, inputs[i : i + 1], max_len=12)[0] for i in range(4)]

        assert batched == alone
        assert len(batched[0][0]) == 1, "eos_at=1 emits one real token before </s>"
        assert len(batched[3][0]) == 0, "eos_at=0 emits </s> immediately"
        assert len(batched[2][0]) == 12, "eos_at=99 runs to max_len"

    def test_a_finished_beam_group_is_frozen_while_the_batch_runs_on(self) -> None:
        from aicls.ocr.models.seq2seq import beam_search_batch

        model = _ScriptedModel(eos_at=[0, 2, 99])
        inputs = scripted_inputs(3)

        batched = beam_search_batch(model, inputs, beam=3, max_len=10)
        alone = [
            beam_search_batch(model, inputs[i : i + 1], beam=3, max_len=10)[0] for i in range(3)
        ]

        assert batched == alone

    def test_finished_rows_do_not_keep_accumulating_score(self) -> None:
        """A finished row's score is its own mean log probability, not the batch's step count."""
        from aicls.ocr.models.seq2seq import greedy_decode_batch

        model = _ScriptedModel(eos_at=[2, 99])
        inputs = scripted_inputs(2)

        batched = greedy_decode_batch(model, inputs, max_len=20)
        alone = greedy_decode_batch(model, inputs[0:1], max_len=20)[0]

        assert batched[0] == alone
        assert math.isfinite(batched[0][1])


# --------------------------------------------------------------------------------------------
# 3b. the *frozen reference* on the early-stopping path
#
# The reference comparisons in section 1 run randomly initialised toy models, in which `</s>`
# fires by accident if at all: they therefore pin the batched decoder against the wave-6
# algorithm only in the regime where every row runs to `max_len`. That is exactly the regime the
# 18.5 s/sample measurement came from, and exactly *not* the regime a trained model decodes in.
# Section 3 covers early stopping but compares the new code against itself (batched vs. one row
# at a time), so a shared misreading of the wave-6 stopping rule would pass both sections.
#
# These tests close that gap: the scripted model forces a known `</s>` schedule and the oracle is
# the frozen wave-6 implementation. `_ScriptedModel` computes its logits elementwise from
# (seed, position, prefix sum), so a row's logits are bit-for-bit independent of the batch it is
# decoded in -- any difference here is the decoding *algorithm*, never floating-point kernel
# selection.
# --------------------------------------------------------------------------------------------
class TestReferenceEquivalenceWithEarlyStopping:
    @pytest.mark.parametrize("incremental", [True, False])
    def test_batched_greedy_matches_the_reference_when_rows_stop_at_different_steps(
        self, incremental: bool
    ) -> None:
        from aicls.ocr.models.seq2seq import greedy_decode_batch

        model = _ScriptedModel(eos_at=[0, 1, 4, 9, 99])
        inputs = scripted_inputs(5)
        expected = [reference_greedy_decode(model, inputs[i : i + 1], max_len=12) for i in range(5)]

        actual = greedy_decode_batch(model, inputs, max_len=12, incremental=incremental)

        assert [ids for ids, _ in actual] == [ids for ids, _ in expected]
        for (_, got), (_, want) in zip(actual, expected, strict=True):
            assert got == pytest.approx(want, abs=1e-6)
        assert [len(ids) for ids, _ in actual] == [0, 1, 4, 9, 12], (
            "the schedule must really produce a mixture of early and max_len rows"
        )

    @pytest.mark.parametrize("incremental", [True, False])
    def test_batched_beam_matches_the_reference_when_groups_finish_at_different_steps(
        self, incremental: bool
    ) -> None:
        from aicls.ocr.models.seq2seq import beam_search_batch

        model = _ScriptedModel(eos_at=[0, 2, 5, 99])
        inputs = scripted_inputs(4)
        expected = [
            reference_beam_search(model, inputs[i : i + 1], beam=3, max_len=10) for i in range(4)
        ]

        actual = beam_search_batch(model, inputs, beam=3, max_len=10, incremental=incremental)

        assert [[ids for ids, _ in group] for group in actual] == [
            [ids for ids, _ in group] for group in expected
        ]
        for got_group, want_group in zip(actual, expected, strict=True):
            for (_, got), (_, want) in zip(got_group, want_group, strict=True):
                assert got == pytest.approx(want, abs=1e-6)
        assert (
            min(len(ids) for ids, _ in actual[1]) < 10 < max(len(ids) for ids, _ in actual[3]) + 1
        )

    def test_beam_with_a_length_penalty_matches_the_reference_with_early_stopping(self) -> None:
        """Length normalisation and the `</s>` schedule interact; pin them together."""
        from aicls.ocr.models.seq2seq import beam_search_batch

        model = _ScriptedModel(eos_at=[1, 3, 99])
        inputs = scripted_inputs(3)
        expected = [
            reference_beam_search(model, inputs[i : i + 1], beam=4, max_len=9, length_penalty=0.6)
            for i in range(3)
        ]

        actual = beam_search_batch(model, inputs, beam=4, max_len=9, length_penalty=0.6)

        assert [[ids for ids, _ in group] for group in actual] == [
            [ids for ids, _ in group] for group in expected
        ]
        for got_group, want_group in zip(actual, expected, strict=True):
            for (_, got), (_, want) in zip(got_group, want_group, strict=True):
                assert got == pytest.approx(want, abs=1e-6)

    def test_batch_order_does_not_change_a_row_s_output(self) -> None:
        """Decoding the same rows in a different order must permute the results, nothing else."""
        from aicls.ocr.models.seq2seq import beam_search_batch, greedy_decode_batch

        model = _ScriptedModel(eos_at=[0, 3, 99, 6])
        inputs = scripted_inputs(4)
        reversed_inputs = inputs.flip(0)

        forward_greedy = greedy_decode_batch(model, inputs, max_len=11)
        backward_greedy = greedy_decode_batch(model, reversed_inputs, max_len=11)
        forward_beam = beam_search_batch(model, inputs, beam=3, max_len=9)
        backward_beam = beam_search_batch(model, reversed_inputs, beam=3, max_len=9)

        assert forward_greedy == list(reversed(backward_greedy))
        assert forward_beam == list(reversed(backward_beam))

    def test_a_batch_of_one_reproduces_the_reference_on_the_early_stop_path(self) -> None:
        """The per-sample wrappers, on a model that actually emits `</s>`."""
        from aicls.ocr.models.seq2seq import beam_search, greedy_decode

        model = _ScriptedModel(eos_at=[2])
        sample = scripted_inputs(1)

        assert greedy_decode(model, sample, max_len=10) == pytest.approx(
            reference_greedy_decode(model, sample, max_len=10)
        )
        assert [ids for ids, _ in beam_search(model, sample, beam=3, max_len=10)] == [
            ids for ids, _ in reference_beam_search(model, sample, beam=3, max_len=10)
        ]

    def test_mixed_length_stroke_batch_matches_the_reference(self) -> None:
        """Section 3's mask test, but against the wave-6 oracle rather than against ourselves."""
        from aicls.ocr.models.seq2seq import beam_search_batch, greedy_decode_batch

        model = build_eval_model(family="seq2seq_stroke", encoder="stroke_transformer")
        inputs = stroke_batch((5, 12, 3, 9))
        expected_greedy = [
            reference_greedy_decode(model, inputs[i : i + 1], max_len=10) for i in range(4)
        ]
        expected_beam = [
            reference_beam_search(model, inputs[i : i + 1], beam=3, max_len=8) for i in range(4)
        ]

        actual_greedy = greedy_decode_batch(model, inputs, max_len=10)
        actual_beam = beam_search_batch(model, inputs, beam=3, max_len=8)

        assert [ids for ids, _ in actual_greedy] == [ids for ids, _ in expected_greedy]
        assert [[ids for ids, _ in g] for g in actual_beam] == [
            [ids for ids, _ in g] for g in expected_beam
        ]


# --------------------------------------------------------------------------------------------
# 4. the batched entry points keep the per-sample contract
# --------------------------------------------------------------------------------------------
class TestBatchedApiContract:
    def test_batched_greedy_returns_one_result_per_row_in_input_order(self) -> None:
        from aicls.ocr.models.seq2seq import greedy_decode_batch

        model = build_eval_model()
        results = greedy_decode_batch(model, image_batch(5), max_len=6)

        assert len(results) == 5
        assert all(isinstance(ids, list) and isinstance(score, float) for ids, score in results)
        assert all(len(ids) <= 6 for ids, _ in results)

    def test_batched_beam_returns_k_sorted_per_row(self) -> None:
        from aicls.ocr.models.seq2seq import beam_search_batch

        model = build_eval_model()
        groups = beam_search_batch(model, image_batch(3), beam=4, max_len=6)

        assert len(groups) == 3
        for group in groups:
            assert len(group) == 4
            scores = [score for _, score in group]
            assert scores == sorted(scores, reverse=True)

    def test_batched_beam_one_matches_batched_greedy(self) -> None:
        from aicls.ocr.models.seq2seq import beam_search_batch, greedy_decode_batch

        model = build_eval_model()
        inputs = image_batch(3)

        greedy = greedy_decode_batch(model, inputs, max_len=7)
        beam = beam_search_batch(model, inputs, beam=1, max_len=7)

        assert [ids for ids, _ in greedy] == [group[0][0] for group in beam]

    def test_beam_width_is_clamped_to_the_vocabulary(self) -> None:
        from aicls.ocr.models.seq2seq import beam_search_batch

        model = build_model(tiny_config(), vocab_size=3).eval()
        groups = beam_search_batch(model, image_batch(2), beam=8, max_len=4)

        assert all(len(group) <= 3 for group in groups)

    def test_batched_beam_rejects_a_non_positive_width(self) -> None:
        from aicls.core.errors import AiclsError
        from aicls.ocr.models.seq2seq import beam_search_batch

        model = build_eval_model()
        with pytest.raises(AiclsError):
            beam_search_batch(model, image_batch(2), beam=0, max_len=4)

    def test_zero_max_len_yields_an_empty_hypothesis(self) -> None:
        from aicls.ocr.models.seq2seq import beam_search_batch, greedy_decode_batch

        model = build_eval_model()
        greedy = greedy_decode_batch(model, image_batch(2), max_len=0)
        beam = beam_search_batch(model, image_batch(2), beam=3, max_len=0)

        assert greedy == [([], 0.0), ([], 0.0)]
        assert beam == [[([], 0.0)], [([], 0.0)]]

    def test_per_sample_entry_points_still_reject_a_batch(self) -> None:
        from aicls.core.errors import AiclsError
        from aicls.ocr.models.seq2seq import beam_search, greedy_decode

        model = build_eval_model()
        with pytest.raises(AiclsError):
            beam_search(model, image_batch(3), beam=2, max_len=4)
        with pytest.raises(AiclsError):
            greedy_decode(model, image_batch(3), max_len=4)


# --------------------------------------------------------------------------------------------
# 5. device and dtype
# --------------------------------------------------------------------------------------------
class _FixedMemory:
    """Delegates to ``inner`` but answers ``encode`` with a memory tensor handed to it.

    The equivalence claim of this whole file is about the **decoder**. On CUDA the encoder is not
    batch-size invariant — measured on the RTX 5090 with the real ``offline_resnet34_tfm``
    architecture, ``encode(batch)[i]`` and ``encode(batch[i:i+1])[0]`` differ by 1.2e-3 relative
    because cuDNN runs the convolutions in TF32 by default and picks different algorithms per
    batch shape (4.7e-6 with ``cudnn.allow_tf32 = False``; 8.4e-7 on CPU — never bitwise, in any
    setting). Comparing a batched decode against a per-sample decode end to end would therefore
    measure the encoder, not the change under test. Holding the memory fixed measures the change
    under test, and on the real 16-sample excerpt batch it agreed 16/16 for greedy and 16/16 for
    every beam of every sample.
    """

    def __init__(self, inner: Any, memory: Any) -> None:
        self._inner = inner
        self._memory = memory
        self.vocab_size = inner.vocab_size

    def encode(self, inputs: Any) -> Any:
        """Ignore ``inputs``; hand back the memory this shim was built with."""
        del inputs
        return self._memory

    def decode(self, memory: Any, tgt_ids: Any) -> Any:
        """Delegate to the wrapped model."""
        return self._inner.decode(memory, tgt_ids)

    def init_decode_state(self, memory: Any, *, max_len: int) -> Any:
        """Delegate to the wrapped model."""
        return self._inner.init_decode_state(memory, max_len=max_len)

    def decode_step(self, state: Any, token_ids: Any) -> Any:
        """Delegate to the wrapped model."""
        return self._inner.decode_step(state, token_ids)


@pytest.mark.gpu
class TestDecodingOnCuda:
    """Real hardware: the batched decoders on the RTX 5090, in fp32 and under bf16 autocast."""

    def _cuda_model(self, **overrides: Any) -> Any:
        if not torch.cuda.is_available():  # pragma: no cover - guarded by the gpu marker
            pytest.skip("torch.cuda.is_available() is False")
        return build_eval_model(**overrides).to("cuda")

    def test_batched_decoding_on_cuda_matches_the_reference_given_the_same_memory(self) -> None:
        """The wave-6 decoder and the batched one produce the same tokens on the GPU."""
        from aicls.ocr.models.seq2seq import beam_search_batch, greedy_decode_batch

        model = self._cuda_model()
        inputs = image_batch(4).to("cuda")
        with torch.no_grad():
            memory = model.encode(inputs)

        batched_greedy = greedy_decode_batch(_FixedMemory(model, memory), inputs, max_len=10)
        batched_beam = beam_search_batch(_FixedMemory(model, memory), inputs, beam=4, max_len=10)
        for row in range(4):
            shim = _FixedMemory(model, memory[row : row + 1])
            reference_ids, _ = reference_greedy_decode(shim, inputs[row : row + 1], max_len=10)
            assert batched_greedy[row][0] == reference_ids
            reference_beams = reference_beam_search(shim, inputs[row : row + 1], beam=4, max_len=10)
            assert [ids for ids, _ in batched_beam[row]] == [ids for ids, _ in reference_beams]

    def test_batched_beam_on_cuda_matches_full_recompute_on_cuda(self) -> None:
        from aicls.ocr.models.seq2seq import beam_search_batch

        model = self._cuda_model()
        inputs = image_batch(4).to("cuda")

        cached = beam_search_batch(model, inputs, beam=4, max_len=10, incremental=True)
        plain = beam_search_batch(model, inputs, beam=4, max_len=10, incremental=False)

        assert [[ids for ids, _ in g] for g in cached] == [[ids for ids, _ in g] for g in plain]

    def test_batched_decoding_runs_under_bf16_autocast_on_cuda(self) -> None:
        from aicls.ocr.models.seq2seq import beam_search_batch, greedy_decode_batch

        model = self._cuda_model()
        inputs = image_batch(4).to("cuda")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            greedy = greedy_decode_batch(model, inputs, max_len=8)
            beam = beam_search_batch(model, inputs, beam=4, max_len=8)

        assert len(greedy) == 4
        assert all(math.isfinite(score) for _, score in greedy)
        assert all(len(group) == 4 for group in beam)
