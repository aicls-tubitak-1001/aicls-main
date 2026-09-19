"""The wave-6 per-sample decoders, frozen verbatim as the correctness oracle.

This is a **copy** of ``aicls.ocr.models.seq2seq.greedy_decode`` / ``beam_search`` as they stood at
git ``036266d`` — the revision that produced the ``ocr_excerpt_wiring`` run on the RTX 5090 whose
validation cost 18.5 s per sample. Nothing here is imported by the library; it exists so that
:mod:`tests.unit.ocr.test_decoding_equivalence` can assert, token for token, that the batched and
incremental replacement decodes *exactly* what the shipped code decoded before it was optimised.

Do not "fix", tidy or speed this file up. Its only value is being the old algorithm. If a change
to the real decoders makes a test here fail, the real decoders changed the model's output — and
with it every CER the project has ever reported.

The module name starts with an underscore so pytest does not collect it as a test module.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from aicls.ocr.tokenizer import BOS_ID, EOS_ID

#: Frozen copy of ``aicls.ocr.models.seq2seq.DEFAULT_LENGTH_PENALTY``.
REFERENCE_LENGTH_PENALTY = 1.0


def reference_single_sample_memory(model: Any, inputs: Tensor) -> Tensor:
    """Encode ``inputs`` and verify it holds exactly one sample."""
    from aicls.ocr.types import OcrInputError

    if inputs.shape[0] != 1:
        raise OcrInputError(
            details={"reason": "decoding_batch_size", "batch": int(inputs.shape[0])}
        )
    with torch.no_grad():
        return model.encode(inputs)


def reference_prefix(tokens: list[int], memory: Tensor) -> Tensor:
    """The ``(1, T)`` decoder prefix, built on the device the encoder memory is on."""
    return torch.tensor([tokens], dtype=torch.long, device=memory.device)


def reference_normalised(score: float, tokens: list[int], length_penalty: float) -> float:
    """Length-normalised score of a hypothesis (``<s>`` excluded from the length)."""
    length = max(len(tokens) - 1, 1)
    return score / (length**length_penalty)


def reference_greedy_decode(
    model: Any,
    inputs: Tensor,
    *,
    max_len: int = 160,
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,
) -> tuple[list[int], float]:
    """Wave-6 greedy decoding: one sample, one decoder call per generated token."""
    memory = reference_single_sample_memory(model, inputs)
    tokens = [bos_id]
    total = 0.0
    with torch.no_grad():
        for _ in range(max_len):
            logits = model.decode(memory, reference_prefix(tokens, memory))[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)[0]
            best = int(torch.argmax(log_probs).item())
            total += float(log_probs[best].item())
            tokens.append(best)
            if best == eos_id:
                break
    generated = [token for token in tokens[1:] if token != eos_id]
    return generated, reference_normalised(total, tokens, REFERENCE_LENGTH_PENALTY)


def reference_beam_search(
    model: Any,
    inputs: Tensor,
    *,
    beam: int = 4,
    max_len: int = 160,
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,
    length_penalty: float = REFERENCE_LENGTH_PENALTY,
) -> list[tuple[list[int], float]]:
    """Wave-6 beam search: one sample, one decoder call per live beam per token."""
    from aicls.ocr.types import OcrInputError

    if beam < 1:
        raise OcrInputError(details={"reason": "beam_width", "beam": int(beam)})
    memory = reference_single_sample_memory(model, inputs)
    width = min(beam, model.vocab_size)

    beams: list[tuple[list[int], float, bool]] = [([bos_id], 0.0, False)]
    with torch.no_grad():
        for _ in range(max_len):
            if all(finished for _, _, finished in beams):
                break
            candidates: list[tuple[list[int], float, bool]] = []
            for tokens, score, finished in beams:
                if finished:
                    candidates.append((tokens, score, True))
                    continue
                logits = model.decode(memory, reference_prefix(tokens, memory))[:, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)[0]
                top_scores, top_ids = torch.topk(log_probs, k=width)
                for rank in range(width):
                    token = int(top_ids[rank].item())
                    candidates.append(
                        ([*tokens, token], score + float(top_scores[rank].item()), token == eos_id)
                    )
            candidates.sort(
                key=lambda item: reference_normalised(item[1], item[0], length_penalty),
                reverse=True,
            )
            beams = candidates[:width]

    scored = [
        (
            [token for token in tokens[1:] if token != eos_id],
            reference_normalised(score, tokens, length_penalty),
        )
        for tokens, score, _ in beams
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:width]


__all__ = [
    "REFERENCE_LENGTH_PENALTY",
    "reference_beam_search",
    "reference_greedy_decode",
    "reference_normalised",
    "reference_prefix",
    "reference_single_sample_memory",
]
