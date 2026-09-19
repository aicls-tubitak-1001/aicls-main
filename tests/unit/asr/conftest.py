"""Fixtures for the ``aicls.asr`` unit tests.

Everything here is synthetic and offline. The two model fixtures build a *tiny randomly
initialised* Hugging Face checkpoint (a few thousand parameters) with ``save_pretrained`` and hand
back the directory, which is how the ADR-0012 gate is honoured: the loaders are exercised for
real, against a real local checkpoint, with no download and no pre-trained weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# Every special token the tiny Whisper decoder may emit, plus the byte-level alphabet.
_WHISPER_SPECIALS = (
    "<|endoftext|>",
    "<|startoftranscript|>",
    "<|tr|>",
    "<|transcribe|>",
    "<|notimestamps|>",
)

# A Turkish CTC alphabet: the letters a recording of a spoken command can contain.
_CTC_LETTERS = "abcçdefgğhıijklmnoöprsştuüvyz"


@pytest.fixture(autouse=True)
def _offline_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse any Hugging Face network access for the whole ``asr`` test package."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def _write_byte_level_vocab(directory: Path, extra_tokens: tuple[str, ...]) -> int:
    """Write a GPT-2 style byte-level ``vocab.json``/``merges.txt`` and return the vocab size."""
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

    vocab: dict[str, int] = {}
    for char in sorted(set(bytes_to_unicode().values())):
        vocab[char] = len(vocab)
    for token in extra_tokens:
        vocab.setdefault(token, len(vocab))
    (directory / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (directory / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    return len(vocab)


@pytest.fixture(scope="session")
def tiny_whisper_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A local directory holding a tiny random Whisper checkpoint plus its processor."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    directory = tmp_path_factory.mktemp("tiny-whisper")
    vocab_size = _write_byte_level_vocab(directory, _WHISPER_SPECIALS)
    tokenizer = transformers.WhisperTokenizer(
        str(directory / "vocab.json"), str(directory / "merges.txt")
    )
    tokenizer.save_pretrained(directory)
    transformers.WhisperFeatureExtractor(
        feature_size=80, sampling_rate=16000, hop_length=160, chunk_length=1, n_fft=400
    ).save_pretrained(directory)

    start_id = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    end_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    config = transformers.WhisperConfig(
        vocab_size=vocab_size,
        num_mel_bins=80,
        encoder_layers=1,
        encoder_attention_heads=1,
        decoder_layers=1,
        decoder_attention_heads=1,
        d_model=16,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        # chunk_length=1 s / hop 160 = 100 mel frames; the encoder halves them.
        max_source_positions=50,
        max_target_positions=32,
        decoder_start_token_id=start_id,
        bos_token_id=end_id,
        eos_token_id=end_id,
        pad_token_id=end_id,
    )
    torch.manual_seed(1234)
    transformers.WhisperForConditionalGeneration(config).save_pretrained(directory)
    return directory


@pytest.fixture(scope="session")
def tiny_wav2vec2_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A local directory holding a tiny random Wav2Vec2 CTC checkpoint plus its processor."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    directory = tmp_path_factory.mktemp("tiny-wav2vec2")
    vocab: dict[str, Any] = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4}
    for letter in _CTC_LETTERS:
        vocab.setdefault(letter, len(vocab))
    (directory / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    transformers.Wav2Vec2CTCTokenizer(
        str(directory / "vocab.json"),
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
    ).save_pretrained(directory)
    transformers.Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True
    ).save_pretrained(directory)

    config = transformers.Wav2Vec2Config(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=16,
        conv_dim=(16, 16, 16),
        conv_stride=(5, 2, 2),
        conv_kernel=(10, 3, 3),
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
    )
    torch.manual_seed(1234)
    transformers.Wav2Vec2ForCTC(config).save_pretrained(directory)
    return directory
