"""Tiny CPU tests for the OCR model zoo (encoders, decoders, seq2seq, beam search).

Everything runs on random weights at a toy size (d_model=32, 2 layers); no checkpoint is ever
downloaded and torchvision backbones are always built with ``weights=None`` (ADR-0012).
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
import types
from typing import Any

import numpy as np
import pytest

from aicls.ocr import OcrModelConfig, build_model

torch = pytest.importorskip("torch", reason="CPU torch is a dev dependency")

VOCAB = 24
IMAGE_SIZE = (32, 64)


def tiny_config(**overrides: Any) -> OcrModelConfig:
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


def image_batch(batch: int = 2) -> Any:
    generator = torch.Generator().manual_seed(1234)
    return torch.rand(batch, 1, *IMAGE_SIZE, generator=generator)


def stroke_batch(batch: int = 2, length: int = 9) -> Any:
    from aicls.ocr.features import STROKE_FEATURE_DIM

    generator = torch.Generator().manual_seed(1234)
    return torch.rand(batch, length, STROKE_FEATURE_DIM, generator=generator)


class TestBuildModel:
    def test_seq2seq_forward_shapes_tiny(self) -> None:
        model = build_model(tiny_config(), vocab_size=VOCAB).eval()
        tgt = torch.zeros(2, 5, dtype=torch.long)
        with torch.no_grad():
            logits = model(image_batch(), tgt)
        assert tuple(logits.shape) == (2, 5, VOCAB)

    def test_stroke_family_forward_shapes_tiny(self) -> None:
        model = build_model(
            tiny_config(family="seq2seq_stroke", encoder="stroke_transformer"), VOCAB
        ).eval()
        tgt = torch.zeros(2, 4, dtype=torch.long)
        with torch.no_grad():
            logits = model(stroke_batch(), tgt)
        assert tuple(logits.shape) == (2, 4, VOCAB)

    @pytest.mark.parametrize("decoder", ["transformer", "lstm_attn", "gru_attn"])
    def test_every_decoder_family_builds_and_runs(self, decoder: str) -> None:
        model = build_model(tiny_config(decoder=decoder), VOCAB).eval()
        with torch.no_grad():
            logits = model(image_batch(1), torch.zeros(1, 3, dtype=torch.long))
        assert tuple(logits.shape) == (1, 3, VOCAB)

    def test_stroke_bilstm_encoder_builds_and_runs(self) -> None:
        model = build_model(
            tiny_config(family="seq2seq_stroke", encoder="stroke_bilstm", decoder="lstm_attn"),
            VOCAB,
        ).eval()
        with torch.no_grad():
            logits = model(stroke_batch(1), torch.zeros(1, 3, dtype=torch.long))
        assert tuple(logits.shape) == (1, 3, VOCAB)

    def test_encoder_family_mismatch_is_rejected(self) -> None:
        from aicls.core.errors import AiclsError

        with pytest.raises(AiclsError):
            build_model(tiny_config(family="seq2seq_stroke", encoder="cnn_small"), VOCAB)

    def test_invalid_vocab_size_is_rejected(self) -> None:
        from aicls.core.errors import AiclsError

        with pytest.raises(AiclsError):
            build_model(tiny_config(), vocab_size=0)

    def test_parameter_count_is_small_at_toy_size(self) -> None:
        model = build_model(tiny_config(), VOCAB)
        params = sum(p.numel() for p in model.parameters())
        assert 0 < params < 2_000_000


class TestTorchvisionEncoders:
    def test_torchvision_encoders_built_with_weights_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The backbone factory must always pass ``weights=None`` (no implicit download)."""
        calls: list[dict[str, Any]] = []

        def fake_resnet18(**kwargs: Any) -> Any:
            calls.append(kwargs)
            return torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1))

        fake_models = types.SimpleNamespace(resnet18=fake_resnet18)
        fake_torchvision = types.ModuleType("torchvision")
        fake_torchvision.models = fake_models  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)
        monkeypatch.setitem(sys.modules, "torchvision.models", fake_models)  # type: ignore[arg-type]

        from aicls.ocr import torch_utils

        monkeypatch.setitem(torch_utils._MODULES, "torchvision.models", fake_models)  # type: ignore[arg-type]

        from aicls.ocr.models.encoders import build_torchvision_backbone

        backbone, _channels = build_torchvision_backbone("resnet18")
        assert backbone is not None
        assert calls == [{"weights": None}]

    def test_real_torchvision_resnet18_encoder_runs(self) -> None:
        pytest.importorskip("torchvision", reason="torchvision ships with the CPU dev stack")
        model = build_model(tiny_config(encoder="resnet18"), VOCAB).eval()
        with torch.no_grad():
            logits = model(image_batch(1), torch.zeros(1, 2, dtype=torch.long))
        assert tuple(logits.shape) == (1, 2, VOCAB)

    def test_unknown_backbone_is_rejected(self) -> None:
        from aicls.core.errors import AiclsError
        from aicls.ocr.models.encoders import build_torchvision_backbone

        with pytest.raises(AiclsError):
            build_torchvision_backbone("resnet999")


class TestDecoding:
    def test_greedy_decode_stops_at_eos(self) -> None:
        from aicls.ocr.models.seq2seq import greedy_decode

        model = build_model(tiny_config(), VOCAB).eval()
        ids, score = greedy_decode(model, image_batch(1), max_len=6)
        assert isinstance(ids, list)
        assert len(ids) <= 6
        assert isinstance(score, float)

    def test_beam_search_returns_k_sorted(self) -> None:
        from aicls.ocr.models.seq2seq import beam_search

        model = build_model(tiny_config(), VOCAB).eval()
        hypotheses = beam_search(model, image_batch(1), beam=4, max_len=6)

        assert len(hypotheses) == 4
        scores = [score for _, score in hypotheses]
        assert scores == sorted(scores, reverse=True)
        assert all(isinstance(ids, list) for ids, _ in hypotheses)
        assert len({tuple(ids) for ids, _ in hypotheses}) == 4

    def test_beam_search_is_deterministic(self) -> None:
        from aicls.ocr.models.seq2seq import beam_search

        model = build_model(tiny_config(), VOCAB).eval()
        first = beam_search(model, image_batch(1), beam=3, max_len=5)
        second = beam_search(model, image_batch(1), beam=3, max_len=5)
        assert [ids for ids, _ in first] == [ids for ids, _ in second]

    def test_beam_one_matches_greedy(self) -> None:
        from aicls.ocr.models.seq2seq import beam_search, greedy_decode

        model = build_model(tiny_config(), VOCAB).eval()
        greedy_ids, _ = greedy_decode(model, image_batch(1), max_len=5)
        (beam_ids, _), *_ = beam_search(model, image_batch(1), beam=1, max_len=5)
        assert beam_ids == greedy_ids

    def test_decoding_rejects_a_batch(self) -> None:
        from aicls.core.errors import AiclsError
        from aicls.ocr.models.seq2seq import beam_search

        model = build_model(tiny_config(), VOCAB).eval()
        with pytest.raises(AiclsError):
            beam_search(model, image_batch(3), beam=2, max_len=4)


class _MemoryDeviceSpy:
    """Stand-in model whose encoder memory lives on a device that is not ``cpu``.

    The point of the ``meta`` device here is that it is a *different* device on a CPU-only host,
    so it exposes on this machine the mismatch a CUDA box would raise:
    :func:`aicls.ocr.models.seq2seq.greedy_decode` and :func:`beam_search` must build the decoder
    prefix on ``memory.device``, not on the default device.

    :meth:`decode` returns real CPU logits so decoding can continue past the assertion — the
    device of the *prefix* is what is under test. It honours the prefix's **batch** dimension:
    beam search now decodes all beams of a sample in one call, so a stand-in that always answered
    with one row would be lying about the interface it stands in for.
    """

    def __init__(self, vocab_size: int = VOCAB) -> None:
        self.vocab_size = vocab_size
        self.prefix_devices: list[Any] = []

    def encode(self, inputs: Any) -> Any:
        del inputs
        return torch.zeros(1, 4, 8, device="meta")

    def decode(self, memory: Any, tgt_ids: Any) -> Any:
        del memory
        self.prefix_devices.append(tgt_ids.device)
        return torch.zeros(int(tgt_ids.shape[0]), int(tgt_ids.shape[1]), self.vocab_size)


class TestDecodingDevice:
    """The decoder prefix must be created on the device the encoder memory is on."""

    def test_greedy_decode_builds_the_prefix_on_the_memory_device(self) -> None:
        from aicls.ocr.models.seq2seq import greedy_decode

        spy = _MemoryDeviceSpy()
        greedy_decode(spy, image_batch(1), max_len=3)

        assert spy.prefix_devices, "decode was never called"
        assert all(device.type == "meta" for device in spy.prefix_devices), spy.prefix_devices

    def test_beam_search_builds_the_prefix_on_the_memory_device(self) -> None:
        from aicls.ocr.models.seq2seq import beam_search

        spy = _MemoryDeviceSpy()
        beam_search(spy, image_batch(1), beam=2, max_len=3)

        assert spy.prefix_devices, "decode was never called"
        assert all(device.type == "meta" for device in spy.prefix_devices), spy.prefix_devices


@pytest.mark.gpu
class TestDecodingOnCuda:
    """The real thing: model and input on the GPU (AICLS_TEST_GPU=1, workstation/container)."""

    def _cuda_model(self) -> Any:
        if not torch.cuda.is_available():  # pragma: no cover - guarded by the gpu marker
            pytest.skip("torch.cuda.is_available() is False")
        return build_model(tiny_config(), VOCAB).eval().to("cuda")

    def test_greedy_decode_runs_on_cuda(self) -> None:
        from aicls.ocr.models.seq2seq import greedy_decode

        model = self._cuda_model()
        ids, score = greedy_decode(model, image_batch(1).to("cuda"), max_len=6)
        assert isinstance(ids, list)
        assert isinstance(score, float)

    def test_beam_search_runs_on_cuda(self) -> None:
        from aicls.ocr.models.seq2seq import beam_search

        model = self._cuda_model()
        hypotheses = beam_search(model, image_batch(1).to("cuda"), beam=3, max_len=6)
        assert len(hypotheses) == 3

    def test_torch_recognizer_recognizes_on_cuda(self) -> None:
        from aicls.ocr import Ink, Stroke, TorchRecognizer
        from aicls.ocr.tokenizer import default_tokenizer

        tokenizer = default_tokenizer()
        cfg = tiny_config(max_len=6)
        model = build_model(cfg, tokenizer.vocab_size).eval().to("cuda")
        recognizer = TorchRecognizer(model, tokenizer, cfg, device="cuda")
        recognizer.warmup()
        ink = Ink(strokes=[Stroke.from_xyt([0, 10, 20], [0, 15, 0], [0, 8, 16])])
        assert isinstance(recognizer.recognize(ink, beam=2).latex, str)


class TestTraining:
    @pytest.mark.timeout(60)
    def test_tiny_forward_backward_step_on_cpu(self) -> None:
        """A synthetic forward/backward pass must finish well under five seconds on CPU."""
        torch.manual_seed(1234)
        model = build_model(tiny_config(), VOCAB).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        images = image_batch(2)
        targets = torch.randint(0, VOCAB, (2, 5))

        started = time.perf_counter()
        logits = model(images, targets[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB), targets[:, 1:].reshape(-1)
        )
        loss.backward()
        optimizer.step()
        elapsed = time.perf_counter() - started

        assert torch.isfinite(loss)
        assert elapsed < 5.0, f"forward/backward took {elapsed:.2f}s"
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(g is not None and torch.isfinite(g).all() for g in grads)


class TestTorchRecognizer:
    def _write_checkpoint(self, tmp_path: Path) -> Path:
        from aicls.ocr.tokenizer import default_tokenizer

        tokenizer = default_tokenizer()
        model_dir = tmp_path / "ocr-0.0.1"
        model_dir.mkdir()
        cfg = tiny_config(max_len=6)
        model = build_model(cfg, tokenizer.vocab_size)
        torch.save(model.state_dict(), model_dir / "model.pt")
        tokenizer.save(model_dir / "vocab.json")
        (model_dir / "config.json").write_text(
            cfg.model_dump_json(indent=2) if hasattr(cfg, "model_dump_json") else "{}",
            encoding="utf-8",
        )
        (model_dir / "model_version.txt").write_text("ocr-0.0.1", encoding="utf-8")
        return model_dir

    def test_load_and_recognize_ink(self, tmp_path: Path) -> None:
        from aicls.ocr import Ink, Stroke, TorchRecognizer

        model_dir = self._write_checkpoint(tmp_path)
        recognizer = TorchRecognizer.load(model_dir, device="cpu")
        recognizer.warmup()
        ink = Ink(strokes=[Stroke.from_xyt([0, 10, 20], [0, 15, 0], [0, 8, 16])])
        recognition = recognizer.recognize(ink, beam=2)

        assert recognition.model_version == "ocr-0.0.1"
        assert isinstance(recognition.latex, str)
        assert len(recognition.alternatives) <= 2
        assert 0.0 <= recognition.confidence <= 1.0

    def test_recognize_an_image(self, tmp_path: Path) -> None:
        from aicls.ocr import Image, TorchRecognizer

        recognizer = TorchRecognizer.load(self._write_checkpoint(tmp_path), device="cpu")
        recognition = recognizer.recognize(Image(np.zeros((20, 40), dtype=np.uint8)), beam=1)
        assert isinstance(recognition.latex, str)

    def test_load_rejects_a_missing_directory(self, tmp_path: Path) -> None:
        from aicls.core.errors import NotFoundError
        from aicls.ocr import TorchRecognizer

        with pytest.raises(NotFoundError):
            TorchRecognizer.load(tmp_path / "nope", device="cpu")

    def test_load_rejects_a_directory_without_weights(self, tmp_path: Path) -> None:
        from aicls.core.errors import NotFoundError
        from aicls.ocr import TorchRecognizer

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(NotFoundError):
            TorchRecognizer.load(empty, device="cpu")

    def test_get_recognizer_torch_loads_the_checkpoint(self, tmp_path: Path) -> None:
        from aicls.ocr import TorchRecognizer, get_recognizer

        recognizer = get_recognizer("torch", model_dir=self._write_checkpoint(tmp_path))
        assert isinstance(recognizer, TorchRecognizer)

    def test_invalid_beam_rejected(self, tmp_path: Path) -> None:
        from aicls.ocr import Ink, OcrInputError, Stroke, TorchRecognizer

        recognizer = TorchRecognizer.load(self._write_checkpoint(tmp_path), device="cpu")
        ink = Ink(strokes=[Stroke.from_xyt([0, 1], [0, 1], [0, 1])])
        with pytest.raises(OcrInputError):
            recognizer.recognize(ink, beam=0)

    def test_empty_ink_rejected(self, tmp_path: Path) -> None:
        from aicls.ocr import Ink, OcrInputError, TorchRecognizer

        recognizer = TorchRecognizer.load(self._write_checkpoint(tmp_path), device="cpu")
        with pytest.raises(OcrInputError):
            recognizer.recognize(Ink(strokes=[]))

    def test_raw_numpy_array_is_wrapped(self, tmp_path: Path) -> None:
        from aicls.ocr import TorchRecognizer

        recognizer = TorchRecognizer.load(self._write_checkpoint(tmp_path), device="cpu")
        recognition = recognizer.recognize(np.zeros((12, 24), dtype=np.uint8), beam=1)
        assert recognition.model_version == "ocr-0.0.1"

    def test_safetensors_checkpoint_is_supported(self, tmp_path: Path) -> None:
        safetensors_torch = pytest.importorskip("safetensors.torch")
        from aicls.ocr import Ink, Stroke, TorchRecognizer
        from aicls.ocr.tokenizer import default_tokenizer

        tokenizer = default_tokenizer()
        model_dir = tmp_path / "ocr-safe"
        model_dir.mkdir()
        cfg = tiny_config(max_len=4)
        model = build_model(cfg, tokenizer.vocab_size)
        safetensors_torch.save_file(
            {k: v.contiguous() for k, v in model.state_dict().items()},
            str(model_dir / "model.safetensors"),
        )
        tokenizer.save(model_dir / "vocab.json")
        (model_dir / "config.json").write_text(cfg.model_dump_json(), encoding="utf-8")

        recognizer = TorchRecognizer.load(model_dir, device="cpu")
        ink = Ink(strokes=[Stroke.from_xyt([0, 5], [0, 5], [0, 5])])
        assert recognizer.recognize(ink, beam=1).model_version == "ocr-safe"

    def test_recognize_moves_the_input_to_the_model_device(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``load(device="cuda")`` puts the weights on the GPU; the input must follow them.

        ``meta`` stands in for ``cuda`` on this CPU-only host: it is simply a device that is not
        the default one, so an input tensor left on ``cpu`` is visible here exactly as the
        device-mismatch ``RuntimeError`` is on the workstation.
        """
        from aicls.ocr import Image, TorchRecognizer
        from aicls.ocr.tokenizer import default_tokenizer

        seen: dict[str, Any] = {}

        def fake_beam_search(model: Any, tensor: Any, **kw: Any) -> list[tuple[list[int], float]]:
            del model, kw
            seen["device"] = tensor.device
            return [([3, 4], -0.5)]

        monkeypatch.setattr("aicls.ocr.models.seq2seq.beam_search", fake_beam_search)
        tokenizer = default_tokenizer()
        cfg = tiny_config(max_len=4)
        recognizer = TorchRecognizer(
            build_model(cfg, tokenizer.vocab_size), tokenizer, cfg, device="meta"
        )
        recognizer.recognize(Image(np.zeros((12, 24), dtype=np.uint8)), beam=1)

        assert seen["device"].type == "meta"

    def test_constructing_a_recognizer_puts_the_module_in_eval_mode(self) -> None:
        """Inference must never run dropout, however the recogniser was built.

        Only ``TorchRecognizer.load()`` used to call ``eval()``. A recogniser built directly from
        a module — what these tests and the API replay path do — therefore ran with dropout
        active, and since the batched decoders landed it also lost the key/value cache silently,
        because ``seq2seq._supports_incremental`` refuses the cache in training mode.
        """
        from aicls.ocr import TorchRecognizer
        from aicls.ocr.tokenizer import default_tokenizer

        tokenizer = default_tokenizer()
        cfg = tiny_config(max_len=4, dropout=0.5)
        model = build_model(cfg, tokenizer.vocab_size).train()
        assert model.training is True

        TorchRecognizer(model, tokenizer, cfg, device="cpu")

        assert model.training is False

    def test_a_recognizer_accepts_a_module_stand_in_without_eval(self) -> None:
        """The eval() call must not become a new requirement on hand-written test doubles."""
        from aicls.ocr import TorchRecognizer
        from aicls.ocr.tokenizer import default_tokenizer

        def bare(inputs: Any, tgt_ids: Any) -> Any:
            del inputs, tgt_ids
            return torch.zeros(1, 1, VOCAB)

        assert TorchRecognizer(bare, default_tokenizer(), tiny_config(max_len=4)) is not None

    def test_warmup_runs_on_the_model_device(self) -> None:
        from aicls.ocr import TorchRecognizer
        from aicls.ocr.tokenizer import default_tokenizer

        seen: dict[str, Any] = {}

        def spy(inputs: Any, tgt_ids: Any) -> Any:
            seen["inputs"] = inputs.device
            seen["tgt_ids"] = tgt_ids.device
            return torch.zeros(1, 1, VOCAB)

        cfg = tiny_config(max_len=4)
        recognizer = TorchRecognizer(spy, default_tokenizer(), cfg, device="meta")
        recognizer.warmup()

        assert seen["inputs"].type == "meta"
        assert seen["tgt_ids"].type == "meta"


class TestStrokeFamilyRecognizer:
    def _stroke_checkpoint(self, tmp_path: Path) -> Path:
        from aicls.ocr.tokenizer import default_tokenizer

        tokenizer = default_tokenizer()
        model_dir = tmp_path / "ocr-stroke-0.0.1"
        model_dir.mkdir()
        cfg = tiny_config(family="seq2seq_stroke", encoder="stroke_transformer", max_len=4)
        model = build_model(cfg, tokenizer.vocab_size)
        torch.save(model.state_dict(), model_dir / "model.pt")
        tokenizer.save(model_dir / "vocab.json")
        (model_dir / "config.json").write_text(cfg.model_dump_json(), encoding="utf-8")
        return model_dir

    def test_recognizes_ink_through_stroke_features(self, tmp_path: Path) -> None:
        from aicls.ocr import Ink, Stroke, TorchRecognizer

        recognizer = TorchRecognizer.load(self._stroke_checkpoint(tmp_path), device="cpu")
        recognizer.warmup()
        ink = Ink(strokes=[Stroke.from_xyt([0, 10, 20], [0, 15, 0], [0, 8, 16])])
        recognition = recognizer.recognize(ink, beam=2)
        assert recognition.model_version == "ocr-stroke-0.0.1"
        assert isinstance(recognition.latex, str)

    def test_stroke_model_refuses_a_raster(self, tmp_path: Path) -> None:
        from aicls.ocr import Image, OcrInputError, TorchRecognizer

        recognizer = TorchRecognizer.load(self._stroke_checkpoint(tmp_path), device="cpu")
        with pytest.raises(OcrInputError):
            recognizer.recognize(Image(np.zeros((8, 8), dtype=np.uint8)))
