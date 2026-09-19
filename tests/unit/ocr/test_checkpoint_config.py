"""Checkpoint metadata handling (torch-free part of `aicls.ocr.torch_recognizer`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aicls.ocr import OcrInputError, OcrModelConfig
from aicls.ocr.torch_recognizer import CONFIG_FILES, WEIGHT_FILES, _first_existing, _load_config


class TestLoadConfig:
    def test_missing_config_falls_back_to_defaults(self) -> None:
        assert _load_config(None) == OcrModelConfig()

    def test_json_config(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"d_model": 64, "heads": 4}), encoding="utf-8")
        cfg = _load_config(path)
        assert (cfg.d_model, cfg.heads) == (64, 4)

    def test_yaml_config_with_a_model_block(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(
            "model:\n  family: seq2seq_stroke\n  encoder: stroke_bilstm\n  d_model: 32\n",
            encoding="utf-8",
        )
        cfg = _load_config(path)
        assert cfg.family == "seq2seq_stroke"
        assert cfg.encoder == "stroke_bilstm"

    def test_a_null_means_not_specified_not_none(self, tmp_path: Path) -> None:
        """The trainer writes its whole schema out, so optional keys arrive explicitly null.

        Every checkpoint the training loop had ever written was unreadable here because of this:
        ``model.max_len: null`` replaced the field's default of 160 with a value it cannot hold,
        and TorchRecognizer.load raised a pydantic error before it reached the weights. Found on
        2026-09-09 by pointing the reporting path at a real checkpoint for the first time, with
        eight hours of training still to run. ``aicls.ml.config`` resolves the same key the same
        way (``self.max_len or DEFAULT_MAX_LEN``); this is the reader agreeing with the writer.
        """
        path = tmp_path / "config.yaml"
        path.write_text(
            "model:\n  encoder: resnet34\n  d_model: 256\n  max_len: null\n  weights: null\n",
            encoding="utf-8",
        )

        cfg = _load_config(path)

        assert cfg.max_len == OcrModelConfig().max_len
        assert cfg.d_model == 256

    def test_unknown_keys_are_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"d_model": 32, "optimizer": "adamw"}), encoding="utf-8")
        assert _load_config(path).d_model == 32

    def test_malformed_json_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text("{oops", encoding="utf-8")
        with pytest.raises(OcrInputError):
            _load_config(path)

    def test_non_mapping_payload_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(OcrInputError):
            _load_config(path)


class TestFileLookup:
    def test_first_existing_prefers_the_declared_order(self, tmp_path: Path) -> None:
        (tmp_path / "model.pt").write_bytes(b"x")
        (tmp_path / "model.pth").write_bytes(b"x")
        assert _first_existing(tmp_path, WEIGHT_FILES) == tmp_path / "model.pt"

    def test_first_existing_returns_none(self, tmp_path: Path) -> None:
        assert _first_existing(tmp_path, CONFIG_FILES) is None
