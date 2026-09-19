"""Encoders for both OCR families (proposal A.2.1 encoder list, ADR-0013).

* Offline (``seq2seq_image``): a small ResNet-style CNN written here (``cnn_small``, the default
  for CI and for the smoke configs) or a torchvision backbone (``resnet18``, ``resnet34``,
  ``vgg16``, ``efficientnet_b0``). Torchvision backbones are **always** constructed with
  ``weights=None``: pre-trained weights come from a local checkpoint after the
  ``WEIGHTS_DOWNLOAD`` gate, never from the network (ADR-0012,
  ``test_torchvision_encoders_built_with_weights_none``).
* Online (``seq2seq_stroke``): a Transformer or BiLSTM over the ``(T, F)`` trace features of
  :func:`aicls.ocr.features.normalize_strokes`.

Every encoder returns a ``(B, S, d_model)`` memory sequence for the decoder's cross attention.

Importing this module imports torch — which is why nothing in :mod:`aicls.ocr` imports it at
module level: :func:`aicls.ocr.models.build_model` pulls it in on demand
(``test_importing_aicls_ocr_does_not_import_torch``).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from aicls.ocr.config import IMAGE_ENCODERS, STROKE_ENCODERS, OcrModelConfig
from aicls.ocr.torch_utils import require_torchvision
from aicls.ocr.types import OcrInputError

#: Torchvision backbones supported by the offline family (all built with ``weights=None``).
TORCHVISION_BACKBONES: tuple[str, ...] = ("resnet18", "resnet34", "vgg16", "efficientnet_b0")


def sinusoidal_position_table(length: int, dim: int) -> Tensor:
    """Classic sinusoidal position table of shape ``(length, dim)``."""
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / max(dim, 1))
    )
    table = torch.zeros(length, dim, dtype=torch.float32)
    table[:, 0::2] = torch.sin(positions * divisor)
    table[:, 1::2] = torch.cos(positions * divisor[: table[:, 1::2].shape[1]])
    return table


def _strip_classifier(network: Any) -> Any:
    """Remove the classification head, keeping the convolutional feature map."""
    if hasattr(network, "fc") and hasattr(network, "layer4"):  # torchvision ResNet
        return nn.Sequential(*list(network.children())[:-2])
    if hasattr(network, "features"):  # VGG / EfficientNet
        return network.features
    return network


def build_torchvision_backbone(name: str) -> tuple[Any, int]:
    """Instantiate a torchvision feature extractor with random weights.

    Args:
        name: One of :data:`TORCHVISION_BACKBONES`.

    Returns:
        ``(module, channels)``: the classifier-free backbone and its output channel count
        (measured with one dry-run forward pass, so a stripped module never lies about it).

    Raises:
        OcrInputError: If ``name`` is not a supported backbone.
        TorchUnavailableError: If torchvision is not installed.
    """
    if name not in TORCHVISION_BACKBONES:
        raise OcrInputError(
            details={"reason": "unknown_backbone", "name": name, "allowed": TORCHVISION_BACKBONES}
        )
    models = require_torchvision()
    factory = getattr(models, name, None)
    if factory is None:  # pragma: no cover - torchvision always ships these
        raise OcrInputError(details={"reason": "backbone_missing_in_torchvision", "name": name})
    # weights=None: random init only. Pre-trained checkpoints are loaded from a local file after
    # the WEIGHTS_DOWNLOAD gate (ADR-0012); anything else here would trigger a download.
    network = factory(weights=None)
    backbone = _strip_classifier(network)
    with torch.no_grad():
        probe = backbone(torch.zeros(1, 3, 64, 64))
    if probe.ndim != 4:  # pragma: no cover - defensive
        raise OcrInputError(details={"reason": "backbone_output_rank", "name": name})
    return backbone, int(probe.shape[1])


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a projection shortcut (ResNet basic block)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Residual forward pass."""
        identity = self.shortcut(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + identity)


class SmallResNet(nn.Module):
    """Compact ResNet-style feature extractor (3 stages, /8 spatial reduction)."""

    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.stage1 = ResidualBlock(width, width, stride=1)
        self.stage2 = ResidualBlock(width, width * 2, stride=2)
        self.stage3 = ResidualBlock(width * 2, width * 4, stride=2)
        self.out_channels = width * 4

    def forward(self, x: Tensor) -> Tensor:
        """Return the ``(B, C, h, w)`` feature map."""
        return self.stage3(self.stage2(self.stage1(self.stem(x))))


class ImageEncoder(nn.Module):
    """Raster encoder: backbone -> 1x1 projection -> 2-D positional encoding -> sequence."""

    def __init__(self, cfg: OcrModelConfig) -> None:
        super().__init__()
        self.expects_rgb = cfg.encoder != "cnn_small"
        if cfg.encoder == "cnn_small":
            backbone: Any = SmallResNet(1, max(8, cfg.d_model // 4))
            channels = backbone.out_channels
        else:
            backbone, channels = build_torchvision_backbone(cfg.encoder)
        self.backbone = backbone
        self.project = nn.Conv2d(channels, cfg.d_model, 1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.d_model = cfg.d_model

    def forward(self, images: Tensor) -> Tensor:
        """Map ``(B, 1, H, W)`` rasters to a ``(B, S, d_model)`` memory sequence."""
        if images.ndim != 4:
            raise OcrInputError(details={"reason": "image_batch_rank", "shape": list(images.shape)})
        if self.expects_rgb and images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        features = self.project(self.backbone(images))
        _batch, dim, height, width = features.shape
        rows = sinusoidal_position_table(height, dim // 2).to(features.device)
        cols = sinusoidal_position_table(width, dim - dim // 2).to(features.device)
        grid = torch.cat(
            [
                rows[:, None, :].expand(height, width, dim // 2),
                cols[None, :, :].expand(height, width, dim - dim // 2),
            ],
            dim=2,
        )
        sequence = features.flatten(2).transpose(1, 2)
        sequence = sequence + grid.reshape(1, height * width, dim)
        return self.dropout(sequence)


class StrokeTransformerEncoder(nn.Module):
    """Transformer encoder over ``(B, T, F)`` trace features."""

    def __init__(self, cfg: OcrModelConfig) -> None:
        super().__init__()
        self.input = nn.Linear(cfg.stroke_feature_dim, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.heads,
            dim_feedforward=cfg.feedforward_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.layers, enable_nested_tensor=False
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.d_model = cfg.d_model

    def forward(self, features: Tensor) -> Tensor:
        """Map ``(B, T, F)`` features to a ``(B, T, d_model)`` memory sequence."""
        hidden = self.input(features)
        positions = sinusoidal_position_table(hidden.shape[1], hidden.shape[2])
        hidden = hidden + positions.to(hidden.device).unsqueeze(0)
        return self.dropout(self.encoder(hidden))


class StrokeBiLstmEncoder(nn.Module):
    """Bidirectional LSTM over ``(B, T, F)`` trace features."""

    def __init__(self, cfg: OcrModelConfig) -> None:
        super().__init__()
        hidden = max(1, cfg.d_model // 2)
        self.lstm = nn.LSTM(
            input_size=cfg.stroke_feature_dim,
            hidden_size=hidden,
            num_layers=cfg.layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.layers > 1 else 0.0,
        )
        self.project = nn.Linear(hidden * 2, cfg.d_model)
        self.d_model = cfg.d_model

    def forward(self, features: Tensor) -> Tensor:
        """Map ``(B, T, F)`` features to a ``(B, T, d_model)`` memory sequence."""
        output, _ = self.lstm(features)
        return self.project(output)


def build_encoder(cfg: OcrModelConfig) -> nn.Module:
    """Build the encoder module described by ``cfg``.

    Args:
        cfg: Model configuration.

    Returns:
        An ``nn.Module`` mapping the family's input tensor to ``(B, S, d_model)``.

    Raises:
        OcrInputError: If the encoder does not belong to the configured family.
    """
    if cfg.is_stroke_family:
        if cfg.encoder not in STROKE_ENCODERS:
            raise OcrInputError(
                details={
                    "reason": "encoder_family_mismatch",
                    "family": cfg.family,
                    "encoder": cfg.encoder,
                    "allowed": STROKE_ENCODERS,
                }
            )
        if cfg.encoder == "stroke_transformer":
            return StrokeTransformerEncoder(cfg)
        return StrokeBiLstmEncoder(cfg)
    if cfg.encoder not in IMAGE_ENCODERS:
        raise OcrInputError(
            details={
                "reason": "encoder_family_mismatch",
                "family": cfg.family,
                "encoder": cfg.encoder,
                "allowed": IMAGE_ENCODERS,
            }
        )
    return ImageEncoder(cfg)


__all__ = [
    "TORCHVISION_BACKBONES",
    "ImageEncoder",
    "ResidualBlock",
    "SmallResNet",
    "StrokeBiLstmEncoder",
    "StrokeTransformerEncoder",
    "build_encoder",
    "build_torchvision_backbone",
    "sinusoidal_position_table",
]
