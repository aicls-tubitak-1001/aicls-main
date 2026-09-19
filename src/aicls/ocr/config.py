"""Model configuration for the OCR encoder-decoder families (ADR-0006, ADR-0013).

:class:`OcrModelConfig` mirrors the ``model:`` block of ``training/configs/ocr/*.yaml`` so the
same object can be built from a YAML training config, written into a checkpoint's
``config.json`` and passed to :func:`aicls.ocr.models.build_model`. It is a plain pydantic model:
importing it never touches torch.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aicls.ocr.features import STROKE_FEATURE_DIM

#: Encoders that consume rasters (``family: seq2seq_image``).
IMAGE_ENCODERS: tuple[str, ...] = ("cnn_small", "resnet18", "resnet34", "vgg16", "efficientnet_b0")
#: Encoders that consume trace features (``family: seq2seq_stroke``).
STROKE_ENCODERS: tuple[str, ...] = ("stroke_transformer", "stroke_bilstm")

ModelFamily = Literal["seq2seq_image", "seq2seq_stroke"]
EncoderName = Literal[
    "cnn_small",
    "resnet18",
    "resnet34",
    "vgg16",
    "efficientnet_b0",
    "stroke_transformer",
    "stroke_bilstm",
]
DecoderName = Literal["transformer", "lstm_attn", "gru_attn"]


class OcrModelConfig(BaseModel):
    """Architecture description of one OCR model.

    Attributes:
        family: ``seq2seq_image`` (offline raster) or ``seq2seq_stroke`` (online trace).
        encoder: Backbone name; torchvision backbones are always built with ``weights=None``.
        decoder: Autoregressive decoder family.
        d_model: Hidden size shared by the encoder projection and the decoder.
        layers: Decoder layer count (also the encoder depth for the stroke encoders).
        heads: Attention heads; must divide ``d_model``.
        ff_dim: Feed-forward width; ``4 * d_model`` when omitted.
        dropout: Dropout probability.
        max_len: Maximum decoded sequence length.
        image_size: ``(H, W)`` raster size for the image family.
        stroke_feature_dim: Input width of the stroke family features.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    family: ModelFamily = "seq2seq_image"
    encoder: EncoderName = "cnn_small"
    decoder: DecoderName = "transformer"
    d_model: int = Field(default=256, ge=8, le=2048)
    layers: int = Field(default=6, ge=1, le=24)
    heads: int = Field(default=8, ge=1, le=32)
    ff_dim: int | None = Field(default=None, ge=8, le=8192)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    max_len: int = Field(default=160, ge=1, le=1024)
    image_size: tuple[int, int] = (96, 384)
    stroke_feature_dim: int = Field(default=STROKE_FEATURE_DIM, ge=1, le=512)

    @field_validator("image_size")
    @classmethod
    def _positive_size(cls, value: tuple[int, int]) -> tuple[int, int]:
        if value[0] <= 0 or value[1] <= 0:
            raise ValueError("image_size must be positive")
        return value

    @model_validator(mode="after")
    def _check_dimensions(self) -> OcrModelConfig:  # noqa: N804 - "after" validators take self
        if self.d_model % self.heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by heads ({self.heads})")
        return self

    @property
    def feedforward_dim(self) -> int:
        """Feed-forward width actually used by the model."""
        return self.ff_dim if self.ff_dim is not None else 4 * self.d_model

    @property
    def is_stroke_family(self) -> bool:
        """True for the online (trace) family."""
        return self.family == "seq2seq_stroke"


__all__ = [
    "IMAGE_ENCODERS",
    "STROKE_ENCODERS",
    "DecoderName",
    "EncoderName",
    "ModelFamily",
    "OcrModelConfig",
]
