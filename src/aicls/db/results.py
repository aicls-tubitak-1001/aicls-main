"""Plain records returned by the repositories (no ORM objects leak into these).

The API layer converts them straight into its pydantic response models, so the field names match
docs/mimari/api.md (``{items, next_cursor}`` pagination, ``{mean_ms, p50_ms, p95_ms, n,
by_modality}`` latency statistics, the contribution dashboard payload).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from aicls.db.enums import InputKind, NluSource, ResultKind, coerce_enum

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Page(Generic[T]):
    """One page of a cursor-paginated listing (docs/mimari/api.md §1).

    Attributes:
        items: Rows of this page, newest first.
        next_cursor: Opaque cursor for the following page, or ``None`` when exhausted.
    """

    items: list[T]
    next_cursor: str | None = None

    def __len__(self) -> int:
        """Number of items on this page."""
        return len(self.items)


@dataclass(frozen=True, slots=True)
class LatencyStats:
    """Aggregated latency of measured requests (contractual "< 3 s" metric)."""

    mean_ms: float
    p50_ms: float
    p95_ms: float
    n: int
    by_modality: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContribStats:
    """Voice-contribution dashboard figures (``GET /contribute/stats``)."""

    hours_accepted: float
    utterances: int
    speakers: int
    by_university: dict[str, int] = field(default_factory=dict)
    by_gender: dict[str, int] = field(default_factory=dict)
    by_age_band: dict[str, int] = field(default_factory=dict)
    qc_pass_rate: float = 0.0


@dataclass(frozen=True, slots=True)
class LoadReport:
    """Outcome of a book ingest (``POST /admin/book/load``, İP-8 100 % tagging criterion)."""

    units: int
    triples: int
    aliases: int
    tagged_ratio: float = 1.0


@dataclass(frozen=True, slots=True)
class OperationCreate:
    """Everything the pipeline knows about one finished operation.

    ``OperationRepository.add`` also accepts a plain mapping with the same keys, so the API layer
    may pass ``model_dump()`` output directly.
    """

    input_kind: InputKind
    source_text: str | None = None
    input_latex: str | None = None
    command: dict[str, Any] = field(default_factory=dict)
    result_kind: ResultKind | None = None
    result_latex: str | None = None
    result_srepr: str | None = None
    steps: dict[str, Any] | None = None
    plot_asset_id: str | None = None
    error_code: str | None = None
    timings: dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False
    model_versions: dict[str, Any] = field(default_factory=dict)
    nlu_source: NluSource | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> OperationCreate:
        """Build from a mapping, accepting the column aliases used by the API schemas.

        ``command``/``command_json``, ``timings``/``timings_json``, ``steps``/``steps_json`` and
        ``model_versions``/``model_versions_json`` are treated as the same field.

        Raises:
            InputValidationError: If an enum-valued key holds an unknown value.
        """

        def pick(*names: str, default: Any = None) -> Any:
            for name in names:
                if name in data and data[name] is not None:
                    return data[name]
            return default

        input_kind = coerce_enum(InputKind, pick("input_kind", default=InputKind.TEXT))
        raw_result_kind = pick("result_kind")
        raw_nlu_source = pick("nlu_source")
        return cls(
            input_kind=input_kind,
            source_text=pick("source_text", "text"),
            input_latex=pick("input_latex", "latex"),
            command=dict(pick("command", "command_json", default={})),
            result_kind=(
                coerce_enum(ResultKind, raw_result_kind) if raw_result_kind is not None else None
            ),
            result_latex=pick("result_latex"),
            result_srepr=pick("result_srepr"),
            steps=pick("steps", "steps_json"),
            plot_asset_id=pick("plot_asset_id"),
            error_code=pick("error_code"),
            timings=dict(pick("timings", "timings_json", default={})),
            cache_hit=bool(pick("cache_hit", default=False)),
            model_versions=dict(pick("model_versions", "model_versions_json", default={})),
            nlu_source=(
                coerce_enum(NluSource, raw_nlu_source) if raw_nlu_source is not None else None
            ),
        )


__all__ = [
    "ContribStats",
    "LatencyStats",
    "LoadReport",
    "OperationCreate",
    "Page",
]
