"""Shared request/response models: problem documents, pages and stage timings (api.md §1).

Every *request* model forbids unknown fields, so a client that invents a key gets a
``VALIDATION_ERROR`` instead of having it silently ignored; every *response* model is explicit,
so nothing an ORM row happens to carry can leak into JSON.
"""

from __future__ import annotations

from typing import Annotated, Any, Generic, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

T = TypeVar("T")

#: Conservative e-mail shape. ``pydantic.EmailStr`` needs the optional ``email-validator``
#: package, which the approved dependency set does not carry, so the constraint is spelled out
#: here: one local part, one domain with at least one dot, no whitespace, at most 255 characters.
EMAIL_PATTERN = r"^[^@\s]{1,64}@[^@\s.]{1,63}(?:\.[^@\s.]{1,63})+$"

#: A normalised e-mail address (trimmed and lower-cased before validation).
EmailAddress: TypeAlias = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, to_lower=True, min_length=6, max_length=255, pattern=EMAIL_PATTERN
    ),
]

#: Platform roles (docs/mimari/api.md §1 "Roller").
RoleName: TypeAlias = Literal["student", "instructor", "contributor", "admin"]

#: How the user expressed the request (``operations.input_kind``).
InputKindName: TypeAlias = Literal["text", "latex", "ink", "audio"]

#: Shape of a CAS result value.
ResultKindName: TypeAlias = Literal["expr", "number", "set", "bool", "plot", "steps"]

#: Outcome discriminator of ``ComputeOut``.
ComputeKindName: TypeAlias = Literal["result", "clarification", "book", "ui"]

#: Where the command came from.
NluSourceName: TypeAlias = Literal["rules", "asr+rules", "llm"]

#: Largest page a client may ask for (mirrors ``aicls.db.repositories.base.MAX_PAGE_SIZE``).
MAX_LIMIT = 200
#: Default page size.
DEFAULT_LIMIT = 50


class ApiModel(BaseModel):
    """Base of every response model (unknown fields are a programming error)."""

    model_config = ConfigDict(extra="forbid")


class RequestModel(BaseModel):
    """Base of every request model (unknown fields are rejected, whitespace trimmed)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ProblemOut(ApiModel):
    """RFC 9457 problem document (api.md §2); documented for OpenAPI, built by hand at runtime.

    ``extra="allow"`` because the structured ``details`` of the raising error are merged into the
    document; the fields spelled out here are the ones the web client reads by name.
    ``construct_name`` carries the wire name ``construct`` as an alias, because a pydantic field
    literally called ``construct`` would shadow a :class:`~pydantic.BaseModel` attribute.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    type: str = Field(examples=["https://ai-cls.example/errors/PARSE_ERROR"])
    title: str = Field(examples=["LaTeX ifadesi ayrıştırılamadı"])
    status: int = Field(examples=[422])
    code: str = Field(examples=["PARSE_ERROR"])
    detail_tr: str = Field(default="", examples=["Konum 8: ..."])
    instance: str | None = None
    position: int | None = None
    construct_name: str | None = Field(default=None, alias="construct")
    suggestions_tr: list[str] | None = None
    errors: list[dict[str, Any]] | None = None


class Page(ApiModel, Generic[T]):
    """One page of a cursor-paginated listing (``{items, next_cursor}``)."""

    items: list[T]
    next_cursor: str | None = None


class TimingsOut(ApiModel):
    """Per-stage durations in milliseconds (ADR-0014); unknown stages are allowed."""

    model_config = ConfigDict(extra="allow")

    nlu: float | None = None
    parse: float | None = None
    cas: float | None = None
    render: float | None = None
    total: float = 0.0


def timings_out(stages: dict[str, float], total_ms: float) -> TimingsOut:
    """Build a :class:`TimingsOut` from a :class:`~aicls.core.timing.StageTimer` snapshot."""
    rounded = {name: round(value, 3) for name, value in stages.items()}
    rounded["total"] = round(total_ms, 3)
    return TimingsOut.model_validate(rounded)


__all__ = [
    "DEFAULT_LIMIT",
    "EMAIL_PATTERN",
    "MAX_LIMIT",
    "ApiModel",
    "ComputeKindName",
    "EmailAddress",
    "InputKindName",
    "NluSourceName",
    "Page",
    "ProblemOut",
    "RequestModel",
    "ResultKindName",
    "RoleName",
    "TimingsOut",
    "timings_out",
]
