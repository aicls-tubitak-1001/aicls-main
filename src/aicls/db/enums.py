"""Enumerated column vocabularies (docs/mimari/veri-modeli.md §3).

Every enumerated column is stored as a dialect-neutral ``VARCHAR(32)`` with a CHECK constraint
(``sa.Enum(..., native_enum=False)``) and mapped to a :class:`~enum.StrEnum` here, so the same
schema compiles on SQLite and MySQL (ADR-0003, dialect rule 3). Values — not member names — are
what reaches the database; :func:`coerce_enum` accepts either form because neighbouring packages
(``aicls.book``, ``aicls.api``) define their own equivalent string enums.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypeVar

from aicls.core.errors import InputValidationError
from aicls.core.messages_tr import msg

#: Length of every non-native enum column (docs/mimari/veri-modeli.md §1 rule 3).
ENUM_LENGTH = 32

E = TypeVar("E", bound=StrEnum)


class Role(StrEnum):
    """Platform role of a user (RBAC matrix, docs/mimari/api.md §1)."""

    STUDENT = "student"
    INSTRUCTOR = "instructor"
    CONTRIBUTOR = "contributor"
    ADMIN = "admin"


class University(StrEnum):
    """Partner universities of the project; ``OTHER`` for everyone else."""

    MUS = "mus"
    MERSIN = "mersin"
    AMASYA = "amasya"
    OTHER = "other"


class WorkspaceRole(StrEnum):
    """Role of a member inside a shared workspace."""

    VIEWER = "viewer"
    EDITOR = "editor"


class InputKind(StrEnum):
    """How the user expressed the request that produced an operation."""

    TEXT = "text"
    LATEX = "latex"
    INK = "ink"
    AUDIO = "audio"


class ResultKind(StrEnum):
    """Shape of a CAS result (docs/mimari/api.md §3.3)."""

    EXPR = "expr"
    NUMBER = "number"
    SET = "set"
    BOOL = "bool"
    PLOT = "plot"
    STEPS = "steps"


class AssetKind(StrEnum):
    """Kind of binary artefact stored under ``AICLS_DATA_ROOT``."""

    PLOT_PNG = "plot_png"
    INK_JSON = "ink_json"
    INK_PNG = "ink_png"
    AUDIO_WAV = "audio_wav"
    BOOK_FIGURE = "book_figure"
    SCREENSHOT = "screenshot"


class NluSource(StrEnum):
    """Which layer produced the command of an operation (İP-4)."""

    RULES = "rules"
    ASR_RULES = "asr+rules"
    LLM = "llm"


class AgeBand(StrEnum):
    """Coarse age band of a voice contributor (KVKK: bands only, never a birth date)."""

    B18_24 = "18-24"
    B25_34 = "25-34"
    B35_44 = "35-44"
    B45_PLUS = "45+"


class Gender(StrEnum):
    """Self-declared gender of a voice contributor (``NA`` = not answered)."""

    F = "f"
    M = "m"
    NA = "na"


class ReviewStatus(StrEnum):
    """State of a recording in the human QC queue (İP-3 data collection)."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FIXED = "fixed"


class Split(StrEnum):
    """Dataset split a recording belongs to (speaker-disjoint, ADR-0008)."""

    UNASSIGNED = "unassigned"
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class UnitType(StrEnum):
    """Type of a textbook content unit (İP-8 vocabulary, ADR-0019)."""

    DEFINITION = "Definition"
    THEOREM = "Theorem"
    LEMMA = "Lemma"
    COROLLARY = "Corollary"
    PROOF = "Proof"
    EXAMPLE = "Example"
    EXERCISE = "Exercise"
    FIGURE = "Figure"
    TABLE = "Table"
    REMARK = "Remark"
    SECTION = "Section"


class TagStatus(StrEnum):
    """Whether a book unit was tagged automatically or reviewed by a human."""

    AUTO = "auto"
    REVIEWED = "reviewed"


class ModelComponent(StrEnum):
    """Trainable component a model version belongs to (İP-3)."""

    OCR = "ocr"
    ASR = "asr"


class EvalKind(StrEnum):
    """Evaluation runner that produced a report (docs/ml/degerlendirme.md)."""

    PARSER = "parser"
    CAS = "cas"
    NLU = "nlu"
    OCR = "ocr"
    ASR = "asr"
    BOOK = "book"
    LATENCY = "latency"


class Modality(StrEnum):
    """Input modality of a latency sample (mirrors :class:`InputKind`)."""

    TEXT = "text"
    LATEX = "latex"
    INK = "ink"
    AUDIO = "audio"


class FeedbackCategory(StrEnum):
    """Category of a pilot-study feedback entry (İP-9)."""

    CRITICAL = "critical"
    USABILITY = "usability"
    SUGGESTION = "suggestion"


class FeedbackStatus(StrEnum):
    """Triage state of a feedback entry (İP-9)."""

    OPEN = "open"
    TRIAGED = "triaged"
    FIXED = "fixed"
    DEFERRED = "deferred"


#: Predicates allowed in ``book_triples.predicate`` (ADR-0019). Stored as ``VARCHAR(32)`` rather
#: than an enum column because ``aicls.book`` owns the vocabulary and may extend it.
BOOK_PREDICATES: tuple[str, ...] = (
    "aicls:partOf",
    "aicls:requires",
    "aicls:illustrates",
    "aicls:proves",
    "aicls:next",
    "aicls:hasFigure",
    "skos:related",
)


def coerce_enum(enum_cls: type[E], value: Any) -> E:
    """Return ``value`` as a member of ``enum_cls``.

    Accepts a member, its value (``"student"``) or its name in any case (``"STUDENT"``) so that
    equivalent string enums defined by other packages interoperate with the repositories.

    Args:
        enum_cls: Target :class:`~enum.StrEnum` subclass.
        value: Member, value or name to convert.

    Returns:
        The matching enum member.

    Raises:
        InputValidationError: If ``value`` matches no member.
    """
    if isinstance(value, enum_cls):
        return value
    text = str(getattr(value, "value", value))
    try:
        return enum_cls(text)
    except ValueError:
        pass
    for member in enum_cls:
        if member.name.casefold() == text.casefold():
            return member
    raise InputValidationError(
        msg("VALIDATION_ERROR"),
        details={"field": enum_cls.__name__, "value": text, "allowed": [m.value for m in enum_cls]},
    )


__all__ = [
    "BOOK_PREDICATES",
    "ENUM_LENGTH",
    "AgeBand",
    "AssetKind",
    "EvalKind",
    "FeedbackCategory",
    "FeedbackStatus",
    "Gender",
    "InputKind",
    "Modality",
    "ModelComponent",
    "NluSource",
    "ResultKind",
    "ReviewStatus",
    "Role",
    "Split",
    "TagStatus",
    "UnitType",
    "University",
    "WorkspaceRole",
    "coerce_enum",
]
