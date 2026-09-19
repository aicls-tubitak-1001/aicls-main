"""Pure-function behaviour: enum coercion, records and the ``no raw SQL`` linter."""

from __future__ import annotations

from enum import StrEnum

import pytest

from aicls.core.errors import InputValidationError
from aicls.db.enums import (
    BOOK_PREDICATES,
    AgeBand,
    InputKind,
    NluSource,
    ResultKind,
    Role,
    UnitType,
    University,
    coerce_enum,
)
from aicls.db.results import ContribStats, LatencyStats, LoadReport, OperationCreate, Page


class TestCoerceEnum:
    """``coerce_enum`` is what lets neighbouring packages pass their own string enums."""

    def test_accepts_members_values_and_names(self) -> None:
        assert coerce_enum(Role, Role.ADMIN) is Role.ADMIN
        assert coerce_enum(Role, "instructor") is Role.INSTRUCTOR
        assert coerce_enum(Role, "INSTRUCTOR") is Role.INSTRUCTOR
        assert coerce_enum(Role, "Instructor") is Role.INSTRUCTOR

    def test_accepts_an_equivalent_enum_from_another_package(self) -> None:
        class TheirUnitType(StrEnum):
            DEFINITION = "Definition"

        assert coerce_enum(UnitType, TheirUnitType.DEFINITION) is UnitType.DEFINITION

    def test_values_with_punctuation_round_trip(self) -> None:
        assert coerce_enum(NluSource, "asr+rules") is NluSource.ASR_RULES
        assert coerce_enum(AgeBand, "45+") is AgeBand.B45_PLUS
        assert coerce_enum(University, "mus") is University.MUS

    def test_unknown_values_raise_with_the_allowed_list(self) -> None:
        with pytest.raises(InputValidationError) as excinfo:
            coerce_enum(Role, "root")
        assert excinfo.value.details["allowed"] == ["student", "instructor", "contributor", "admin"]
        assert excinfo.value.code.value == "VALIDATION_ERROR"


def test_book_predicates_match_the_adr_vocabulary() -> None:
    """ADR-0019 fixes the predicate vocabulary flattened into ``book_triples``."""
    assert BOOK_PREDICATES == (
        "aicls:partOf",
        "aicls:requires",
        "aicls:illustrates",
        "aicls:proves",
        "aicls:next",
        "aicls:hasFigure",
        "skos:related",
    )


class TestRecords:
    """Plain records handed to the API layer."""

    def test_page_is_frozen_and_sized(self) -> None:
        page = Page(items=[1, 2, 3], next_cursor="01ABC")
        assert len(page) == 3
        with pytest.raises(AttributeError):
            page.items = []  # type: ignore[misc]

    def test_stats_records_have_the_api_field_names(self) -> None:
        latency = LatencyStats(mean_ms=1.0, p50_ms=2.0, p95_ms=3.0, n=4)
        assert latency.by_modality == {}
        contrib = ContribStats(hours_accepted=1.5, utterances=10, speakers=2)
        assert (contrib.by_university, contrib.qc_pass_rate) == ({}, 0.0)
        assert LoadReport(units=1, triples=2, aliases=3).tagged_ratio == 1.0

    def test_operation_create_from_mapping_reads_aliases_and_coerces_enums(self) -> None:
        created = OperationCreate.from_mapping(
            {
                "input_kind": "INK",
                "text": "türevini al",
                "latex": "x^2",
                "command_json": {"op": "diff"},
                "result_kind": "expr",
                "steps_json": {"1": "..."},
                "timings": {"total": 1.0},
                "model_versions_json": {"ocr": "ocr-0.1.0"},
                "nlu_source": "asr+rules",
                "cache_hit": 1,
            }
        )
        assert created.input_kind is InputKind.INK
        assert created.source_text == "türevini al"
        assert created.input_latex == "x^2"
        assert created.command == {"op": "diff"}
        assert created.result_kind is ResultKind.EXPR
        assert created.steps == {"1": "..."}
        assert created.timings == {"total": 1.0}
        assert created.model_versions == {"ocr": "ocr-0.1.0"}
        assert created.nlu_source is NluSource.ASR_RULES
        assert created.cache_hit is True

    def test_operation_create_from_an_empty_mapping_uses_text_defaults(self) -> None:
        created = OperationCreate.from_mapping({})
        assert created.input_kind is InputKind.TEXT
        assert (created.result_kind, created.nlu_source, created.steps) == (None, None, None)
        assert created.command == {} and created.timings == {}


def test_lint_no_raw_sql_is_green_for_the_db_package(load_script: object) -> None:
    """``tools/lint_no_raw_sql.py`` must stay green (docs/guvenlik-gizlilik.md §5)."""
    module = load_script("tools/lint_no_raw_sql.py")  # type: ignore[operator]
    from pathlib import Path

    root = Path(module.__file__).resolve().parents[1]
    problems = module.scan_paths([root / "src" / "aicls" / "db"], root)
    assert problems == []
