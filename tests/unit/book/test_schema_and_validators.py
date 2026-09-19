"""Schema vocabulary, coverage arithmetic and the untagged-heading refusal."""

from __future__ import annotations

import pytest

from aicls.book import (
    ContentUnit,
    Predicate,
    TagResult,
    Triple,
    UnitType,
    UntaggedHeadingError,
    UntaggedSpan,
    coverage,
    require_full_coverage,
    unit_id,
    untagged_headings,
)
from aicls.book.ingest.validators import REQUIRED_COVERAGE
from aicls.book.schema import SLUG_BY_TYPE, TYPE_BY_SLUG
from aicls.core.errors import ErrorCode, InputValidationError
from aicls.db.enums import BOOK_PREDICATES
from aicls.db.enums import UnitType as DbUnitType


def test_unit_type_matches_the_database_vocabulary() -> None:
    """``aicls.book.UnitType`` and ``aicls.db.enums.UnitType`` carry the same values."""
    assert [member.value for member in UnitType] == [member.value for member in DbUnitType]


def test_predicate_matches_the_database_allow_list() -> None:
    """Every predicate is one of the CURIEs ``book_triples.predicate`` accepts (ADR-0019)."""
    assert {member.value for member in Predicate} == set(BOOK_PREDICATES)


def test_slug_table_is_a_bijection() -> None:
    """Each unit type mints exactly one identifier prefix."""
    assert set(SLUG_BY_TYPE) == set(UnitType)
    assert len(TYPE_BY_SLUG) == len(SLUG_BY_TYPE)
    assert unit_id(UnitType.DEFINITION, "2.4") == "tanim-2.4"


def test_content_unit_rejects_unknown_fields() -> None:
    """``extra="forbid"`` keeps a typo from silently becoming an ignored field."""
    with pytest.raises(ValueError, match="titel_tr"):
        ContentUnit(id="x", type=UnitType.REMARK, titel_tr="yanlış")  # type: ignore[call-arg]


def test_triple_key_is_the_dedup_key() -> None:
    """``Triple.key`` renders the predicate as its stored CURIE."""
    triple = Triple(subject="a", predicate=Predicate.REQUIRES, object="b")
    assert triple.key() == ("a", "aicls:requires", "b")
    assert Triple(subject="a", predicate="aicls:requires", object="b") == triple


def test_coverage_is_the_tagged_share() -> None:
    """Coverage counts tagged nodes against tagged plus untagged headings."""
    node = ContentUnit(id="tanim-1.1", type=UnitType.DEFINITION)
    span = UntaggedSpan(3, "Tanım: Süreklilik")
    assert coverage(TagResult(nodes=[node])) == REQUIRED_COVERAGE
    assert coverage(TagResult(nodes=[node], untagged_spans=[span])) == 0.5
    assert coverage(TagResult(untagged_spans=[span])) == 0.0
    assert coverage(TagResult()) == REQUIRED_COVERAGE


def test_require_full_coverage_raises_and_lists_the_headings() -> None:
    """The refusal carries the heading list the operator needs, as an RFC 9457 detail."""
    node = ContentUnit(id="tanim-1.1", type=UnitType.DEFINITION)
    spans = [UntaggedSpan(3, "Tanım: Süreklilik", "missing_number", "a.md")]
    good = TagResult(nodes=[node])
    bad = TagResult(nodes=[node], untagged_spans=spans)
    assert require_full_coverage(good) == 1.0
    with pytest.raises(UntaggedHeadingError) as excinfo:
        require_full_coverage(good, bad)
    error = excinfo.value
    assert isinstance(error, InputValidationError)
    assert error.code is ErrorCode.VALIDATION_ERROR
    assert error.headings == ["Tanım: Süreklilik"]
    assert error.details["untagged_headings"] == ["Tanım: Süreklilik"]
    assert error.details["untagged"][0]["line"] == 3
    assert error.details["untagged"][0]["source"] == "a.md"
    assert error.tagged_ratio == pytest.approx(2 / 3)
    assert error.to_problem()["status"] == 422


def test_untagged_headings_collects_across_results() -> None:
    """The helper flattens the spans of several files in order."""
    left = TagResult(untagged_spans=[UntaggedSpan(1, "a")])
    right = TagResult(untagged_spans=[UntaggedSpan(2, "b")])
    assert [span.text for span in untagged_headings(left, right)] == ["a", "b"]


def test_tag_result_unit_ids() -> None:
    """``TagResult.unit_ids`` is the set the loader checks its edges against."""
    result = TagResult(nodes=[ContentUnit(id="tanim-1.1", type=UnitType.DEFINITION)])
    assert result.unit_ids() == {"tanim-1.1"}
