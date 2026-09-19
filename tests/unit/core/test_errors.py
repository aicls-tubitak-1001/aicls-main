"""AiclsError hierarchy, ErrorCode enum and RFC 9457 problem-details rendering."""

from __future__ import annotations

import pytest

from aicls.core import messages_tr
from aicls.core.errors import (
    ERROR_TYPE_BASE,
    HTTP_STATUS_BY_CODE,
    AiclsError,
    ErrorCode,
    InputValidationError,
    NotFoundError,
)


def test_error_code_is_str_enum_with_all_contract_codes() -> None:
    expected = {
        "PARSE_ERROR",
        "UNSUPPORTED_CONSTRUCT",
        "UNSUPPORTED_PROBLEM",
        "CAS_TIMEOUT",
        "NLU_NO_INTENT",
        "OCR_INPUT_INVALID",
        "ASR_INPUT_INVALID",
        "UPLOAD_INVALID",
        "CONSENT_REQUIRED",
        "AUTH_INVALID",
        "FORBIDDEN",
        "NOT_FOUND",
        "RATE_LIMITED",
        "TRAINING_BUSY",
        "VALIDATION_ERROR",
        "GATE_BLOCKED",
        "INTERNAL",
    }
    assert {c.value for c in ErrorCode} == expected
    assert ErrorCode.PARSE_ERROR == "PARSE_ERROR"
    assert isinstance(ErrorCode.INTERNAL, str)
    # every code has an HTTP status, a title and a default message
    for code in ErrorCode:
        assert code in HTTP_STATUS_BY_CODE
        assert code.value in messages_tr.TITLES
        assert code.value in messages_tr.MESSAGES


def test_aicls_error_defaults_from_code() -> None:
    err = AiclsError(ErrorCode.UNSUPPORTED_PROBLEM)
    assert err.code is ErrorCode.UNSUPPORTED_PROBLEM
    assert err.message_tr == messages_tr.MSG_UNSUPPORTED_PROBLEM
    assert err.http_status == 422
    assert err.details == {}
    assert err.exit_code == 1
    assert str(err) == "UNSUPPORTED_PROBLEM: Bu problem tipi şu an için desteklenmemektedir."


def test_aicls_error_accepts_string_code_and_overrides() -> None:
    err = AiclsError("PARSE_ERROR", "Konum 8: } bekleniyor", {"position": 8}, 400)
    assert err.code is ErrorCode.PARSE_ERROR
    assert err.message_tr == "Konum 8: } bekleniyor"
    assert err.details == {"position": 8}
    assert err.http_status == 400


def test_aicls_error_unknown_code_maps_to_internal() -> None:
    err = AiclsError("NOT_A_CODE")
    assert err.code is ErrorCode.INTERNAL
    assert err.details["original_code"] == "NOT_A_CODE"
    assert err.http_status == 500


def test_to_problem_is_rfc9457_dict() -> None:
    err = AiclsError(
        ErrorCode.PARSE_ERROR,
        "Konum 8: ifade tamamlanmamış, } bekleniyor",
        {"position": 8, "errors": [{"loc": ["body", "latex"], "msg": "x"}]},
    )
    problem = err.to_problem(instance="01J8R2ZQ7G4V0X9K3M2N5P6Q7R")
    assert problem == {
        "type": f"{ERROR_TYPE_BASE}/PARSE_ERROR",
        "title": "LaTeX ifadesi ayrıştırılamadı",
        "status": 422,
        "detail_tr": "Konum 8: ifade tamamlanmamış, } bekleniyor",
        "code": "PARSE_ERROR",
        "instance": "01J8R2ZQ7G4V0X9K3M2N5P6Q7R",
        "position": 8,
        "errors": [{"loc": ["body", "latex"], "msg": "x"}],
    }


def test_to_problem_without_instance_and_reserved_key_collision() -> None:
    err = AiclsError(ErrorCode.INTERNAL, details={"status": "weird", "extra": 1})
    problem = err.to_problem()
    assert "instance" not in problem
    assert problem["status"] == 500  # reserved key wins
    assert problem["details"] == {"status": "weird"}
    assert problem["extra"] == 1


def test_details_are_copied_not_shared() -> None:
    d = {"a": 1}
    err = AiclsError(ErrorCode.INTERNAL, details=d)
    d["a"] = 2
    assert err.details == {"a": 1}


def test_subclasses_fix_code_and_status() -> None:
    nf = NotFoundError("Kayıt yok", details={"id": "x"})
    assert nf.code is ErrorCode.NOT_FOUND
    assert nf.http_status == 404
    assert nf.message_tr == "Kayıt yok"
    vf = InputValidationError()
    assert vf.code is ErrorCode.VALIDATION_ERROR
    assert vf.http_status == 422
    assert isinstance(vf, AiclsError)


def test_error_is_raiseable_and_catchable_as_aicls_error() -> None:
    with pytest.raises(AiclsError) as ei:
        raise NotFoundError()
    assert ei.value.code is ErrorCode.NOT_FOUND


def test_repr_is_informative() -> None:
    err = AiclsError(ErrorCode.FORBIDDEN, details={"role": "student"})
    text = repr(err)
    assert "FORBIDDEN" in text and "role" in text
