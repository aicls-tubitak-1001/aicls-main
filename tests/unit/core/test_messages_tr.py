"""messages_tr: the single source of Turkish user-facing strings, including the contract text."""

from __future__ import annotations

import pytest

from aicls.core import messages_tr
from aicls.core.errors import ErrorCode
from aicls.core.messages_tr import MSG_UNSUPPORTED_PROBLEM, msg


def test_unsupported_problem_message_is_contract_text() -> None:
    assert MSG_UNSUPPORTED_PROBLEM == "Bu problem tipi şu an için desteklenmemektedir."
    assert messages_tr.MESSAGES["UNSUPPORTED_PROBLEM"] == MSG_UNSUPPORTED_PROBLEM
    assert msg("UNSUPPORTED_PROBLEM") == MSG_UNSUPPORTED_PROBLEM


def test_msg_formats_keyword_arguments() -> None:
    text = msg("PARSE_ERROR_AT", position=8, reason="} bekleniyor")
    assert text == "Konum 8: } bekleniyor"


def test_msg_accepts_error_code_enum() -> None:
    assert msg(ErrorCode.NOT_FOUND) == messages_tr.MESSAGES["NOT_FOUND"]


def test_msg_unknown_key_raises_key_error() -> None:
    with pytest.raises(KeyError):
        msg("NO_SUCH_MESSAGE")


def test_msg_missing_placeholder_raises_key_error() -> None:
    with pytest.raises(KeyError):
        msg("PARSE_ERROR_AT", position=1)


def test_all_message_constants_are_registered() -> None:
    seen = 0
    for name, value in vars(messages_tr).items():
        if name.startswith("MSG_"):
            key = name.removeprefix("MSG_")
            assert messages_tr.MESSAGES[key] == value, name
            assert isinstance(value, str) and value.strip(), name
            seen += 1
    assert seen >= len(ErrorCode)


def test_titles_cover_every_code_and_are_turkish_short() -> None:
    for code in ErrorCode:
        title = messages_tr.TITLES[code.value]
        assert title and len(title) < 60
