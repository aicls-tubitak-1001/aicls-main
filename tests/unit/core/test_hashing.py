"""sha256 helpers and the order-sensitive JSONL fingerprint."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from aicls.core.errors import AiclsError, ErrorCode
from aicls.core.hashing import fingerprint_jsonl, sha256_bytes, sha256_file, sha256_json


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    p = tmp_path / "blob.bin"
    data = bytes(range(256)) * 1000
    p.write_bytes(data)
    assert sha256_file(p) == hashlib.sha256(data).hexdigest()
    assert sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def test_sha256_file_missing_raises_aicls_error(tmp_path: Path) -> None:
    with pytest.raises(AiclsError) as ei:
        sha256_file(tmp_path / "nope.bin")
    assert ei.value.code is ErrorCode.NOT_FOUND


def test_sha256_json_is_key_order_independent_and_unicode_safe() -> None:
    a = sha256_json({"b": 1, "a": [1, 2, {"z": "ş"}]})
    b = sha256_json({"a": [1, 2, {"z": "ş"}], "b": 1})
    assert a == b
    assert a != sha256_json({"a": [1, 2, {"z": "s"}], "b": 1})
    # canonical form: sorted keys, compact separators, no ascii escaping
    canon = '{"a":[1,2,{"z":"ş"}],"b":1}'.encode()
    assert a == hashlib.sha256(canon).hexdigest()


def test_sha256_json_serialises_paths_and_sets_deterministically() -> None:
    obj = {"p": Path("x") / "y", "s": {3, 1, 2}}
    assert sha256_json(obj) == sha256_json({"s": {2, 3, 1}, "p": Path("x") / "y"})


def test_sha256_json_rejects_unserialisable() -> None:
    with pytest.raises(AiclsError) as ei:
        sha256_json({"f": object()})
    assert ei.value.code is ErrorCode.VALIDATION_ERROR


def test_fingerprint_jsonl_is_order_sensitive(tmp_path: Path) -> None:
    lines = ['{"id": 1, "text": "türev"}', '{"id": 2, "text": "integral"}']
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text("\n".join(lines) + "\n", encoding="utf-8")
    p2.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    f1 = fingerprint_jsonl(p1)
    f2 = fingerprint_jsonl(p2)
    assert f1.startswith("sha256:") and len(f1) == len("sha256:") + 64
    assert f1 != f2, "row order is part of the dataset identity (moduller.md: sıra dahil)"


def test_fingerprint_jsonl_ignores_line_ending_style_and_trailing_newline(tmp_path: Path) -> None:
    lines = ['{"a": 1}', '{"a": 2}']
    lf = tmp_path / "lf.jsonl"
    crlf = tmp_path / "crlf.jsonl"
    no_final = tmp_path / "nf.jsonl"
    lf.write_bytes(("\n".join(lines) + "\n").encode())
    crlf.write_bytes(("\r\n".join(lines) + "\r\n").encode())
    no_final.write_bytes("\n".join(lines).encode())
    assert fingerprint_jsonl(lf) == fingerprint_jsonl(crlf) == fingerprint_jsonl(no_final)


def test_fingerprint_jsonl_sensitive_to_content_and_line_count(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text('{"a": 1}\n', encoding="utf-8")
    b.write_text('{"a": 1}\n{"a": 1}\n', encoding="utf-8")
    assert fingerprint_jsonl(a) != fingerprint_jsonl(b)


def test_fingerprint_jsonl_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(AiclsError):
        fingerprint_jsonl(tmp_path / "missing.jsonl")


def test_sha256_json_serialises_bytes_as_hex() -> None:
    from aicls.core.hashing import canonical_json

    assert canonical_json({"b": b"\x00\xff"}) == '{"b":"00ff"}'
