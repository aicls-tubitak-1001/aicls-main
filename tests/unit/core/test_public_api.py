"""The aicls.core package re-exports its public contract (modules/core.json public_api)."""

from __future__ import annotations

import importlib
import sys

from aicls import core

CORE_MODULES = (
    "settings",
    "paths",
    "logging",
    "errors",
    "messages_tr",
    "ids",
    "hashing",
    "timing",
    "turkish",
    "gates",
    "latex_safety",
)


def test_public_api_symbols_exported() -> None:
    expected = [
        "get_settings",
        "Settings",
        "data_root",
        "models_root",
        "runs_root",
        "repo_root",
        "package_resource",
        "ErrorCode",
        "AiclsError",
        "msg",
        "MSG_UNSUPPORTED_PROBLEM",
        "ulid",
        "short_hash",
        "sha256_file",
        "sha256_json",
        "fingerprint_jsonl",
        "StageTimer",
        "tr_casefold",
        "tr_lower",
        "strip_diacritics",
        "TR_SUFFIXES",
        "require_gate",
        "GateError",
        "GateEntry",
        "read_gate_ledger",
        "sanitize_latex",
        "get_logger",
        "configure_logging",
    ]
    for name in expected:
        assert hasattr(core, name), name
        assert name in core.__all__, name


def test_core_modules_import_cleanly_and_have_docstrings() -> None:
    for mod in CORE_MODULES:
        m = importlib.import_module(f"aicls.core.{mod}")
        assert m.__doc__, mod


def test_core_does_not_import_ml_stack() -> None:
    """Importing every core module must not pull torch/transformers (ADR-0002 layering)."""
    before = {name for name in ("torch", "transformers", "torchvision") if name in sys.modules}
    for mod in CORE_MODULES:
        importlib.import_module(f"aicls.core.{mod}")
    after = {name for name in ("torch", "transformers", "torchvision") if name in sys.modules}
    assert after == before
