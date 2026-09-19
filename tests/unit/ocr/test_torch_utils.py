"""Lazy torch/torchvision loader tests (clear typed errors instead of ImportError)."""

from __future__ import annotations

import builtins

import pytest

from aicls.core.errors import ErrorCode
from aicls.ocr import torch_utils
from aicls.ocr.torch_utils import (
    TorchUnavailableError,
    require_torch,
    require_torchvision,
    reset_module_cache,
)


def _blocking_import(blocked: str) -> object:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == blocked or name.startswith(f"{blocked}."):
            raise ImportError(f"No module named {blocked!r}")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    return fake_import


class TestRequireTorch:
    def test_returns_the_module_and_caches_it(self) -> None:
        pytest.importorskip("torch")
        assert require_torch() is require_torch()

    def test_missing_torch_raises_a_typed_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(torch_utils._MODULES, "torch", None)
        monkeypatch.setattr(builtins, "__import__", _blocking_import("torch"))
        try:
            with pytest.raises(TorchUnavailableError) as exc:
                require_torch()
        finally:
            monkeypatch.undo()
        assert exc.value.code is ErrorCode.INTERNAL
        assert exc.value.details["missing_package"] == "torch"
        assert exc.value.message_tr


class TestRequireTorchvision:
    def test_missing_torchvision_raises_a_typed_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(torch_utils._MODULES, "torchvision.models", None)
        monkeypatch.setattr(builtins, "__import__", _blocking_import("torchvision"))
        try:
            with pytest.raises(TorchUnavailableError) as exc:
                require_torchvision()
        finally:
            monkeypatch.undo()
        assert exc.value.details["missing_package"] == "torchvision"


def test_reset_module_cache_forgets_everything() -> None:
    pytest.importorskip("torch")
    require_torch()
    reset_module_cache()
    assert all(value is None for value in torch_utils._MODULES.values())
    assert require_torch() is not None
