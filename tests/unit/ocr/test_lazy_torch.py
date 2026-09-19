"""Importing `aicls.ocr` must never pull torch in (ADR-0002 layering, API cold-start budget)."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

PROBE = (
    "import sys;"
    "import aicls.ocr as ocr;"
    "assert 'torch' not in sys.modules, 'aicls.ocr imported torch';"
    "assert 'torchvision' not in sys.modules, 'aicls.ocr imported torchvision';"
    "assert 'transformers' not in sys.modules, 'aicls.ocr imported transformers';"
    "ocr.get_recognizer('stub').warmup();"
    "assert 'torch' not in sys.modules, 'the stub recognizer imported torch';"
    "print('ok')"
)


@pytest.mark.timeout(90)
def test_importing_aicls_ocr_does_not_import_torch(repo_root: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    proc = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root),
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_public_api_is_exported() -> None:
    from aicls import ocr

    expected = {
        "Recognizer",
        "Recognition",
        "Ink",
        "Stroke",
        "Image",
        "OcrInputError",
        "OcrModelConfig",
        "PreprocessConfig",
        "LatexTokenizer",
        "StubRecognizer",
        "TorchRecognizer",
        "get_recognizer",
        "available_recognizers",
        "read_inkml",
        "write_inkml",
        "ink_from_canvas_json",
        "ink_to_canvas_json",
        "normalize_strokes",
        "render_ink",
        "render_ink_to_image",
        "preprocess_image",
        "postprocess_latex",
        "build_model",
    }
    assert expected <= set(ocr.__all__)
    for name in expected:
        assert hasattr(ocr, name), name


def test_build_model_raises_a_clear_error_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """When torch is absent the lazy loader must fail with an AiclsError, not ImportError."""
    import builtins

    from aicls.core.errors import AiclsError
    from aicls.ocr import OcrModelConfig, build_model, torch_utils

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    # drop the memoised module so require_torch() really re-imports under the patched hook
    monkeypatch.setitem(torch_utils._MODULES, "torch", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        with pytest.raises(AiclsError) as exc:
            build_model(OcrModelConfig(d_model=8, layers=1, heads=2), vocab_size=10)
    finally:
        monkeypatch.undo()
    assert exc.value.details.get("missing_package") == "torch"
