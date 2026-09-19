"""Shared pytest fixtures and marker policy (ADR-0010, docs/gelistirme/test-stratejisi.md §4).

Provided here (scaffold level):

* ``repo_root`` / ``load_script`` — locate the repository and import ``tools/``/``scripts/`` files
  by path (they are not packages).
* ``data_root`` — a temporary ``AICLS_DATA_ROOT`` (outside the repo, never OneDrive).
* ``test_env`` — monkeypatched ``AICLS_*`` environment for a hermetic test profile
  (``AICLS_ENV=test``, SQLite ``:memory:``, stub models, fixed JWT secret, contribute enabled with
  ``ethics_approval_ref="TEST"``, all gates empty).
* ``settings`` — ``aicls.core.settings.get_settings()`` under ``test_env`` when the core package is
  implemented; otherwise a plain mapping of the same keys so scaffold tests still run.
* ``seed`` / ``rng`` — deterministic seeding (1234) for ``random``, ``numpy`` and ``torch`` (if loaded).
* ``freeze_time`` — freezegun at a fixed instant.

Marker policy: tests under ``tests/unit`` get ``unit`` automatically; ``gpu``/``data``/``mysql``
tests are skipped unless their environment switch is present; ``browser`` is always skipped in
pytest (Playwright runs from ``web/``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import importlib.util
import os
from pathlib import Path
import random
import sys
import types
from typing import Any

import pytest

# typer >= 0.27 forces colour and box-drawing in its rich help whenever GITHUB_ACTIONS (or
# FORCE_COLOR / PY_COLORS) is set, so on CI `aicls --help` came back with ANSI codes inside the
# option names and the tests that read the help as plain text failed there and nowhere else
# (2026-09-10). typer reads this switch once, when typer.rich_utils is first imported, which is
# why it is set here before any test module can import aicls.cli.
os.environ.setdefault("_TYPER_FORCE_DISABLE_TERMINAL", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 1234
TEST_JWT_SECRET = "0123456789abcdef" * 4  # 64 hex chars = 32 bytes


# --------------------------------------------------------------------------------------------
# hypothesis profiles (PR: 50 examples, nightly: 1000) — registered only if hypothesis exists
# --------------------------------------------------------------------------------------------
try:
    from hypothesis import HealthCheck
    from hypothesis import settings as _hyp_settings

    _hyp_settings.register_profile(
        "pr", max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    _hyp_settings.register_profile(
        "nightly", max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    _hyp_settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "pr"))
except ImportError:  # pragma: no cover - hypothesis is a dev dependency
    pass


# --------------------------------------------------------------------------------------------
# marker policy
# --------------------------------------------------------------------------------------------
def _cuda_is_really_available() -> bool:
    """Whether a CUDA device actually exists in this interpreter.

    ``AICLS_TEST_GPU=1`` says "I want the gpu tests"; it does not say the machine can run them.
    Setting it in the Windows dev venv, whose torch is ``2.11.0+cpu``, made every ``gpu`` test
    **fail** with ``Torch not compiled with CUDA enabled`` instead of skipping (measured
    2026-09-08). Torch is imported only when the switch is on, so an ordinary run still never
    pays for it.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - torch is a gated extra
        return False
    return bool(torch.cuda.is_available())


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-mark unit tests and skip environment-gated markers when their switch is absent."""
    unit_dir = REPO_ROOT / "tests" / "unit"
    skip_gpu = pytest.mark.skip(reason="AICLS_TEST_GPU=1 not set, or torch reports no CUDA device")
    skip_data = pytest.mark.skip(reason="real dataset under AICLS_DATA_ROOT required (data marker)")
    skip_mysql = pytest.mark.skip(reason="AICLS_TEST_MYSQL_URL not set (nightly mysql service)")
    skip_browser = pytest.mark.skip(reason="browser tests run via Playwright from web/")
    have_gpu = os.environ.get("AICLS_TEST_GPU") == "1" and _cuda_is_really_available()
    have_mysql = bool(os.environ.get("AICLS_TEST_MYSQL_URL"))
    have_data = bool(os.environ.get("AICLS_DATA_ROOT")) and os.environ.get("AICLS_TEST_DATA") == "1"
    for item in items:
        path = Path(str(item.path))
        if unit_dir in path.parents and not any(m.name == "unit" for m in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        if item.get_closest_marker("gpu") and not have_gpu:
            item.add_marker(skip_gpu)
        if item.get_closest_marker("mysql") and not have_mysql:
            item.add_marker(skip_mysql)
        if item.get_closest_marker("data") and not have_data:
            item.add_marker(skip_data)
        if item.get_closest_marker("browser"):
            item.add_marker(skip_browser)


# --------------------------------------------------------------------------------------------
# repository helpers
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def load_script() -> Callable[[str], types.ModuleType]:
    """Import a ``tools/`` or ``scripts/`` file by repo-relative path (they are not packages)."""

    def _load(rel_path: str) -> types.ModuleType:
        path = REPO_ROOT / rel_path
        name = "aicls_script_" + path.stem
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None, rel_path
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    return _load


# --------------------------------------------------------------------------------------------
# environment / settings
# --------------------------------------------------------------------------------------------
@pytest.fixture
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary AICLS_DATA_ROOT (pytest basetemp lives in %TEMP%, outside OneDrive)."""
    root = tmp_path_factory.mktemp("aicls-data")
    for sub in ("models", "runs", "mlruns", "audio/raw"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def test_env(monkeypatch: pytest.MonkeyPatch, data_root: Path) -> dict[str, str]:
    """Hermetic ``AICLS_*`` environment for the test profile; returns the mapping that was set."""
    env = {
        "AICLS_ENV": "test",
        "AICLS_PROFILE": "full",
        "AICLS_DATA_ROOT": str(data_root),
        "AICLS_MODELS_ROOT": str(data_root / "models"),
        "AICLS_DB_URL": "sqlite+pysqlite:///:memory:",
        "AICLS_JWT_SECRET": TEST_JWT_SECRET,
        "AICLS_CORS_ORIGINS": '["http://testserver"]',
        "AICLS_MODELS_ENABLED": "false",
        "AICLS_LLM_FALLBACK": "false",
        "AICLS_LLM_ENDPOINT": "",
        "AICLS_LLM_API_KEY": "",
        "AICLS_CONTRIB_ENABLED": "true",
        "AICLS_ETHICS_APPROVAL_REF": "TEST",
        "AICLS_COMPUTE_TIMEOUT_S": "2.0",
        "AICLS_SANDBOX_WORKERS": "1",
        "AICLS_RATE_LIMIT_ENABLED": "false",
        "AICLS_LOG_LEVEL": "WARNING",
        "AICLS_GATE_DATASET_DOWNLOAD": "",
        "AICLS_GATE_WEIGHTS_DOWNLOAD": "",
        "AICLS_GATE_CUDA_INSTALL": "",
        "AICLS_GATE_TRAINING": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    # Never let a developer's .env leak approvals into tests.
    monkeypatch.delenv("AICLS_TEST_MYSQL_URL", raising=False)
    return env


@pytest.fixture
def settings(test_env: dict[str, str]) -> Any:
    """``aicls.core.settings.get_settings()`` under ``test_env`` (or the env mapping until core lands)."""
    try:
        from aicls.core import settings as core_settings
    except ImportError:
        return test_env
    get = getattr(core_settings, "get_settings", None)
    if get is None:
        return test_env
    cache_clear = getattr(get, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()
    result = get()
    if callable(cache_clear):
        cache_clear()
    return result


# --------------------------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------------------------
def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    # Seed torch only when a test already imported it (importing torch costs seconds).
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)


@pytest.fixture(autouse=True)
def seed() -> int:
    """Seed random/numpy/torch with 1234 before every test (autouse)."""
    _seed_everything(DEFAULT_SEED)
    return DEFAULT_SEED


@pytest.fixture
def rng(seed: int) -> Any:
    """``numpy.random.default_rng(1234)``."""
    np = pytest.importorskip("numpy")
    return np.random.default_rng(seed)


@pytest.fixture
def freeze_time() -> Iterator[Any]:
    """Freeze the clock at 2026-09-07T12:00:00Z (freezegun)."""
    freezegun = pytest.importorskip("freezegun")
    with freezegun.freeze_time("2026-09-07T12:00:00Z") as frozen:
        yield frozen
