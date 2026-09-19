"""Path helpers: roots derived from settings, repo root discovery, package resources."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

import pytest

from aicls.core import paths
from aicls.core.errors import AiclsError, ErrorCode
from aicls.core.settings import get_settings


def test_roots_follow_settings(test_env: dict[str, str]) -> None:
    get_settings.cache_clear()
    root = Path(test_env["AICLS_DATA_ROOT"])
    assert paths.data_root() == root
    assert paths.models_root() == root / "models"
    assert paths.runs_root() == root / "runs"
    get_settings.cache_clear()


def test_repo_root_contains_pyproject() -> None:
    root = paths.repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "aicls" / "core").is_dir()


def test_find_repo_root_walks_up(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert paths.find_repo_root(nested) == tmp_path


def test_find_repo_root_raises_when_absent(tmp_path: Path) -> None:
    with pytest.raises(AiclsError) as ei:
        paths.find_repo_root(tmp_path)
    assert ei.value.code is ErrorCode.NOT_FOUND


def test_package_resource_returns_existing_file() -> None:
    p = paths.package_resource("aicls", "py.typed")
    assert p.is_file()


def test_package_resource_missing_raises() -> None:
    with pytest.raises(AiclsError) as ei:
        paths.package_resource("aicls.core", "does-not-exist.yaml")
    assert ei.value.code is ErrorCode.NOT_FOUND
    assert ei.value.details["package"] == "aicls.core"


def test_package_resource_unknown_package_raises() -> None:
    with pytest.raises(AiclsError) as ei:
        paths.package_resource("aicls.no_such_pkg", "x.yaml")
    assert ei.value.code is ErrorCode.NOT_FOUND


def test_ensure_dir_creates(tmp_path: Path) -> None:
    target = tmp_path / "x" / "y"
    assert paths.ensure_dir(target) == target
    assert target.is_dir()


# --------------------------------------------------------------------- Windows long paths
def test_long_path_is_identity_off_windows(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("POSIX behaviour")
    assert paths.long_path(tmp_path / "a" / "b") == tmp_path / "a" / "b"


@pytest.mark.skipif(os.name != "nt", reason="the \\\\?\\ prefix is a Windows API feature")
def test_long_path_prefixes_and_normalises_on_windows(tmp_path: Path) -> None:
    prefixed = paths.long_path(tmp_path / "a" / ".." / "b")
    assert str(prefixed).startswith("\\\\?\\")
    assert str(prefixed).endswith("\\b")
    assert ".." not in str(prefixed)
    # idempotent: prefixing an already prefixed path must not stack prefixes
    assert paths.long_path(prefixed) == prefixed
    assert str(paths.long_path(prefixed)).count("\\\\?\\") == 1


@pytest.mark.skipif(os.name != "nt", reason="MAX_PATH is a Windows limit")
def test_long_path_escapes_the_260_character_limit(tmp_path: Path) -> None:
    """Without ``LongPathsEnabled`` a plain ``mkdir`` past ~248 characters fails (WinError 206)."""
    deep = tmp_path
    while len(str(deep)) < 300:
        deep = deep / ("d" * 40)

    paths.long_path(deep).mkdir(parents=True, exist_ok=True)
    assert paths.long_path(deep).is_dir()
    payload = paths.long_path(deep) / "note.txt"
    payload.write_text("ok", encoding="utf-8")
    assert payload.read_text(encoding="utf-8") == "ok"

    shutil.rmtree(paths.long_path(tmp_path / ("d" * 40)), ignore_errors=True)
