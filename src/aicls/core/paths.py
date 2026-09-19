"""Filesystem roots and package resources.

Data-bearing roots come from :class:`aicls.core.settings.Settings` (outside OneDrive, ADR-0008 /
ADR-0011); :func:`repo_root` locates the checkout by walking up to ``pyproject.toml``;
:func:`package_resource` resolves files shipped inside the ``aicls`` package (grammars, YAML,
JSON-LD contexts) through :mod:`importlib.resources`; :func:`long_path` is what lets a writer
touch a path past the Windows 260-character ``MAX_PATH`` limit.
"""

from __future__ import annotations

import atexit
from contextlib import ExitStack
from importlib import resources
import os
from pathlib import Path

from aicls.core import messages_tr
from aicls.core.errors import NotFoundError
from aicls.core.settings import get_settings

_REPO_MARKER = "pyproject.toml"
#: Windows extended-length path prefix — the only way past ``MAX_PATH`` without a registry change.
_WINDOWS_LONG_PREFIX = "\\\\?\\"
_resource_stack = ExitStack()
atexit.register(_resource_stack.close)


def data_root() -> Path:
    """``AICLS_DATA_ROOT``: datasets, audio, runs, SQLite dev database."""
    return get_settings().data_root


def models_root() -> Path:
    """``AICLS_MODELS_ROOT`` (default ``AICLS_DATA_ROOT/models``)."""
    root = get_settings().models_root
    if root is None:  # pragma: no cover - the model validator always fills it
        return data_root() / "models"
    return root


def runs_root() -> Path:
    """``AICLS_DATA_ROOT/runs``: training/evaluation run directories (ADR-0007)."""
    return get_settings().runs_root


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: this file) until a ``pyproject.toml`` is found.

    Raises:
        NotFoundError: If no marker is found up to the filesystem root.
    """
    origin = (start or Path(__file__)).resolve()
    for candidate in (origin, *origin.parents):
        if (candidate / _REPO_MARKER).is_file():
            return candidate
    raise NotFoundError(
        messages_tr.msg("REPO_ROOT_NOT_FOUND", start=str(origin)),
        details={"start": str(origin), "marker": _REPO_MARKER},
    )


def repo_root() -> Path:
    """Repository root of the source checkout (the directory holding ``pyproject.toml``)."""
    return find_repo_root()


def package_resource(pkg: str, name: str) -> Path:
    """Return a real filesystem path to resource ``name`` inside package ``pkg``.

    Works for the src layout and regular installs; when the package is loaded from an archive
    the resource is extracted to a temporary file that lives until interpreter exit.

    Raises:
        NotFoundError: If the package cannot be imported or the resource does not exist.
    """
    try:
        traversable = resources.files(pkg).joinpath(name)
    except (ModuleNotFoundError, TypeError) as exc:
        raise NotFoundError(
            messages_tr.msg("RESOURCE_NOT_FOUND", package=pkg, name=name),
            details={"package": pkg, "name": name, "reason": str(exc)},
        ) from exc
    if not traversable.is_file():
        raise NotFoundError(
            messages_tr.msg("RESOURCE_NOT_FOUND", package=pkg, name=name),
            details={"package": pkg, "name": name},
        )
    if isinstance(traversable, Path):
        return traversable
    return _resource_stack.enter_context(resources.as_file(traversable))  # pragma: no cover


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def long_path(path: Path | str) -> Path:
    """The form of ``path`` to hand the filesystem on a machine with ``MAX_PATH`` in force.

    Windows refuses a path over 260 characters (248 for ``mkdir``) unless the process opts out
    with the ``\\\\?\\`` extended-length prefix or the machine has ``LongPathsEnabled`` set —
    which this workstation does not (measured 2026-09-08). A run directory under a deep
    ``AICLS_DATA_ROOT`` crosses that easily, and ``aicls.ml.checkpoint`` failed there with
    ``FileNotFoundError [WinError 3]``. Every filesystem call that may touch such a path goes
    through this helper.

    The returned path is absolute and normalised, so it is **not** interchangeable with the
    argument for display or comparison: keep the caller's own path for logs, reports and return
    values, and use this one only for the filesystem call.

    Args:
        path: Any path, absolute or relative to the current working directory.

    Returns:
        On Windows the ``\\\\?\\``-prefixed absolute path (``\\\\?\\UNC\\server\\share...`` for a
        UNC path); on every other platform the path unchanged. Prefixing is idempotent.
    """
    candidate = Path(path)
    # `os.name` (not `sys.platform`) so a type checker keeps both branches reachable.
    if os.name != "nt":
        return candidate  # pragma: no cover - POSIX branch; this project runs on Windows
    text = os.path.normpath(str(candidate if candidate.is_absolute() else Path.cwd() / candidate))
    if text.startswith(_WINDOWS_LONG_PREFIX):
        return Path(text)
    if text.startswith("\\\\"):  # pragma: no cover - no UNC share in the test environment
        return Path(f"{_WINDOWS_LONG_PREFIX}UNC{text[1:]}")
    return Path(_WINDOWS_LONG_PREFIX + text)


__all__ = [
    "data_root",
    "ensure_dir",
    "find_repo_root",
    "long_path",
    "models_root",
    "package_resource",
    "repo_root",
    "runs_root",
]
