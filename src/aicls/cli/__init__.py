"""Typer CLI entry point (``aicls``): serve, db, data, train, eval, book, models, report.

The console script declared by ``pyproject.toml`` is :data:`aicls.cli.main.app`. Importing this
package eagerly imports **nothing**: :data:`app` is resolved on first attribute access through
:pep:`562` module ``__getattr__``, so ``import aicls.cli`` never pays for typer, torch, SymPy or
FastAPI. That matters because :mod:`aicls.cli.train` and :mod:`aicls.cli.evaluate` sit on top of
:mod:`aicls.ml` and :mod:`aicls.eval`; a plain re-export here would make every importer of the
package pay for the whole tree.

Only ``app`` is re-exported. The root callback in :mod:`aicls.cli.main` is also called ``main``,
and binding that function here would shadow the ``aicls.cli.main`` **submodule** — the name
Python itself sets on this package as soon as anything imports it. Reach the callback through
``aicls.cli.main.main``.

``tests/integration/test_cli_contracts.py`` pins both halves: the lazy name resolves to the real
Typer application, and importing the package alone leaves ``typer`` and ``torch`` out of
``sys.modules``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time cost is exactly what this module avoids
    from aicls.cli.main import app

__all__ = ["app"]


def __getattr__(name: str) -> Any:
    """Resolve :data:`app` from :mod:`aicls.cli.main` on first access.

    Args:
        name: Attribute being looked up on the package.

    Returns:
        The Typer application.

    Raises:
        AttributeError: For any other name, so a typo still fails loudly and the ``main``
            submodule keeps resolving through the normal import machinery.
    """
    if name != "app":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = importlib.import_module("aicls.cli.main").app
    globals()["app"] = value
    return value


def __dir__() -> list[str]:
    """List the lazily exported name alongside the module's own globals."""
    return sorted(set(globals()) | {"app"})
