"""Loading of the package-resource YAML files under ``aicls/nlu/inventory``.

The inventory, the lexicon and the number table ship *inside* the wheel
(``[tool.setuptools.package-data]`` lists ``**/*.yaml``), so they are read through
:func:`aicls.core.paths.package_resource` and never through a repository-relative path. Loading
is cached per absolute path, which keeps ``parse_command`` allocation-free after the first call
while still letting tests point at a temporary inventory file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]  # PyYAML ships no stubs and types-PyYAML is absent

from aicls.nlu.errors import InventoryError

#: Package the YAML resources live in.
RESOURCE_PACKAGE = "aicls.nlu"

#: Resource names, relative to :data:`RESOURCE_PACKAGE`.
COMMANDS_RESOURCE = "inventory/commands.yaml"
LEXICON_RESOURCE = "inventory/lexicon.yaml"
NUMBERS_RESOURCE = "inventory/numbers.yaml"


def resource_path(name: str) -> Path:
    """Absolute path of a packaged YAML resource.

    Args:
        name: Resource name relative to :data:`RESOURCE_PACKAGE`
            (for example ``"inventory/commands.yaml"``).

    Returns:
        A real filesystem path, extracted from the archive if necessary.

    Raises:
        aicls.core.errors.NotFoundError: If the resource does not exist.
    """
    from aicls.core.paths import package_resource

    return package_resource(RESOURCE_PACKAGE, name)


@lru_cache(maxsize=8)
def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping from ``path`` (cached by absolute path).

    Args:
        path: File to read; must contain a top-level mapping.

    Returns:
        The parsed mapping.

    Raises:
        aicls.nlu.errors.InventoryError: If the file is not valid YAML or is not a mapping.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InventoryError("resource_unreadable", details={"path": str(path)}) from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise InventoryError(
            "invalid_yaml", details={"path": str(path), "error": str(exc)[:200]}
        ) from exc
    if not isinstance(data, dict):
        raise InventoryError(
            "not_a_mapping", details={"path": str(path), "type": type(data).__name__}
        )
    return data


__all__ = [
    "COMMANDS_RESOURCE",
    "LEXICON_RESOURCE",
    "NUMBERS_RESOURCE",
    "RESOURCE_PACKAGE",
    "load_yaml",
    "resource_path",
]
