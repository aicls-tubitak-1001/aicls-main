"""The packaged command inventory: schema, loader and the YAML resources themselves."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from aicls.nlu.inventory.schema import (
    MIN_EXAMPLES_PER_INTENT,
    Defaults,
    Example,
    IntentSpec,
    Inventory,
    SlotSpec,
    SlotType,
    Weights,
    build_inventory,
    example_texts,
)
from aicls.nlu.resources import COMMANDS_RESOURCE, load_yaml, resource_path


@lru_cache(maxsize=4)
def _load_cached(path: str) -> Inventory:
    return build_inventory(load_yaml(Path(path)), path=path)


def load_inventory(path: Path | str | None = None) -> Inventory:
    """Load (and cache) the command inventory.

    Args:
        path: Optional override, for tests and for experimenting with a bigger inventory; the
            packaged ``inventory/commands.yaml`` when omitted.

    Returns:
        The validated :class:`~aicls.nlu.inventory.schema.Inventory`.

    Raises:
        aicls.nlu.errors.InventoryError: If the YAML is malformed or fails schema validation.
    """
    resolved = Path(path) if path is not None else resource_path(COMMANDS_RESOURCE)
    return _load_cached(str(resolved))


__all__ = [
    "COMMANDS_RESOURCE",
    "MIN_EXAMPLES_PER_INTENT",
    "Defaults",
    "Example",
    "IntentSpec",
    "Inventory",
    "SlotSpec",
    "SlotType",
    "Weights",
    "build_inventory",
    "example_texts",
    "load_inventory",
]
