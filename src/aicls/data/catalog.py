"""Read-only view over ``catalog.yaml``: what a dataset or checkpoint is and which gate it needs.

The catalog is **descriptive**. Nothing in this module (or anywhere else in :mod:`aicls.data`)
fetches bytes: the gated scripts under ``training/scripts/`` read these entries, print the URL,
the size and the licence, ask :func:`aicls.core.gates.require_gate` for permission and only then
download (ADR-0012). Keeping the description here means an approval request can be prepared —
and reviewed — without any code that could accidentally start a download.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
import yaml  # type: ignore[import-untyped]  # PyYAML ships no stubs and types-PyYAML is absent

from aicls.core.paths import package_resource
from aicls.data.errors import invalid_data, missing_path

#: Name of the catalog resource inside this package.
CATALOG_RESOURCE = "catalog.yaml"

EntryKind = Literal["dataset", "weights"]


class CatalogEntry(BaseModel):
    """One catalog row: a dataset or a set of pre-trained weights.

    Attributes:
        name: Catalog key (``"mathwriting"``, ``"whisper-small"``).
        kind: ``"dataset"`` or ``"weights"``.
        version: Version label (``mathwriting-2024.09``, ``whisper-small``).
        url: Download URL, or ``None`` for data produced inside the project.
        license: SPDX identifier or ``"internal"``.
        expected_bytes: Expected download size, used for the disk check of the gated script.
        sha256: Archive digest; ``None`` until a gated download has produced one.
        gate: Approval-gate name (``DATASET_DOWNLOAD``/``WEIGHTS_DOWNLOAD``) or ``None``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = ""
    kind: EntryKind = "dataset"
    version: str
    title: str = ""
    source: str = ""
    url: str | None = None
    license: str
    expected_bytes: int = Field(default=0, ge=0)
    sha256: str | None = None
    gate: str | None = None
    citation: str = ""
    splits: list[str] = Field(default_factory=list)
    format: str = ""
    notes_tr: str = ""

    @property
    def expected_gib(self) -> float:
        """Expected size in GiB (for the approval dossier and the disk check)."""
        return self.expected_bytes / float(1 << 30)

    @property
    def archive_name(self) -> str:
        """Local file name of this entry's archive: the basename of its URL.

        The name decides the file the gated download script writes and the file the (ungated)
        preparation script streams, so both read it from here instead of guessing a suffix —
        MathWriting is distributed as ``mathwriting-2024.tgz``, not as a zip.

        Returns:
            The URL basename (``"mathwriting-2024.tgz"``), or :attr:`version` when the entry has
            no URL or the URL ends in something that is not a usable file name.
        """
        candidate = self.url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1] if self.url else ""
        if not candidate or candidate in {".", ".."} or "\\" in candidate:
            return self.version
        return candidate

    def describe(self) -> str:
        """One-line English description used by the gated download scripts."""
        return (
            f"{self.name} ({self.version}, {self.kind}) — licence {self.license}, "
            f"{self.expected_gib:.2f} GiB, gate {self.gate or 'none'}, url {self.url or 'n/a'}"
        )


class Catalog(BaseModel):
    """The parsed ``catalog.yaml``: datasets and weights, keyed by name."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
    datasets: dict[str, CatalogEntry] = Field(default_factory=dict)
    weights: dict[str, CatalogEntry] = Field(default_factory=dict)

    def entries(self) -> list[CatalogEntry]:
        """Every entry, datasets first, each in file order."""
        return [*self.datasets.values(), *self.weights.values()]

    def get(self, name: str) -> CatalogEntry:
        """Look a name up in both sections.

        Raises:
            NotFoundError: If ``name`` is not in the catalog.
        """
        entry = self.datasets.get(name) or self.weights.get(name)
        if entry is None:
            raise missing_path(
                package_resource("aicls.data", CATALOG_RESOURCE),
                "catalog_entry_missing",
                requested=name,
                available=sorted([*self.datasets, *self.weights]),
            )
        return entry


@lru_cache(maxsize=1)
def load_catalog() -> Catalog:
    """Parse ``catalog.yaml`` once per process.

    Returns:
        The catalog; entry ``name`` fields are filled in from their keys.

    Raises:
        InputValidationError: If the file is not valid YAML or has an unexpected shape.
    """
    path = package_resource("aicls.data", CATALOG_RESOURCE)
    try:
        payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - the shipped file is valid
        raise invalid_data("catalog_not_yaml", path=str(path), error=str(exc)) from exc
    if not isinstance(payload, dict):  # pragma: no cover - defensive
        raise invalid_data("catalog_not_a_mapping", path=str(path))
    for section in ("datasets", "weights"):
        rows = payload.get(section) or {}
        if not isinstance(rows, dict):  # pragma: no cover - defensive
            raise invalid_data("catalog_section_not_a_mapping", section=section)
        for key, row in rows.items():
            row["name"] = key
    try:
        return Catalog.model_validate(payload)
    except ValueError as exc:  # pragma: no cover - the shipped file is valid
        raise invalid_data("catalog_invalid", path=str(path), error=str(exc)) from exc


def catalog_entry(name: str) -> CatalogEntry:
    """Return the catalog entry called ``name``.

    Args:
        name: ``"mathwriting"``, ``"audio_commands"``, ``"whisper-small"``, ...

    Raises:
        NotFoundError: If no such entry exists.
    """
    return load_catalog().get(name)


__all__ = [
    "CATALOG_RESOURCE",
    "Catalog",
    "CatalogEntry",
    "EntryKind",
    "catalog_entry",
    "load_catalog",
]
