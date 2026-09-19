"""``aicls models`` — the checkpoint registry (``models/registry.yaml`` -> ``model_versions``).

Weights never live in git (ADR-0007): ``models/registry.yaml`` is a committed *description* of
checkpoints that sit under ``AICLS_MODELS_ROOT``, and this command is what turns that description
into the ``model_versions`` rows the API reads to decide which checkpoint serves a request.

Two invariants are enforced on import rather than trusted:

* the checkpoint file must exist and its SHA-256 must equal the digest in the YAML
  (``--no-verify`` skips it, which is what a registry seeded before a gated download needs);
* the row's ``path`` is the checkpoint **directory** under ``AICLS_MODELS_ROOT`` — the same shape
  :class:`aicls.api.model_registry.ModelRegistry` probes with ``is_dir()``.

``config_hash``, ``git_sha`` and ``run_id`` are read out of the run manifest that produced the
checkpoint when it is present, so a promoted model can always be traced back to a training run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field
import typer
import yaml  # type: ignore[import-untyped]  # PyYAML ships no stubs and types-PyYAML is absent

from aicls.cli.common import cli_logger, db_session, emit, use_json

app = typer.Typer(
    name="models",
    help="Import, list and promote trained checkpoints (models/registry.yaml).",
    no_args_is_help=True,
)

#: Default registry file, relative to the repository root.
REGISTRY_RELATIVE_PATH = Path("models") / "registry.yaml"

#: Components a registry entry may declare.
COMPONENTS = ("ocr", "asr")


class RegistryEntry(BaseModel):
    """One row of ``models/registry.yaml`` (the schema documented in that file's header)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    kind: str
    path: str = Field(min_length=1)
    checkpoint: str = "model.safetensors"
    sha256: str = ""
    version: str = ""
    run_manifest: str = "run_manifest.json"
    config: str = ""
    metrics_file: str = ""
    normalizer: str = ""
    created_at: str = ""
    gate_refs: list[str] = Field(default_factory=list)
    notes: str = ""

    @property
    def resolved_version(self) -> str:
        """``version`` when given, else the ``<kind>-`` prefix stripped from ``id``."""
        return self.version or self.id.removeprefix(f"{self.kind}-")


class RegistryFile(BaseModel):
    """The parsed ``models/registry.yaml`` document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    models: list[RegistryEntry] = Field(default_factory=list)


def load_registry(path: Path) -> RegistryFile:
    """Parse a model registry file.

    Raises:
        aicls.core.errors.NotFoundError: If the file does not exist.
        aicls.core.errors.InputValidationError: If it is not valid YAML, not a mapping, or an
            entry does not match the documented schema.
    """
    from aicls.core.errors import InputValidationError, NotFoundError
    from aicls.core.messages_tr import MSG_VALIDATION_ERROR, msg

    target = Path(path)
    if not target.is_file():
        raise NotFoundError(msg("FILE_NOT_FOUND", path=str(target)), details={"path": str(target)})
    try:
        document: Any = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise InputValidationError(
            MSG_VALIDATION_ERROR,
            details={"reason": "registry_not_yaml", "path": str(target), "error": str(exc)},
        ) from exc
    if not isinstance(document, dict):
        raise InputValidationError(
            MSG_VALIDATION_ERROR, details={"reason": "registry_not_a_mapping", "path": str(target)}
        )
    try:
        return RegistryFile.model_validate(document)
    except ValueError as exc:
        raise InputValidationError(
            MSG_VALIDATION_ERROR,
            details={"reason": "registry_invalid", "path": str(target), "error": str(exc)},
        ) from exc


def _provenance(directory: Path, entry: RegistryEntry) -> dict[str, Any]:
    """Read ``config_hash``/``git_sha``/``run_id`` from the checkpoint's run manifest, if any."""
    manifest = directory / entry.run_manifest if entry.run_manifest else None
    if manifest is None or not manifest.is_file():
        return {"config_hash": "", "git_sha": "", "run_id": None}
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # a broken manifest must not lose the row
        cli_logger(__name__).warning(
            "cli.models.manifest_unreadable", path=str(manifest), error=str(exc)[:200]
        )
        return {"config_hash": "", "git_sha": "", "run_id": None}
    return {
        "config_hash": str(document.get("config_hash", "")),
        "git_sha": str(document.get("git_sha", "")),
        "run_id": document.get("run_id"),
    }


def _metrics(root: Path, entry: RegistryEntry) -> dict[str, Any]:
    """Read the evaluation metrics the entry points at (``metrics_file``), or ``{}``."""
    if not entry.metrics_file:
        return {}
    path = Path(entry.metrics_file)
    candidate = path if path.is_absolute() else root / path
    if not candidate.is_file():
        cli_logger(__name__).warning(
            "cli.models.metrics_missing", path=str(candidate), model=entry.id
        )
        return {}
    try:
        document = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
        return {}
    metrics = document.get("metrics") if isinstance(document, dict) else None
    return dict(metrics) if isinstance(metrics, dict) else {}


@app.command("import")
def import_registry(
    registry: Annotated[
        Path | None,
        typer.Argument(help="Registry file; models/registry.yaml by default."),
    ] = None,
    models_root: Annotated[
        Path | None,
        typer.Option("--models-root", help="Checkpoint root; AICLS_MODELS_ROOT by default."),
    ] = None,
    verify: Annotated[
        bool,
        typer.Option("--verify/--no-verify", help="Check that each checkpoint's SHA-256 matches."),
    ] = True,
    db_url: Annotated[str, typer.Option("--db-url", help="Database URL; AICLS_DB_URL.")] = "",
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Register every checkpoint described by ``models/registry.yaml``.

    Raises:
        aicls.core.errors.InputValidationError: If an entry declares an unknown component, or —
            under ``--verify`` — its checkpoint is missing or its digest does not match.
    """
    from aicls.core.errors import InputValidationError
    from aicls.core.hashing import sha256_file
    from aicls.core.messages_tr import MSG_VALIDATION_ERROR
    from aicls.core.paths import find_repo_root
    from aicls.core.settings import get_settings
    from aicls.db import ModelVersionRepository

    use_json(json_output)
    root = find_repo_root()
    source = Path(registry) if registry is not None else root / REGISTRY_RELATIVE_PATH
    document = load_registry(source)
    settings = get_settings()
    default_root = Path(settings.models_root or "")
    weights_root = Path(models_root) if models_root is not None else default_root

    imported: list[dict[str, Any]] = []
    with db_session(db_url) as session:
        repo = ModelVersionRepository(session)
        for entry in document.models:
            if entry.kind not in COMPONENTS:
                raise InputValidationError(
                    MSG_VALIDATION_ERROR,
                    details={
                        "reason": "unknown_component",
                        "id": entry.id,
                        "kind": entry.kind,
                        "expected": list(COMPONENTS),
                    },
                )
            directory = weights_root / entry.path
            checkpoint = directory / entry.checkpoint
            digest = ""
            if verify:
                if not checkpoint.is_file():
                    raise InputValidationError(
                        MSG_VALIDATION_ERROR,
                        details={
                            "reason": "checkpoint_missing",
                            "id": entry.id,
                            "path": str(checkpoint),
                            "hint": "--no-verify registers an entry whose weights are not here yet",
                        },
                    )
                digest = sha256_file(checkpoint)
                if entry.sha256 and digest != entry.sha256:
                    raise InputValidationError(
                        MSG_VALIDATION_ERROR,
                        details={
                            "reason": "checkpoint_digest_mismatch",
                            "id": entry.id,
                            "expected": entry.sha256,
                            "actual": digest,
                        },
                    )
            provenance = _provenance(directory, entry)
            row = repo.register(
                entry.kind,
                entry.resolved_version,
                str(directory),
                digest or entry.sha256,
                _metrics(root, entry),
                provenance["run_id"],
                config_hash=provenance["config_hash"],
                git_sha=provenance["git_sha"],
            )
            imported.append(
                {
                    "id": entry.id,
                    "component": row.component.value,
                    "version": row.version,
                    "path": row.path,
                    "sha256": row.sha256,
                    "verified": bool(verify),
                    "is_active": row.is_active,
                    "config_hash": row.config_hash,
                    "git_sha": row.git_sha,
                }
            )
    payload = {
        "registry": source.as_posix(),
        "models_root": weights_root.as_posix(),
        "schema_version": document.schema_version,
        "imported": imported,
        "count": len(imported),
    }
    emit(
        payload,
        *(f"registered {row['component']} {row['version']} -> {row['path']}" for row in imported),
        f"{len(imported)} model version(s) from {source}",
    )


@app.command("list")
def list_versions(
    component: Annotated[
        str, typer.Option("--component", help="Filter by component: ocr or asr.")
    ] = "",
    db_url: Annotated[str, typer.Option("--db-url", help="Database URL; AICLS_DB_URL.")] = "",
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """List registered checkpoints, newest first."""
    from aicls.db import ModelVersionRepository

    use_json(json_output)
    lines: list[str] = []
    with db_session(db_url) as session:
        rows = ModelVersionRepository(session).list_versions(component or None)
        lines = [
            f"{'*' if row.is_active else ' '} {row.component.value} {row.version} "
            f"sha256={row.sha256[:12]} {row.path}"
            for row in rows
        ]
        versions = [
            {
                "component": row.component.value,
                "version": row.version,
                "path": row.path,
                "sha256": row.sha256,
                "is_active": row.is_active,
                "metrics": dict(row.metrics_json),
                "config_hash": row.config_hash,
                "git_sha": row.git_sha,
                "run_id": row.run_id,
                "created_at": row.created_at,
            }
            for row in rows
        ]
    emit({"component": component or None, "count": len(versions), "versions": versions}, *lines)


@app.command("activate")
def activate(
    component: Annotated[str, typer.Option("--component", help="Component: ocr or asr.")],
    version: Annotated[str, typer.Option("--version", help="Version to promote.")],
    db_url: Annotated[str, typer.Option("--db-url", help="Database URL; AICLS_DB_URL.")] = "",
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Promote one version and deactivate every other version of the component.

    Raises:
        aicls.core.errors.NotFoundError: If the version is not registered.
    """
    from aicls.db import ModelVersionRepository

    use_json(json_output)
    with db_session(db_url) as session:
        row = ModelVersionRepository(session).activate(component, version)
        payload = {
            "component": row.component.value,
            "version": row.version,
            "path": row.path,
            "sha256": row.sha256,
            "is_active": row.is_active,
            "promoted_at": row.promoted_at,
        }
    emit(payload, f"activated {payload['component']} {payload['version']} ({payload['path']})")


__all__ = [
    "COMPONENTS",
    "REGISTRY_RELATIVE_PATH",
    "RegistryEntry",
    "RegistryFile",
    "activate",
    "app",
    "import_registry",
    "list_versions",
    "load_registry",
]
