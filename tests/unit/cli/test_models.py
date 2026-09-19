"""``aicls models``: the checkpoint registry (``models/registry.yaml`` -> ``model_versions``).

Weights are never in git, so the registry is a description that must be checked against the bytes
on disk: the import verifies the SHA-256 of every checkpoint it registers, and refuses the whole
file when one does not match. ``activate`` is what the API's
:class:`aicls.api.model_registry.ModelRegistry` reads, so exactly one version per component may
be active.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.unit.cli.conftest import payload_of

pytestmark = pytest.mark.usefixtures("cli_env")


@pytest.fixture
def models_root(tmp_path: Path) -> Path:
    """A checkpoint directory holding one fake OCR checkpoint and its run manifest."""
    directory = tmp_path / "models" / "ocr-0.1.0"
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(b"not-a-real-checkpoint")
    manifest = {
        "run_id": "01JRUNID",
        "git_sha": "abcdef1234567890abcdef1234567890abcdef12",
        "config_hash": "0" * 64,
    }
    (directory / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8", newline="\n"
    )
    return tmp_path / "models"


@pytest.fixture
def registry_file(tmp_path: Path, models_root: Path) -> Path:
    """A registry file describing that checkpoint with its real digest."""
    from aicls.core.hashing import sha256_file

    digest = sha256_file(models_root / "ocr-0.1.0" / "model.safetensors")
    path = tmp_path / "registry.yaml"
    path.write_text(
        "schema_version: 1\n"
        "models:\n"
        '  - id: "ocr-0.1.0"\n'
        "    kind: ocr\n"
        '    path: "ocr-0.1.0"\n'
        '    checkpoint: "model.safetensors"\n'
        f'    sha256: "{digest}"\n'
        "    normalizer: latex_v1\n"
        '    notes: "wiring check fixture, not a trained model"\n',
        encoding="utf-8",
        newline="\n",
    )
    return path


def test_cli_models_import_seed_registry(
    runner: CliRunner, app_module: object, registry_file: Path, models_root: Path, db_url: str
) -> None:
    """Importing the registry file creates the ``model_versions`` rows the API reads."""
    result = runner.invoke(
        app_module,
        [
            "models",
            "import",
            str(registry_file),
            "--models-root",
            str(models_root),
            "--db-url",
            db_url,
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = payload_of(result.stdout)
    assert payload["count"] == 1
    entry = payload["imported"][0]
    assert entry["component"] == "ocr"
    assert entry["version"] == "0.1.0"
    assert entry["verified"] is True
    assert entry["is_active"] is False, "importing never promotes; that is a separate decision"
    assert entry["git_sha"].startswith("abcdef"), "provenance comes from the run manifest"
    assert Path(entry["path"]).name == "ocr-0.1.0"

    listed = runner.invoke(app_module, ["models", "list", "--db-url", db_url, "--json"])
    assert listed.exit_code == 0
    versions = payload_of(listed.stdout)["versions"]
    assert [row["version"] for row in versions] == ["0.1.0"]


def test_models_import_is_idempotent(
    runner: CliRunner, app_module: object, registry_file: Path, models_root: Path, db_url: str
) -> None:
    """Re-importing the same file updates the row instead of duplicating the version."""
    args = [
        "models",
        "import",
        str(registry_file),
        "--models-root",
        str(models_root),
        "--db-url",
        db_url,
        "--json",
    ]
    assert runner.invoke(app_module, args).exit_code == 0
    assert runner.invoke(app_module, args).exit_code == 0
    listed = runner.invoke(app_module, ["models", "list", "--db-url", db_url, "--json"])
    assert len(payload_of(listed.stdout)["versions"]) == 1


def test_models_import_refuses_a_digest_mismatch(
    runner: CliRunner, app_module: object, registry_file: Path, models_root: Path, db_url: str
) -> None:
    """The digest is the only thing tying a registry row to actual bytes; a mismatch is fatal."""
    (models_root / "ocr-0.1.0" / "model.safetensors").write_bytes(b"different-bytes")
    result = runner.invoke(
        app_module,
        [
            "models",
            "import",
            str(registry_file),
            "--models-root",
            str(models_root),
            "--db-url",
            db_url,
        ],
    )
    assert result.exit_code == 1
    assert "checkpoint_digest_mismatch" in result.stderr


def test_models_import_refuses_a_missing_checkpoint(
    runner: CliRunner, app_module: object, registry_file: Path, tmp_path: Path, db_url: str
) -> None:
    """A registry pointing at weights that are not there is a mistake, not an empty success."""
    result = runner.invoke(
        app_module,
        [
            "models",
            "import",
            str(registry_file),
            "--models-root",
            str(tmp_path / "empty"),
            "--db-url",
            db_url,
        ],
    )
    assert result.exit_code == 1
    assert "checkpoint_missing" in result.stderr


def test_models_activate_promotes_exactly_one_version(
    runner: CliRunner, app_module: object, registry_file: Path, models_root: Path, db_url: str
) -> None:
    """``activate`` is what the request path reads; exactly one version may serve."""
    runner.invoke(
        app_module,
        [
            "models",
            "import",
            str(registry_file),
            "--models-root",
            str(models_root),
            "--db-url",
            db_url,
        ],
    )
    result = runner.invoke(
        app_module,
        [
            "models",
            "activate",
            "--component",
            "ocr",
            "--version",
            "0.1.0",
            "--db-url",
            db_url,
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert payload_of(result.stdout)["is_active"] is True

    listed = runner.invoke(app_module, ["models", "list", "--db-url", db_url, "--json"])
    active = [row for row in payload_of(listed.stdout)["versions"] if row["is_active"]]
    assert len(active) == 1


def test_models_activate_unknown_version_is_not_found(
    runner: CliRunner, app_module: object, db_url: str
) -> None:
    """Promoting a version nobody registered must fail loudly."""
    result = runner.invoke(
        app_module,
        ["models", "activate", "--component", "ocr", "--version", "9.9.9", "--db-url", db_url],
    )
    assert result.exit_code == 1
    assert "NOT_FOUND" in result.stderr


def test_committed_registry_file_parses() -> None:
    """The committed ``models/registry.yaml`` must always match the documented schema."""
    from aicls.cli.models import REGISTRY_RELATIVE_PATH, load_registry
    from aicls.core.paths import find_repo_root

    document = load_registry(find_repo_root() / REGISTRY_RELATIVE_PATH)
    assert document.schema_version == 1
    assert document.models == [], "no checkpoint exists yet: every training gate is closed"


def test_load_registry_refuses_a_malformed_file(tmp_path: Path) -> None:
    """A registry that does not match the documented schema is a typo, not an empty import."""
    from aicls.cli.models import load_registry
    from aicls.core.errors import InputValidationError, NotFoundError

    with pytest.raises(NotFoundError):
        load_registry(tmp_path / "absent.yaml")

    not_a_mapping = tmp_path / "list.yaml"
    not_a_mapping.write_text("- a\n- b\n", encoding="utf-8", newline="\n")
    with pytest.raises(InputValidationError) as shape:
        load_registry(not_a_mapping)
    assert shape.value.details["reason"] == "registry_not_a_mapping"

    unknown_key = tmp_path / "extra.yaml"
    unknown_key.write_text(
        'schema_version: 1\nmodels:\n  - id: "ocr-1"\n    kind: ocr\n    path: "p"\n'
        "    nonsense: 1\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(InputValidationError) as invalid:
        load_registry(unknown_key)
    assert invalid.value.details["reason"] == "registry_invalid"


def test_models_import_refuses_an_unknown_component(
    runner: CliRunner, app_module: object, tmp_path: Path, db_url: str
) -> None:
    """Only ``ocr`` and ``asr`` exist (:class:`aicls.db.enums.ModelComponent`)."""
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        'schema_version: 1\nmodels:\n  - id: "nlu-0.1.0"\n    kind: nlu\n    path: "nlu-0.1.0"\n',
        encoding="utf-8",
        newline="\n",
    )
    result = runner.invoke(
        app_module, ["models", "import", str(registry), "--no-verify", "--db-url", db_url]
    )
    assert result.exit_code == 1
    assert "unknown_component" in result.stderr


def test_models_import_without_verification_records_the_declared_digest(
    runner: CliRunner, app_module: object, tmp_path: Path, db_url: str
) -> None:
    """``--no-verify`` seeds a registry whose weights have not been downloaded yet (ADR-0012)."""
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        'schema_version: 1\nmodels:\n  - id: "asr-0.2.0"\n    kind: asr\n'
        '    path: "asr-0.2.0"\n    sha256: "' + "b" * 64 + '"\n',
        encoding="utf-8",
        newline="\n",
    )
    result = runner.invoke(
        app_module,
        [
            "models",
            "import",
            str(registry),
            "--models-root",
            str(tmp_path / "absent"),
            "--no-verify",
            "--db-url",
            db_url,
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    entry = payload_of(result.stdout)["imported"][0]
    assert entry["component"] == "asr"
    assert entry["verified"] is False
    assert entry["sha256"] == "b" * 64
