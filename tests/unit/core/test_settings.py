"""Settings: env-prefix loading, overrides, derived defaults and profile validation."""

from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr, ValidationError
import pytest

from aicls.core import settings as core_settings
from aicls.core.settings import Settings, get_settings, settings_override


def test_settings_override_via_env(test_env: dict[str, str]) -> None:
    get_settings.cache_clear()
    s = get_settings()
    assert s.env == "test"
    assert s.profile == "full"
    assert s.data_root == Path(test_env["AICLS_DATA_ROOT"])
    assert s.models_root == Path(test_env["AICLS_MODELS_ROOT"])
    assert s.db_url == "sqlite+pysqlite:///:memory:"
    assert isinstance(s.jwt_secret, SecretStr)
    assert s.jwt_secret.get_secret_value() == test_env["AICLS_JWT_SECRET"]
    assert s.cors_origins == ["http://testserver"]
    assert s.models_enabled is False
    assert s.contrib_enabled is True
    assert s.ethics_approval_ref == "TEST"
    assert s.compute_timeout_s == 2.0
    assert s.sandbox_workers == 1
    assert s.log_level == "WARNING"
    assert s.rate_limit_enabled is False
    get_settings.cache_clear()


def test_settings_llm_fallback_accepts_short_env_alias(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AICLS_LLM_FALLBACK", "true")
    assert Settings().llm_fallback_enabled is True
    monkeypatch.delenv("AICLS_LLM_FALLBACK")
    monkeypatch.setenv("AICLS_LLM_FALLBACK_ENABLED", "true")
    assert Settings().llm_fallback_enabled is True


def test_settings_gate_fields_default_empty(test_env: dict[str, str]) -> None:
    s = Settings()
    assert (
        s.gate_dataset_download
        == s.gate_weights_download
        == s.gate_cuda_install
        == s.gate_training
        == ""
    )
    assert s.gate_value("TRAINING") == ""


def test_settings_gate_value_reads_env(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AICLS_GATE_TRAINING", "approved")
    assert Settings().gate_value("TRAINING") == "approved"
    assert Settings().is_gate_env_approved("TRAINING") is True
    assert Settings().is_gate_env_approved("CUDA_INSTALL") is False


def test_settings_derived_defaults(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("AICLS_MODELS_ROOT")
    monkeypatch.delenv("AICLS_DB_URL")
    monkeypatch.setenv("AICLS_DATA_ROOT", str(tmp_path))
    s = Settings()
    assert s.models_root == tmp_path / "models"
    assert s.db_url.startswith("sqlite+pysqlite:///")
    assert s.db_url.endswith("aicls-dev.db")
    assert s.runs_root == tmp_path / "runs"


def test_settings_get_is_cached(test_env: dict[str, str]) -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()


def test_settings_override_context_manager(test_env: dict[str, str]) -> None:
    get_settings.cache_clear()
    with settings_override(profile="collect", sandbox_workers=7):
        s = get_settings()
        assert s.profile == "collect"
        assert s.sandbox_workers == 7
        assert s.is_collect_profile is True
    assert get_settings().profile == "full"
    get_settings.cache_clear()


def test_settings_prod_requires_strong_jwt_secret(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AICLS_ENV", "prod")
    monkeypatch.setenv("AICLS_JWT_SECRET", "short")
    with pytest.raises(ValidationError):
        Settings()
    monkeypatch.setenv("AICLS_JWT_SECRET", "x" * 32)
    assert Settings().is_prod is True


def test_settings_rejects_invalid_profile(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AICLS_PROFILE", "half")
    with pytest.raises(ValidationError):
        Settings()


def test_settings_rejects_nonpositive_limits(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AICLS_SANDBOX_WORKERS", "0")
    with pytest.raises(ValidationError):
        Settings()


def test_settings_log_level_normalised(
    test_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AICLS_LOG_LEVEL", "debug")
    assert Settings().log_level == "DEBUG"
    monkeypatch.setenv("AICLS_LOG_LEVEL", "loud")
    with pytest.raises(ValidationError):
        Settings()


def test_settings_default_data_root_is_outside_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("AICLS_DATA_ROOT", "AICLS_MODELS_ROOT", "AICLS_DB_URL"):
        monkeypatch.delenv(key, raising=False)
    root = core_settings.default_data_root()
    assert root.is_absolute()
    assert root.name == "aicls-data"
