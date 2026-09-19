"""Application settings loaded from ``AICLS_*`` environment variables and ``.env``.

:class:`Settings` is a pydantic-settings model (docs/mimari/moduller.md, docs/gelistirme/kurulum.md
§2). :func:`get_settings` returns a process-wide cached instance; tests call
``get_settings.cache_clear()`` or use :func:`settings_override`.

Approval-gate fields (``gate_*``) are read here so that a ``.env`` entry counts as the
environment condition of the triple lock (ADR-0012); they must stay empty in CI.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EnvName = Literal["dev", "test", "prod"]
ProfileName = Literal["full", "collect"]
GateName = Literal["DATASET_DOWNLOAD", "WEIGHTS_DOWNLOAD", "CUDA_INSTALL", "TRAINING"]

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_MIN_JWT_SECRET_LEN = 32
_APPROVED = "approved"


def default_data_root() -> Path:
    """Default ``AICLS_DATA_ROOT`` outside OneDrive (ADR-0011).

    Windows: ``D:\\aicls-data`` when a D: drive exists, else ``C:\\aicls-data``.
    Other platforms: ``~/aicls-data``.
    """
    if os.name == "nt":
        d_drive = Path("D:/")
        return (d_drive if d_drive.exists() else Path("C:/")) / "aicls-data"
    return Path.home() / "aicls-data"


class Settings(BaseSettings):
    """Runtime configuration (env prefix ``AICLS_``; ``.env`` in the working directory).

    Field names follow docs/mimari/moduller.md. ``models_root`` and ``db_url`` derive from
    ``data_root`` when not given. ``jwt_secret`` may be empty in ``dev``/``test`` but must be at
    least 32 characters in ``prod``.
    """

    model_config = SettingsConfigDict(
        env_prefix="AICLS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    env: EnvName = "dev"
    profile: ProfileName = "full"
    data_root: Path = Field(default_factory=default_data_root)
    models_root: Path | None = None
    db_url: str = ""
    jwt_secret: SecretStr = SecretStr("")
    cors_origins: list[str] = Field(default_factory=list)
    rate_limit_enabled: bool = True
    models_enabled: bool = False
    llm_fallback_enabled: bool = False
    llm_fallback: bool | None = None  # short alias AICLS_LLM_FALLBACK (.env.example, conftest)
    llm_endpoint: str = ""
    llm_api_key: SecretStr = SecretStr("")
    contrib_enabled: bool = False
    ethics_approval_ref: str = ""
    compute_timeout_s: float = Field(default=2.5, gt=0)
    sandbox_workers: int = Field(default=2, ge=1)
    log_level: str = "INFO"
    gate_dataset_download: str = ""
    gate_weights_download: str = ""
    gate_cuda_install: str = ""
    gate_training: str = ""

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: Any) -> str:
        level = str(value).strip().upper()
        if level not in _LOG_LEVELS:
            raise ValueError(f"log_level must be one of {_LOG_LEVELS}, got {value!r}")
        return level

    @field_validator("data_root", mode="after")
    @classmethod
    def _absolute_data_root(cls, value: Path) -> Path:
        return value.expanduser().resolve() if not value.is_absolute() else value.expanduser()

    @model_validator(mode="after")
    def _derive_and_check(self) -> Settings:  # noqa: N804 - "after" validators are instance methods
        if self.llm_fallback is not None:
            self.llm_fallback_enabled = self.llm_fallback_enabled or self.llm_fallback
        if self.models_root is None:
            self.models_root = self.data_root / "models"
        if not self.db_url:
            self.db_url = f"sqlite+pysqlite:///{(self.data_root / 'aicls-dev.db').as_posix()}"
        if self.env == "prod" and len(self.jwt_secret.get_secret_value()) < _MIN_JWT_SECRET_LEN:
            raise ValueError(
                f"AICLS_JWT_SECRET must be at least {_MIN_JWT_SECRET_LEN} characters in prod"
            )
        return self

    # ----------------------------------------------------------------------------------------
    # convenience
    # ----------------------------------------------------------------------------------------
    @property
    def runs_root(self) -> Path:
        """``AICLS_DATA_ROOT/runs`` (training and evaluation runs, ADR-0007)."""
        return self.data_root / "runs"

    @property
    def is_prod(self) -> bool:
        """True when ``env == "prod"``."""
        return self.env == "prod"

    @property
    def is_collect_profile(self) -> bool:
        """True when only the voice data-collection routers are enabled."""
        return self.profile == "collect"

    def gate_value(self, name: GateName | str) -> str:
        """Raw value of ``AICLS_GATE_<name>`` (empty string when unset).

        Raises:
            AttributeError: If ``name`` is not one of the four gate names.
        """
        attr = f"gate_{str(name).lower()}"
        value = getattr(self, attr)
        if not isinstance(value, str):  # pragma: no cover - typed fields guarantee str
            raise AttributeError(attr)
        return value

    def is_gate_env_approved(self, name: GateName | str) -> bool:
        """True when ``AICLS_GATE_<name>`` equals ``"approved"`` (case-insensitive, trimmed)."""
        return self.gate_value(name).strip().lower() == _APPROVED


_overrides: dict[str, Any] = {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached :class:`Settings` (built on first call).

    Tests reset with ``get_settings.cache_clear()`` after changing the environment.
    """
    return Settings(**_overrides)


@contextmanager
def settings_override(**changes: Any) -> Iterator[Settings]:
    """Temporarily construct :func:`get_settings` with explicit field values.

    Init keyword arguments take priority over environment variables in pydantic-settings, so this
    is a clean way for tests to flip a flag without touching ``os.environ``::

        with settings_override(profile="collect"):
            assert get_settings().is_collect_profile

    Yields:
        The overridden :class:`Settings` instance.
    """
    previous = dict(_overrides)
    _overrides.update(changes)
    get_settings.cache_clear()
    try:
        yield get_settings()
    finally:
        _overrides.clear()
        _overrides.update(previous)
        get_settings.cache_clear()


__all__ = [
    "EnvName",
    "GateName",
    "ProfileName",
    "Settings",
    "default_data_root",
    "get_settings",
    "settings_override",
]
