"""Which OCR/ASR checkpoint the request path uses (``model_versions.is_active``).

The registry is the single place that decides between the deterministic stubs and a trained
checkpoint. It answers "stub" whenever ``AICLS_MODELS_ENABLED`` is false, whenever no version is
promoted, or whenever the promoted version's directory is gone — the API must keep serving the
deterministic path while a model is being trained (ADR-0018), never fail because a checkpoint is
missing.

``torch`` is never imported here, and — this is the part a lazy ``import`` inside a function does
**not** buy — no *static* import path from :mod:`aicls.api` to ``torch`` exists at all. The
``no-torch-in-core`` contract of ADR-0002 is checked on the import graph, not on
``sys.modules``: :mod:`aicls.ocr.factory` and :mod:`aicls.asr.interfaces` both reach ``torch``
statically, so they are resolved through :func:`importlib.import_module` at the moment a trained
checkpoint is actually promoted. The stubs come from the torch-free submodules
(:mod:`aicls.ocr.stub`, :mod:`aicls.asr.stub`) rather than through the package ``__init__``,
which does carry the heavy edge. This is the wave-2 lesson recorded in
``tests/integration/test_layering_contracts.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
import importlib
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Literal, TypeAlias

from aicls.core.errors import AiclsError
from aicls.core.logging import get_logger
from aicls.core.settings import Settings
from aicls.db.enums import ModelComponent
from aicls.db.repositories.registry import ModelVersionRepository

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlalchemy.orm import Session

    from aicls.asr.types import Transcriber
    from aicls.ocr.types import Recognizer

#: State reported by ``GET /health/ready`` for one component.
ModelState: TypeAlias = Literal["stub", "loaded", "absent", "training"]

#: Factory that opens a database session (``session_scope(engine)``).
SessionFactory: TypeAlias = Callable[[], AbstractContextManager["Session"]]

_logger = get_logger(__name__)


class ModelRegistry:
    """Resolve the active recogniser and transcriber, with a cached instance per component.

    Args:
        settings: Runtime settings; ``models_enabled`` is the master switch.
        session_factory: Opens a session to read ``model_versions``; when omitted the registry
            never touches the database and always answers with the stubs.

    Example:
        >>> from aicls.core.settings import Settings
        >>> registry = ModelRegistry(Settings(models_enabled=False))
        >>> registry.state("ocr")
        'stub'
    """

    def __init__(self, settings: Settings, session_factory: SessionFactory | None = None) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._lock = threading.Lock()
        self._recognizer: Recognizer | None = None
        self._transcriber: Transcriber | None = None
        self._versions: dict[str, str | None] = {"ocr": None, "asr": None}
        self._states: dict[str, ModelState] = {"ocr": "stub", "asr": "stub"}

    # -- resolution --------------------------------------------------------------------------
    def _active(self, component: ModelComponent) -> tuple[str | None, Path | None]:
        """Return ``(version, checkpoint directory)`` promoted for ``component``."""
        if not self._settings.models_enabled or self._session_factory is None:
            return None, None
        try:
            with self._session_factory() as session:
                row = ModelVersionRepository(session).active(component)
                if row is None:
                    return None, None
                return row.version, Path(row.path) if row.path else None
        except AiclsError as exc:  # pragma: no cover - a broken registry must not 500 a request
            _logger.warning("api.registry.lookup_failed", component=component.value, error=str(exc))
            return None, None

    def get_recognizer(self) -> Recognizer:
        """Return the active handwriting recogniser (the stub unless a checkpoint is promoted)."""
        with self._lock:
            if self._recognizer is None:
                self._recognizer = self._build_recognizer()
            return self._recognizer

    def _build_recognizer(self) -> Recognizer:
        from aicls.ocr.stub import StubRecognizer

        version, model_dir = self._active(ModelComponent.OCR)
        if version is not None and model_dir is not None and model_dir.is_dir():
            factory = importlib.import_module("aicls.ocr.factory")
            try:
                recognizer: Recognizer = factory.get_recognizer("torch", model_dir=model_dir)
            except AiclsError as exc:
                _logger.warning("api.registry.ocr_load_failed", version=version, error=str(exc))
            else:
                self._versions["ocr"], self._states["ocr"] = version, "loaded"
                return recognizer
        stub = StubRecognizer()
        self._versions["ocr"] = stub.model_version
        self._states["ocr"] = "absent" if version is not None else "stub"
        return stub

    def get_transcriber(self) -> Transcriber:
        """Return the active speech transcriber (the stub unless a checkpoint is promoted)."""
        with self._lock:
            if self._transcriber is None:
                self._transcriber = self._build_transcriber()
            return self._transcriber

    def _build_transcriber(self) -> Transcriber:
        from aicls.asr.stub import StubTranscriber

        version, model_dir = self._active(ModelComponent.ASR)
        if version is not None and model_dir is not None and model_dir.is_dir():
            backend = "wav2vec2" if "wav2vec" in version.lower() else "whisper"
            interfaces = importlib.import_module("aicls.asr.interfaces")
            try:
                built = interfaces.get_transcriber(backend, model_dir=model_dir)
                transcriber: Transcriber = built
            except AiclsError as exc:
                _logger.warning("api.registry.asr_load_failed", version=version, error=str(exc))
            else:
                self._versions["asr"], self._states["asr"] = version, "loaded"
                return transcriber
        stub = StubTranscriber()
        self._versions["asr"] = stub.model_version
        self._states["asr"] = "absent" if version is not None else "stub"
        return stub

    # -- introspection -----------------------------------------------------------------------
    def reload(self) -> None:
        """Drop the cached instances so the next call re-reads ``model_versions``."""
        with self._lock:
            self._recognizer = None
            self._transcriber = None
            self._versions = {"ocr": None, "asr": None}
            self._states = {"ocr": "stub", "asr": "stub"}
        _logger.info("api.registry.reloaded")

    def set_recognizer(self, recognizer: Recognizer, version: str = "override") -> None:
        """Install a recogniser directly (a warm swap, and what the tests inject)."""
        with self._lock:
            self._recognizer = recognizer
            self._versions["ocr"] = version
            self._states["ocr"] = "loaded"

    def set_transcriber(self, transcriber: Transcriber, version: str = "override") -> None:
        """Install a transcriber directly (a warm swap, and what the tests inject)."""
        with self._lock:
            self._transcriber = transcriber
            self._versions["asr"] = version
            self._states["asr"] = "loaded"

    def warmup(self) -> None:
        """Instantiate and warm both components (called from the lifespan)."""
        self.get_recognizer().warmup()
        self.get_transcriber().warmup()

    def version(self, component: str) -> str | None:
        """Version string reported for ``component`` (``"ocr"`` / ``"asr"``)."""
        self._ensure(component)
        return self._versions.get(component)

    def state(self, component: str) -> ModelState:
        """Readiness state of ``component`` (``stub``/``loaded``/``absent``/``training``)."""
        self._ensure(component)
        return self._states.get(component, "stub")

    def versions(self) -> dict[str, str | None]:
        """``{"ocr": ..., "asr": ...}`` for ``ComputeOut.model_versions``."""
        return {"ocr": self.version("ocr"), "asr": self.version("asr")}

    def states(self) -> dict[str, ModelState]:
        """``{"ocr": ..., "asr": ...}`` for ``GET /health/ready``."""
        return {"ocr": self.state("ocr"), "asr": self.state("asr")}

    def _ensure(self, component: str) -> None:
        """Resolve ``component`` once so its version and state are known."""
        if component == "ocr":
            self.get_recognizer()
        elif component == "asr":
            self.get_transcriber()


__all__ = ["ModelRegistry", "ModelState", "SessionFactory"]
