"""Operation, asset and model-output repositories (docs/mimari/veri-modeli.md §3.1–§3.2)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy import select

from aicls.core.errors import NotFoundError
from aicls.core.messages_tr import msg
from aicls.db.base import utcnow
from aicls.db.enums import AssetKind, InputKind, NluSource, ResultKind, coerce_enum
from aicls.db.models import Asset, Operation, Recognition, Transcript
from aicls.db.repositories.base import Repository
from aicls.db.results import OperationCreate, Page


class AssetRepository(Repository):
    """Queries over ``assets`` (plots, ink, audio, book figures, screenshots)."""

    def add(
        self,
        kind: AssetKind | str,
        path: str,
        *,
        sha256: str = "",
        size_bytes: int = 0,
        mime: str = "application/octet-stream",
        owner_id: str | None = None,
        retention_until: Any = None,
    ) -> Asset:
        """Register a stored file and return the row.

        Args:
            kind: Asset kind (``plot_png``, ``audio_wav``, ...).
            path: Path relative to ``AICLS_DATA_ROOT`` — never an absolute path.
            sha256: Digest of the file contents (bare hex or ``sha256:``-prefixed).
            size_bytes: File size in bytes.
            mime: MIME type.
            owner_id: Owning user, if the asset is personal data.
            retention_until: Naive UTC datetime after which the file may be purged.
        """
        asset = Asset(
            kind=coerce_enum(AssetKind, kind),
            path=path,
            sha256=sha256,
            bytes=size_bytes,
            mime=mime,
            owner_id=owner_id,
            retention_until=retention_until,
        )
        self.session.add(asset)
        self.session.flush()
        return asset

    def get(self, asset_id: str) -> Asset | None:
        """Return the asset with ``asset_id`` or ``None``."""
        return self.session.get(Asset, asset_id)

    def expired(self, *, now: Any = None, limit: int | None = None) -> list[Asset]:
        """Return assets whose ``retention_until`` has passed (retention job, §4)."""
        moment = now if now is not None else utcnow()
        stmt = (
            select(Asset)
            .where(Asset.retention_until.is_not(None), Asset.retention_until <= moment)
            .order_by(Asset.id)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self.session.scalars(stmt))


class OperationRepository(Repository):
    """Queries over ``operations`` — the per-workspace history of the pipeline."""

    def add(self, ws_id: str, user_id: str, op: OperationCreate | Mapping[str, Any]) -> Operation:
        """Append one finished operation to a workspace and return it.

        Args:
            ws_id: Owning workspace id.
            user_id: User who issued the request.
            op: An :class:`~aicls.db.results.OperationCreate` or a mapping with the same keys.
        """
        data = op if isinstance(op, OperationCreate) else OperationCreate.from_mapping(op)
        operation = Operation(
            workspace_id=ws_id,
            user_id=user_id,
            input_kind=coerce_enum(InputKind, data.input_kind),
            source_text=data.source_text,
            input_latex=data.input_latex,
            command_json=dict(data.command),
            result_kind=(
                coerce_enum(ResultKind, data.result_kind) if data.result_kind is not None else None
            ),
            result_latex=data.result_latex,
            result_srepr=data.result_srepr,
            steps_json=data.steps,
            plot_asset_id=data.plot_asset_id,
            error_code=data.error_code,
            timings_json=dict(data.timings),
            cache_hit=data.cache_hit,
            model_versions_json=dict(data.model_versions),
            nlu_source=(
                coerce_enum(NluSource, data.nlu_source) if data.nlu_source is not None else None
            ),
        )
        self.session.add(operation)
        self.session.flush()
        return operation

    def get(self, op_id: str) -> Operation | None:
        """Return the operation with ``op_id`` or ``None``."""
        return self.session.get(Operation, op_id)

    def page(
        self, ws_id: str, cursor: str | None = None, limit: int | None = None
    ) -> Page[Operation]:
        """Return one page of a workspace's operations, newest first."""
        stmt = select(Operation).where(Operation.workspace_id == ws_id)
        return self.cursor_page(stmt, Operation.id, cursor=cursor, limit=limit)

    def delete(self, ws_id: str, op_id: str) -> bool:
        """Delete one operation of a workspace; ``False`` when it does not exist there."""
        stmt = select(Operation).where(Operation.id == op_id, Operation.workspace_id == ws_id)
        operation = self.session.scalars(stmt).one_or_none()
        if operation is None:
            return False
        self.session.delete(operation)
        self.session.flush()
        return True


class RecognitionRepository(Repository):
    """Queries over ``recognitions`` and ``transcripts`` (İP-3 outputs and corrections)."""

    def add(
        self,
        latex: str,
        *,
        user_id: str | None = None,
        operation_id: str | None = None,
        ink_asset_id: str | None = None,
        image_asset_id: str | None = None,
        confidence: float = 0.0,
        alternatives: Mapping[str, Any] | None = None,
        model_version: str = "",
        elapsed_ms: int = 0,
    ) -> Recognition:
        """Store one handwriting-recognition result and return it."""
        recognition = Recognition(
            user_id=user_id,
            operation_id=operation_id,
            ink_asset_id=ink_asset_id,
            image_asset_id=image_asset_id,
            latex=latex,
            confidence=confidence,
            alternatives_json=dict(alternatives or {}),
            model_version=model_version,
            elapsed_ms=elapsed_ms,
        )
        self.session.add(recognition)
        self.session.flush()
        return recognition

    def get(self, rec_id: str) -> Recognition | None:
        """Return the recognition with ``rec_id`` or ``None``."""
        return self.session.get(Recognition, rec_id)

    def set_correction(self, rec_id: str, latex: str) -> None:
        """Record the user's corrected LaTeX and stamp ``corrected_at`` (İP-3 feedback loop).

        Raises:
            NotFoundError: If no such recognition exists.
        """
        recognition = self.session.get(Recognition, rec_id)
        if recognition is None:
            raise NotFoundError(msg("NOT_FOUND"), details={"table": "recognitions", "id": rec_id})
        recognition.corrected_latex = latex
        recognition.corrected_at = utcnow()
        self.session.flush()

    def corrections(self, *, limit: int | None = None) -> list[Recognition]:
        """Return corrected recognitions, newest first (retraining candidates)."""
        stmt = (
            select(Recognition)
            .where(Recognition.corrected_at.is_not(None))
            .order_by(Recognition.id.desc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self.session.scalars(stmt))

    def add_transcript(
        self,
        text: str,
        *,
        user_id: str | None = None,
        operation_id: str | None = None,
        audio_asset_id: str | None = None,
        confidence: float = 0.0,
        model_version: str = "",
        elapsed_ms: int = 0,
    ) -> Transcript:
        """Store one speech-recognition result and return it."""
        transcript = Transcript(
            user_id=user_id,
            operation_id=operation_id,
            audio_asset_id=audio_asset_id,
            text=text,
            confidence=confidence,
            model_version=model_version,
            elapsed_ms=elapsed_ms,
        )
        self.session.add(transcript)
        self.session.flush()
        return transcript

    def set_transcript_correction(self, transcript_id: str, text: str) -> None:
        """Record the user's corrected transcript text.

        Raises:
            NotFoundError: If no such transcript exists.
        """
        transcript = self.session.get(Transcript, transcript_id)
        if transcript is None:
            raise NotFoundError(
                msg("NOT_FOUND"), details={"table": "transcripts", "id": transcript_id}
            )
        transcript.corrected_text = text
        self.session.flush()


__all__ = ["AssetRepository", "OperationRepository", "RecognitionRepository"]
