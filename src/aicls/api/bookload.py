"""``POST /admin/book/load``: ingest a zipped Markdown book directory (İP-8).

The archive is expanded into a temporary directory with an explicit zip-slip guard — a member
whose resolved path escapes the extraction root is refused rather than written — and only then
handed to :func:`aicls.book.load_markdown_dir`, which refuses the whole batch when any heading
went untagged (the "100 % tagged" criterion of İP-8).
"""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING
import zipfile

from starlette.datastructures import UploadFile

from aicls.api.uploads import read_upload
from aicls.core import messages_tr
from aicls.core.errors import AiclsError, ErrorCode, InputValidationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import Request
    from sqlalchemy.orm import Session

    from aicls.db.results import LoadReport

#: Largest accepted archive (a textbook chapter set is text, not media).
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024

#: Largest accepted uncompressed size (zip-bomb guard).
MAX_UNPACKED_BYTES = 64 * 1024 * 1024

#: File suffixes copied out of the archive.
ALLOWED_SUFFIXES: frozenset[str] = frozenset({".md", ".markdown"})


def _safe_members(archive: zipfile.ZipFile, root: Path) -> list[zipfile.ZipInfo]:
    """Return the Markdown members whose destination stays inside ``root``.

    Raises:
        InputValidationError: If a member escapes the extraction root (zip slip) or the
            uncompressed total exceeds :data:`MAX_UNPACKED_BYTES`.
    """
    members: list[zipfile.ZipInfo] = []
    total = 0
    resolved_root = root.resolve()
    for info in archive.infolist():
        if info.is_dir():
            continue
        target = (root / info.filename).resolve()
        if not target.is_relative_to(resolved_root):
            raise InputValidationError(
                messages_tr.MSG_VALIDATION_ERROR,
                details={"reason": "zip_slip", "member": info.filename[:120]},
            )
        if Path(info.filename).suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        total += info.file_size
        if total > MAX_UNPACKED_BYTES:
            raise InputValidationError(
                messages_tr.MSG_VALIDATION_ERROR,
                details={"reason": "archive_too_large", "limit": MAX_UNPACKED_BYTES},
            )
        members.append(info)
    if not members:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "no_markdown_members", "allowed": sorted(ALLOWED_SUFFIXES)},
        )
    return members


async def load_book_archive(request: Request, session: Session) -> LoadReport:
    """Read the uploaded archive and load every chapter it contains.

    Args:
        request: The multipart request carrying a ``file`` (or ``archive``) part.
        session: Open database session; the caller commits.

    Returns:
        The :class:`~aicls.db.results.LoadReport` of the ingest.

    Raises:
        aicls.core.errors.AiclsError: ``UPLOAD_INVALID`` when no archive part was sent or it is
            not a readable zip; ``VALIDATION_ERROR`` for an unsafe or empty archive.
        aicls.book.UntaggedHeadingError: When tagging coverage is below 1.0 (422 with the
            untagged headings listed).
    """
    from aicls.book.loader import load_markdown_dir
    from aicls.db.repositories.book import BookRepository

    form = await request.form()
    upload = form.get("file") or form.get("archive")
    if not isinstance(upload, UploadFile):
        raise AiclsError(
            ErrorCode.UPLOAD_INVALID,
            messages_tr.MSG_UPLOAD_INVALID,
            {"reason": "archive_part_missing", "expected": ["file", "archive"]},
        )
    payload = await read_upload(upload, max_bytes=MAX_ARCHIVE_BYTES, field="file")
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise AiclsError(
            ErrorCode.UPLOAD_INVALID, messages_tr.MSG_UPLOAD_INVALID, {"reason": "not_a_zip"}
        ) from exc
    with archive, tempfile.TemporaryDirectory(prefix="aicls-book-") as tmp:
        root = Path(tmp)
        flat = root / "chapters"
        flat.mkdir()
        for info in _safe_members(archive, root):
            name = Path(info.filename).name
            (flat / name).write_bytes(archive.read(info))
        return load_markdown_dir(flat, BookRepository(session))


__all__ = [
    "ALLOWED_SUFFIXES",
    "MAX_ARCHIVE_BYTES",
    "MAX_UNPACKED_BYTES",
    "load_book_archive",
]
