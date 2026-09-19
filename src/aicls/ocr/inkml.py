"""InkML reading and writing (ADR-0013, MathWriting-compatible subset).

Only the part of the W3C InkML vocabulary MathWriting uses is supported: an ``<ink>`` root, free
``<annotation type="...">`` metadata and ``<trace>`` elements holding comma-separated ``x y t``
points. That is enough to read the corpus and to round-trip whatever the browser canvas sends.

Security (docs/guvenlik-gizlilik.md §6): parsing goes through :mod:`defusedxml` with DTDs,
entity definitions and external references all disabled, so an uploaded document can neither read
local files (XXE) nor blow up memory (billion laughs); payload, stroke and point budgets are
enforced before any array is allocated (``test_inkml_parser_rejects_external_entities``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.etree.ElementTree import ParseError

from defusedxml import DTDForbidden, EntitiesForbidden, ExternalReferenceForbidden
from defusedxml.ElementTree import fromstring as defused_fromstring
import numpy as np

from aicls.core import messages_tr
from aicls.core.errors import NotFoundError
from aicls.core.logging import get_logger
from aicls.ocr.types import (
    MAX_POINTS_PER_INK,
    MAX_STROKES_PER_INK,
    Ink,
    OcrInputError,
    Stroke,
)

if TYPE_CHECKING:
    import numpy.typing as npt

_log = get_logger(__name__)

#: Maximum accepted InkML payload size in bytes (docs/mimari/api.md §1).
MAX_INKML_BYTES = 1024 * 1024
#: InkML namespace used when writing.
INKML_NS = "http://www.w3.org/2003/InkML"
#: Annotation key holding the ground-truth LaTeX label (MathWriting convention).
LABEL_ANNOTATION = "label"
#: Annotation keys accepted for the writer identifier.
WRITER_ANNOTATIONS = ("writerId", "writer_id")
#: Prefix under which free-form :attr:`Ink.meta` entries are stored.
META_PREFIX = "meta:"
#: Number of significant digits used when writing coordinates (round-trips float32 exactly).
_FLOAT_FORMAT = "{:.9g}"


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _read_source(src: Path | str | bytes) -> bytes:
    """Return the raw document bytes for a path, string or byte payload."""
    if isinstance(src, bytes | bytearray):
        data = bytes(src)
    elif isinstance(src, Path):
        if not src.is_file():
            raise NotFoundError(
                messages_tr.msg("FILE_NOT_FOUND", path=str(src)), details={"path": str(src)}
            )
        data = src.read_bytes()
    else:
        data = src.encode("utf-8")
    if len(data) > MAX_INKML_BYTES:
        raise OcrInputError(
            details={"reason": "too_large", "size": len(data), "limit": MAX_INKML_BYTES}
        )
    return data


def _parse_trace(text: str) -> npt.NDArray[np.float32]:
    """Parse one ``<trace>`` body into an ``(N, 3)`` array (missing ``t`` becomes the index)."""
    rows: list[tuple[float, float, float]] = []
    for index, chunk in enumerate(text.replace("\n", " ").split(",")):
        fields = chunk.split()
        if not fields:
            continue
        if len(fields) < 2:
            raise OcrInputError(details={"reason": "trace_point_arity", "point": chunk.strip()})
        try:
            x = float(fields[0])
            y = float(fields[1])
            t = float(fields[2]) if len(fields) > 2 else float(index)
        except ValueError as exc:
            raise OcrInputError(
                details={"reason": "trace_not_numeric", "point": chunk.strip()}
            ) from exc
        rows.append((x, y, t))
    return np.array(rows, dtype=np.float32).reshape(-1, 3)


def read_inkml(src: Path | str | bytes) -> Ink:
    """Parse an InkML document into an :class:`~aicls.ocr.types.Ink`.

    Args:
        src: A filesystem path, an XML string or the raw bytes of the document.

    Returns:
        The parsed ink; ``label``/``writer_id``/``meta`` are filled from ``<annotation>``.

    Raises:
        NotFoundError: If ``src`` is a path that does not exist.
        OcrInputError: If the payload is too large, malformed, uses a DTD/entities/external
            references, has a wrong root element or exceeds the stroke/point budgets.
    """
    data = _read_source(src)
    try:
        root = defused_fromstring(
            data.decode("utf-8", errors="strict"),
            forbid_dtd=True,
            forbid_entities=True,
            forbid_external=True,
        )
    except DTDForbidden as exc:
        raise OcrInputError(details={"reason": "xml_dtd_forbidden", "error": str(exc)}) from exc
    except EntitiesForbidden as exc:
        raise OcrInputError(
            details={"reason": "xml_entities_forbidden", "error": str(exc)}
        ) from exc
    except ExternalReferenceForbidden as exc:
        raise OcrInputError(
            details={"reason": "xml_external_forbidden", "error": str(exc)}
        ) from exc
    except (ParseError, UnicodeDecodeError, ValueError) as exc:
        raise OcrInputError(details={"reason": "xml_malformed", "error": str(exc)}) from exc

    if _strip_ns(root.tag) != "ink":
        raise OcrInputError(details={"reason": "root_element", "tag": _strip_ns(root.tag)})

    label: str | None = None
    writer_id: str | None = None
    meta: dict[str, Any] = {}
    for node in root.iter():
        if _strip_ns(node.tag) != "annotation":
            continue
        key = node.get("type", "")
        value = (node.text or "").strip()
        if key == LABEL_ANNOTATION:
            label = value
        elif key in WRITER_ANNOTATIONS:
            writer_id = value
        elif key == "t0_ms":
            try:
                meta["t0_ms"] = float(value)
            except ValueError:  # pragma: no cover - defensive, malformed metadata is ignored
                _log.warning("inkml.t0_ms_invalid", value=value)
        elif key.startswith(META_PREFIX):
            meta[key[len(META_PREFIX) :]] = value

    strokes: list[Stroke] = []
    total_points = 0
    for node in root.iter():
        if _strip_ns(node.tag) != "trace":
            continue
        if len(strokes) >= MAX_STROKES_PER_INK:
            raise OcrInputError(
                details={"reason": "too_many_strokes", "limit": MAX_STROKES_PER_INK}
            )
        points = _parse_trace(node.text or "")
        if points.shape[0] == 0:
            continue
        total_points += int(points.shape[0])
        if total_points > MAX_POINTS_PER_INK:
            raise OcrInputError(
                details={
                    "reason": "too_many_points",
                    "points": total_points,
                    "limit": MAX_POINTS_PER_INK,
                }
            )
        strokes.append(Stroke(points))

    return Ink(strokes=strokes, label=label or None, writer_id=writer_id or None, meta=meta)


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def write_inkml(ink: Ink) -> bytes:
    """Serialise an ink back to an InkML document.

    The output is UTF-8 with LF line endings and 9 significant digits per coordinate, which makes
    :func:`read_inkml` reproduce the exact float32 values (``test_inkml_read_write_roundtrip``).

    Args:
        ink: The sample to serialise.

    Returns:
        The document bytes.
    """
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', f'<ink xmlns="{INKML_NS}">']
    if ink.label:
        lines.append(f'  <annotation type="{LABEL_ANNOTATION}">{_escape(ink.label)}</annotation>')
    if ink.writer_id:
        lines.append(
            f'  <annotation type="{WRITER_ANNOTATIONS[0]}">{_escape(ink.writer_id)}</annotation>'
        )
    for key, value in sorted(ink.meta.items()):
        if key == "t0_ms":
            lines.append(
                f'  <annotation type="t0_ms">{_FLOAT_FORMAT.format(float(value))}</annotation>'
            )
        else:
            lines.append(
                f'  <annotation type="{META_PREFIX}{_escape(str(key))}">'
                f"{_escape(str(value))}</annotation>"
            )
    for stroke in ink.strokes:
        body = ", ".join(
            " ".join(_FLOAT_FORMAT.format(float(value)) for value in point)
            for point in stroke.points
        )
        lines.append(f"  <trace>{body}</trace>")
    lines.append("</ink>")
    return ("\n".join(lines) + "\n").encode("utf-8")


__all__ = ["INKML_NS", "MAX_INKML_BYTES", "read_inkml", "write_inkml"]
