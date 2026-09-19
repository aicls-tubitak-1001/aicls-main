"""Cross-module contracts the API introduces (KNOWN_ISSUES: "Cross-module contract testing").

Every boundary ``aicls.api`` adds is pinned here, in the shape the *other* side actually uses:

* **api ↔ web** — the payload shapes ``web/src/api/types.ts`` and the two socket hooks declare;
* **api ↔ core** — every :class:`~aicls.core.errors.ErrorCode` maps to the HTTP status of
  api.md §2, with the Turkish text from :mod:`aicls.core.messages_tr` and no other source;
* **api ↔ mathengine/latex2sympy** — an out-of-grammar construct becomes the İP-4 contract
  sentence rather than a 500 (the explicit "recognised but not computable" path KNOWN_ISSUES
  item 2 asked the API to own);
* **api ↔ ocr/asr** — the upload budgets are the modules' own constants, not copies;
* **api ↔ ml/eval/data** — the layering forbids the import, and torch never enters the process.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from aicls.api.errors import STATUS_TO_CODE, AuthError, ForbiddenError, RateLimitedError
from aicls.api.pipeline import as_unsupported
from aicls.api.schemas import ComputeOut, RecognizeOut
from aicls.api.uploads import MAX_INK_BYTES, check_image_bytes, check_wav_bytes
from aicls.asr.audio_io import MAX_WAV_BYTES
from aicls.core import messages_tr
from aicls.core.errors import HTTP_STATUS_BY_CODE, AiclsError, ErrorCode
from aicls.mathengine.errors import UnsupportedProblem
from aicls.ocr.types import MAX_IMAGE_BYTES, MAX_POINTS_PER_INK

pytestmark = pytest.mark.integration

#: Probe run in a clean interpreter: import the whole API surface, then assert torch is absent.
NO_TORCH_PROBE = """
import importlib, sys

for name in (
    "aicls.api",
    "aicls.api.app",
    "aicls.api.pipeline",
    "aicls.api.model_registry",
    "aicls.api.responses",
    "aicls.api.routers",
    "aicls.api.uploads",
    "aicls.api.ws.ink",
    "aicls.api.ws.audio",
):
    importlib.import_module(name)
assert "torch" not in sys.modules, sorted(m for m in sys.modules if m.startswith("torch"))
assert "transformers" not in sys.modules
assert "torchvision" not in sys.modules
print("ok")
"""


# ---------------------------------------------------------------------------------------------
# api <-> core: the error model
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("code", list(ErrorCode))
def test_every_error_code_has_a_status_a_title_and_a_turkish_detail(code: ErrorCode) -> None:
    """The api.md §2 table is complete: status, title and Turkish text for every code."""
    error = AiclsError(code)
    problem = error.to_problem(instance="01J000000000000000000000")
    assert problem["status"] == HTTP_STATUS_BY_CODE[code]
    assert problem["code"] == code.value
    assert problem["title"] == messages_tr.TITLES[code.value]
    assert problem["detail_tr"] == messages_tr.msg(code)
    assert problem["type"].endswith(f"/{code.value}")
    assert problem["instance"] == "01J000000000000000000000"


def test_api_error_types_carry_the_documented_codes() -> None:
    """The three API-owned errors map onto the shared vocabulary."""
    assert AuthError().http_status == 401
    assert AuthError().code is ErrorCode.AUTH_INVALID
    assert ForbiddenError().http_status == 403
    assert ForbiddenError().code is ErrorCode.FORBIDDEN
    limited = RateLimitedError(7)
    assert limited.http_status == 429
    assert limited.retry_after == 7
    assert "7" in limited.message_tr


def test_status_to_code_table_agrees_with_the_core_table() -> None:
    """A framework exception maps to a code whose own status is the same number."""
    for status, code in STATUS_TO_CODE.items():
        if code in (ErrorCode.VALIDATION_ERROR, ErrorCode.UPLOAD_INVALID):
            continue  # these cover several statuses on purpose (400/405/413/415/422)
        assert HTTP_STATUS_BY_CODE[code] == status, f"{status} -> {code}"


# ---------------------------------------------------------------------------------------------
# api <-> mathengine / latex2sympy: recognised but not computable
# ---------------------------------------------------------------------------------------------
def test_unsupported_construct_becomes_the_contract_sentence() -> None:
    """``UNSUPPORTED_CONSTRUCT`` from the parser is answered with the frozen İP-4 sentence."""
    from aicls.latex2sympy import UnsupportedConstruct

    translated = as_unsupported(UnsupportedConstruct("\\binom", note="v1.1"))
    assert isinstance(translated, UnsupportedProblem)
    assert translated.code is ErrorCode.UNSUPPORTED_PROBLEM
    assert translated.http_status == 422
    assert translated.message_tr == messages_tr.MSG_UNSUPPORTED_PROBLEM
    assert translated.details["construct"] == "\\binom"
    assert translated.details["original_code"] == "UNSUPPORTED_CONSTRUCT"


def test_other_errors_pass_through_untouched() -> None:
    """A parse error keeps its own code and position; only the unsupported case is rewritten."""
    from aicls.latex2sympy import ParseError

    original = ParseError(line=1, col=8, expected=["LBRACE"])
    assert as_unsupported(original) is original
    assert original.details["position"] == 8


def test_contract_sentence_is_the_frozen_wording() -> None:
    """The İP-4 risk-plan sentence must not drift (the API is one of its two consumers)."""
    assert messages_tr.MSG_UNSUPPORTED_PROBLEM == "Bu problem tipi şu an için desteklenmemektedir."


# ---------------------------------------------------------------------------------------------
# api <-> ocr / asr: the upload budgets are the modules' own constants
# ---------------------------------------------------------------------------------------------
def test_upload_limits_come_from_the_owning_modules() -> None:
    """The API never re-declares a limit that ``aicls.ocr``/``aicls.asr`` already owns."""
    assert MAX_IMAGE_BYTES == 2 * 1024 * 1024
    assert MAX_WAV_BYTES == 10 * 1024 * 1024
    assert MAX_POINTS_PER_INK == 20_000
    assert MAX_INK_BYTES == 1024 * 1024


def test_upload_guards_reject_the_wrong_container() -> None:
    """Magic-byte checks refuse a renamed file before any decoder runs."""
    from aicls.asr.errors import AsrInputError
    from aicls.ocr.types import OcrInputError

    with pytest.raises(OcrInputError) as image:
        check_image_bytes(b"GIF89a bu bir gif")
    assert image.value.details["reason"] == "bad_magic"

    with pytest.raises(AsrInputError) as audio:
        check_wav_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    assert audio.value.details["reason"] == "not_riff_wave"


# ---------------------------------------------------------------------------------------------
# api <-> web: the payload shapes the TypeScript client declares
# ---------------------------------------------------------------------------------------------
def _typescript_alias_targets(source: str) -> dict[str, str]:
    """Map each ``export type X = Schema<"Y">...`` alias to the schema name ``Y`` it looks up.

    The web client no longer hand-writes its DTOs: every domain type is a lookup into the
    generated ``components["schemas"]`` table, so a schema the backend stops emitting becomes a
    TypeScript compile error rather than a silent drift. This scan checks the other direction —
    that the names being looked up are ones the API actually serves.
    """
    pattern = re.compile(r'export type (\w+)\s*=[^;]*?Schema<"(\w+)">', re.DOTALL)
    return {alias: schema for alias, schema in pattern.findall(source)}


@pytest.mark.parametrize(
    ("alias", "model"),
    [("ComputeResponse", ComputeOut), ("RecognizeResponse", RecognizeOut)],
)
def test_web_client_types_are_derived_from_the_api_models(
    repo_root: Path, alias: str, model: type
) -> None:
    """Every domain type the web client reads resolves to a schema the API serves.

    This is the wave-1 class of bug (the web client believing in a field the wire never sends),
    now guarded structurally: the client derives its types from the committed OpenAPI document,
    so the assertion is that the *schema name* it looks up is one this response model publishes
    under that exact name.
    """
    source = (repo_root / "web" / "src" / "api" / "types.ts").read_text(encoding="utf-8")
    aliases = _typescript_alias_targets(source)
    assert aliases, "web/src/api/types.ts no longer derives its types from the generated schema"

    served = model.model_json_schema(by_alias=True, ref_template="{model}")["title"]
    targets = set(aliases.values())
    assert served in targets, (
        f"the web client never looks up {served}; "
        f"it derives {sorted(targets)[:8]}… from the generated schema"
    )


def test_openapi_document_is_checked_in_and_matches_the_web_input(repo_root: Path) -> None:
    """``docs/api/openapi.json`` and the ``openapi-typescript`` input are the same document."""
    docs = repo_root / "docs" / "api" / "openapi.json"
    web = repo_root / "web" / "src" / "api" / "generated" / "openapi.json"
    assert docs.is_file() and web.is_file()
    assert json.loads(docs.read_text(encoding="utf-8")) == json.loads(
        web.read_text(encoding="utf-8")
    )


def test_ws_envelope_shape_matches_the_web_hooks(repo_root: Path) -> None:
    """Both socket hooks expect ``{type, seq, payload}`` and the server sends exactly that."""
    from aicls.api.ws.protocol import WsEnvelope, envelope, slow_down

    assert set(WsEnvelope.model_fields) == {"type", "seq", "payload"}
    assert set(envelope("partial", {"latex": "x"}, 3)) == {"type", "seq", "payload"}
    assert slow_down(250)["payload"]["retry_after_ms"] == 250

    # The message unions live in the shared socket layer; the two feature hooks re-export it.
    wire = (repo_root / "web" / "src" / "api" / "ws.ts").read_text(encoding="utf-8")
    for message in ("stroke", "pen_up", "clear", "finalize", "ping"):
        assert f'"{message}"' in wire, f"the ink client no longer sends {message}"
    for message in ("partial", "final", "error", "slow_down", "pong"):
        assert f'"{message}"' in wire, f"the ink client no longer handles {message}"
    for message in ("start", "stop", "cancel", "command"):
        assert f'"{message}"' in wire, f"the audio client no longer knows {message}"

    for hook, module in (
        (("features", "workspace", "useInkSocket.ts"), "useInkSocket"),
        (("features", "voice", "useAudioSocket.ts"), "useAudioSocket"),
    ):
        source = repo_root.joinpath("web", "src", *hook).read_text(encoding="utf-8")
        assert module in source and "@/api/sockets" in source, (
            f"{module} no longer binds the shared socket layer"
        )


def test_canvas_json_contract_is_the_agreed_shape() -> None:
    """The web Canvas payload (``{version, canvas:{w,h}, strokes}``) still round-trips."""
    from aicls.ocr.canvas import ink_from_canvas_json

    ink = ink_from_canvas_json(
        {
            "version": 1,
            "canvas": {"w": 800, "h": 300},
            "strokes": [{"points": [{"x": 1.0, "y": 2.0, "t": 1700000000000}]}],
        }
    )
    assert ink.n_strokes == 1
    assert ink.meta["canvas_width"] == 800.0
    assert ink.meta["canvas_height"] == 300.0


# ---------------------------------------------------------------------------------------------
# api <-> ml / eval / data: the layering
# ---------------------------------------------------------------------------------------------
def test_api_does_not_import_torch(repo_root: Path) -> None:
    """Importing the whole API surface must not pull torch into the process (ADR-0002).

    Run in a subprocess: another module of the suite legitimately imports torch (the ``ocr``/
    ``asr`` wrapper tests), so ``sys.modules`` of *this* process proves nothing.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    proc = subprocess.run(
        [sys.executable, "-c", NO_TORCH_PROBE],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root),
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ok" in proc.stdout


@pytest.mark.parametrize("forbidden", ["aicls.ml", "aicls.eval", "aicls.data"])
def test_api_never_imports_the_training_packages(forbidden: str) -> None:
    """The ``api-no-training`` contract of ADR-0002, checked on the real import graph."""
    import grimp

    graph = grimp.build_graph("aicls", include_external_packages=False)
    chain = graph.find_shortest_chain(importer="aicls.api", imported=forbidden)
    assert chain is None, f"aicls.api reaches {forbidden} via {chain}"
