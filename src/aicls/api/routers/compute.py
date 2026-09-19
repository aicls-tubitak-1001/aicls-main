"""The command path: text, typed LaTeX, a parse preview and rendered plots (api.md §3.4).

``POST /compute/text`` and ``POST /compute/latex`` answer the *user's question*: a result, a
clarification (200, not an error), a textbook unit or a client-side action.
``POST /compute/parse`` answers the *editor's question* — "is what I am typing a formula yet?" —
and therefore reports the parser's own diagnosis, including ``UNSUPPORTED_CONSTRUCT``, as a 200
payload with ``ok: false``.
"""

from __future__ import annotations

from fastapi import APIRouter, Response

from aicls.api.deps import AppSettings, CurrentUser, DbSession, Pipeline, Timer
from aicls.api.responses import compute_out
from aicls.api.schemas import ComputeLatexIn, ComputeOut, ComputeTextIn, ParseIn, ParseOut
from aicls.core import messages_tr
from aicls.core.errors import AiclsError, NotFoundError
from aicls.db.repositories.compute import AssetRepository
from aicls.latex2sympy import normalize_latex, parse_latex, to_latex, to_sympy
from aicls.nlu.types import ParseContext

router = APIRouter(prefix="/compute", tags=["compute"])


@router.post("/text", response_model=ComputeOut)
def compute_text(
    body: ComputeTextIn,
    user: CurrentUser,
    session: DbSession,
    pipeline: Pipeline,
    timer: Timer,
) -> ComputeOut:
    """Turkish text → NLU → CAS (or the book layer, or a UI action)."""
    context = ParseContext(
        last_latex=body.context.last_latex if body.context else None,
        active_var=body.context.active_var if body.context else None,
        modality="text",
    )
    outcome = pipeline.run_text(
        body.text,
        context,
        user,
        body.workspace_id,
        session=session,
        timer=timer,
        confirm_intent=body.confirm_intent,
    )
    return compute_out(outcome, timer)


@router.post("/latex", response_model=ComputeOut)
def compute_latex(
    body: ComputeLatexIn,
    user: CurrentUser,
    session: DbSession,
    pipeline: Pipeline,
    timer: Timer,
) -> ComputeOut:
    """Execute a typed CAS command with no NLU in the way (the deterministic path)."""
    outcome = pipeline.run_latex(
        body.command, user, body.workspace_id, session=session, timer=timer
    )
    return compute_out(outcome, timer)


@router.post("/parse", response_model=ParseOut)
def compute_parse(body: ParseIn, _user: CurrentUser, timer: Timer) -> ParseOut:
    """Parse a LaTeX fragment and report the diagnosis; never computes anything."""
    with timer.stage("parse"):
        normalised = normalize_latex(body.latex)
        try:
            node = parse_latex(normalised.text)
        except AiclsError as exc:
            return ParseOut(
                ok=False,
                repairs=list(normalised.repairs),
                error=exc.to_problem(),
            )
        try:
            srepr = repr(to_sympy(node))
        except AiclsError:
            srepr = None
        return ParseOut(
            ok=True,
            ast={"node": type(node).__name__, "repr": repr(node)[:2000]},
            srepr=srepr,
            canonical_latex=to_latex(node),
            repairs=list(normalised.repairs),
        )


@router.get("/plots/{asset_id}.png", response_class=Response)
def get_plot(
    asset_id: str, user: CurrentUser, session: DbSession, settings: AppSettings
) -> Response:
    """Serve a rendered plot to the user who produced it."""
    asset = AssetRepository(session).get(asset_id)
    missing = NotFoundError(messages_tr.MSG_NOT_FOUND, details={"table": "assets", "id": asset_id})
    if asset is None or asset.mime != "image/png":
        raise missing
    if asset.owner_id is not None and asset.owner_id != user.id and user.role.value != "admin":
        raise missing
    path = settings.data_root / asset.path
    if not path.is_file():
        raise missing
    return Response(content=path.read_bytes(), media_type="image/png")


__all__ = ["router"]
