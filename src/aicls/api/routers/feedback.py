"""Pilot-study feedback and the System Usability Scale (İP-9, api.md §3.9).

The SUS score is computed by :func:`aicls.db.repositories.registry.sus_score`, which is the copy
of the frozen ADR-0015 definition that sits below the API in the layering — ``aicls.eval`` owns
the authoritative one but the API may not import it (``api-no-training`` contract).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status

from aicls.api.deps import CurrentUser, DbSession, StaffUser
from aicls.api.schemas import FeedbackIn, FeedbackOut, Page, SusIn, SusOut
from aicls.db.models import Feedback
from aicls.db.repositories.registry import FeedbackRepository, sus_score

router = APIRouter(prefix="/feedback", tags=["feedback"])


def feedback_out(row: Feedback) -> FeedbackOut:
    """Convert a ``feedback`` row into its API shape."""
    return FeedbackOut(
        id=row.id,
        category=row.category.value,
        text=row.text,
        page=row.page,
        status=row.status.value,
        created_at=row.created_at.isoformat(),
    )


@router.post("", response_model=FeedbackOut, status_code=status.HTTP_201_CREATED)
def post_feedback(body: FeedbackIn, user: CurrentUser, session: DbSession) -> FeedbackOut:
    """Record one feedback entry in state ``open``."""
    row = FeedbackRepository(session).add_feedback(
        user.id,
        body.category,
        body.text,
        body.page,
        screenshot_asset_id=body.screenshot,
    )
    return feedback_out(row)


@router.post("/sus", response_model=SusOut, status_code=status.HTTP_201_CREATED)
def post_sus(body: SusIn, user: CurrentUser, session: DbSession) -> SusOut:
    """Store a SUS questionnaire and return its score (contract target > 75).

    Raises:
        InputValidationError: If the ten answers are not all in 1..5.
    """
    score = sus_score(body.answers)
    FeedbackRepository(session).add_sus(
        user.id,
        body.cohort,
        body.answers,
        score=score,
        collected_week=body.collected_week,
    )
    return SusOut(score=score)


@router.get("", response_model=Page[FeedbackOut])
def list_feedback(
    _user: StaffUser,
    session: DbSession,
    status_filter: Annotated[str | None, Query(alias="status", max_length=16)] = None,
    category: Annotated[str | None, Query(max_length=16)] = None,
    cursor: Annotated[str | None, Query(max_length=26)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> Page[FeedbackOut]:
    """One page of feedback entries (instructor/admin)."""
    page = FeedbackRepository(session).page(
        status=status_filter, category=category, cursor=cursor, limit=limit
    )
    return Page[FeedbackOut](
        items=[feedback_out(row) for row in page.items], next_cursor=page.next_cursor
    )


__all__ = ["feedback_out", "router"]
