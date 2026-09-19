"""REST routers and the profile tables that decide which of them are mounted (ADR-0018).

``full`` is the workstation: everything. ``collect`` is the small VM that hosts the voice
data-collection tab; it must be reachable from the public internet, so it gets only what that
tab needs — identity, the contribution flow, feedback, health and the quality-control subset of
the administration surface. It never exposes the CAS, the book graph or the model registry,
because it has neither the GPU nor the data.
"""

from __future__ import annotations

from fastapi import APIRouter

from aicls.api.routers import (
    admin,
    asr,
    auth,
    book,
    compute,
    contribute,
    feedback,
    health,
    ocr,
    telemetry,
    users,
    workspaces,
)

#: Routers mounted on the ``full`` profile, in the order they appear in api.md §3.
FULL_ROUTERS: tuple[APIRouter, ...] = (
    health.router,
    auth.router,
    users.router,
    workspaces.router,
    compute.router,
    ocr.router,
    asr.router,
    book.router,
    contribute.router,
    contribute.withdraw_router,
    feedback.router,
    telemetry.router,
    admin.router,
    admin.qc_router,
)

#: Routers mounted on the ``collect`` profile (ADR-0018 §Karar).
COLLECT_ROUTERS: tuple[APIRouter, ...] = (
    health.router,
    auth.router,
    users.router,
    contribute.router,
    contribute.withdraw_router,
    feedback.router,
    admin.qc_router,
)

#: Profile name -> routers.
ROUTERS_BY_PROFILE: dict[str, tuple[APIRouter, ...]] = {
    "full": FULL_ROUTERS,
    "collect": COLLECT_ROUTERS,
}


def routers_for(profile: str) -> tuple[APIRouter, ...]:
    """Return the routers mounted for ``profile`` (unknown profiles fall back to ``full``)."""
    return ROUTERS_BY_PROFILE.get(profile, FULL_ROUTERS)


__all__ = [
    "COLLECT_ROUTERS",
    "FULL_ROUTERS",
    "ROUTERS_BY_PROFILE",
    "admin",
    "asr",
    "auth",
    "book",
    "compute",
    "contribute",
    "feedback",
    "health",
    "ocr",
    "routers_for",
    "telemetry",
    "users",
    "workspaces",
]
