"""Client-side latency telemetry (ADR-0014, api.md §3.10).

The contract metric is "command → *rendered* result", which no server-side stopwatch can see.
The browser therefore reports ``client_submit_ts``/``client_render_ts`` from
``performance.now()``, and this endpoint merges them with the server stages already stored on the
operation before writing one ``latency_samples`` row. ``GET /admin/metrics/latency`` then reports
mean/p50/p95 from exactly that table.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, status

from aicls.api.deps import CurrentUser, DbSession
from aicls.api.schemas import LatencySampleIn
from aicls.core import messages_tr
from aicls.core.errors import NotFoundError
from aicls.db.repositories.compute import OperationRepository
from aicls.db.repositories.registry import LatencyRepository

router = APIRouter(prefix="/telemetry", tags=["telemetry"])


@router.post("/latency", status_code=status.HTTP_204_NO_CONTENT)
def post_latency(
    request: Request, body: LatencySampleIn, user: CurrentUser, session: DbSession
) -> None:
    """Store one end-to-end latency sample.

    Raises:
        NotFoundError: If ``operation_id`` names an operation the caller did not produce.
    """
    stages: dict[str, float] = {}
    server_ms = 0.0
    if body.operation_id:
        operation = OperationRepository(session).get(body.operation_id)
        if operation is None or operation.user_id != user.id:
            raise NotFoundError(
                messages_tr.MSG_NOT_FOUND,
                details={"table": "operations", "id": body.operation_id},
            )
        stages = {
            key: float(value)
            for key, value in dict(operation.timings_json or {}).items()
            if isinstance(value, int | float)
        }
        server_ms = float(stages.get("total", 0.0))
    client_total = max(0.0, body.client_render_ts - body.client_submit_ts)
    LatencyRepository(session).add(
        {
            "operation_id": body.operation_id,
            "request_id": getattr(request.state, "request_id", ""),
            "modality": body.modality,
            "endpoint": "/telemetry/latency",
            "server_ms": server_ms,
            "stage": stages,
            "client_total_ms": client_total,
            "render_ms": body.render_ms,
        }
    )


__all__ = ["router"]
