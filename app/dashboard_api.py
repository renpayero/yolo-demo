"""API del dashboard: KPIs por zona, línea de tiempo, personas en zonas y alertas de un rango horario."""
from __future__ import annotations

import time
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request

from app.metrics import build_dashboard
from app.storage import Storage

router = APIRouter(prefix="/api", tags=["dashboard"])

MAX_RANGE_S = 31 * 24 * 3600
DEFAULT_RANGE_S = 3600


@router.get("/dashboard")
def dashboard(
    request: Request,
    start: Annotated[float | None, Query(alias="from", description="epoch UTC en segundos")] = None,
    end: Annotated[float | None, Query(alias="to", description="epoch UTC en segundos")] = None,
) -> dict[str, Any]:
    now = time.time()
    end = now if end is None else end
    start = end - DEFAULT_RANGE_S if start is None else start
    if start >= end:
        raise HTTPException(status_code=422, detail="El inicio del rango tiene que ser anterior al fin")
    if end - start > MAX_RANGE_S:
        raise HTTPException(status_code=422, detail="El rango no puede superar los 31 días")

    storage: Storage = request.app.state.storage
    camera_id = request.app.state.cfg.camera.id
    sessions, visits, alerts = storage.dashboard_rows(camera_id, start, end)
    return build_dashboard(storage.list_zones(camera_id), sessions, visits, alerts, lo=start, hi=end, now=now)
