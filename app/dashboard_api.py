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


HEATMAP_W, HEATMAP_H = 64, 48


def _range(start: float | None, end: float | None) -> tuple[float, float, float]:
    now = time.time()
    end = now if end is None else end
    start = end - DEFAULT_RANGE_S if start is None else start
    if start >= end:
        raise HTTPException(status_code=422, detail="El inicio del rango tiene que ser anterior al fin")
    if end - start > MAX_RANGE_S:
        raise HTTPException(status_code=422, detail="El rango no puede superar los 31 días")
    return start, end, now


@router.get("/heatmap")
def heatmap(
    request: Request,
    start: Annotated[float | None, Query(alias="from")] = None,
    end: Annotated[float | None, Query(alias="to")] = None,
) -> dict[str, Any]:
    """Mapa de calor de permanencia: segundos-persona por celda (cada posición registrada vale 1 s)."""
    start, end, _ = _range(start, end)
    storage: Storage = request.app.state.storage
    grid = storage.heatmap(request.app.state.cfg.camera.id, start, end, HEATMAP_W, HEATMAP_H)
    return {
        "width": HEATMAP_W,
        "height": HEATMAP_H,
        "grid": grid,
        "max_s": max(max(row) for row in grid),
        "total_s": sum(sum(row) for row in grid),
    }


@router.get("/dashboard")
def dashboard(
    request: Request,
    start: Annotated[float | None, Query(alias="from", description="epoch UTC en segundos")] = None,
    end: Annotated[float | None, Query(alias="to", description="epoch UTC en segundos")] = None,
) -> dict[str, Any]:
    start, end, now = _range(start, end)
    storage: Storage = request.app.state.storage
    camera_id = request.app.state.cfg.camera.id
    sessions, visits, alerts = storage.dashboard_rows(camera_id, start, end)
    return build_dashboard(
        storage.list_zones(camera_id), sessions, visits, alerts, lo=start, hi=end, now=now,
        person_names=storage.person_names(),
    )
