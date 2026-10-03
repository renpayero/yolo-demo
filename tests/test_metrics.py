"""Métricas del dashboard: operaciones con intervalos y armado del resumen por zona."""
import pytest

from app.metrics import (
    AlertRow,
    SessionRow,
    VisitRow,
    build_dashboard,
    clip,
    gaps,
    intersect,
    merge,
    step_counts,
    total,
)
from app.zones import Zone

SQUARE = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
LIJADO = Zone(1, "Lijado", "work", SQUARE)
EMBALAJE = Zone(2, "Embalaje", "work", SQUARE)


# ---------- Intervalos ----------

def test_merge_unions_overlapping_touching_and_unsorted():
    assert merge([(30, 40), (10, 20), (15, 25), (25, 28), (50, 50)]) == [(10, 28), (30, 40)]


def test_clip_to_range():
    assert clip([(0, 10), (20, 30), (40, 50)], 5, 45) == [(5, 10), (20, 30), (40, 45)]
    assert clip([(0, 10)], 10, 20) == []


def test_intersect_and_total():
    a = [(0, 10), (20, 30)]
    b = [(5, 25)]
    assert intersect(a, b) == [(5, 10), (20, 25)]
    assert total(intersect(a, b)) == 10


def test_gaps_inside_range():
    assert gaps([(10, 20), (30, 40)], 0, 50) == [(0, 10), (20, 30), (40, 50)]
    assert gaps([], 0, 50) == [(0, 50)]
    assert gaps([(0, 50)], 0, 50) == []


def test_step_counts():
    assert step_counts([(0, 10), (5, 15), (10, 20)]) == [(0, 1), (5, 2), (10, 2), (15, 1), (20, 0)]


# ---------- Resumen ----------

SESSIONS = [SessionRow(0, 100), SessionRow(150, None)]   # sin video entre 100 y 150; la segunda sigue abierta
VISITS = [
    VisitRow(1, 7, 10, 30),
    VisitRow(1, 8, 20, 40),        # se superpone con la anterior: dos personas a la vez
    VisitRow(1, 9, 90, 100),
    VisitRow(1, 7, 170, None),     # abierta: cuenta hasta "ahora"
    VisitRow(99, 7, 10, 20),       # zona borrada: no entra en ninguna zona actual
]
ALERTS = [
    AlertRow(1, "dwell", 1, "Lijado", 7, 35.0, 40.0, "Permanencia"),
    AlertRow(2, "empty", 2, "Embalaje", None, 60.0, None, "Vacía"),
]


def zone_kpis(result, zone_id):
    return next(z for z in result["zones"] if z["id"] == zone_id)


def test_dashboard_kpis_per_zone():
    result = build_dashboard([LIJADO, EMBALAJE], SESSIONS, VISITS, ALERTS, lo=0, hi=1000, now=200)

    assert result["range"] == {"from": 0, "to": 200, "now": 200}   # el futuro no cuenta
    assert result["summary"]["monitored_s"] == 150                  # 100 + 50 (hasta ahora)
    assert result["unmonitored"] == [(100, 150)]

    lijado = zone_kpis(result, 1)
    assert lijado["occupied_s"] == 30 + 10 + 30                     # [10,40] ∪ [90,100] ∪ [170,200]
    assert lijado["empty_s"] == 150 - 70
    assert lijado["occupancy_pct"] == pytest.approx(70 / 150 * 100)
    assert lijado["visits"] == 4
    assert lijado["avg_visit_s"] == pytest.approx((20 + 20 + 10 + 30) / 4)
    assert lijado["max_visit_s"] == 30
    assert lijado["person_s"] == 20 + 20 + 10 + 30                  # tiempo-persona: suma, no unión
    assert lijado["alerts"] == 1
    assert result["timeline"][0]["intervals"] == [(10, 40), (90, 100), (170, 200)]

    embalaje = zone_kpis(result, 2)
    assert (embalaje["occupied_s"], embalaje["empty_s"], embalaje["occupancy_pct"]) == (0, 150, 0)
    assert (embalaje["visits"], embalaje["avg_visit_s"], embalaje["max_visit_s"]) == (0, None, None)
    assert embalaje["alerts"] == 1

    assert result["summary"]["visits"] == 4 and result["summary"]["alerts"] == 2
    assert [a["id"] for a in result["alerts"]] == [2, 1]           # más recientes primero


def test_dashboard_clips_to_range():
    result = build_dashboard([LIJADO], SESSIONS, VISITS, ALERTS, lo=25, hi=95, now=200)
    lijado = zone_kpis(result, 1)
    assert result["summary"]["monitored_s"] == 70
    assert lijado["occupied_s"] == (40 - 25) + (95 - 90)
    assert lijado["visits"] == 1                    # solo la que empezó dentro del rango (t=90)
    assert lijado["person_s"] == (30 - 25) + (40 - 25) + (95 - 90)
    assert lijado["alerts"] == 1                    # la de t=35


def test_dashboard_without_video_has_no_percentage():
    result = build_dashboard([LIJADO], SESSIONS, VISITS, ALERTS, lo=110, hi=140, now=200)
    lijado = zone_kpis(result, 1)
    assert result["summary"]["monitored_s"] == 0
    assert lijado["occupancy_pct"] is None and lijado["empty_s"] == 0
    assert result["unmonitored"] == [(110, 140)]


def test_headcount_series_totals_and_per_zone():
    visits = [VisitRow(1, 7, 10, 30), VisitRow(1, 8, 20, 40), VisitRow(2, 9, 25, 35)]
    result = build_dashboard([LIJADO, EMBALAJE], [SessionRow(0, 100)], visits, [], lo=0, hi=100, now=200)
    series = [(p["t"], p["total"], p["zones"]) for p in result["headcount"]]
    assert series == [
        (0, 0, {1: 0, 2: 0}),
        (10, 1, {1: 1, 2: 0}),
        (20, 2, {1: 2, 2: 0}),
        (25, 3, {1: 2, 2: 1}),
        (30, 2, {1: 1, 2: 1}),
        (35, 1, {1: 1, 2: 0}),
        (40, 0, {1: 0, 2: 0}),
        (100, 0, {1: 0, 2: 0}),
    ]
    assert result["summary"]["peak_people"] == 3
