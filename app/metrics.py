"""Métricas del dashboard a partir de lo registrado (sesiones, visitas, alertas). Lógica pura.

Definiciones (para un rango [desde, hasta]):
- Tiempo monitoreado: tiempo con video (sesiones). Sin video no se sabe si una zona estaba vacía.
- Tiempo ocupada: unión de las visitas de la zona (con dos personas a la vez cuenta una sola vez),
  dentro del tiempo monitoreado.
- Tiempo vacía: monitoreado - ocupada. % de ocupación: ocupada / monitoreado.
- Tiempo-persona: suma de las visitas (dos personas 10 minutos = 20 minutos-persona).
- Visitas, promedio y máximo: visitas que empezaron dentro del rango; las abiertas cuentan hasta ahora.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.zones import Zone

Interval = tuple[float, float]


@dataclass(frozen=True)
class SessionRow:
    start: float
    end: float | None          # None: sigue abierta (hay video ahora)


@dataclass(frozen=True)
class VisitRow:
    zone_id: int
    track_id: int
    start: float
    end: float | None          # None: la persona sigue en la zona
    person_id: int | None = None


@dataclass(frozen=True)
class AlertRow:
    id: int
    kind: str
    zone_id: int | None
    zone_name: str | None
    track_id: int | None
    ts: float
    resolved_ts: float | None
    message: str
    person_id: int | None = None


# ---------- Operaciones con intervalos ----------

def merge(intervals: Iterable[Interval]) -> list[Interval]:
    """Unión: ordena y junta los que se superponen o se tocan. Descarta los de duración cero."""
    out: list[Interval] = []
    for start, end in sorted(i for i in intervals if i[1] > i[0]):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def clip(intervals: Iterable[Interval], lo: float, hi: float) -> list[Interval]:
    return [(max(s, lo), min(e, hi)) for s, e in intervals if e > lo and s < hi]


def intersect(a: Sequence[Interval], b: Sequence[Interval]) -> list[Interval]:
    """Intersección de dos listas ya unidas (`merge`)."""
    out: list[Interval] = []
    i = j = 0
    while i < len(a) and j < len(b):
        start, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if start < end:
            out.append((start, end))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def total(intervals: Iterable[Interval]) -> float:
    return sum(e - s for s, e in intervals)


def gaps(intervals: Sequence[Interval], lo: float, hi: float) -> list[Interval]:
    """Los huecos de una lista unida dentro de [lo, hi]."""
    out: list[Interval] = []
    cursor = lo
    for start, end in intervals:
        if start > cursor:
            out.append((cursor, min(start, hi)))
        cursor = max(cursor, end)
    if cursor < hi:
        out.append((cursor, hi))
    return out


def step_counts(intervals: Iterable[Interval]) -> list[tuple[float, int]]:
    """Cuántos intervalos están abiertos a partir de cada instante en que eso cambia."""
    deltas: dict[float, int] = defaultdict(int)
    for start, end in intervals:
        deltas[start] += 1
        deltas[end] -= 1
    count = 0
    out = []
    for t in sorted(deltas):
        count += deltas[t]
        out.append((t, count))
    return out


# ---------- Resumen del dashboard ----------

def build_dashboard(
    zones: Sequence[Zone],
    sessions: Iterable[SessionRow],
    visits: Iterable[VisitRow],
    alerts: Iterable[AlertRow],
    lo: float,
    hi: float,
    now: float,
    person_names: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    hi = min(hi, now)
    monitored = merge(clip(((s.start, s.end if s.end is not None else now) for s in sessions), lo, hi))
    monitored_s = total(monitored)

    visits = list(visits)
    visits_by_zone: dict[int, list[Interval]] = defaultdict(list)
    for v in visits:
        visits_by_zone[v.zone_id].append((v.start, v.end if v.end is not None else now))
    alerts_in_range = sorted((a for a in alerts if lo <= a.ts < hi), key=lambda a: a.ts, reverse=True)

    kpis, timeline = [], []
    for zone in zones:
        zone_visits = visits_by_zone.get(zone.id, [])
        occupied = intersect(merge(clip(zone_visits, lo, hi)), monitored)
        started = [e - s for s, e in zone_visits if lo <= s < hi]
        occupied_s = total(occupied)
        kpis.append({
            "id": zone.id,
            "name": zone.name,
            "kind": zone.kind,
            "color": zone.color,
            "occupied_s": occupied_s,
            "empty_s": monitored_s - occupied_s,
            "occupancy_pct": occupied_s / monitored_s * 100 if monitored_s > 0 else None,
            "visits": len(started),
            "avg_visit_s": sum(started) / len(started) if started else None,
            "max_visit_s": max(started) if started else None,
            "person_s": total(clip(zone_visits, lo, hi)),
            "alerts": sum(1 for a in alerts_in_range if a.zone_id == zone.id),
        })
        timeline.append({
            "id": zone.id,
            "name": zone.name,
            "color": zone.color,
            "intervals": occupied,
            "alerts": [{"ts": a.ts, "kind": a.kind} for a in alerts_in_range if a.zone_id == zone.id],
        })

    headcount = _headcount(zones, visits_by_zone, lo, hi)
    return {
        "range": {"from": lo, "to": hi, "now": now},
        "summary": {
            "monitored_s": monitored_s,
            "visits": sum(k["visits"] for k in kpis),
            "alerts": len(alerts_in_range),
            "active_alerts": sum(1 for a in alerts_in_range if a.resolved_ts is None),
            "peak_people": max((p["total"] for p in headcount), default=0),
        },
        "zones": kpis,
        "persons": _per_person(zones, visits, alerts_in_range, person_names or {}, lo, hi, now),
        "timeline": timeline,
        "unmonitored": gaps(monitored, lo, hi),
        "headcount": headcount,
        "alerts": [
            {
                "id": a.id, "kind": a.kind, "zone_id": a.zone_id, "zone_name": a.zone_name,
                "track_id": a.track_id, "ts": a.ts, "resolved_ts": a.resolved_ts, "message": a.message,
            }
            for a in alerts_in_range
        ],
    }


def _per_person(
    zones: Sequence[Zone],
    visits: Sequence[VisitRow],
    alerts: Sequence[AlertRow],
    person_names: Mapping[int, str],
    lo: float,
    hi: float,
    now: float,
) -> list[dict[str, Any]]:
    """Tiempo y visitas por zona de cada persona identificada (solo las que tuvieron actividad en el rango)."""
    out = []
    for person_id, name in sorted(person_names.items(), key=lambda item: item[1].lower()):
        rows = []
        for zone in zones:
            intervals = [
                (v.start, v.end if v.end is not None else now)
                for v in visits if v.person_id == person_id and v.zone_id == zone.id
            ]
            time_s = total(clip(intervals, lo, hi))
            started = [e - s for s, e in intervals if lo <= s < hi]
            if time_s > 0 or started:
                rows.append({
                    "zone_id": zone.id,
                    "zone_name": zone.name,
                    "color": zone.color,
                    "visits": len(started),
                    "time_s": time_s,
                    "avg_visit_s": sum(started) / len(started) if started else None,
                })
        if rows:
            out.append({
                "id": person_id,
                "name": name,
                "time_s": sum(r["time_s"] for r in rows),
                "visits": sum(r["visits"] for r in rows),
                "alerts": sum(1 for a in alerts if a.person_id == person_id),
                "zones": rows,
            })
    return out


def _headcount(
    zones: Sequence[Zone], visits_by_zone: dict[int, list[Interval]], lo: float, hi: float
) -> list[dict[str, Any]]:
    """Personas en cada zona y en total a lo largo del rango (función escalonada)."""
    deltas: dict[float, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for zone in zones:
        for start, end in clip(visits_by_zone.get(zone.id, []), lo, hi):
            deltas[start][zone.id] += 1
            if end < hi:  # si llega al final del rango, sigue adentro: no se descuenta
                deltas[end][zone.id] -= 1
    counts = {zone.id: 0 for zone in zones}
    series = []
    for t in sorted(set(deltas) | {lo, hi}):
        for zone_id, delta in deltas.get(t, {}).items():
            counts[zone_id] += delta
        series.append({"t": t, "total": sum(counts.values()), "zones": dict(counts)})
    return series
