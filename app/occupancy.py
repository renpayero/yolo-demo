"""Motor de ocupación: decide entradas y salidas de zonas a partir del punto de cada persona seguida.

Lógica pura (sin OpenCV ni YOLO): recibe (timestamp, {track_id: punto}) y devuelve eventos, así se puede
probar con secuencias simuladas.

- Entrada: la persona tiene que estar dentro de forma continua durante `enter_debounce_s` (evita visitas falsas
  de alguien que pasa rozando el borde). La visita empieza cuando entró, no cuando se confirmó.
- Salida: recién cuando no se la ve dentro durante más de `exit_grace_s`, así un parpadeo del detector o una
  oclusión breve no cortan la visita. La visita dura desde la entrada hasta la última vez que se la vio dentro.
- Presentes: el conteo en vivo solo incluye a quien se vio dentro en el último `PRESENCE_WINDOW_S`; una visita
  en período de gracia sigue abierta, pero no se cuenta (si no, un ID perdido se contaría dos veces).
- Reasignación: si el tracker pierde a alguien y lo reencuentra con otro ID en el mismo lugar, la visita
  continúa con el ID nuevo en vez de cerrar una y abrir otra.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from app.zones import Point, Zone, point_in_polygon

PRESENCE_WINDOW_S = 0.5   # tapa parpadeos del detector sin dejar "fantasmas" en el conteo
REBIND_DISTANCE = 0.05   # distancia normalizada máxima entre donde se perdió un track y donde aparece el nuevo


@dataclass(frozen=True)
class ZoneEvent:
    kind: Literal["enter", "exit", "rebind"]
    zone_id: int
    track_id: int
    ts: float           # entrada: cuándo entró · salida: última vez que se lo vio dentro · rebind: ahora
    enter_ts: float
    duration_s: float   # solo en las salidas
    visit_id: int
    previous_track_id: int | None = None   # solo en rebind


@dataclass(frozen=True)
class Occupant:
    track_id: int
    enter_ts: float
    dwell_s: float
    present: bool       # visto dentro hace menos de PRESENCE_WINDOW_S


@dataclass(frozen=True)
class ZoneStatus:
    zone: Zone
    occupants: tuple[Occupant, ...]   # visitas abiertas (incluye las que están en período de gracia)
    empty_since: float | None         # None mientras hay alguna visita abierta

    @property
    def present(self) -> tuple[Occupant, ...]:
        return tuple(o for o in self.occupants if o.present)


@dataclass
class _Visit:
    visit_id: int
    enter_ts: float
    last_seen: float
    last_point: Point


class OccupancyEngine:
    def __init__(
        self,
        zones: Iterable[Zone],
        enter_debounce_s: float,
        exit_grace_s: float,
        start_ts: float,
    ) -> None:
        self._debounce = enter_debounce_s
        self._grace = exit_grace_s
        self._now = start_ts
        self._next_visit_id = 1
        self._zones: dict[int, Zone] = {}
        self._visits: dict[int, dict[int, _Visit]] = {}        # zona → track → visita
        self._candidates: dict[int, dict[int, float]] = {}     # zona → track → desde cuándo está dentro
        self._empty_since: dict[int, float | None] = {}
        self._last_exit: dict[int, float] = {}                 # última salida mientras la zona tenía gente
        self.set_zones(zones, start_ts)

    def set_zones(self, zones: Iterable[Zone], ts: float) -> list[ZoneEvent]:
        """Reemplaza las zonas en caliente. Las visitas de zonas borradas se cierran; las editadas siguen."""
        new = {zone.id: zone for zone in zones}
        events: list[ZoneEvent] = []
        for zone_id in [z for z in self._zones if z not in new]:
            events += [self._close(zone_id, track_id) for track_id in list(self._visits[zone_id])]
            for table in (self._visits, self._candidates, self._empty_since, self._last_exit):
                table.pop(zone_id, None)
        for zone_id in new:
            if zone_id not in self._zones:
                self._visits[zone_id] = {}
                self._candidates[zone_id] = {}
                self._empty_since[zone_id] = ts
        self._zones = new
        return events

    def update(self, ts: float, points: Mapping[int, Point]) -> list[ZoneEvent]:
        self._now = ts
        events: list[ZoneEvent] = []
        for zone_id, zone in self._zones.items():
            inside = {tid for tid, point in points.items() if point_in_polygon(point, zone.polygon)}
            events += self._update_zone(zone_id, inside, points, ts)
        return events

    def close_all(self) -> list[ZoneEvent]:
        """Cierra todas las visitas sin esperar la gracia (al apagar el sistema)."""
        return [self._close(zone_id, tid) for zone_id in self._zones for tid in list(self._visits[zone_id])]

    def snapshot(self) -> list[ZoneStatus]:
        return [
            ZoneStatus(
                zone=zone,
                occupants=tuple(self._occupant(tid, v) for tid, v in self._visits[zone_id].items()),
                empty_since=self._empty_since[zone_id],
            )
            for zone_id, zone in sorted(self._zones.items())
        ]

    def track_zones(self) -> dict[int, list[tuple[Zone, float]]]:
        """Para cada persona presente, las zonas donde está y hace cuánto (para las etiquetas del video)."""
        result: dict[int, list[tuple[Zone, float]]] = {}
        for zone_id, zone in sorted(self._zones.items()):
            for tid, visit in self._visits[zone_id].items():
                if self._is_present(visit):
                    result.setdefault(tid, []).append((zone, visit.last_seen - visit.enter_ts))
        return result

    # ---------- Internos ----------

    def _is_present(self, visit: _Visit) -> bool:
        return self._now - visit.last_seen <= PRESENCE_WINDOW_S

    def _occupant(self, track_id: int, visit: _Visit) -> Occupant:
        return Occupant(track_id, visit.enter_ts, visit.last_seen - visit.enter_ts, self._is_present(visit))

    def _update_zone(
        self, zone_id: int, inside: set[int], points: Mapping[int, Point], ts: float
    ) -> list[ZoneEvent]:
        visits = self._visits[zone_id]
        candidates = self._candidates[zone_id]
        events: list[ZoneEvent] = []

        for tid in sorted(inside):
            point = points[tid]
            if tid in visits:
                visits[tid].last_seen, visits[tid].last_point = ts, point
                continue
            orphan = self._find_orphan(visits, points, point)
            if orphan is not None:
                events.append(self._rebind(zone_id, orphan, tid, point, ts))
                candidates.pop(tid, None)
                continue
            first_seen = candidates.setdefault(tid, ts)
            if ts - first_seen >= self._debounce:
                del candidates[tid]
                events.append(self._open(zone_id, tid, first_seen, point, ts))

        for tid in [t for t in candidates if t not in inside]:
            del candidates[tid]  # salió antes de confirmar la entrada

        expired = [tid for tid, v in visits.items() if tid not in inside and ts - v.last_seen > self._grace]
        events += [self._close(zone_id, tid) for tid in expired]
        return events

    @staticmethod
    def _find_orphan(visits: dict[int, _Visit], points: Mapping[int, Point], point: Point) -> int | None:
        """Visita abierta cuyo track ya no se ve en ningún lado y que se perdió cerca de `point`."""
        best, best_distance = None, REBIND_DISTANCE
        for tid, visit in visits.items():
            if tid in points:
                continue  # sigue en cuadro: es otra persona
            distance = math.dist(visit.last_point, point)
            if distance <= best_distance:
                best, best_distance = tid, distance
        return best

    def _open(self, zone_id: int, track_id: int, first_seen: float, point: Point, ts: float) -> ZoneEvent:
        visits = self._visits[zone_id]
        if not visits:
            self._empty_since[zone_id] = None
            self._last_exit.pop(zone_id, None)
        visit = _Visit(self._next_visit_id, first_seen, ts, point)
        self._next_visit_id += 1
        visits[track_id] = visit
        return ZoneEvent("enter", zone_id, track_id, first_seen, first_seen, 0.0, visit.visit_id)

    def _rebind(self, zone_id: int, old_tid: int, new_tid: int, point: Point, ts: float) -> ZoneEvent:
        visits = self._visits[zone_id]
        visit = visits.pop(old_tid)
        visit.last_seen, visit.last_point = ts, point
        visits[new_tid] = visit
        return ZoneEvent("rebind", zone_id, new_tid, ts, visit.enter_ts, 0.0, visit.visit_id, old_tid)

    def _close(self, zone_id: int, track_id: int) -> ZoneEvent:
        visit = self._visits[zone_id].pop(track_id)
        last_exit = max(self._last_exit.get(zone_id, visit.last_seen), visit.last_seen)
        self._last_exit[zone_id] = last_exit
        if not self._visits[zone_id]:
            self._empty_since[zone_id] = last_exit
        return ZoneEvent(
            "exit",
            zone_id,
            track_id,
            visit.last_seen,
            visit.enter_ts,
            visit.last_seen - visit.enter_ts,
            visit.visit_id,
        )
