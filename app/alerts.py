"""Motor de alertas por zona. Lógica pura: evalúa el estado del motor de ocupación en cada frame.

Reglas (configurables por zona):
- dwell: alguien permanece en una estación de trabajo más de `max_dwell_s`. Una alerta por visita; se resuelve
  cuando esa visita termina.
- empty: una estación de trabajo queda vacía más de `max_empty_s` (estación parada). Una alerta por período
  vacío; se resuelve cuando entra alguien. Solo cuenta el tiempo con video: sin señal no se sabe si está vacía.
- restricted: cualquier ingreso a una zona restringida. Se resuelve cuando la persona sale.
"""
from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from app.formatting import format_duration
from app.occupancy import ZoneEvent, ZoneStatus
from app.zones import Zone

AlertKind = Literal["dwell", "empty", "restricted"]
AlertKey = tuple[str, int, int]   # (tipo, zona, visita) — en "empty" la visita es 0


@dataclass(frozen=True)
class Alert:
    uid: int
    kind: str                       # AlertKind o, para posturas, "arms_raised" | "trunk_flexion" | "fall"
    zone_id: int | None             # las de postura pueden no estar en ninguna zona
    zone_name: str | None
    ts: float
    message: str
    track_id: int | None = None
    visit_id: int | None = None
    value_s: float | None = None    # permanencia o tiempo vacía al dispararse


@dataclass(frozen=True)
class AlertChange:
    action: Literal["raise", "resolve"]
    alert: Alert
    ts: float


class AlertEngine:
    def __init__(self, next_uid: Callable[[], int] | None = None) -> None:
        self._active: dict[AlertKey, Alert] = {}
        counter = itertools.count(1)
        # Numeración compartida con las alertas de postura: el registro las identifica por este número.
        self._next_uid = next_uid or (lambda: next(counter))
        self._names: Mapping[int, str] = {}

    def _who(self, track_id: int) -> str:
        name = self._names.get(track_id)
        return f"{name} (ID {track_id})" if name else f"persona {track_id}"

    @property
    def active(self) -> list[Alert]:
        return sorted(self._active.values(), key=lambda a: a.ts)

    def resolve_all(self, ts: float) -> list[AlertChange]:
        """Al apagar el sistema: ninguna alerta queda abierta en la base."""
        return [change for key in list(self._active) for change in self._resolve(key, ts)]

    def evaluate(
        self,
        ts: float,
        statuses: Iterable[ZoneStatus],
        events: Iterable[ZoneEvent],
        monitoring_since: float | None,
        names: Mapping[int, str] | None = None,   # track → nombre, si la persona está identificada
    ) -> list[AlertChange]:
        statuses = list(statuses)
        zones = {s.zone.id: s.zone for s in statuses}
        self._names = names or {}
        changes: list[AlertChange] = []

        for event in events:
            zone = zones.get(event.zone_id)
            if event.kind == "enter" and zone is not None and zone.kind == "restricted":
                message = f"Ingreso a zona restringida «{zone.name}» ({self._who(event.track_id)})"
                changes += self._raise(
                    ("restricted", zone.id, event.visit_id), "restricted", zone, ts, message,
                    track_id=event.track_id, visit_id=event.visit_id,
                )
            elif event.kind == "exit":
                changes += self._resolve(("restricted", event.zone_id, event.visit_id), ts)
                changes += self._resolve(("dwell", event.zone_id, event.visit_id), ts)

        for status in statuses:
            changes += self._check_dwell(status, ts)
            changes += self._check_empty(status, ts, monitoring_since)

        for key, alert in list(self._active.items()):
            if alert.zone_id not in zones:  # zona borrada
                changes += self._resolve(key, ts)
        return changes

    def _check_dwell(self, status: ZoneStatus, ts: float) -> list[AlertChange]:
        zone = status.zone
        if zone.kind != "work" or not zone.max_dwell_s:
            return []
        changes: list[AlertChange] = []
        for occupant in status.occupants:
            if occupant.dwell_s > zone.max_dwell_s:
                message = (
                    f"Permanencia prolongada en {zone.name}: {self._who(occupant.track_id)} "
                    f"lleva más de {format_duration(zone.max_dwell_s)}"
                )
                changes += self._raise(
                    ("dwell", zone.id, occupant.visit_id), "dwell", zone, ts, message,
                    track_id=occupant.track_id, visit_id=occupant.visit_id, value_s=occupant.dwell_s,
                )
        return changes

    def _check_empty(self, status: ZoneStatus, ts: float, monitoring_since: float | None) -> list[AlertChange]:
        zone = status.zone
        key: AlertKey = ("empty", zone.id, 0)
        if zone.kind != "work" or not zone.max_empty_s or status.empty_since is None:
            return self._resolve(key, ts)
        if monitoring_since is None:
            return []
        empty_for = ts - max(status.empty_since, monitoring_since)
        if empty_for <= zone.max_empty_s:
            return []
        message = f"{zone.name} sin nadie hace más de {format_duration(zone.max_empty_s)}"
        return self._raise(key, "empty", zone, ts, message, value_s=empty_for)

    def _raise(
        self,
        key: AlertKey,
        kind: AlertKind,
        zone: Zone,
        ts: float,
        message: str,
        **extra: float | int | None,
    ) -> list[AlertChange]:
        if key in self._active:
            return []
        alert = Alert(uid=self._next_uid(), kind=kind, zone_id=zone.id, zone_name=zone.name, ts=ts, message=message,
                      **extra)  # type: ignore[arg-type]
        self._active[key] = alert
        return [AlertChange("raise", alert, ts)]

    def _resolve(self, key: AlertKey, ts: float) -> list[AlertChange]:
        alert = self._active.pop(key, None)
        return [AlertChange("resolve", alert, ts)] if alert else []
