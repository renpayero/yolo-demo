"""Registro persistente de sesiones, visitas y alertas.

El pipeline nunca espera a la base: encola operaciones y un único hilo escritor las aplica en orden (un solo
escritor evita bloqueos de SQLite). Los ids internos de los motores (visit_id, uid de alerta) se traducen a
ids de la base dentro de ese hilo, así una salida siempre actualiza la fila que creó su entrada.
"""
from __future__ import annotations

import logging
import queue
import sqlite3
import threading
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from app.alerts import AlertChange
from app.occupancy import ZoneEvent

log = logging.getLogger(__name__)

Operation = Callable[[sqlite3.Connection], None]


class Recorder:
    def __init__(self, db_path: Path, camera_id: str) -> None:
        self._db_path = db_path
        self._camera_id = camera_id
        self._queue: queue.Queue[Operation | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        # Solo los toca el hilo escritor:
        self._session_id: int | None = None
        self._visit_rows: dict[int, int] = {}   # visit_id del motor → visits.id
        self._alert_rows: dict[int, int] = {}   # uid de la alerta → alerts.id

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Termina de escribir lo pendiente y cierra."""
        self._queue.put(None)
        self._thread.join(timeout=10)

    def flush(self) -> None:
        """Espera a que todo lo encolado esté escrito (para tests y apagado)."""
        self._queue.join()

    # ---------- API para el pipeline (no bloquea) ----------

    def session_start(self, ts: float) -> None:
        def op(con: sqlite3.Connection) -> None:
            cur = con.execute(
                "INSERT INTO sessions (camera_id, start_ts, last_heartbeat) VALUES (?, ?, ?)",
                (self._camera_id, ts, ts),
            )
            self._session_id = cur.lastrowid

        self._queue.put(op)

    def session_heartbeat(self, ts: float) -> None:
        def op(con: sqlite3.Connection) -> None:
            if self._session_id is not None:
                con.execute("UPDATE sessions SET last_heartbeat = ? WHERE id = ?", (ts, self._session_id))

        self._queue.put(op)

    def session_end(self, ts: float) -> None:
        def op(con: sqlite3.Connection) -> None:
            if self._session_id is not None:
                con.execute(
                    "UPDATE sessions SET end_ts = ?, last_heartbeat = ? WHERE id = ?", (ts, ts, self._session_id)
                )
                self._session_id = None

        self._queue.put(op)

    def zone_events(
        self,
        events: Iterable[ZoneEvent],
        zone_names: Mapping[int, str],
        persons: Mapping[int, int] | None = None,   # track → persona identificada
    ) -> None:
        persons = persons or {}
        for event in events:
            if event.kind == "enter":
                zone_name = zone_names.get(event.zone_id, f"zona {event.zone_id}")
                self._queue.put(self._visit_open_op(event, zone_name, persons.get(event.track_id)))
            elif event.kind == "exit":
                self._queue.put(self._visit_close_op(event))

    def positions(self, ts: float, points: Mapping[int, tuple[float, float]]) -> None:
        """Dónde está cada persona (el pipeline lo manda una vez por segundo): alimenta el mapa de calor."""
        rows = [(self._camera_id, ts, track_id, x, y) for track_id, (x, y) in points.items()]
        if not rows:
            return

        def op(con: sqlite3.Connection) -> None:
            con.executemany("INSERT INTO positions (camera_id, ts, track_id, x, y) VALUES (?, ?, ?, ?, ?)", rows)

        self._queue.put(op)

    def visit_person(self, visit_id: int, person_id: int) -> None:
        """La persona se identificó con la visita ya abierta: se le asigna desde el comienzo."""
        def op(con: sqlite3.Connection) -> None:
            row_id = self._visit_rows.get(visit_id)
            if row_id is not None:
                con.execute("UPDATE visits SET person_id = ? WHERE id = ?", (person_id, row_id))

        self._queue.put(op)

    def alert_changes(self, changes: Iterable[AlertChange], persons: Mapping[int, int] | None = None) -> None:
        persons = persons or {}
        for change in changes:
            if change.action == "raise":
                op = self._alert_raise_op(change, persons.get(change.alert.track_id))
            else:
                op = self._alert_resolve_op(change)
            self._queue.put(op)

    # ---------- Operaciones (corren en el hilo escritor) ----------

    def _visit_open_op(self, event: ZoneEvent, zone_name: str, person_id: int | None) -> Operation:
        def op(con: sqlite3.Connection) -> None:
            cur = con.execute(
                "INSERT INTO visits (camera_id, session_id, zone_id, zone_name, track_id, person_id, start_ts)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (self._camera_id, self._session_id, event.zone_id, zone_name, event.track_id, person_id,
                 event.enter_ts),
            )
            self._visit_rows[event.visit_id] = cur.lastrowid

        return op

    def _visit_close_op(self, event: ZoneEvent) -> Operation:
        def op(con: sqlite3.Connection) -> None:
            row_id = self._visit_rows.pop(event.visit_id, None)
            if row_id is not None:
                con.execute(
                    "UPDATE visits SET end_ts = ?, duration_s = ?, closed_by = 'exit' WHERE id = ?",
                    (event.ts, event.duration_s, row_id),
                )

        return op

    def _alert_raise_op(self, change: AlertChange, person_id: int | None) -> Operation:
        alert = change.alert

        def op(con: sqlite3.Connection) -> None:
            cur = con.execute(
                "INSERT INTO alerts (camera_id, kind, zone_id, zone_name, track_id, person_id, ts, value_s, message)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (self._camera_id, alert.kind, alert.zone_id, alert.zone_name, alert.track_id, person_id, alert.ts,
                 alert.value_s, alert.message),
            )
            self._alert_rows[alert.uid] = cur.lastrowid

        return op

    def _alert_resolve_op(self, change: AlertChange) -> Operation:
        def op(con: sqlite3.Connection) -> None:
            row_id = self._alert_rows.pop(change.alert.uid, None)
            if row_id is not None:
                con.execute("UPDATE alerts SET resolved_ts = ? WHERE id = ?", (change.ts, row_id))

        return op

    # ---------- Hilo escritor ----------

    def _run(self) -> None:
        # Una transacción por operación: hay pocos eventos por segundo y, si una falla, no arrastra a las demás.
        con = sqlite3.connect(self._db_path, timeout=5)
        try:
            while True:
                op = self._queue.get()
                try:
                    if op is None:
                        break
                    with con:
                        op(con)
                except Exception:  # un registro fallido no puede tirar abajo la demo
                    log.exception("No se pudo guardar un evento en la base")
                finally:
                    self._queue.task_done()
        finally:
            con.close()
