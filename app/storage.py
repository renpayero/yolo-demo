"""Persistencia en SQLite (archivo en el volumen /data).

El esquema se versiona con PRAGMA user_version: cada entrada de MIGRATIONS lleva la base a la versión siguiente.
Cada operación abre su propia conexión, así se puede usar desde cualquier hilo. Las escrituras frecuentes
(visitas, alertas, sesiones) no pasan por acá sino por `Recorder`, que escribe desde un solo hilo.
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from app.metrics import AlertRow, SessionRow, VisitRow
from app.zones import Zone, ZoneSpec

MAX_DASHBOARD_ALERTS = 500

MIGRATIONS: tuple[str, ...] = (
    # 1: zonas
    """
    CREATE TABLE zones (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id   TEXT NOT NULL,
        name        TEXT NOT NULL,
        kind        TEXT NOT NULL CHECK (kind IN ('work', 'restricted')),
        polygon     TEXT NOT NULL,          -- JSON [[x, y], ...] normalizado 0-1
        created_at  REAL NOT NULL,
        updated_at  REAL NOT NULL,
        UNIQUE (camera_id, name)
    );
    """,
    # 2: umbrales de alerta, sesiones de monitoreo, visitas y alertas. Tiempos en epoch UTC (segundos).
    """
    ALTER TABLE zones ADD COLUMN max_dwell_s REAL;
    ALTER TABLE zones ADD COLUMN max_empty_s REAL;

    -- Períodos con video: "tiempo vacía" solo cuenta mientras el sistema estaba mirando.
    CREATE TABLE sessions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id       TEXT NOT NULL,
        start_ts        REAL NOT NULL,
        end_ts          REAL,               -- NULL mientras hay video
        last_heartbeat  REAL NOT NULL
    );

    CREATE TABLE visits (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id   TEXT NOT NULL,
        session_id  INTEGER REFERENCES sessions (id),
        zone_id     INTEGER NOT NULL,
        zone_name   TEXT NOT NULL,          -- copia: la historia conserva el nombre aunque la zona cambie
        track_id    INTEGER NOT NULL,
        person_id   INTEGER,                -- identificación (opcional)
        start_ts    REAL NOT NULL,
        end_ts      REAL,                   -- NULL mientras la visita sigue abierta
        duration_s  REAL,
        closed_by   TEXT                    -- 'exit' | 'restart' (cortada por un reinicio)
    );
    CREATE INDEX visits_by_time ON visits (camera_id, start_ts);

    CREATE TABLE alerts (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id    TEXT NOT NULL,
        kind         TEXT NOT NULL,         -- 'dwell' | 'empty' | 'restricted'
        zone_id      INTEGER,
        zone_name    TEXT,
        track_id     INTEGER,
        person_id    INTEGER,
        ts           REAL NOT NULL,
        resolved_ts  REAL,
        value_s      REAL,
        message      TEXT NOT NULL
    );
    CREATE INDEX alerts_by_time ON alerts (camera_id, ts);
    """,
)


class ZoneNameTaken(ValueError):
    pass


@dataclass(frozen=True)
class RecoveryReport:
    sessions: int
    visits: int
    alerts: int


def _row_to_zone(row: sqlite3.Row) -> Zone:
    return Zone(
        id=row["id"],
        name=row["name"],
        kind=row["kind"],
        polygon=tuple((float(x), float(y)) for x, y in json.loads(row["polygon"])),
        max_dwell_s=row["max_dwell_s"],
        max_empty_s=row["max_empty_s"],
    )


def _spec_values(spec: ZoneSpec) -> tuple[str, str, str, float | None, float | None]:
    return (spec.name, spec.kind, json.dumps([list(p) for p in spec.polygon]), spec.max_dwell_s, spec.max_empty_s)


class Storage:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=5)
        con.row_factory = sqlite3.Row
        try:
            with con:  # commit si todo sale bien, rollback si hay excepción
                yield con
        finally:
            con.close()

    def _migrate(self) -> None:
        with self._tx() as con:
            con.execute("PRAGMA journal_mode=WAL")
            version = con.execute("PRAGMA user_version").fetchone()[0]
            for number, sql in enumerate(MIGRATIONS[version:], start=version + 1):
                con.executescript(sql)
                con.execute(f"PRAGMA user_version = {number}")

    # ---------- Zonas ----------

    def list_zones(self, camera_id: str) -> list[Zone]:
        with self._tx() as con:
            rows = con.execute("SELECT * FROM zones WHERE camera_id = ? ORDER BY id", (camera_id,)).fetchall()
        return [_row_to_zone(r) for r in rows]

    def create_zone(self, camera_id: str, spec: ZoneSpec) -> Zone:
        now = time.time()
        try:
            with self._tx() as con:
                cur = con.execute(
                    "INSERT INTO zones (camera_id, name, kind, polygon, max_dwell_s, max_empty_s, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (camera_id, *_spec_values(spec), now, now),
                )
                row = con.execute("SELECT * FROM zones WHERE id = ?", (cur.lastrowid,)).fetchone()
        except sqlite3.IntegrityError as exc:
            raise ZoneNameTaken(spec.name) from exc
        return _row_to_zone(row)

    def update_zone(self, zone_id: int, spec: ZoneSpec) -> Zone | None:
        try:
            with self._tx() as con:
                cur = con.execute(
                    "UPDATE zones SET name = ?, kind = ?, polygon = ?, max_dwell_s = ?, max_empty_s = ?, updated_at = ?"
                    " WHERE id = ?",
                    (*_spec_values(spec), time.time(), zone_id),
                )
                if cur.rowcount == 0:
                    return None
                row = con.execute("SELECT * FROM zones WHERE id = ?", (zone_id,)).fetchone()
        except sqlite3.IntegrityError as exc:
            raise ZoneNameTaken(spec.name) from exc
        return _row_to_zone(row)

    def delete_zone(self, zone_id: int) -> bool:
        with self._tx() as con:
            return con.execute("DELETE FROM zones WHERE id = ?", (zone_id,)).rowcount > 0

    def replace_zones(self, camera_id: str, specs: Sequence[ZoneSpec]) -> list[Zone]:
        """Importación: reemplaza todas las zonas de la cámara en una sola transacción."""
        now = time.time()
        try:
            with self._tx() as con:
                con.execute("DELETE FROM zones WHERE camera_id = ?", (camera_id,))
                con.executemany(
                    "INSERT INTO zones (camera_id, name, kind, polygon, max_dwell_s, max_empty_s, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [(camera_id, *_spec_values(spec), now, now) for spec in specs],
                )
        except sqlite3.IntegrityError as exc:
            raise ZoneNameTaken("nombres repetidos en el archivo importado") from exc
        return self.list_zones(camera_id)

    # ---------- Lectura para el dashboard ----------

    def dashboard_rows(
        self, camera_id: str, lo: float, hi: float
    ) -> tuple[list[SessionRow], list[VisitRow], list[AlertRow]]:
        """Sesiones y visitas que se superponen con [lo, hi] y alertas disparadas en ese rango."""
        overlap = "camera_id = ? AND start_ts < ? AND (end_ts IS NULL OR end_ts > ?)"
        with self._tx() as con:
            sessions = [
                SessionRow(r["start_ts"], r["end_ts"])
                for r in con.execute(f"SELECT start_ts, end_ts FROM sessions WHERE {overlap}", (camera_id, hi, lo))
            ]
            visits = [
                VisitRow(r["zone_id"], r["track_id"], r["start_ts"], r["end_ts"], r["person_id"])
                for r in con.execute(
                    f"SELECT zone_id, track_id, start_ts, end_ts, person_id FROM visits WHERE {overlap}",
                    (camera_id, hi, lo),
                )
            ]
            alerts = [
                AlertRow(r["id"], r["kind"], r["zone_id"], r["zone_name"], r["track_id"], r["ts"],
                         r["resolved_ts"], r["message"])
                for r in con.execute(
                    "SELECT * FROM alerts WHERE camera_id = ? AND ts >= ? AND ts < ? ORDER BY ts DESC LIMIT ?",
                    (camera_id, lo, hi, MAX_DASHBOARD_ALERTS),
                )
            ]
        return sessions, visits, alerts

    # ---------- Recuperación tras un corte ----------

    def recover_unclosed(self, camera_id: str) -> RecoveryReport:
        """Cierra lo que quedó abierto si la app se cortó sin apagarse bien (corte de luz, docker kill...).

        Se toma como fin el último heartbeat de la sesión: es lo último que se sabe con certeza.
        """
        with self._tx() as con:
            visits = con.execute(
                """
                UPDATE visits SET
                    end_ts = MAX(start_ts, COALESCE(
                        (SELECT last_heartbeat FROM sessions s WHERE s.id = visits.session_id), start_ts)),
                    closed_by = 'restart'
                WHERE camera_id = ? AND end_ts IS NULL
                """,
                (camera_id,),
            ).rowcount
            con.execute(
                "UPDATE visits SET duration_s = end_ts - start_ts WHERE camera_id = ? AND duration_s IS NULL",
                (camera_id,),
            )
            alerts = con.execute(
                """
                UPDATE alerts SET resolved_ts = MAX(ts, COALESCE(
                    (SELECT MAX(last_heartbeat) FROM sessions s WHERE s.camera_id = alerts.camera_id), ts))
                WHERE camera_id = ? AND resolved_ts IS NULL
                """,
                (camera_id,),
            ).rowcount
            sessions = con.execute(
                "UPDATE sessions SET end_ts = last_heartbeat WHERE camera_id = ? AND end_ts IS NULL", (camera_id,)
            ).rowcount
        return RecoveryReport(sessions=sessions, visits=visits, alerts=alerts)
