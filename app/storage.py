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

import numpy as np

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
    # 3: personas registradas con consentimiento y sus huellas faciales (nunca fotos).
    """
    CREATE TABLE persons (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL UNIQUE,
        consent_ts  REAL NOT NULL,          -- cuándo se registró su consentimiento
        created_ts  REAL NOT NULL
    );
    CREATE TABLE face_embeddings (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id   INTEGER NOT NULL REFERENCES persons (id),
        vector      BLOB NOT NULL,          -- 128 float32 normalizados
        created_ts  REAL NOT NULL
    );
    CREATE INDEX face_embeddings_by_person ON face_embeddings (person_id);
    """,
    # 4: posiciones (una por persona y por segundo) para el mapa de calor de permanencia.
    """
    CREATE TABLE positions (
        camera_id  TEXT NOT NULL,
        ts         REAL NOT NULL,
        track_id   INTEGER NOT NULL,
        x          REAL NOT NULL,     -- punto de referencia normalizado 0-1
        y          REAL NOT NULL
    );
    CREATE INDEX positions_by_time ON positions (camera_id, ts);
    """,
)

POSITIONS_RETENTION_DAYS = 30

EMBEDDING_DIM = 128


class ZoneNameTaken(ValueError):
    pass


class PersonNameTaken(ValueError):
    pass


@dataclass(frozen=True)
class Person:
    id: int
    name: str
    consent_ts: float
    samples: int


@dataclass(frozen=True)
class DeletionReport:
    visits_anonymized: int
    alerts_anonymized: int


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

    # ---------- Personas registradas ----------

    def list_persons(self) -> list[Person]:
        with self._tx() as con:
            rows = con.execute(
                "SELECT p.id, p.name, p.consent_ts, COUNT(e.id) AS samples FROM persons p"
                " LEFT JOIN face_embeddings e ON e.person_id = p.id GROUP BY p.id ORDER BY p.name"
            ).fetchall()
        return [Person(r["id"], r["name"], r["consent_ts"], r["samples"]) for r in rows]

    def person_names(self) -> dict[int, str]:
        with self._tx() as con:
            return {r["id"]: r["name"] for r in con.execute("SELECT id, name FROM persons")}

    def create_person(self, name: str, embeddings: np.ndarray) -> Person:
        """Alta con su consentimiento y sus huellas, todo junto: sin huellas no queda ninguna persona a medias."""
        now = time.time()
        try:
            with self._tx() as con:
                person_id = con.execute(
                    "INSERT INTO persons (name, consent_ts, created_ts) VALUES (?, ?, ?)", (name, now, now)
                ).lastrowid
                self._insert_embeddings(con, person_id, embeddings, now)
        except sqlite3.IntegrityError as exc:
            raise PersonNameTaken(name) from exc
        return Person(person_id, name, now, len(embeddings))

    def add_embeddings(self, person_id: int, embeddings: np.ndarray) -> bool:
        with self._tx() as con:
            if con.execute("SELECT 1 FROM persons WHERE id = ?", (person_id,)).fetchone() is None:
                return False
            self._insert_embeddings(con, person_id, embeddings, time.time())
        return True

    @staticmethod
    def _insert_embeddings(con: sqlite3.Connection, person_id: int, embeddings: np.ndarray, now: float) -> None:
        vectors = np.asarray(embeddings, dtype=np.float32).reshape(-1, EMBEDDING_DIM)
        con.executemany(
            "INSERT INTO face_embeddings (person_id, vector, created_ts) VALUES (?, ?, ?)",
            [(person_id, v.tobytes(), now) for v in vectors],
        )

    def load_embeddings(self) -> dict[int, np.ndarray]:
        out: dict[int, list[np.ndarray]] = {}
        with self._tx() as con:
            for r in con.execute("SELECT person_id, vector FROM face_embeddings"):
                out.setdefault(r["person_id"], []).append(np.frombuffer(r["vector"], dtype=np.float32))
        return {pid: np.vstack(vectors) for pid, vectors in out.items()}

    def delete_person(self, person_id: int) -> DeletionReport | None:
        """Borra a la persona y sus huellas, y deja su historial anónimo (las visitas siguen contando)."""
        with self._tx() as con:
            if con.execute("SELECT 1 FROM persons WHERE id = ?", (person_id,)).fetchone() is None:
                return None
            visits = con.execute("UPDATE visits SET person_id = NULL WHERE person_id = ?", (person_id,)).rowcount
            alerts = con.execute("UPDATE alerts SET person_id = NULL WHERE person_id = ?", (person_id,)).rowcount
            con.execute("DELETE FROM face_embeddings WHERE person_id = ?", (person_id,))
            con.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        return DeletionReport(visits_anonymized=visits, alerts_anonymized=alerts)

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
                         r["resolved_ts"], r["message"], r["person_id"])
                for r in con.execute(
                    "SELECT * FROM alerts WHERE camera_id = ? AND ts >= ? AND ts < ? ORDER BY ts DESC LIMIT ?",
                    (camera_id, lo, hi, MAX_DASHBOARD_ALERTS),
                )
            ]
        return sessions, visits, alerts

    def heatmap(self, camera_id: str, lo: float, hi: float, width: int, height: int) -> list[list[int]]:
        """Segundos-persona en cada celda de una grilla `width × height` sobre la imagen (1 posición = 1 s)."""
        grid = [[0] * width for _ in range(height)]
        with self._tx() as con:
            rows = con.execute(
                "SELECT MIN(CAST(x * ? AS INTEGER), ?) AS cx, MIN(CAST(y * ? AS INTEGER), ?) AS cy, COUNT(*) AS n"
                " FROM positions WHERE camera_id = ? AND ts >= ? AND ts < ? GROUP BY cx, cy",
                (width, width - 1, height, height - 1, camera_id, lo, hi),
            ).fetchall()
        for r in rows:
            if 0 <= r["cx"] < width and 0 <= r["cy"] < height:
                grid[r["cy"]][r["cx"]] = r["n"]
        return grid

    def prune_positions(self, now: float, days: int = POSITIONS_RETENTION_DAYS) -> int:
        """Las posiciones solo sirven para el mapa de calor: se guardan un tiempo y se borran."""
        with self._tx() as con:
            return con.execute("DELETE FROM positions WHERE ts < ?", (now - days * 86400,)).rowcount

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
