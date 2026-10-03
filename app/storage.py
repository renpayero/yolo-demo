"""Persistencia en SQLite (archivo en el volumen /data).

El esquema se versiona con PRAGMA user_version: cada entrada de MIGRATIONS lleva la base a la versión siguiente.
Cada operación abre su propia conexión, así se puede usar desde cualquier hilo.
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from app.zones import Point, Zone

MIGRATIONS: tuple[str, ...] = (
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
)


class ZoneNameTaken(ValueError):
    pass


def _row_to_zone(row: sqlite3.Row) -> Zone:
    return Zone(
        id=row["id"],
        name=row["name"],
        kind=row["kind"],
        polygon=tuple((float(x), float(y)) for x, y in json.loads(row["polygon"])),
    )


class Storage:
    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self._path, timeout=5)
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

    def create_zone(self, camera_id: str, name: str, kind: str, polygon: Sequence[Point]) -> Zone:
        now = time.time()
        try:
            with self._tx() as con:
                cur = con.execute(
                    "INSERT INTO zones (camera_id, name, kind, polygon, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (camera_id, name, kind, json.dumps([list(p) for p in polygon]), now, now),
                )
                row = con.execute("SELECT * FROM zones WHERE id = ?", (cur.lastrowid,)).fetchone()
        except sqlite3.IntegrityError as exc:
            raise ZoneNameTaken(name) from exc
        return _row_to_zone(row)

    def update_zone(self, zone_id: int, name: str, kind: str, polygon: Sequence[Point]) -> Zone | None:
        try:
            with self._tx() as con:
                cur = con.execute(
                    "UPDATE zones SET name = ?, kind = ?, polygon = ?, updated_at = ? WHERE id = ?",
                    (name, kind, json.dumps([list(p) for p in polygon]), time.time(), zone_id),
                )
                if cur.rowcount == 0:
                    return None
                row = con.execute("SELECT * FROM zones WHERE id = ?", (zone_id,)).fetchone()
        except sqlite3.IntegrityError as exc:
            raise ZoneNameTaken(name) from exc
        return _row_to_zone(row)

    def delete_zone(self, zone_id: int) -> bool:
        with self._tx() as con:
            return con.execute("DELETE FROM zones WHERE id = ?", (zone_id,)).rowcount > 0

    def replace_zones(self, camera_id: str, zones: Sequence[tuple[str, str, Sequence[Point]]]) -> list[Zone]:
        """Importación: reemplaza todas las zonas de la cámara en una sola transacción."""
        now = time.time()
        try:
            with self._tx() as con:
                con.execute("DELETE FROM zones WHERE camera_id = ?", (camera_id,))
                con.executemany(
                    "INSERT INTO zones (camera_id, name, kind, polygon, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    [(camera_id, n, k, json.dumps([list(p) for p in poly]), now, now) for n, k, poly in zones],
                )
        except sqlite3.IntegrityError as exc:
            raise ZoneNameTaken("nombres repetidos en el archivo importado") from exc
        return self.list_zones(camera_id)
