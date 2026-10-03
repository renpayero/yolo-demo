"""Zonas poligonales con nombre. Geometría pura (sin OpenCV), en coordenadas normalizadas 0-1.

Las coordenadas normalizadas hacen que una zona marcada sobre la imagen sirva igual con cualquier resolución.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.domain import BBox

Point = tuple[float, float]

ZONE_KINDS = ("work", "restricted")
ANCHORS = ("center", "torso", "feet")
ZONE_PALETTE = ("#2f81f7", "#3fb950", "#d29922", "#39c5cf", "#f778ba", "#e3b341", "#79c0ff", "#a371f7")
RESTRICTED_COLOR = "#f85149"
TORSO_RATIO = 0.35  # altura del pecho, medida desde arriba de la caja de la persona


@dataclass(frozen=True)
class Zone:
    id: int
    name: str
    kind: str                       # "work" (estación de trabajo) | "restricted"
    polygon: tuple[Point, ...]
    max_dwell_s: float | None = None   # alerta si alguien se queda más (solo zonas de trabajo)
    max_empty_s: float | None = None   # alerta si la estación queda vacía más (solo zonas de trabajo)

    @property
    def color(self) -> str:
        """Color de la zona (hex), el mismo en el video y en la web."""
        if self.kind == "restricted":
            return RESTRICTED_COLOR
        return ZONE_PALETTE[(self.id - 1) % len(ZONE_PALETTE)]


@dataclass(frozen=True)
class ZoneSpec:
    """Datos de una zona para crearla o editarla (todavía sin id)."""
    name: str
    kind: str
    polygon: tuple[Point, ...]
    max_dwell_s: float | None = None
    max_empty_s: float | None = None


def point_in_polygon(point: Point, polygon: tuple[Point, ...]) -> bool:
    """Ray casting: cuenta cuántas veces una semirrecta horizontal desde el punto cruza los lados."""
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def anchor_point(bbox: BBox, width: int, height: int, mode: str) -> Point:
    """Punto de la persona que decide si está dentro de una zona, normalizado 0-1.

    feet: pies (cámara en altura mirando el piso) · torso: pecho (webcam de escritorio, no se ven los pies)
    · center: centro de la caja.
    """
    x1, y1, x2, y2 = bbox
    if mode == "center":
        y = (y1 + y2) / 2
    elif mode == "feet":
        y = y2
    elif mode == "torso":
        y = y1 + TORSO_RATIO * (y2 - y1)
    else:
        raise ValueError(f"Punto de referencia desconocido: {mode!r} (opciones: {', '.join(ANCHORS)})")
    return ((x1 + x2) / 2 / width, y / height)
