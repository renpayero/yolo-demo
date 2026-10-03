"""Tipos de datos compartidos. Sin dependencias de OpenCV ni de YOLO."""
from __future__ import annotations

from dataclasses import dataclass

BBox = tuple[float, float, float, float]          # x1, y1, x2, y2 en píxeles
Keypoint = tuple[float, float, float]             # x, y, confianza (formato COCO, 17 puntos)


@dataclass(frozen=True)
class Detection:
    track_id: int
    bbox: BBox
    confidence: float
    keypoints: tuple[Keypoint, ...] | None = None
