"""Carga de config.yaml, con valores por defecto para todo lo que no se especifique."""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

from app.zones import ANCHORS


@dataclass(frozen=True)
class CameraConfig:
    id: str = "cam1"
    name: str = "Webcam"
    source: str = "rtsp://mediamtx:8554/webcam"
    anchor: str = "torso"

    def __post_init__(self) -> None:
        if self.anchor not in ANCHORS:
            raise ValueError(f"config.yaml: camera.anchor debe ser uno de {', '.join(ANCHORS)}")


@dataclass(frozen=True)
class OccupancyConfig:
    enter_debounce_s: float = 0.5
    exit_grace_s: float = 3.0


@dataclass(frozen=True)
class DetectorConfig:
    model: str = "auto"
    confidence: float = 0.4
    image_size: int = 640
    track_buffer: int = 60


@dataclass(frozen=True)
class IdentificationConfig:
    enabled: bool = False          # apagada por defecto: se prende desde la pestaña Personas
    # Similitud coseno mínima. Medido con LFW (25 registrados, 465 fotos propias, 300 desconocidos), foto a foto:
    # 0.40 → 100 % reconocidas, 1.3 % de desconocidos aceptados; 0.45 → 99.8 % y 0.3 %; 0.50 → 97.4 % y 0 %.
    match_threshold: float = 0.45
    min_margin: float = 0.08       # ventaja mínima sobre la segunda persona más parecida
    votes_needed: int = 3          # coincidencias (de las últimas 5) para ponerle nombre a un track
    min_face_px: int = 40          # caras más chicas no se intentan reconocer
    checks_per_frame: int = 2      # cuántos tracks se revisan por frame (cuida los fps)
    check_interval_s: float = 0.25 # tracks sin nombre: cada cuánto se los vuelve a mirar
    # Tracks con nombre: cada cuánto se verifica que siga siendo la misma persona. Con 2 contradicciones seguidas
    # para retirar el nombre, tras un cambio de ID del tracker el nombre equivocado dura como mucho ~2 s.
    recheck_interval_s: float = 1.0


@dataclass(frozen=True)
class StreamConfig:
    max_width: int = 960
    fps: int = 15
    jpeg_quality: int = 75


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    occupancy: OccupancyConfig = field(default_factory=OccupancyConfig)
    identification: IdentificationConfig = field(default_factory=IdentificationConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)


T = TypeVar("T")


def _section(cls: type[T], raw: dict[str, Any], name: str) -> T:
    values = raw.get(name) or {}
    known = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"config.yaml: clave(s) desconocida(s) en '{name}': {', '.join(sorted(unknown))}")
    return cls(**values)


def load_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    raw = raw or {}
    camera = _section(CameraConfig, raw, "camera")
    if source := os.environ.get("CAMERA_SOURCE"):
        camera = dataclasses.replace(camera, source=source)
    return AppConfig(
        camera=camera,
        detector=_section(DetectorConfig, raw, "detector"),
        occupancy=_section(OccupancyConfig, raw, "occupancy"),
        identification=_section(IdentificationConfig, raw, "identification"),
        stream=_section(StreamConfig, raw, "stream"),
    )
