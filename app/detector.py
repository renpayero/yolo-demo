"""Detección y seguimiento de personas.

El resto de la app solo conoce `PersonDetector`. Cambiar Ultralytics por otro detector (por ejemplo, por la
licencia AGPL en producción) es escribir otra clase con los mismos métodos.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml

from app.config import DetectorConfig
from app.domain import Detection

log = logging.getLogger(__name__)

MODELS_DIR = Path("/models")
PERSON_CLASS = 0


class PersonDetector(Protocol):
    device: str
    model_name: str

    def track(self, image: np.ndarray) -> list[Detection]: ...

    def reset(self) -> None: ...


def _resolve_weights(model: str, device: str) -> Path:
    if model == "auto":
        return MODELS_DIR / ("yolo26s.pt" if device.startswith("cuda") else "yolo26n.pt")
    return Path(model)


def _tracker_config(track_buffer: int) -> Path:
    """Copia la config de ByteTrack que trae Ultralytics, cambiando solo `track_buffer`."""
    import ultralytics

    base = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml"
    cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
    cfg["track_buffer"] = track_buffer
    out = Path(tempfile.gettempdir()) / "bytetrack_demo.yaml"
    out.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return out


class UltralyticsPersonDetector:
    """YOLO (detección) + ByteTrack: personas con un ID estable mientras se las sigue viendo.

    Se usa el modelo de detección y no el de pose: el de pose se entrena con COCO-keypoints, que ignora a las
    personas chicas, y en cámaras de techo pierde a la mayoría (en el video de prueba: 2 de 7 contra 7 de 7).
    """

    def __init__(self, cfg: DetectorConfig) -> None:
        import torch
        from ultralytics import YOLO

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        weights = _resolve_weights(cfg.model, self.device)
        self.model_name = weights.stem
        self._cfg = cfg
        self._tracker = _tracker_config(cfg.track_buffer)
        self._model = YOLO(str(weights), task="detect")
        log.info("Detector %s en %s", self.model_name, self.device)

    def track(self, image: np.ndarray) -> list[Detection]:
        result = self._model.track(
            image,
            persist=True,
            classes=[PERSON_CLASS],
            conf=self._cfg.confidence,
            imgsz=self._cfg.image_size,
            tracker=str(self._tracker),
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        # Si el tracker todavía no confirmó ningún track, las cajas vienen sin ID: no hay a quién seguir.
        if boxes is None or boxes.id is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        return [
            Detection(
                track_id=ids[i],
                bbox=(float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])),
                confidence=float(confs[i]),
            )
            for i in range(len(ids))
        ]

    def reset(self) -> None:
        """Olvida los tracks (al reconectar la fuente, los IDs viejos ya no significan nada)."""
        predictor = getattr(self._model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()
