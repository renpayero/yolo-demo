"""Pipeline completo con un detector falso (sin YOLO): fuente → detección → JPEG + estado."""
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.capture import FrameSource
from app.config import AppConfig
from app.domain import Detection
from app.pipeline import Pipeline
from app.storage import Storage

WHOLE_FRAME = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
RIGHT_EDGE = ((0.9, 0.0), (1.0, 0.0), (1.0, 1.0), (0.9, 1.0))


class FakeDetector:
    device = "cpu"
    model_name = "fake"

    def __init__(self) -> None:
        self.resets = 0

    def track(self, image: np.ndarray) -> list[Detection]:
        return [Detection(track_id=7, bbox=(5.0, 5.0, 40.0, 40.0), confidence=0.9)]

    def reset(self) -> None:
        self.resets += 1


@pytest.fixture
def video_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    for _ in range(30):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "vision.db")


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def zone_state(pipeline, name):
    return next(z for z in pipeline.state["zones"] if z["name"] == name)


def test_pipeline_publishes_jpeg_and_state(video_file, storage):
    detector = FakeDetector()
    source = FrameSource(str(video_file))
    pipeline = Pipeline(AppConfig(), source, lambda: detector, storage)
    assert pipeline.state["phase"] == "starting"
    source.start()
    pipeline.start()
    try:
        assert wait_until(lambda: pipeline.state["phase"] == "running")
        state = pipeline.state
        assert state["people"] == 1
        assert state["device"] == "cpu" and state["model"] == "fake"
        assert state["camera"]["connected"] and state["camera"]["width"] == 64

        seq, jpeg = pipeline.hub.latest()
        assert jpeg is not None and jpeg[:2] == b"\xff\xd8"  # JPEG válido
        assert detector.resets >= 1  # se resetea el tracker al abrir la fuente
    finally:
        pipeline.stop()
        source.stop()


def test_person_inside_zone_is_counted_and_zones_reload(video_file, storage):
    storage.create_zone("cam1", "Todo", "work", WHOLE_FRAME)
    source = FrameSource(str(video_file))
    pipeline = Pipeline(AppConfig(), source, FakeDetector, storage)
    source.start()
    pipeline.start()
    try:
        # La persona falsa está siempre en (5,5)-(40,40): adentro de "Todo".
        assert wait_until(lambda: zone_state(pipeline, "Todo")["count"] == 1)
        assert zone_state(pipeline, "Todo")["occupants"][0]["track_id"] == 7

        # Zona nueva en caliente, lejos de la persona: aparece vacía.
        storage.create_zone("cam1", "Borde", "restricted", RIGHT_EDGE)
        pipeline.request_zones_reload()
        assert wait_until(lambda: any(z["name"] == "Borde" for z in pipeline.state["zones"]))
        borde = zone_state(pipeline, "Borde")
        assert borde["count"] == 0 and borde["kind"] == "restricted" and borde["empty_for_s"] is not None
    finally:
        pipeline.stop()
        source.stop()


def test_unexpected_error_does_not_stop_the_pipeline(video_file, storage, monkeypatch):
    source = FrameSource(str(video_file))
    pipeline = Pipeline(AppConfig(), source, FakeDetector, storage)
    source.start()
    pipeline.start()
    try:
        assert wait_until(lambda: pipeline.state["phase"] == "running")

        def broken_list_zones(camera_id):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(storage, "list_zones", broken_list_zones)
        pipeline.request_zones_reload()
        time.sleep(0.3)
        before = pipeline.hub.latest()[0]
        assert wait_until(lambda: pipeline.hub.latest()[0] > before)  # el video sigue saliendo
        assert pipeline._thread.is_alive()
    finally:
        pipeline.stop()
        source.stop()


def test_pipeline_shows_no_signal_placeholder(storage):
    source = FrameSource("/no/existe.mp4")
    pipeline = Pipeline(AppConfig(), source, FakeDetector, storage)
    source.start()
    pipeline.start()
    try:
        assert wait_until(lambda: pipeline.state["phase"] == "no_signal")
        _, jpeg = pipeline.hub.latest()
        assert jpeg is not None  # el navegador ve el cartel "sin señal", no una imagen rota
    finally:
        pipeline.stop()
        source.stop()


def test_pipeline_reports_detector_load_error(video_file, storage):
    def broken_factory():
        raise RuntimeError("no hay pesos")

    source = FrameSource(str(video_file))
    pipeline = Pipeline(AppConfig(), source, broken_factory, storage)
    source.start()
    pipeline.start()
    try:
        assert wait_until(lambda: pipeline.state["phase"] == "error")
        assert "no hay pesos" in pipeline.state["error"]
    finally:
        pipeline.stop()
        source.stop()
