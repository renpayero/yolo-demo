"""Pipeline completo con un detector falso (sin YOLO): fuente → detección → JPEG + estado."""
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.capture import FrameSource
from app.config import AppConfig, IdentificationConfig, PostureConfig
from app.domain import Detection
from app.faces import Face, Recognizer
from app.pipeline import Pipeline
from app.recorder import Recorder
from app.storage import Storage
from app.zones import ZoneSpec

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
    storage.create_zone("cam1", ZoneSpec("Todo", "work", WHOLE_FRAME))
    source = FrameSource(str(video_file))
    pipeline = Pipeline(AppConfig(), source, FakeDetector, storage)
    source.start()
    pipeline.start()
    try:
        # La persona falsa está siempre en (5,5)-(40,40): adentro de "Todo".
        assert wait_until(lambda: zone_state(pipeline, "Todo")["count"] == 1)
        assert zone_state(pipeline, "Todo")["occupants"][0]["track_id"] == 7

        # Zona nueva en caliente, lejos de la persona: aparece vacía.
        storage.create_zone("cam1", ZoneSpec("Borde", "restricted", RIGHT_EDGE))
        pipeline.request_zones_reload()
        assert wait_until(lambda: any(z["name"] == "Borde" for z in pipeline.state["zones"]))
        borde = zone_state(pipeline, "Borde")
        assert borde["count"] == 0 and borde["kind"] == "restricted" and borde["empty_for_s"] is not None
    finally:
        pipeline.stop()
        source.stop()


def test_alerts_and_history_are_recorded_and_closed_on_shutdown(video_file, storage):
    storage.create_zone("cam1", ZoneSpec("Todo", "work", WHOLE_FRAME, max_dwell_s=1))
    recorder = Recorder(storage.path, "cam1")
    source = FrameSource(str(video_file))
    pipeline = Pipeline(AppConfig(), source, FakeDetector, storage, recorder)
    recorder.start()
    source.start()
    pipeline.start()
    try:
        assert wait_until(lambda: pipeline.state["alerts"]["active"])
        [alert] = pipeline.state["alerts"]["active"]
        assert alert["kind"] == "dwell" and alert["track_id"] == 7 and "Todo" in alert["message"]
    finally:
        pipeline.stop()
        recorder.stop()
        source.stop()

    con = sqlite3.connect(storage.path)
    sessions = con.execute("SELECT start_ts, end_ts FROM sessions").fetchall()
    visits = con.execute("SELECT zone_name, track_id, end_ts, closed_by FROM visits").fetchall()
    alerts = con.execute("SELECT kind, resolved_ts FROM alerts").fetchall()
    con.close()
    assert len(sessions) == 1 and sessions[0][1] is not None          # sesión cerrada al apagar
    assert [(v[0], v[1], v[3]) for v in visits] == [("Todo", 7, "exit")] and visits[0][2] is not None
    assert alerts[0][0] == "dwell" and alerts[0][1] is not None        # alerta resuelta al apagar


ANA = np.eye(128, dtype=np.float32)[0]


class AlwaysAna:
    """Motor facial falso: siempre ve una cara grande, que es la de Ana."""

    def detect(self, image):
        return [Face((0, 0, 100, 100), 0.99, None)]

    def embed(self, image, face):
        return ANA


def test_identification_names_the_track_its_visit_and_can_be_turned_off(video_file, storage):
    storage.create_zone("cam1", ZoneSpec("Todo", "work", WHOLE_FRAME))
    ana = storage.create_person("Ana", ANA[None])
    cfg = AppConfig(identification=IdentificationConfig(
        enabled=True, min_face_px=5, check_interval_s=0.0, checks_per_frame=1, votes_needed=2,
    ))
    recorder = Recorder(storage.path, "cam1")
    source = FrameSource(str(video_file))
    pipeline = Pipeline(cfg, source, FakeDetector, storage, recorder, Recognizer(cfg.identification, AlwaysAna))
    recorder.start()
    source.start()
    pipeline.start()
    try:
        def occupant_names():
            return [(o["track_id"], o["name"]) for o in zone_state(pipeline, "Todo")["occupants"]]

        assert wait_until(lambda: occupant_names() == [(7, "Ana")])
        assert pipeline.state["identification"] == {"enabled": True, "persons": 1, "identified_now": 1}

        samples = pipeline.enroll()          # captura de registro con la misma cámara
        assert samples.shape == (5, 128)

        pipeline.set_identification(False)   # apagada: todos anónimos en el acto
        assert wait_until(lambda: occupant_names() == [(7, None)])
    finally:
        pipeline.stop()
        recorder.stop()
        source.stop()

    con = sqlite3.connect(storage.path)
    assert con.execute("SELECT person_id FROM visits").fetchall() == [(ana.id,)]
    con.close()


BETO = np.eye(128, dtype=np.float32)[1]


class SwitchableFace:
    """Motor facial falso cuya cara se puede cambiar a mitad de la prueba (otra persona hereda el track)."""

    face = ANA

    def detect(self, image):
        return [Face((0, 0, 100, 100), 0.99, None)]

    def embed(self, image, face):
        return SwitchableFace.face


def test_track_handover_splits_the_visit_between_people(video_file, storage):
    storage.create_zone("cam1", ZoneSpec("Todo", "work", WHOLE_FRAME))
    ana = storage.create_person("Ana", ANA[None])
    beto = storage.create_person("Beto", BETO[None])
    cfg = AppConfig(identification=IdentificationConfig(
        enabled=True, min_face_px=5, check_interval_s=0.0, checks_per_frame=1, votes_needed=2,
        recheck_interval_s=0.3,
    ))
    SwitchableFace.face = ANA
    recorder = Recorder(storage.path, "cam1")
    source = FrameSource(str(video_file))
    pipeline = Pipeline(cfg, source, FakeDetector, storage, recorder, Recognizer(cfg.identification, SwitchableFace))
    recorder.start()
    source.start()
    pipeline.start()

    def names():
        return [o["name"] for o in zone_state(pipeline, "Todo")["occupants"]]

    try:
        assert wait_until(lambda: names() == ["Ana"])
        SwitchableFace.face = BETO           # mismo track 7, otra cara
        assert wait_until(lambda: names() == ["Beto"])
    finally:
        pipeline.stop()
        recorder.stop()
        source.stop()

    con = sqlite3.connect(storage.path)
    visits = con.execute("SELECT person_id, start_ts, end_ts FROM visits ORDER BY id").fetchall()
    con.close()
    assert [v[0] for v in visits] == [ana.id, beto.id]
    assert visits[0][2] <= visits[1][1]       # la de Ana termina antes de que empiece la de Beto


def arms_up_pose():
    """Pose falsa sobre la caja de FakeDetector: de pie, con la muñeca izquierda sobre la cabeza."""
    points = [(0.0, 0.0, 0.0)] * 17
    points[5], points[6] = (15.0, 15.0, 0.9), (30.0, 15.0, 0.9)     # hombros
    points[11], points[12] = (17.0, 30.0, 0.9), (28.0, 30.0, 0.9)   # caderas
    points[9], points[10] = (14.0, 5.0, 0.9), (31.0, 25.0, 0.9)     # muñecas: izquierda arriba
    return [((5.0, 5.0, 40.0, 40.0), tuple(points))]


class FakePose:
    def __init__(self, device):
        self.device = device

    def estimate(self, image):
        return arms_up_pose()


def test_posture_alert_and_heatmap_positions(video_file, storage):
    cfg = AppConfig(posture=PostureConfig(arms_raised_s=0.5))
    recorder = Recorder(storage.path, "cam1")
    source = FrameSource(str(video_file))
    pipeline = Pipeline(cfg, source, FakeDetector, storage, recorder, pose_factory=FakePose)
    recorder.start()
    source.start()
    pipeline.start()
    try:
        assert wait_until(lambda: any(a["kind"] == "arms_raised" for a in pipeline.state["alerts"]["active"]))
        assert pipeline.state["posture"] == {"enabled": True}
        time.sleep(1.2)  # al menos dos muestras de posición
    finally:
        pipeline.stop()
        recorder.stop()
        source.stop()

    con = sqlite3.connect(storage.path)
    kinds = con.execute("SELECT kind, resolved_ts IS NOT NULL FROM alerts").fetchall()
    positions = con.execute("SELECT track_id, x, y FROM positions").fetchall()
    con.close()
    assert kinds == [("arms_raised", 1)]               # registrada y resuelta al apagar
    assert len(positions) >= 2 and all(p[0] == 7 for p in positions)
    grid = storage.heatmap("cam1", 0, time.time() + 1, 64, 48)
    assert sum(map(sum, grid)) == len(positions)


def test_pose_model_failure_keeps_the_demo_running(video_file, storage):
    def broken_pose(device):
        raise RuntimeError("sin pesos de pose")

    source = FrameSource(str(video_file))
    pipeline = Pipeline(AppConfig(), source, FakeDetector, storage, pose_factory=broken_pose)
    source.start()
    pipeline.start()
    try:
        assert wait_until(lambda: pipeline.state["phase"] == "running")
        assert pipeline.state["posture"] == {"enabled": False}
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
