import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.alerts import Alert, AlertChange
from app.config import AppConfig
from app.dashboard_api import router
from app.occupancy import ZoneEvent
from app.recorder import Recorder
from app.storage import Storage
from app.zones import ZoneSpec

SQUARE = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))


@pytest.fixture
def client(tmp_path):
    storage = Storage(tmp_path / "vision.db")
    zone = storage.create_zone("cam1", ZoneSpec("Lijado", "work", SQUARE, max_dwell_s=30))

    t0 = time.time() - 600  # hace 10 minutos
    recorder = Recorder(storage.path, "cam1")
    recorder.start()
    recorder.session_start(t0)
    recorder.zone_events([ZoneEvent("enter", zone.id, 7, t0 + 60, t0 + 60, 0.0, 1)], {zone.id: "Lijado"})
    recorder.zone_events([ZoneEvent("exit", zone.id, 7, t0 + 120, t0 + 60, 60.0, 1)], {zone.id: "Lijado"})
    alert = Alert(1, "dwell", zone.id, "Lijado", t0 + 90, "Permanencia prolongada", track_id=7, visit_id=1)
    recorder.alert_changes([AlertChange("raise", alert, t0 + 90), AlertChange("resolve", alert, t0 + 120)])
    recorder.stop()

    app = FastAPI()
    app.include_router(router)
    app.state.cfg = AppConfig()
    app.state.storage = storage
    return TestClient(app), t0


def test_default_range_is_last_hour_with_data(client):
    http, t0 = client
    data = http.get("/api/dashboard").json()
    assert data["range"]["to"] - data["range"]["from"] == pytest.approx(3600, abs=1)
    [lijado] = data["zones"]
    assert lijado["name"] == "Lijado" and lijado["visits"] == 1 and lijado["occupied_s"] == pytest.approx(60)
    assert data["summary"]["monitored_s"] == pytest.approx(600, abs=2)  # sesión abierta: hasta ahora
    assert [a["message"] for a in data["alerts"]] == ["Permanencia prolongada"]
    assert data["timeline"][0]["intervals"] == [[pytest.approx(t0 + 60), pytest.approx(t0 + 120)]]


def test_explicit_range_before_the_data_is_empty(client):
    http, t0 = client
    data = http.get("/api/dashboard", params={"from": t0 - 7200, "to": t0 - 3600}).json()
    assert data["summary"]["monitored_s"] == 0 and data["zones"][0]["visits"] == 0
    assert data["unmonitored"] == [[t0 - 7200, t0 - 3600]]


@pytest.mark.parametrize(
    "params",
    [{"from": 200, "to": 100}, {"from": 0, "to": 32 * 24 * 3600}],
)
def test_invalid_ranges(client, params):
    http, _ = client
    assert http.get("/api/dashboard", params=params).status_code == 422
