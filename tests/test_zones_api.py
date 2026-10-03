import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import AppConfig
from app.storage import Storage
from app.zones_api import router

TRIANGLE = [[0.1, 0.1], [0.5, 0.1], [0.3, 0.5]]


class FakePipeline:
    def __init__(self) -> None:
        self.reloads = 0

    def request_zones_reload(self) -> None:
        self.reloads += 1


@pytest.fixture
def ctx(tmp_path):
    app = FastAPI()
    app.include_router(router)
    app.state.cfg = AppConfig()
    app.state.storage = Storage(tmp_path / "vision.db")
    app.state.pipeline = FakePipeline()
    return TestClient(app), app.state.pipeline


def test_create_list_update_delete(ctx):
    client, pipeline = ctx
    r = client.post("/api/zones", json={"name": "  Lijado ", "polygon": TRIANGLE})
    assert r.status_code == 201
    zone = r.json()
    assert zone["name"] == "Lijado" and zone["kind"] == "work" and zone["color"].startswith("#")

    r = client.put(f"/api/zones/{zone['id']}", json={"name": "Lijado", "kind": "restricted", "polygon": TRIANGLE})
    assert r.status_code == 200 and r.json()["kind"] == "restricted"
    assert [z["name"] for z in client.get("/api/zones").json()] == ["Lijado"]

    assert client.delete(f"/api/zones/{zone['id']}").status_code == 204
    assert client.get("/api/zones").json() == []
    assert pipeline.reloads == 3  # cada cambio avisa al pipeline


@pytest.mark.parametrize(
    "body",
    [
        {"name": "", "polygon": TRIANGLE},
        {"name": "   ", "polygon": TRIANGLE},
        {"name": "A", "polygon": TRIANGLE[:2]},                       # menos de 3 vértices
        {"name": "A", "polygon": [[0.1, 0.1], [1.2, 0.1], [0.3, 0.5]]},  # fuera de 0-1
        {"name": "A", "kind": "otra", "polygon": TRIANGLE},
    ],
)
def test_invalid_zones_are_rejected(ctx, body):
    client, pipeline = ctx
    assert client.post("/api/zones", json=body).status_code == 422
    assert pipeline.reloads == 0


def test_duplicate_name_and_missing_zone(ctx):
    client, _ = ctx
    client.post("/api/zones", json={"name": "Lijado", "polygon": TRIANGLE})
    r = client.post("/api/zones", json={"name": "Lijado", "polygon": TRIANGLE})
    assert r.status_code == 409 and "Lijado" in r.json()["detail"]
    assert client.put("/api/zones/999", json={"name": "X", "polygon": TRIANGLE}).status_code == 404
    assert client.delete("/api/zones/999").status_code == 404


def test_alert_thresholds(ctx):
    client, _ = ctx
    r = client.post("/api/zones", json={"name": "Lijado", "polygon": TRIANGLE, "max_dwell_s": 30, "max_empty_s": 60})
    assert (r.json()["max_dwell_s"], r.json()["max_empty_s"]) == (30, 60)

    # En una restringida cualquier ingreso ya es alerta: los umbrales se descartan.
    r = client.post("/api/zones", json={"name": "Prohibido", "kind": "restricted", "polygon": TRIANGLE,
                                        "max_dwell_s": 30, "max_empty_s": 60})
    assert (r.json()["max_dwell_s"], r.json()["max_empty_s"]) == (None, None)

    for bad in (0, -5, 100000):
        r = client.post("/api/zones", json={"name": f"X{bad}", "polygon": TRIANGLE, "max_dwell_s": bad})
        assert r.status_code == 422


def test_export_then_import_roundtrip(ctx):
    client, _ = ctx
    client.post("/api/zones", json={"name": "Lijado", "polygon": TRIANGLE, "max_dwell_s": 30})
    client.post("/api/zones", json={"name": "Prohibido", "kind": "restricted", "polygon": TRIANGLE})
    exported = client.get("/api/zones/export")
    assert exported.json()["zones"][0]["max_dwell_s"] == 30
    assert "attachment" in exported.headers["content-disposition"]
    data = exported.json()
    assert [z["name"] for z in data["zones"]] == ["Lijado", "Prohibido"]

    client.post("/api/zones", json={"name": "Sobra", "polygon": TRIANGLE})
    imported = client.post("/api/zones/import", json=data)
    assert imported.status_code == 200
    assert [(z["name"], z["kind"]) for z in imported.json()] == [("Lijado", "work"), ("Prohibido", "restricted")]


def test_import_with_duplicate_names_keeps_existing(ctx):
    client, _ = ctx
    client.post("/api/zones", json={"name": "Original", "polygon": TRIANGLE})
    bad = {"version": 1, "zones": [{"name": "A", "polygon": TRIANGLE}, {"name": "A", "polygon": TRIANGLE}]}
    assert client.post("/api/zones/import", json=bad).status_code == 409
    assert [z["name"] for z in client.get("/api/zones").json()] == ["Original"]
