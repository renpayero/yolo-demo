import sqlite3

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.faces import EnrollmentError
from app.persons_api import router
from app.storage import Storage

ANA = np.tile(np.eye(128, dtype=np.float32)[0], (5, 1))


class FakePipeline:
    def __init__(self):
        self.samples = ANA
        self.error = None
        self.existing = None
        self.reloads = []
        self.identification = False

    def enroll(self):
        if self.error:
            raise EnrollmentError(self.error)
        return self.samples

    def match_existing(self, samples):
        return self.existing, 0.9

    def reload_persons(self, forget_person_id=None):
        self.reloads.append(forget_person_id)

    @property
    def identification_enabled(self):
        return self.identification

    def set_identification(self, enabled):
        self.identification = enabled


@pytest.fixture
def ctx(tmp_path):
    app = FastAPI()
    app.include_router(router)
    app.state.storage = Storage(tmp_path / "vision.db")
    app.state.pipeline = FakePipeline()
    return TestClient(app), app.state.pipeline, app.state.storage


def test_register_requires_consent(ctx):
    client, pipeline, storage = ctx
    r = client.post("/api/persons", json={"name": "Ana", "consent": False})
    assert r.status_code == 422 and "consentimiento" in r.json()["detail"]
    assert storage.list_persons() == []


def test_register_list_and_add_samples(ctx):
    client, pipeline, storage = ctx
    r = client.post("/api/persons", json={"name": " Ana ", "consent": True})
    assert r.status_code == 201
    person = r.json()
    assert person["name"] == "Ana" and person["samples"] == 5 and person["consent_ts"] > 0
    assert pipeline.reloads == [None]

    pipeline.existing = person["id"]        # más muestras de la misma persona: permitido
    r = client.post(f"/api/persons/{person['id']}/samples")
    assert r.status_code == 200 and r.json()["samples"] == 10
    assert [p["name"] for p in client.get("/api/persons").json()] == ["Ana"]


def test_failed_capture_leaves_nothing_behind(ctx):
    client, pipeline, storage = ctx
    pipeline.error = "Hay más de una cara en la imagen"
    r = client.post("/api/persons", json={"name": "Ana", "consent": True})
    assert r.status_code == 422 and "más de una cara" in r.json()["detail"]
    assert storage.list_persons() == []


def test_duplicate_name_and_duplicate_face(ctx):
    client, pipeline, storage = ctx
    client.post("/api/persons", json={"name": "Ana", "consent": True})
    r = client.post("/api/persons", json={"name": "Ana", "consent": True})
    assert r.status_code == 409 and "Ana" in r.json()["detail"]

    pipeline.existing = 1                   # la cara capturada ya es de Ana
    r = client.post("/api/persons", json={"name": "Otra", "consent": True})
    assert r.status_code == 409 and "ya está registrada como «Ana»" in r.json()["detail"]
    assert len(storage.list_persons()) == 1


def test_delete_anonymizes_history(ctx):
    client, pipeline, storage = ctx
    person_id = client.post("/api/persons", json={"name": "Ana", "consent": True}).json()["id"]
    con = sqlite3.connect(storage.path)
    con.execute("INSERT INTO visits (camera_id, zone_id, zone_name, track_id, person_id, start_ts) "
                "VALUES ('cam1', 1, 'Lijado', 7, ?, 0)", (person_id,))
    con.execute("INSERT INTO alerts (camera_id, kind, person_id, ts, message) VALUES ('cam1', 'dwell', ?, 0, 'x')",
                (person_id,))
    con.commit()

    r = client.delete(f"/api/persons/{person_id}")
    assert r.json() == {"visits_anonymized": 1, "alerts_anonymized": 1}
    assert pipeline.reloads[-1] == person_id
    assert storage.list_persons() == [] and storage.load_embeddings() == {}
    assert con.execute("SELECT count(*) FROM visits WHERE person_id IS NULL").fetchone()[0] == 1  # la visita sigue
    assert client.delete(f"/api/persons/{person_id}").status_code == 404


def test_identification_toggle(ctx):
    client, pipeline, _ = ctx
    assert client.get("/api/identification").json() == {"enabled": False}
    assert client.put("/api/identification", json={"enabled": True}).json() == {"enabled": True}
    assert pipeline.identification is True
