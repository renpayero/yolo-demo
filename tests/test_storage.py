import json
import sqlite3

import numpy as np
import pytest

from app.storage import MIGRATIONS, Storage, ZoneNameTaken
from app.zones import ZoneSpec

SQUARE = ((0.1, 0.1), (0.4, 0.1), (0.4, 0.4))


def spec(name="Lijado", kind="work", dwell=None, empty=None):
    return ZoneSpec(name, kind, SQUARE, dwell, empty)


@pytest.fixture
def storage(tmp_path):
    return Storage(tmp_path / "db" / "vision.db")


def test_schema_version_and_wal(tmp_path):
    path = tmp_path / "vision.db"
    Storage(path)
    Storage(path)  # reabrir no vuelve a aplicar migraciones
    con = sqlite3.connect(path)
    assert con.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)
    assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_existing_v1_database_is_migrated_keeping_zones(tmp_path):
    """La base de la etapa 1 (con zonas ya marcadas) se actualiza sin perder nada."""
    path = tmp_path / "vision.db"
    con = sqlite3.connect(path)
    con.executescript(MIGRATIONS[0])
    con.execute("PRAGMA user_version = 1")
    con.execute(
        "INSERT INTO zones (camera_id, name, kind, polygon, created_at, updated_at) VALUES (?, ?, ?, ?, 0, 0)",
        ("cam1", "Lijado", "work", json.dumps([list(p) for p in SQUARE])),
    )
    con.commit()
    con.close()

    zones = Storage(path).list_zones("cam1")
    assert [(z.name, z.max_dwell_s, z.max_empty_s) for z in zones] == [("Lijado", None, None)]


def test_v2_database_gets_persons_tables(tmp_path):
    path = tmp_path / "vision.db"
    con = sqlite3.connect(path)
    con.executescript(MIGRATIONS[0])
    con.executescript(MIGRATIONS[1])
    con.execute("PRAGMA user_version = 2")
    con.commit()
    con.close()
    storage = Storage(path)
    assert storage.list_persons() == [] and storage.load_embeddings() == {}


def test_embeddings_roundtrip(storage):
    vectors = np.random.default_rng(1).normal(size=(3, 128)).astype(np.float32)
    person = storage.create_person("Ana", vectors)
    loaded = storage.load_embeddings()
    assert list(loaded) == [person.id]
    assert np.allclose(loaded[person.id], vectors)
    assert storage.person_names() == {person.id: "Ana"}


def test_zone_crud(storage):
    zone = storage.create_zone("cam1", spec(dwell=30, empty=60))
    assert zone.id > 0 and zone.polygon == SQUARE
    assert (zone.max_dwell_s, zone.max_empty_s) == (30, 60)

    updated = storage.update_zone(zone.id, spec("Lijado fino", "restricted"))
    assert updated.name == "Lijado fino" and updated.kind == "restricted" and updated.max_dwell_s is None
    assert storage.list_zones("cam1") == [updated]

    assert storage.delete_zone(zone.id)
    assert storage.list_zones("cam1") == []
    assert not storage.delete_zone(zone.id)
    assert storage.update_zone(zone.id, spec("x")) is None


def test_zone_names_are_unique_per_camera(storage):
    storage.create_zone("cam1", spec("Lijado"))
    with pytest.raises(ZoneNameTaken):
        storage.create_zone("cam1", spec("Lijado"))
    storage.create_zone("cam2", spec("Lijado"))  # otra cámara: permitido
    other = storage.create_zone("cam1", spec("Pintura"))
    with pytest.raises(ZoneNameTaken):
        storage.update_zone(other.id, spec("Lijado"))


def test_replace_zones_is_atomic(storage):
    storage.create_zone("cam1", spec("Vieja"))
    with pytest.raises(ZoneNameTaken):
        storage.replace_zones("cam1", [spec("A"), spec("A")])
    assert [z.name for z in storage.list_zones("cam1")] == ["Vieja"]  # no se perdió nada

    zones = storage.replace_zones("cam1", [spec("A", dwell=10), spec("B", "restricted")])
    assert [(z.name, z.kind, z.max_dwell_s) for z in zones] == [("A", "work", 10), ("B", "restricted", None)]
