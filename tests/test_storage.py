import sqlite3

import pytest

from app.storage import MIGRATIONS, Storage, ZoneNameTaken

SQUARE = ((0.1, 0.1), (0.4, 0.1), (0.4, 0.4))


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


def test_zone_crud(storage):
    zone = storage.create_zone("cam1", "Lijado", "work", SQUARE)
    assert zone.id > 0 and zone.polygon == SQUARE

    updated = storage.update_zone(zone.id, "Lijado fino", "restricted", SQUARE[:3])
    assert updated.name == "Lijado fino" and updated.kind == "restricted"
    assert storage.list_zones("cam1") == [updated]

    assert storage.delete_zone(zone.id)
    assert storage.list_zones("cam1") == []
    assert not storage.delete_zone(zone.id)
    assert storage.update_zone(zone.id, "x", "work", SQUARE) is None


def test_zone_names_are_unique_per_camera(storage):
    storage.create_zone("cam1", "Lijado", "work", SQUARE)
    with pytest.raises(ZoneNameTaken):
        storage.create_zone("cam1", "Lijado", "work", SQUARE)
    storage.create_zone("cam2", "Lijado", "work", SQUARE)  # otra cámara: permitido
    other = storage.create_zone("cam1", "Pintura", "work", SQUARE)
    with pytest.raises(ZoneNameTaken):
        storage.update_zone(other.id, "Lijado", "work", SQUARE)


def test_replace_zones_is_atomic(storage):
    storage.create_zone("cam1", "Vieja", "work", SQUARE)
    with pytest.raises(ZoneNameTaken):
        storage.replace_zones("cam1", [("A", "work", SQUARE), ("A", "work", SQUARE)])
    assert [z.name for z in storage.list_zones("cam1")] == ["Vieja"]  # no se perdió nada

    zones = storage.replace_zones("cam1", [("A", "work", SQUARE), ("B", "restricted", SQUARE)])
    assert [(z.name, z.kind) for z in zones] == [("A", "work"), ("B", "restricted")]
