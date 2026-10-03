"""Registro en SQLite: sesiones, visitas y alertas, y recuperación después de un corte."""
import sqlite3

import pytest

from app.alerts import Alert, AlertChange
from app.occupancy import ZoneEvent
from app.recorder import Recorder
from app.storage import Storage

NAMES = {1: "Lijado", 2: "Embalaje"}


@pytest.fixture
def db(tmp_path):
    storage = Storage(tmp_path / "vision.db")
    recorder = Recorder(storage.path, "cam1")
    recorder.start()
    yield storage, recorder
    recorder.stop()


def rows(storage, sql):
    con = sqlite3.connect(storage.path)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql)]
    finally:
        con.close()


def enter(visit_id, zone_id=1, track_id=7, ts=10.0):
    return ZoneEvent("enter", zone_id, track_id, ts, ts, 0.0, visit_id)


def exit_(visit_id, zone_id=1, track_id=7, enter_ts=10.0, last_seen=25.0):
    return ZoneEvent("exit", zone_id, track_id, last_seen, enter_ts, last_seen - enter_ts, visit_id)


def alert(uid, kind="dwell", ts=40.0):
    return Alert(uid, kind, 1, "Lijado", ts, f"alerta {uid}", track_id=7, visit_id=1, value_s=31.0)


def test_session_visits_and_alerts_lifecycle(db):
    storage, recorder = db
    recorder.session_start(5.0)
    recorder.zone_events([enter(1), enter(2, zone_id=2, track_id=8, ts=12.0)], NAMES)
    recorder.session_heartbeat(15.0)
    recorder.zone_events([exit_(1)], NAMES)
    recorder.alert_changes([AlertChange("raise", alert(1), 40.0)])
    recorder.alert_changes([AlertChange("resolve", alert(1), 55.0)])
    recorder.session_end(60.0)
    recorder.flush()

    [session] = rows(storage, "SELECT * FROM sessions")
    assert (session["start_ts"], session["end_ts"], session["last_heartbeat"]) == (5.0, 60.0, 60.0)

    visits = rows(storage, "SELECT * FROM visits ORDER BY id")
    assert [(v["zone_name"], v["track_id"], v["start_ts"]) for v in visits] == [("Lijado", 7, 10.0), ("Embalaje", 8, 12.0)]
    assert (visits[0]["end_ts"], visits[0]["duration_s"], visits[0]["closed_by"]) == (25.0, 15.0, "exit")
    assert visits[1]["end_ts"] is None                     # sigue abierta
    assert all(v["session_id"] == session["id"] for v in visits)

    [row] = rows(storage, "SELECT * FROM alerts")
    assert (row["kind"], row["ts"], row["resolved_ts"], row["value_s"], row["message"]) == ("dwell", 40.0, 55.0, 31.0, "alerta 1")


def test_failed_operation_does_not_lose_the_others(db):
    storage, recorder = db
    recorder.session_start(1.0)
    recorder._queue.put(lambda con: con.execute("INSERT INTO tabla_que_no_existe VALUES (1)"))
    recorder.zone_events([enter(1)], NAMES)
    recorder.flush()
    assert len(rows(storage, "SELECT * FROM sessions")) == 1
    assert len(rows(storage, "SELECT * FROM visits")) == 1


def test_recovery_closes_what_a_crash_left_open(tmp_path):
    storage = Storage(tmp_path / "vision.db")
    recorder = Recorder(storage.path, "cam1")
    recorder.start()
    recorder.session_start(100.0)
    recorder.zone_events([enter(1, ts=110.0)], NAMES)
    recorder.alert_changes([AlertChange("raise", alert(1, ts=150.0), 150.0)])
    recorder.session_heartbeat(170.0)
    recorder.flush()
    recorder.stop()  # "corte": nunca se llamó a session_end ni se cerró la visita

    report = Storage(storage.path).recover_unclosed("cam1")
    assert (report.sessions, report.visits, report.alerts) == (1, 1, 1)

    [session] = rows(storage, "SELECT * FROM sessions")
    assert session["end_ts"] == 170.0                       # último heartbeat
    [visit] = rows(storage, "SELECT * FROM visits")
    assert (visit["end_ts"], visit["duration_s"], visit["closed_by"]) == (170.0, 60.0, "restart")
    [row] = rows(storage, "SELECT * FROM alerts")
    assert row["resolved_ts"] == 170.0

    again = storage.recover_unclosed("cam1")               # idempotente
    assert (again.sessions, again.visits, again.alerts) == (0, 0, 0)
