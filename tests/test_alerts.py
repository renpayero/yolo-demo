"""Alertas sobre secuencias simuladas: motor de ocupación real + motor de alertas, sin video."""
import pytest

from app.alerts import AlertEngine
from app.occupancy import OccupancyEngine
from app.zones import Zone

LEFT = ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0))
RIGHT = ((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0))
IN_LEFT = (0.25, 0.5)
IN_LEFT_2 = (0.10, 0.2)
IN_RIGHT = (0.75, 0.5)


def zone(zone_id=1, kind="work", dwell=None, empty=None, poly=LEFT, name="Lijado"):
    return Zone(zone_id, name, kind, poly, max_dwell_s=dwell, max_empty_s=empty)


class Sim:
    def __init__(self, *zones, monitoring_since=0.0):
        self.occupancy = OccupancyEngine(zones, enter_debounce_s=0.5, exit_grace_s=3.0, start_ts=0.0)
        self.alerts = AlertEngine()
        self.monitoring_since = monitoring_since
        self.changes = []

    def run(self, start, end, points):
        for i in range(round((end - start) * 10) + 1):
            self.step(start + i / 10, points)

    def step(self, ts, points, zones=None):
        events = self.occupancy.update(ts, points) if zones is None else self.occupancy.set_zones(zones, ts)
        self.changes += self.alerts.evaluate(ts, self.occupancy.snapshot(), events, self.monitoring_since)

    def actions(self):
        return [(c.action, c.alert.kind, c.alert.track_id) for c in self.changes]


def test_dwell_alert_once_after_threshold_and_resolved_on_exit():
    sim = Sim(zone(dwell=10))
    sim.run(0.0, 15.0, {7: IN_LEFT})
    assert sim.actions() == [("raise", "dwell", 7)]
    alert = sim.changes[0].alert
    assert sim.changes[0].ts == pytest.approx(10.1)
    assert alert.value_s == pytest.approx(10.1)
    assert "Lijado" in alert.message and "0:10" in alert.message
    assert [a.kind for a in sim.alerts.active] == ["dwell"]

    sim.run(15.1, 19.0, {})                      # sale; la visita se cierra al vencer la gracia
    assert sim.actions() == [("raise", "dwell", 7), ("resolve", "dwell", 7)]
    assert sim.alerts.active == []


def test_no_dwell_alert_below_threshold():
    sim = Sim(zone(dwell=10))
    sim.run(0.0, 9.5, {7: IN_LEFT})
    sim.run(9.6, 15.0, {})
    assert sim.changes == []


def test_each_visit_gets_its_own_dwell_alert():
    sim = Sim(zone(dwell=5))
    sim.run(0.0, 6.0, {1: IN_LEFT, 2: IN_LEFT_2})
    assert sorted(sim.actions()) == [("raise", "dwell", 1), ("raise", "dwell", 2)]


def test_empty_alert_then_resolved_then_new_empty_period():
    sim = Sim(zone(empty=5))
    sim.run(0.0, 5.0, {})
    assert sim.changes == []                     # 5 s exactos todavía no superan el límite
    sim.run(5.1, 5.9, {})
    assert sim.actions() == [("raise", "empty", None)]
    assert "Lijado" in sim.changes[0].alert.message

    sim.run(6.0, 8.0, {7: IN_LEFT})              # llega alguien: se resuelve
    assert sim.actions()[-1] == ("resolve", "empty", None)

    sim.run(8.1, 13.0, {})                       # se fue (visto por última vez en t=8)
    assert len(sim.alerts.active) == 0
    sim.run(13.1, 13.5, {})                      # 5 s desde la salida: alerta de un período nuevo
    assert [a for a, k, _ in sim.actions() if k == "empty"] == ["raise", "resolve", "raise"]


def test_empty_time_only_counts_while_monitoring():
    sim = Sim(zone(empty=5), monitoring_since=None)
    sim.run(0.0, 20.0, {})                       # sin video: no se sabe si está vacía
    assert sim.changes == []

    sim.monitoring_since = 100.0                 # vuelve el video en t=100
    sim.run(100.0, 105.0, {})
    assert sim.changes == []                     # el reloj arrancó en 100, no en 0
    sim.run(105.1, 105.5, {})
    assert sim.actions() == [("raise", "empty", None)]


def test_restricted_zone_alerts_on_entry_and_ignores_thresholds():
    sim = Sim(zone(kind="restricted", dwell=1, empty=1, name="Prohibido"))
    sim.run(0.0, 20.0, {})                       # vacía mucho tiempo: en una restringida es lo normal
    assert sim.changes == []

    sim.run(20.0, 25.0, {7: IN_LEFT})
    assert sim.actions() == [("raise", "restricted", 7)]
    assert sim.changes[0].ts == pytest.approx(20.5)  # apenas se confirma la entrada
    assert "Prohibido" in sim.changes[0].alert.message

    sim.run(25.1, 29.0, {})
    assert sim.actions()[-1] == ("resolve", "restricted", 7)


def test_zone_without_thresholds_never_alerts():
    sim = Sim(zone())
    sim.run(0.0, 30.0, {7: IN_LEFT})
    sim.run(30.1, 60.0, {})
    assert sim.changes == []


def test_deleting_a_zone_resolves_its_alerts():
    sim = Sim(zone(empty=2), zone(2, dwell=1, poly=RIGHT, name="Embalaje"))
    sim.run(0.0, 3.0, {7: IN_RIGHT})
    assert sorted(sim.actions()) == [("raise", "dwell", 7), ("raise", "empty", None)]
    sim.step(3.1, {}, zones=[])
    assert sorted(a for a, _, _ in sim.actions()) == ["raise", "raise", "resolve", "resolve"]
    assert sim.alerts.active == []


def test_removing_threshold_resolves_empty_alert():
    sim = Sim(zone(empty=2))
    sim.run(0.0, 3.0, {})
    assert sim.actions() == [("raise", "empty", None)]
    sim.step(3.1, {}, zones=[zone(empty=None)])
    sim.step(3.2, {})
    assert sim.actions()[-1] == ("resolve", "empty", None)
