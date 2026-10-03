"""Secuencias simuladas de detecciones contra el motor de ocupación (sin video ni modelo)."""
import pytest

from app.occupancy import OccupancyEngine
from app.zones import Zone

LEFT = Zone(id=1, name="Lijado", kind="work", polygon=((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)))
RIGHT = Zone(id=2, name="Embalaje", kind="work", polygon=((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)))
CENTER = Zone(id=3, name="Mesa", kind="work", polygon=((0.3, 0.0), (0.7, 0.0), (0.7, 1.0), (0.3, 1.0)))

IN_LEFT = (0.25, 0.5)
IN_RIGHT = (0.75, 0.5)
IN_LEFT_AND_CENTER = (0.4, 0.5)

DEBOUNCE = 0.5
GRACE = 3.0
FPS = 10


def engine(*zones, start=0.0):
    return OccupancyEngine(zones or (LEFT,), enter_debounce_s=DEBOUNCE, exit_grace_s=GRACE, start_ts=start)


def run(eng, start, end, points):
    """Simula frames a FPS entre start y end (inclusive) con los mismos puntos; devuelve los eventos."""
    events = []
    steps = round((end - start) * FPS)
    for i in range(steps + 1):
        events += eng.update(start + i / FPS, points)
    return events


def kinds(events):
    return [(e.kind, e.zone_id, e.track_id) for e in events]


def occupants(eng, zone_id):
    status = next(s for s in eng.snapshot() if s.zone.id == zone_id)
    return {o.track_id: o.dwell_s for o in status.occupants}


def test_enter_needs_debounce_and_starts_when_first_seen():
    eng = engine()
    assert run(eng, 0.0, 0.4, {7: IN_LEFT}) == []
    events = run(eng, 0.5, 0.5, {7: IN_LEFT})
    assert kinds(events) == [("enter", 1, 7)]
    assert events[0].ts == 0.0  # la visita arranca cuando entró, no cuando se confirmó


def test_passing_through_faster_than_debounce_is_not_a_visit():
    eng = engine()
    events = run(eng, 0.0, 0.3, {7: IN_LEFT}) + run(eng, 0.4, 5.0, {})
    assert events == []


def test_exit_after_grace_with_duration_until_last_seen():
    eng = engine()
    run(eng, 0.0, 5.0, {7: IN_LEFT})
    assert run(eng, 5.1, 8.0, {}) == []          # todavía dentro del período de gracia
    events = run(eng, 8.1, 8.1, {})
    assert kinds(events) == [("exit", 1, 7)]
    assert events[0].ts == pytest.approx(5.0)
    assert events[0].duration_s == pytest.approx(5.0)


def test_detector_blink_within_grace_keeps_single_visit():
    eng = engine()
    events = run(eng, 0.0, 5.0, {7: IN_LEFT})
    events += run(eng, 5.1, 6.0, {})              # 1 s sin verlo (parpadeo / oclusión)
    events += run(eng, 6.1, 10.0, {7: IN_LEFT})
    events += run(eng, 10.1, 14.0, {})
    assert kinds(events) == [("enter", 1, 7), ("exit", 1, 7)]
    assert events[-1].duration_s == pytest.approx(10.0)


def test_person_seen_outside_counts_as_absent():
    eng = engine()
    run(eng, 0.0, 2.0, {7: IN_LEFT})
    events = run(eng, 2.1, 6.0, {7: IN_RIGHT})    # sigue en cuadro, pero fuera de la zona
    assert kinds(events) == [("exit", 1, 7)]
    assert events[0].duration_s == pytest.approx(2.0)


def test_several_people_in_same_zone():
    eng = engine()
    run(eng, 0.0, 1.0, {1: IN_LEFT, 2: IN_LEFT})
    assert set(occupants(eng, 1)) == {1, 2}
    events = run(eng, 1.1, 5.0, {2: IN_LEFT})
    assert kinds(events) == [("exit", 1, 1)]
    assert set(occupants(eng, 1)) == {2}


def test_moving_from_one_zone_to_another():
    eng = engine(LEFT, RIGHT)
    events = run(eng, 0.0, 3.0, {7: IN_LEFT}) + run(eng, 3.1, 8.0, {7: IN_RIGHT})
    assert kinds(events) == [("enter", 1, 7), ("enter", 2, 7), ("exit", 1, 7)]
    assert set(occupants(eng, 2)) == {7}


def test_overlapping_zones_count_independently():
    eng = engine(LEFT, CENTER)
    events = run(eng, 0.0, 1.0, {7: IN_LEFT_AND_CENTER})
    assert sorted(kinds(events)) == [("enter", 1, 7), ("enter", 3, 7)]


def test_empty_zone_tracks_since_when():
    eng = engine(start=100.0)
    status = eng.snapshot()[0]
    assert status.occupants == () and status.empty_since == 100.0

    run(eng, 100.0, 104.0, {7: IN_LEFT})
    assert eng.snapshot()[0].empty_since is None

    run(eng, 104.1, 108.0, {})
    assert eng.snapshot()[0].empty_since == pytest.approx(104.0)  # desde la última vez que se lo vio


def test_zone_empty_since_last_person_left():
    eng = engine()
    run(eng, 0.0, 2.0, {1: IN_LEFT, 2: IN_LEFT})
    run(eng, 2.1, 6.0, {2: IN_LEFT})             # 1 sale en t=2, 2 sigue hasta t=6
    run(eng, 6.1, 10.0, {})
    assert eng.snapshot()[0].empty_since == pytest.approx(6.0)


def test_dwell_time_is_reported_while_inside():
    eng = engine()
    run(eng, 0.0, 12.0, {7: IN_LEFT})
    assert occupants(eng, 1)[7] == pytest.approx(12.0)
    assert present(eng, 1) == {7}
    assert eng.track_zones()[7] == [(LEFT, pytest.approx(12.0))]


def test_removing_a_zone_closes_its_visits():
    eng = engine(LEFT, RIGHT)
    run(eng, 0.0, 2.0, {7: IN_LEFT})
    events = eng.set_zones([RIGHT], ts=2.5)
    assert kinds(events) == [("exit", 1, 7)]
    assert [s.zone.id for s in eng.snapshot()] == [2]


def test_editing_a_zone_keeps_visits_and_uses_new_shape():
    eng = engine()
    run(eng, 0.0, 2.0, {7: IN_LEFT})
    smaller = Zone(id=1, name="Lijado", kind="work", polygon=((0.0, 0.0), (0.1, 0.0), (0.1, 1.0), (0.0, 1.0)))
    assert eng.set_zones([smaller], ts=2.1) == []
    events = run(eng, 2.1, 6.0, {7: IN_LEFT})     # el punto ya no cae en la zona achicada
    assert kinds(events) == [("exit", 1, 7)]


def test_adding_a_zone_at_runtime():
    eng = engine()
    eng.set_zones([LEFT, RIGHT], ts=1.0)
    events = run(eng, 1.0, 2.0, {7: IN_RIGHT})
    assert kinds(events) == [("enter", 2, 7)]


def test_close_all_ends_every_visit_immediately():
    eng = engine(LEFT, RIGHT)
    run(eng, 0.0, 2.0, {1: IN_LEFT, 2: IN_RIGHT})
    events = eng.close_all()
    assert sorted(kinds(events)) == [("exit", 1, 1), ("exit", 2, 2)]
    assert all(e.duration_s == pytest.approx(2.0) for e in events)
    assert all(s.occupants == () for s in eng.snapshot())


def present(eng, zone_id):
    status = next(s for s in eng.snapshot() if s.zone.id == zone_id)
    return {o.track_id for o in status.occupants if o.present}


def test_lost_track_stops_counting_before_grace_ends():
    """Un track perdido sigue con la visita abierta (gracia), pero no se cuenta como presente."""
    eng = engine()
    run(eng, 0.0, 2.0, {1: IN_LEFT, 2: IN_LEFT})
    run(eng, 2.1, 3.5, {2: IN_LEFT})             # 1 dejó de verse hace 1.5 s (< gracia de 3 s)
    assert set(occupants(eng, 1)) == {1, 2}      # visita de 1 todavía abierta
    assert present(eng, 1) == {2}                # pero en el conteo en vivo solo está 2


def test_id_switch_at_same_spot_continues_the_visit():
    """El tracker pierde a la persona y la vuelve a encontrar con otro ID en el mismo lugar."""
    eng = engine()
    events = run(eng, 0.0, 3.0, {1: IN_LEFT})
    events += run(eng, 3.1, 3.4, {})             # oclusión breve
    events += run(eng, 3.5, 8.0, {2: (0.26, 0.5)})  # ID nuevo, casi en el mismo punto
    events += run(eng, 8.1, 12.0, {})
    assert [(e.kind, e.track_id) for e in events] == [("enter", 1), ("rebind", 2), ("exit", 2)]
    assert events[1].previous_track_id == 1
    assert events[-1].duration_s == pytest.approx(8.0)  # una sola visita de 0 a 8 s
    assert len({e.visit_id for e in events}) == 1


def test_no_rebind_if_previous_track_is_still_visible():
    eng = engine()
    run(eng, 0.0, 3.0, {1: IN_LEFT})
    # 1 sigue en cuadro (fuera de la zona) y aparece otra persona donde estaba 1: son dos personas.
    events = run(eng, 3.1, 4.0, {1: IN_RIGHT, 2: IN_LEFT})
    assert ("enter", 1, 2) in kinds(events)
    assert all(e.kind != "rebind" for e in events)


def test_no_rebind_if_new_track_is_far_away():
    eng = engine()
    run(eng, 0.0, 3.0, {1: (0.1, 0.1)})
    events = run(eng, 3.1, 4.0, {2: (0.4, 0.9)})
    assert ("enter", 1, 2) in kinds(events)
    assert all(e.kind != "rebind" for e in events)


def test_split_track_closes_the_old_visit_and_opens_a_new_one():
    """Otra persona heredó el track (se detectó por la cara): cada una se queda con su parte de la visita."""
    eng = engine(LEFT, RIGHT)
    run(eng, 0.0, 10.0, {7: IN_LEFT})
    [old] = [o for o in eng.snapshot()[0].occupants]
    events = eng.split_track(7, end_ts=6.0, start_ts=7.0)
    assert [(e.kind, e.zone_id, e.track_id) for e in events] == [("exit", 1, 7), ("enter", 1, 7)]
    assert events[0].ts == 6.0 and events[0].duration_s == pytest.approx(6.0)
    assert events[1].enter_ts == 7.0 and events[1].visit_id != old.visit_id
    status = eng.snapshot()[0]
    assert status.empty_since is None and [o.dwell_s for o in status.occupants] == [pytest.approx(3.0)]
    assert eng.split_track(99, 1.0, 2.0) == []  # track sin visitas: nada


def test_no_frames_still_closes_after_grace():
    """Si la cámara se cae, el pipeline sigue llamando a update() sin detecciones."""
    eng = engine()
    run(eng, 0.0, 2.0, {7: IN_LEFT})
    assert kinds(eng.update(10.0, {})) == [("exit", 1, 7)]
