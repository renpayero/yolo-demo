"""Posturas a partir de puntos del cuerpo sintéticos (formato COCO), sin modelo ni cámara."""
import itertools

import pytest

from app.config import PostureConfig
from app.domain import Detection
from app.posture import PostureMonitor, PostureReading, attach_keypoints, read_posture

# Índices COCO usados: 5/6 hombros, 7/8 codos, 9/10 muñecas, 11/12 caderas, 15/16 tobillos.


def skeleton(shoulders, hips, wrists=None, conf=0.9):
    """17 puntos; los no indicados quedan con confianza 0 (no visibles)."""
    points = [(0.0, 0.0, 0.0)] * 17
    (ls, rs), (lh, rh) = shoulders, hips
    points[5], points[6] = (*ls, conf), (*rs, conf)
    points[11], points[12] = (*lh, conf), (*rh, conf)
    if wrists:
        lw, rw = wrists
        if lw:
            points[9] = (*lw, conf)
        if rw:
            points[10] = (*rw, conf)
    return tuple(points)


# Persona de pie: hombros en y=100, caderas en y=200 (torso de 100 px, vertical).
STANDING = dict(shoulders=((90, 100), (130, 100)), hips=((95, 200), (125, 200)))
STANDING_BOX = (60, 50, 160, 350)


def test_standing_upright_with_arms_down():
    r = read_posture(skeleton(**STANDING, wrists=((85, 210), (135, 210))), STANDING_BOX)
    assert r.trunk_angle == pytest.approx(0, abs=1)
    assert r.arms_raised is False and r.lying is False


def test_one_arm_above_the_shoulder_counts():
    r = read_posture(skeleton(**STANDING, wrists=((85, 60), (135, 210))), STANDING_BOX)
    assert r.arms_raised is True


def test_wrist_just_at_shoulder_height_is_not_raised():
    r = read_posture(skeleton(**STANDING, wrists=((85, 95), (135, 98))), STANDING_BOX)
    assert r.arms_raised is False


def test_bent_forward_trunk_angle():
    # Hombros adelantados 100 px y 50 px más arriba que la cadera: atan(100/50) = 63,4° con la vertical.
    r = read_posture(skeleton(shoulders=((195, 150), (205, 150)), hips=((95, 200), (105, 200))), (60, 120, 230, 350))
    assert r.trunk_angle == pytest.approx(63.4, abs=1)
    assert r.lying is False                      # caja más alta que ancha: está inclinado, no en el piso


def test_lying_on_the_floor():
    r = read_posture(skeleton(shoulders=((300, 400), (300, 410)), hips=((200, 405), (200, 415))), (100, 370, 380, 440))
    assert r.trunk_angle > 80 and r.lying is True
    assert r.arms_raised is None                 # acostado, "arriba del hombro" no significa nada


def body(points, conf=0.9):
    """Puntos sueltos: {índice COCO: (x, y)}."""
    out = [(0.0, 0.0, 0.0)] * 17
    for i, (x, y) in points.items():
        out[i] = (x, y, conf)
    return tuple(out)


def test_lying_at_the_frame_edge_without_hips_uses_the_whole_body():
    """Tirada contra el borde inferior: se ven cabeza, hombros y brazos, pero no las caderas."""
    kp = body({0: (40, 210), 1: (38, 205), 2: (42, 205), 5: (70, 200), 6: (72, 215),
               7: (100, 198), 8: (104, 218), 9: (130, 200), 10: (135, 220)})
    r = read_posture(kp, (30, 188, 150, 240))
    assert r.trunk_angle is None and r.body_angle > 60 and r.lying is True


def test_foreshortened_torso_lying_diagonal():
    """Torso acortado por la perspectiva (~50°) pero el cuerpo entero, de la cabeza a los pies, está horizontal."""
    kp = body({0: (60, 170), 5: (75, 172), 6: (85, 185), 11: (107, 199), 12: (113, 209),
               13: (160, 205), 14: (165, 212), 15: (205, 215), 16: (210, 222)})
    r = read_posture(kp, (50, 160, 220, 235))
    assert r.trunk_angle < 60 and r.body_angle > 60 and r.lying is True


def test_sitting_on_the_floor_with_legs_forward_is_not_lying():
    """Sentado en el piso: torso vertical y piernas horizontales (forma de L): no es estar tirado."""
    kp = body({0: (50, 60), 5: (40, 90), 6: (60, 90), 11: (42, 150), 12: (58, 150),
               13: (100, 155), 14: (100, 160), 15: (150, 158), 16: (150, 163)})
    r = read_posture(kp, (30, 50, 160, 170))
    assert r.lying is False


def test_missing_keypoints_give_unknown():
    r = read_posture(tuple([(0.0, 0.0, 0.0)] * 17), STANDING_BOX)
    assert r == PostureReading(arms_raised=None, trunk_angle=None, lying=None)
    assert read_posture(None, STANDING_BOX) == r


# ---------- Monitor: condiciones sostenidas → alertas ----------

CFG = PostureConfig(arms_raised_s=5, trunk_flexion_deg=45, trunk_flexion_s=5, fall_s=2)
ARMS_UP = PostureReading(arms_raised=True, trunk_angle=5, lying=False)
NORMAL = PostureReading(arms_raised=False, trunk_angle=5, lying=False)
BENT = PostureReading(arms_raised=False, trunk_angle=60, lying=False)
LYING = PostureReading(arms_raised=None, trunk_angle=85, lying=True)


def run(monitor, start, end, readings, **kwargs):
    changes = []
    for i in range(round((end - start) * 10) + 1):
        changes += monitor.update(start + i / 10, readings, **kwargs)
    return changes


def actions(changes):
    return [(c.action, c.alert.kind, c.alert.track_id) for c in changes]


def test_arms_raised_alert_after_hold_and_resolved_when_lowered():
    m = PostureMonitor(CFG)
    assert run(m, 0.0, 4.9, {7: ARMS_UP}) == []
    changes = run(m, 5.0, 6.0, {7: ARMS_UP}, zones={7: (1, "Lijado")}, names={7: "Ana"})
    assert actions(changes) == [("raise", "arms_raised", 7)]
    alert = changes[0].alert
    assert alert.zone_id == 1 and "Ana" in alert.message and "Lijado" in alert.message
    assert m.ongoing(6.0) == {7: [("arms_raised", pytest.approx(6.0))]}

    changes = run(m, 6.1, 8.0, {7: NORMAL})
    assert actions(changes) == [("resolve", "arms_raised", 7)]   # tras ~1 s con los brazos abajo
    assert m.active == []


def test_short_flickers_do_not_break_the_streak():
    """Un frame sin pose (None) o un parpadeo no reinicia la cuenta, si dura menos de 1 s."""
    m = PostureMonitor(CFG)
    run(m, 0.0, 2.0, {7: ARMS_UP})
    run(m, 2.1, 2.5, {})                       # el modelo de pose no lo vio un momento
    assert actions(run(m, 2.6, 5.5, {7: ARMS_UP})) == [("raise", "arms_raised", 7)]


def test_trunk_flexion_and_fall_are_distinct():
    m = PostureMonitor(CFG)
    changes = run(m, 0.0, 6.0, {7: BENT, 8: LYING})
    assert sorted(actions(changes)) == [("raise", "fall", 8), ("raise", "trunk_flexion", 7)]
    fall = next(c for c in changes if c.alert.kind == "fall")
    assert fall.ts == pytest.approx(2.0, abs=0.11)   # la caída se avisa rápido (2 s)


def test_track_that_disappears_resolves_its_alerts():
    m = PostureMonitor(CFG)
    run(m, 0.0, 3.0, {8: LYING})
    assert actions(run(m, 3.1, 5.0, {})) == [("resolve", "fall", 8)]


def test_keypoints_are_attached_to_the_overlapping_person():
    a = Detection(1, (0, 0, 100, 200), 0.9)
    b = Detection(2, (300, 0, 400, 200), 0.9)
    far = Detection(3, (600, 0, 650, 100), 0.9)
    kp_a, kp_b = ((1.0, 1.0, 0.9),) * 17, ((2.0, 2.0, 0.9),) * 17
    poses = [((305, 5, 398, 205), kp_b), ((2, 0, 101, 190), kp_a), ((900, 0, 950, 50), kp_a)]
    result = attach_keypoints([a, b, far], poses)
    assert [d.keypoints for d in result] == [kp_a, kp_b, None]   # la persona sin pose queda sin puntos


def test_resolve_all_and_shared_ids():
    ids = itertools.count(100)
    m = PostureMonitor(CFG, next_uid=lambda: next(ids))
    changes = run(m, 0.0, 3.0, {8: LYING})
    assert changes[0].alert.uid == 100            # comparte la numeración con las alertas de zonas
    assert actions(m.resolve_all(3.5)) == [("resolve", "fall", 8)]
