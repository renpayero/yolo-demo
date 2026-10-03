"""Ergonomía y caídas a partir de los 17 puntos del cuerpo (formato COCO). Lógica pura.

- Brazos elevados: una muñeca por encima de su hombro (trabajo con las manos sobre la cabeza, riesgo de hombro).
- Tronco flexionado: el torso (cadera → hombros) inclinado respecto de la vertical (riesgo lumbar).
- Persona en el piso: torso casi horizontal y la caja de la persona más ancha que alta (posible caída).

Las tres se miden en la imagen: dependen del ángulo de la cámara. Una condición solo genera alerta si se
sostiene un tiempo, así un gesto o un frame mal leído no disparan nada.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from app.alerts import Alert, AlertChange
from app.config import PostureConfig
from app.domain import BBox, Detection, Keypoint
from app.formatting import format_duration

MIN_CONFIDENCE = 0.5
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP = 5, 6, 9, 10, 11, 12
RAISED_MARGIN = 0.15        # la muñeca tiene que estar sobre el hombro al menos esto × largo del torso
LYING_MIN_ANGLE = 60.0      # torso casi horizontal...
LYING_ASPECT = 1.2          # ...y caja al menos 1,2 veces más ancha que alta
RELEASE_S = 1.0             # una condición termina tras este tiempo sin verla (tapa frames sin pose)

KIND_LABELS = {
    "arms_raised": "Brazos sobre los hombros",
    "trunk_flexion": "Tronco flexionado",
    "fall": "Posible caída",
}


@dataclass(frozen=True)
class PostureReading:
    arms_raised: bool | None    # None: no se ven hombros y muñecas (o la persona está acostada)
    trunk_angle: float | None   # grados respecto de la vertical; None: no se ven hombros y caderas
    lying: bool | None          # None: no se puede saber
    body_angle: float | None = None  # eje del cuerpo entero (todos los puntos visibles) respecto de la vertical


UNKNOWN = PostureReading(None, None, None)
MIN_BODY_POINTS = 5         # puntos visibles para estimar la orientación del cuerpo entero
MIN_ELONGATION = 2.0        # los puntos tienen que estar alineados (no una nube compacta) para que el eje signifique algo


def _body_angle(points: Sequence[Keypoint]) -> float | None:
    """Orientación del cuerpo entero: eje principal de todos los puntos visibles, en grados desde la vertical.

    Sirve cuando no se ve el par hombros-caderas (persona tirada contra el borde del cuadro) o cuando la
    perspectiva acorta el torso: de pie el eje es vertical, tirado es horizontal, sentado o agachado queda en medio.
    """
    visible = [(x, y) for x, y, c in points if c >= MIN_CONFIDENCE]
    if len(visible) < MIN_BODY_POINTS:
        return None
    mx, my = _mean(visible)
    sxx = sum((x - mx) ** 2 for x, _ in visible) / len(visible)
    syy = sum((y - my) ** 2 for _, y in visible) / len(visible)
    sxy = sum((x - mx) * (y - my) for x, y in visible) / len(visible)
    spread = math.hypot((sxx - syy) / 2, sxy)
    major, minor = (sxx + syy) / 2 + spread, (sxx + syy) / 2 - spread
    if minor > 0 and major / minor < MIN_ELONGATION:
        return None
    from_horizontal = abs(math.degrees(0.5 * math.atan2(2 * sxy, sxx - syy)))
    return 90.0 - from_horizontal


def _visible(points: Sequence[Keypoint], *indices: int) -> list[tuple[float, float]]:
    return [(points[i][0], points[i][1]) for i in indices if points[i][2] >= MIN_CONFIDENCE]


def _mean(points: list[tuple[float, float]]) -> tuple[float, float]:
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def read_posture(keypoints: Sequence[Keypoint] | None, bbox: BBox) -> PostureReading:
    if not keypoints:
        return UNKNOWN
    shoulders = _visible(keypoints, L_SHOULDER, R_SHOULDER)
    hips = _visible(keypoints, L_HIP, R_HIP)

    angle = torso = None
    if shoulders and hips:
        (sx, sy), (hx, hy) = _mean(shoulders), _mean(hips)
        torso = math.hypot(sx - hx, sy - hy)
        angle = math.degrees(math.atan2(abs(sx - hx), hy - sy))  # 0 = erguido, 90 = horizontal

    # En el piso: caja horizontal y el cuerpo acostado según el torso o, si el torso no se ve completo o la
    # perspectiva lo acorta, según el eje del cuerpo entero.
    body = _body_angle(keypoints)
    orientations = [a for a in (angle, body) if a is not None]
    lying = None
    if orientations:
        x1, y1, x2, y2 = bbox
        lying = max(orientations) >= LYING_MIN_ANGLE and (x2 - x1) >= LYING_ASPECT * (y2 - y1)

    arms = None
    if not lying and (angle is None or angle < 45):
        reference = torso or 0.25 * (bbox[3] - bbox[1])
        sides = [
            (keypoints[s], keypoints[w]) for s, w in ((L_SHOULDER, L_WRIST), (R_SHOULDER, R_WRIST))
            if keypoints[s][2] >= MIN_CONFIDENCE and keypoints[w][2] >= MIN_CONFIDENCE
        ]
        if sides:
            arms = any(wrist[1] < shoulder[1] - RAISED_MARGIN * reference for shoulder, wrist in sides)
    return PostureReading(arms_raised=arms, trunk_angle=angle, lying=lying, body_angle=body)


def iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def attach_keypoints(
    detections: Sequence[Detection], poses: Sequence[tuple[BBox, tuple[Keypoint, ...]]], min_iou: float = 0.4
) -> list[Detection]:
    """Le pega a cada persona seguida los puntos del cuerpo de la pose que mejor se superpone con su caja.

    El modelo de pose corre aparte (no hace tracking): se asocia por superposición, de a pares, el mejor primero.
    """
    pairs = sorted(
        ((iou(d.bbox, p[0]), i, j) for i, d in enumerate(detections) for j, p in enumerate(poses)), reverse=True
    )
    used_d, used_p, matched = set(), set(), {}
    for score, i, j in pairs:
        if score < min_iou:
            break
        if i in used_d or j in used_p:
            continue
        used_d.add(i)
        used_p.add(j)
        matched[i] = poses[j][1]
    return [replace(d, keypoints=matched[i]) if i in matched else d for i, d in enumerate(detections)]


@dataclass
class _Streak:
    since: float
    last_true: float
    alert: Alert | None = None


class PostureMonitor:
    """Convierte condiciones sostenidas en alertas (y las resuelve cuando la condición termina)."""

    def __init__(self, cfg: PostureConfig, next_uid: Callable[[], int] | None = None) -> None:
        self._cfg = cfg
        counter = itertools.count(1)
        self._next_uid = next_uid or (lambda: next(counter))
        self._streaks: dict[tuple[int, str], _Streak] = {}   # (track, condición) → racha
        self._holds = {"arms_raised": cfg.arms_raised_s, "trunk_flexion": cfg.trunk_flexion_s, "fall": cfg.fall_s}

    @property
    def active(self) -> list[Alert]:
        return sorted((s.alert for s in self._streaks.values() if s.alert), key=lambda a: a.ts)

    def ongoing(self, ts: float) -> dict[int, list[tuple[str, float]]]:
        """Condiciones que se están cumpliendo ahora y hace cuánto (aunque todavía no sean alerta)."""
        out: dict[int, list[tuple[str, float]]] = {}
        for (track_id, kind), streak in self._streaks.items():
            if ts - streak.last_true <= RELEASE_S:
                out.setdefault(track_id, []).append((kind, ts - streak.since))
        return out

    def _conditions(self, reading: PostureReading) -> set[str]:
        found = set()
        if reading.arms_raised:
            found.add("arms_raised")
        if reading.lying:
            found.add("fall")
        elif reading.trunk_angle is not None and reading.trunk_angle >= self._cfg.trunk_flexion_deg:
            found.add("trunk_flexion")
        return found

    def update(
        self,
        ts: float,
        readings: Mapping[int, PostureReading],
        zones: Mapping[int, tuple[int, str]] | None = None,   # track → (zona, nombre) donde está
        names: Mapping[int, str] | None = None,                # track → nombre de la persona identificada
    ) -> list[AlertChange]:
        zones, names = zones or {}, names or {}
        changes: list[AlertChange] = []
        for track_id, reading in readings.items():
            for kind in self._conditions(reading):
                streak = self._streaks.setdefault((track_id, kind), _Streak(since=ts, last_true=ts))
                streak.last_true = ts
                if streak.alert is None and ts - streak.since >= self._holds[kind]:
                    streak.alert = self._alert(kind, track_id, streak, ts, zones.get(track_id), names.get(track_id))
                    changes.append(AlertChange("raise", streak.alert, ts))
        for key in [k for k, s in self._streaks.items() if ts - s.last_true > RELEASE_S]:
            streak = self._streaks.pop(key)
            if streak.alert:
                changes.append(AlertChange("resolve", streak.alert, ts))
        return changes

    def resolve_all(self, ts: float) -> list[AlertChange]:
        changes = [AlertChange("resolve", s.alert, ts) for s in self._streaks.values() if s.alert]
        self._streaks.clear()
        return changes

    def _alert(
        self, kind: str, track_id: int, streak: _Streak, ts: float, zone: tuple[int, str] | None, name: str | None
    ) -> Alert:
        who = f"{name} (ID {track_id})" if name else f"persona {track_id}"
        where = f" en {zone[1]}" if zone else ""
        held = format_duration(self._holds[kind])
        if kind == "fall":
            message = f"Posible caída{where}: {who} en el piso hace más de {held}"
        elif kind == "trunk_flexion":
            message = f"Tronco flexionado (≥ {self._cfg.trunk_flexion_deg:.0f}°){where}: {who} hace más de {held}"
        else:
            message = f"Brazos sobre los hombros{where}: {who} hace más de {held}"
        return Alert(
            uid=self._next_uid(), kind=kind, zone_id=zone[0] if zone else None, zone_name=zone[1] if zone else None,
            ts=ts, message=message, track_id=track_id, value_s=ts - streak.since,
        )
