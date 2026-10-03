"""Reconocimiento facial con OpenCV: YuNet detecta caras y SFace calcula su huella (128 números).

Solo lo usa el hilo del pipeline. Los modelos se cargan recién la primera vez que hacen falta (identificación
prendida o un registro), así con la identificación apagada no se procesa ninguna cara.
Nunca se guardan imágenes: del registro solo quedan las huellas.
"""
from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.config import IdentificationConfig
from app.domain import BBox, Detection
from app.identity import Gallery, IdentityVoter

log = logging.getLogger(__name__)

DETECTOR_PATH = Path("/models/face_detection_yunet_2026may.onnx")
RECOGNIZER_PATH = Path("/models/face_recognition_sface_2021dec.onnx")
DETECTION_SCORE = 0.85
HEAD_SHARE = 0.55          # de cuerpo entero: parte superior de la caja donde se busca la cara
FULL_BODY_ASPECT = 2.0     # caja al menos el doble de alta que de ancha = persona de cuerpo entero
PRUNE_EVERY_S = 30.0
FORGET_AFTER_S = 60.0
# Si un track con nombre desaparece más que esto y vuelve, el tracker pudo habérselo dado a otra persona
# (relevo en un puesto): el nombre queda "en duda" hasta volver a ver la cara.
REAPPEAR_GAP_S = 0.5

ENROLL_SAMPLES = 5
ENROLL_TIMEOUT_S = 6.0
ENROLL_SAMPLE_GAP_S = 0.25
ENROLL_MIN_FACE_PX = 80
ENROLL_MIN_SCORE = 0.9
ENROLL_PROBLEMS = {
    "no_face": "No se ve ninguna cara con claridad. Mirá de frente a la cámara, con buena luz.",
    "many_faces": "Hay más de una cara en la imagen: que quede solo la persona a registrar.",
    "small": "La cara se ve muy chica: acercate a la cámara (a menos de 1,5 m).",
    "no_video": "No hay video de la cámara.",
}


class EnrollmentError(Exception):
    pass


@dataclass(frozen=True)
class Face:
    box: tuple[float, float, float, float]   # x, y, ancho, alto (en la imagen analizada)
    score: float
    row: np.ndarray                          # fila completa de YuNet (caja + 5 puntos + score), para alinear


class FaceEngine:
    def __init__(self) -> None:
        self._detector = cv2.FaceDetectorYN.create(str(DETECTOR_PATH), "", (320, 320), DETECTION_SCORE, 0.3, 50)
        self._recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_PATH), "")
        log.info("Modelos de reconocimiento facial cargados")

    def detect(self, image: np.ndarray) -> list[Face]:
        h, w = image.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(image)
        if faces is None:
            return []
        return [Face((float(f[0]), float(f[1]), float(f[2]), float(f[3])), float(f[14]), f) for f in faces]

    def embed(self, image: np.ndarray, face: Face) -> np.ndarray:
        feature = self._recognizer.feature(self._recognizer.alignCrop(image, face.row)).flatten()
        return feature / np.linalg.norm(feature)


def head_crop(image: np.ndarray, bbox: BBox, min_px: int) -> np.ndarray | None:
    """Recorte donde buscar la cara de una persona, con un poco de margen a los costados.

    De cuerpo entero (caja alta y angosta) la cara está arriba: alcanza con la parte superior. En un plano
    cercano (webcam de escritorio) puede estar en cualquier altura de la caja: se usa entera, porque una cara
    cortada (sin boca ni mentón) el detector no la encuentra.
    """
    x1, y1, x2, y2 = bbox
    margin = 0.1 * (x2 - x1)
    full_body = (y2 - y1) >= FULL_BODY_ASPECT * (x2 - x1)
    share = HEAD_SHARE if full_body else 1.0
    h, w = image.shape[:2]
    left, right = max(0, int(x1 - margin)), min(w, int(x2 + margin))
    top, bottom = max(0, int(y1)), min(h, int(y1 + share * (y2 - y1)))
    if right - left < min_px or bottom - top < min_px:
        return None
    return image[top:bottom, left:right]


@dataclass(frozen=True)
class IdentityChange:
    track_id: int
    person_id: int | None            # None: se le retiró el nombre (era otra persona)
    end_ts: float | None = None      # al retirar: hasta cuándo fue la persona anterior
    start_ts: float | None = None    # al retirar: desde cuándo es otra persona


@dataclass
class _Enrollment:
    future: Future
    deadline: float
    samples: list[np.ndarray] = field(default_factory=list)
    last_sample_at: float = 0.0
    problems: Counter = field(default_factory=Counter)


class Recognizer:
    def __init__(self, cfg: IdentificationConfig, engine_factory: Callable[[], FaceEngine] = FaceEngine) -> None:
        self._cfg = cfg
        self._engine_factory = engine_factory
        self._engine: FaceEngine | None = None
        self._gallery = Gallery({})
        self._voter = IdentityVoter(cfg.votes_needed, window=5)
        self._last_check: dict[int, float] = {}
        self._last_seen: dict[int, float] = {}
        self._confirmed_at: dict[int, float] = {}            # track con nombre → última vez que la cara coincidió
        self._suspects: dict[int, tuple[float, float]] = {}  # track en duda → (cuándo se perdió, cuándo volvió)
        self._next_prune = 0.0
        self._enrollment: _Enrollment | None = None

    def _ready(self) -> FaceEngine:
        if self._engine is None:
            self._engine = self._engine_factory()
        return self._engine

    # ---------- Galería e identidades ----------

    def set_gallery(self, embeddings: Mapping[int, np.ndarray]) -> None:
        self._gallery = Gallery(embeddings)

    def gallery_size(self) -> int:
        return len(self._gallery)

    def identity(self, track_id: int) -> int | None:
        """Nombre confirmado del track; los que están "en duda" no se muestran ni se registran."""
        return None if track_id in self._suspects else self._voter.identity(track_id)

    def assignments(self) -> dict[int, int]:
        return {t: p for t, p in self._voter.assignments().items() if t not in self._suspects}

    def transfer(self, old_track: int, new_track: int, now: float) -> None:
        """Reasignación por posición en el motor de ocupación: el nombre pasa, pero en duda hasta ver la cara."""
        self._voter.transfer(old_track, new_track)
        if self._voter.identity(new_track) is not None:
            self._suspects[new_track] = (self._last_seen.get(old_track, now), now)
        self._suspects.pop(old_track, None)
        self._confirmed_at.pop(old_track, None)

    def forget_person(self, person_id: int) -> None:
        self._voter.forget_person(person_id)
        self._drop_unassigned()

    def clear(self) -> None:
        """Al apagar la identificación: todos los tracks vuelven a ser anónimos."""
        self._voter.clear()
        self._last_check.clear()
        self._suspects.clear()
        self._confirmed_at.clear()

    def _drop_unassigned(self) -> None:
        assigned = self._voter.assignments()
        for table in (self._suspects, self._confirmed_at):
            for track_id in [t for t in table if t not in assigned]:
                del table[track_id]

    def identify(self, image: np.ndarray, detections: Sequence[Detection], now: float) -> list[IdentityChange]:
        """Revisa caras de algunos tracks, en este orden de prioridad:
        1. los que tienen nombre pero están en duda (volvieron tras perderse): se verifican en el acto;
        2. los que todavía no tienen nombre;
        3. los que tienen nombre: se re-verifican de vez en cuando.
        """
        if not len(self._gallery):
            return []
        assigned = self._voter.assignments()
        for det in detections:
            previous = self._last_seen.get(det.track_id)
            if det.track_id in assigned and previous is not None and now - previous > REAPPEAR_GAP_S:
                self._suspects.setdefault(det.track_id, (previous, now))
            self._last_seen[det.track_id] = now
        never = float("-inf")

        def due(det: Detection, interval: float) -> bool:
            return now - self._last_check.get(det.track_id, never) >= interval

        def oldest_first(dets: list[Detection]) -> list[Detection]:
            return sorted(dets, key=lambda d: self._last_check.get(d.track_id, never))

        suspects = oldest_first([d for d in detections if d.track_id in self._suspects])
        pending = oldest_first([d for d in detections if d.track_id not in assigned and due(d, self._cfg.check_interval_s)])
        recheck = oldest_first([
            d for d in detections
            if d.track_id in assigned and d.track_id not in self._suspects and due(d, self._cfg.recheck_interval_s)
        ])

        changes: list[IdentityChange] = []
        for det in (suspects + pending + recheck)[: self._cfg.checks_per_frame]:
            self._last_check[det.track_id] = now
            seen, person = self._look(image, det)
            if seen:  # de espaldas o tapado: no es evidencia ni a favor ni en contra
                self._judge(det.track_id, person, assigned.get(det.track_id), now, changes)
        self._prune(now)
        return changes

    def _judge(
        self, track_id: int, person: int | None, assigned: int | None, now: float, changes: list[IdentityChange]
    ) -> None:
        if track_id in self._suspects:
            lost_at, back_at = self._suspects.pop(track_id)
            if person == assigned:
                self._confirmed_at[track_id] = now
                return
            self._revoke(track_id, end_ts=lost_at, start_ts=back_at, changes=changes)
        elif assigned is not None:
            if person == assigned:
                self._confirmed_at[track_id] = now
            if not self._voter.verify(track_id, person):
                return
            self._revoke(track_id, end_ts=self._confirmed_at.get(track_id, now), start_ts=now, changes=changes)
        # Sin nombre (o recién retirado): la cara vista cuenta como voto por quien es ahora.
        if (identified := self._voter.observe(track_id, person)) is not None:
            self._confirmed_at[track_id] = now
            changes.append(IdentityChange(track_id, identified))

    def _revoke(self, track_id: int, end_ts: float, start_ts: float, changes: list[IdentityChange]) -> None:
        self._voter.forget(track_id)
        self._confirmed_at.pop(track_id, None)
        changes.append(IdentityChange(track_id, None, end_ts=end_ts, start_ts=start_ts))

    def _look(self, image: np.ndarray, det: Detection) -> tuple[bool, int | None]:
        """¿Se ve una cara en el track? Y si se ve, ¿de quién es? (None: de nadie registrado, o ambigua)."""
        crop = head_crop(image, det.bbox, self._cfg.min_face_px)
        if crop is None:
            return False, None
        engine = self._ready()
        faces = [f for f in engine.detect(crop) if min(f.box[2], f.box[3]) >= self._cfg.min_face_px]
        if not faces:
            return False, None
        face = max(faces, key=lambda f: f.box[2] * f.box[3])
        match = self._gallery.match(engine.embed(crop, face), self._cfg.match_threshold, self._cfg.min_margin)
        return True, match.person_id

    def _prune(self, now: float) -> None:
        if now < self._next_prune:
            return
        self._next_prune = now + PRUNE_EVERY_S
        for track_id in [t for t, seen in self._last_seen.items() if now - seen > FORGET_AFTER_S]:
            self._last_seen.pop(track_id, None)
            self._last_check.pop(track_id, None)
            self._voter.forget(track_id)
        self._drop_unassigned()

    # ---------- Registro (captura de muestras) ----------

    def start_enrollment(self, future: Future, now: float) -> None:
        if self._enrollment is not None:
            future.set_exception(EnrollmentError("Ya hay un registro en curso; esperá a que termine."))
            return
        self._enrollment = _Enrollment(future, deadline=now + ENROLL_TIMEOUT_S)

    def step_enrollment(self, image: np.ndarray | None, now: float) -> None:
        """Se llama en cada vuelta del pipeline mientras hay un registro en curso."""
        enrollment = self._enrollment
        if enrollment is None:
            return
        if now >= enrollment.deadline:
            reason = enrollment.problems.most_common(1)[0][0] if enrollment.problems else "no_face"
            self._finish(exception=EnrollmentError(ENROLL_PROBLEMS[reason]))
            return
        if image is None:
            enrollment.problems["no_video"] += 1
            return
        if now - enrollment.last_sample_at < ENROLL_SAMPLE_GAP_S:
            return
        feature, problem = self._enrollment_sample(image)
        if feature is None:
            enrollment.problems[problem] += 1
            return
        enrollment.samples.append(feature)
        enrollment.last_sample_at = now
        if len(enrollment.samples) >= ENROLL_SAMPLES:
            self._finish(result=np.array(enrollment.samples, dtype=np.float32))

    def _enrollment_sample(self, image: np.ndarray) -> tuple[np.ndarray | None, str]:
        engine = self._ready()
        faces = [f for f in engine.detect(image) if f.score >= ENROLL_MIN_SCORE]
        if not faces:
            return None, "no_face"
        if len(faces) > 1:
            return None, "many_faces"
        if min(faces[0].box[2], faces[0].box[3]) < ENROLL_MIN_FACE_PX:
            return None, "small"
        return engine.embed(image, faces[0]), ""

    def _finish(self, result: np.ndarray | None = None, exception: Exception | None = None) -> None:
        future = self._enrollment.future
        self._enrollment = None
        if future.done():  # el pedido HTTP ya se canceló
            return
        if exception is not None:
            future.set_exception(exception)
        else:
            future.set_result(result)

    def match_existing(self, samples: np.ndarray) -> tuple[int | None, float]:
        """¿Las muestras nuevas son de alguien ya registrado? (evita registrar a la misma persona dos veces)"""
        if not len(self._gallery):
            return None, 0.0
        matches = [self._gallery.match(s, self._cfg.match_threshold, self._cfg.min_margin) for s in samples]
        votes = Counter(m.person_id for m in matches if m.person_id is not None)
        if not votes:
            return None, max(m.score for m in matches)
        person, count = votes.most_common(1)[0]
        return (person, max(m.score for m in matches)) if count > len(samples) // 2 else (None, 0.0)
