"""Reconocedor y registro con un motor facial falso (sin modelos): recorte, agenda, votación y captura."""
from concurrent.futures import Future

import numpy as np
import pytest

from app.config import IdentificationConfig
from app.domain import Detection
from app.faces import ENROLL_SAMPLES, EnrollmentError, Face, Recognizer, head_crop

ANA = np.eye(128, dtype=np.float32)[0]
BETO = np.eye(128, dtype=np.float32)[1]


class FakeEngine:
    """Una cara por imagen si hay algo pintado; la "huella" depende del tono: 200 = Ana, 100 = Beto."""

    def __init__(self, faces_per_image=1, face_px=100):
        self.faces_per_image = faces_per_image
        self.face_px = face_px
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        if image.max() == 0:
            return []
        return [Face((0, 0, self.face_px, self.face_px), 0.99, None) for _ in range(self.faces_per_image)]

    def embed(self, image, face):
        return ANA if image.max() >= 150 else BETO


def frame_with(people):
    """Imagen negra con cada persona pintada de un tono en su caja."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    for (x1, y1, x2, y2), tone in people:
        image[y1:y2, x1:x2] = tone
    return image


CFG = IdentificationConfig(min_face_px=40, checks_per_frame=1, check_interval_s=0.0, votes_needed=3)


def recognizer(engine=None, gallery=None):
    rec = Recognizer(CFG, engine_factory=lambda: engine or FakeEngine())
    rec.set_gallery(gallery if gallery is not None else {1: ANA[None], 2: BETO[None]})
    return rec


def test_head_crop_full_body_takes_the_top_of_the_person():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    crop = head_crop(image, (100, 100, 200, 400), min_px=40)        # de cuerpo entero (alto y angosto)
    assert crop.shape[0] == pytest.approx(0.55 * 300, abs=1)       # la parte de arriba
    assert crop.shape[1] == pytest.approx(100 + 2 * 10, abs=1)     # con margen a los costados
    assert head_crop(image, (100, 100, 120, 130), min_px=40) is None  # demasiado chico


def test_head_crop_close_up_keeps_the_whole_box():
    """Plano cercano (webcam): la cara puede estar hasta abajo de la caja; cortarla impide detectarla."""
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    crop = head_crop(image, (403, 101, 838, 666), min_px=40)        # caja real de un retrato
    assert crop.shape[0] == 666 - 101


def test_identifies_a_track_after_enough_votes():
    rec = recognizer()
    det = Detection(7, (100, 50, 300, 450), 0.9)
    image = frame_with([((100, 50, 300, 450), 200)])
    results = [[(c.track_id, c.person_id) for c in rec.identify(image, [det], now=t * 0.1)] for t in range(4)]
    assert results[:2] == [[], []] and results[2] == [(7, 1)]
    assert rec.identity(7) == 1
    assert results[3] == []                       # ya identificado: no se re-verifica hasta que pase el intervalo


def test_strangers_and_unseen_faces_are_not_identified():
    rec = recognizer(gallery={2: BETO[None]})      # solo Beto registrado; el track es Ana
    det = Detection(7, (100, 50, 300, 450), 0.9)
    image = frame_with([((100, 50, 300, 450), 200)])
    for t in range(10):
        rec.identify(image, [det], now=t)
    assert rec.identity(7) is None

    engine = FakeEngine()
    rec = recognizer(engine)
    dark = frame_with([])                         # la persona de espaldas: no hay cara
    for t in range(10):
        rec.identify(dark, [det], now=t)
    assert rec.identity(7) is None


def test_without_gallery_no_face_is_processed():
    engine = FakeEngine()
    rec = recognizer(engine, gallery={})
    rec.identify(frame_with([((100, 50, 300, 450), 200)]), [Detection(7, (100, 50, 300, 450), 0.9)], now=0)
    assert engine.calls == 0                      # ni siquiera se cargan los modelos


def test_checks_are_shared_between_tracks():
    """Con un chequeo por frame, se alternan los tracks: el que hace más que no se mira va primero."""
    rec = recognizer()
    a = Detection(7, (0, 0, 200, 400), 0.9)
    b = Detection(8, (300, 0, 500, 400), 0.9)
    image = frame_with([((0, 0, 200, 400), 200), ((300, 0, 500, 400), 100)])
    for t in range(6):
        rec.identify(image, [a, b], now=t * 0.1)
    assert rec.assignments() == {7: 1, 8: 2}


def test_identified_tracks_are_rechecked_and_fixed():
    """Sin huecos (presencia continua), el ID 7 de Ana pasa a Beto: la re-verificación retira el nombre
    tras dos contradicciones y después se identifica a Beto."""
    cfg = IdentificationConfig(min_face_px=40, checks_per_frame=1, check_interval_s=0.0, votes_needed=3,
                               recheck_interval_s=1.0)
    rec = Recognizer(cfg, engine_factory=FakeEngine)
    rec.set_gallery({1: ANA[None], 2: BETO[None]})
    det = Detection(7, (100, 50, 300, 450), 0.9)
    ana, beto = frame_with([((100, 50, 300, 450), 200)]), frame_with([((100, 50, 300, 450), 100)])

    changes = [c for t in range(3) for c in rec.identify(ana, [det], now=t * 0.1)]
    assert [(c.track_id, c.person_id) for c in changes] == [(7, 1)]
    changes = [c for t in range(3, 40) for c in rec.identify(beto, [det], now=t * 0.1)]
    assert (changes[0].track_id, changes[0].person_id) == (7, None)   # nombre retirado
    assert changes[0].end_ts == pytest.approx(0.2)                     # última vez que se confirmó a Ana
    assert changes[0].start_ts == pytest.approx(2.2, abs=0.15)         # 2 re-verificaciones (1 s c/u) después
    assert (changes[-1].track_id, changes[-1].person_id) == (7, 2) and rec.identity(7) == 2


def identified(rec, image, det, start=0.0):
    for i in range(3):
        rec.identify(image, [det], now=start + i * 0.1)


def test_reappearing_track_is_hidden_until_its_face_is_checked():
    """El track vuelve tras un hueco: el nombre no se muestra hasta verificar la cara (que sí coincide)."""
    rec = recognizer()
    det = Detection(7, (100, 50, 300, 450), 0.9)
    ana = frame_with([((100, 50, 300, 450), 200)])
    identified(rec, ana, det)
    assert rec.identity(7) == 1

    rec.identify(frame_with([]), [], now=0.5)          # se pierde un momento
    changes = rec.identify(ana, [det], now=2.0)         # vuelve: se verifica en el acto y coincide
    assert changes == [] and rec.identity(7) == 1


def test_inherited_track_is_revoked_on_the_first_check():
    """Sale Ana; un segundo después Beto aparece en el mismo lugar y el tracker le da el ID 7."""
    rec = recognizer()
    det = Detection(7, (100, 50, 300, 450), 0.9)
    identified(rec, frame_with([((100, 50, 300, 450), 200)]), det)
    beto = frame_with([((100, 50, 300, 450), 100)])

    # Primer frame del regreso: el nombre de Ana ya no se muestra ni se usa, y la cara lo desmiente en el acto.
    changes = rec.identify(beto, [det], now=1.5)
    assert rec.identity(7) is None and rec.assignments() == {}
    [change] = changes
    assert (change.track_id, change.person_id) == (7, None)
    assert change.end_ts == pytest.approx(0.2) and change.start_ts == pytest.approx(1.5)  # Ana: hasta que se fue


def test_suspect_without_visible_face_stays_anonymous():
    rec = recognizer()
    det = Detection(7, (100, 50, 300, 450), 0.9)
    identified(rec, frame_with([((100, 50, 300, 450), 200)]), det)
    for t in (1.5, 2.0, 2.5):
        rec.identify(frame_with([]), [det], now=t)      # volvió de espaldas: no hay cara que verificar
    assert rec.identity(7) is None


def test_transfer_from_occupancy_rebind_is_also_verified():
    rec = recognizer()
    identified(rec, frame_with([((100, 50, 300, 450), 200)]), Detection(7, (100, 50, 300, 450), 0.9))
    rec.transfer(7, 9, now=0.5)                         # reasignación por posición, no por cara
    assert rec.identity(9) is None                      # en duda hasta verla
    rec.identify(frame_with([((100, 50, 300, 450), 200)]), [Detection(9, (100, 50, 300, 450), 0.9)], now=0.6)
    assert rec.identity(9) == 1


def test_clear_makes_everyone_anonymous():
    rec = recognizer()
    det = Detection(7, (100, 50, 300, 450), 0.9)
    image = frame_with([((100, 50, 300, 450), 200)])
    for t in range(3):
        rec.identify(image, [det], now=t)
    rec.clear()
    assert rec.assignments() == {}


# ---------- Registro ----------

def run_enrollment(rec, image, seconds=8.0, step=0.1):
    future = Future()
    rec.start_enrollment(future, now=0.0)
    t = 0.0
    while not future.done() and t <= seconds:
        rec.step_enrollment(image, now=t)
        t += step
    return future


def test_enrollment_collects_samples():
    rec = recognizer()
    future = run_enrollment(rec, frame_with([((100, 50, 300, 450), 200)]))
    samples = future.result(timeout=0)
    assert samples.shape == (ENROLL_SAMPLES, 128)


@pytest.mark.parametrize(
    ("engine", "image", "message"),
    [
        (FakeEngine(), frame_with([]), "ninguna cara"),
        (FakeEngine(faces_per_image=2), frame_with([((0, 0, 100, 100), 200)]), "más de una cara"),
        (FakeEngine(face_px=50), frame_with([((0, 0, 100, 100), 200)]), "muy chica"),
        (FakeEngine(), None, "No hay video"),
    ],
)
def test_enrollment_explains_why_it_failed(engine, image, message):
    rec = recognizer(engine)
    future = run_enrollment(rec, image)
    with pytest.raises(EnrollmentError, match=message):
        future.result(timeout=0)


def test_only_one_enrollment_at_a_time():
    rec = recognizer()
    first, second = Future(), Future()
    rec.start_enrollment(first, now=0)
    rec.start_enrollment(second, now=0)
    with pytest.raises(EnrollmentError, match="en curso"):
        second.result(timeout=0)
    assert not first.done()


def test_match_existing_detects_an_already_registered_face():
    rec = recognizer()
    assert rec.match_existing(np.array([ANA, ANA, ANA]))[0] == 1
    stranger = np.eye(128, dtype=np.float32)[5]
    assert rec.match_existing(np.array([stranger] * 3))[0] is None
