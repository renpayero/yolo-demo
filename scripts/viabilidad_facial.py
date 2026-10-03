r"""¿Sirve una cámara para reconocimiento facial? Se mide sobre un video real de esa cámara, con el mismo detector,
tracker y motor facial que la app.

Mide: (1) qué tamaño tienen las caras de las personas que se ven, (2) cuántas llegan al mínimo que usa la app y
(3) si las huellas faciales distinguen personas: compara la misma persona en distintos frames (mismo track)
contra personas distintas. Si no se distinguen, a esa distancia/resolución el reconocimiento adivinaría.

Uso (grabar ~1 minuto con la cámara candidata, con gente moviéndose como en el día a día):
    docker compose cp video.mp4 app:/tmp/video.mp4
    Get-Content scripts\viabilidad_facial.py -Raw | docker compose exec -T app python - /tmp/video.mp4
"""
import sys
from collections import defaultdict

import cv2
import numpy as np

from app.config import DetectorConfig, IdentificationConfig
from app.detector import UltralyticsPersonDetector
from app.faces import Face, FaceEngine, head_crop

UPSCALE = 3            # se agranda el recorte para darle al detector de caras la mejor chance posible
MIN_FACES_PER_TRACK = 4

cfg = IdentificationConfig()
engine = FaceEngine()
detector = UltralyticsPersonDetector(DetectorConfig())
permissive = cv2.FaceDetectorYN.create(
    "/models/face_detection_yunet_2026may.onnx", "", (320, 320), 0.5, 0.3, 50
)


def faces_in(image):
    h, w = image.shape[:2]
    permissive.setInputSize((w, h))
    _, rows = permissive.detect(image)
    return [] if rows is None else [Face(tuple(map(float, r[:4])), float(r[14]), r) for r in rows]


def main(path: str) -> None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir {path}")
    detections, usable, face_widths = 0, 0, []
    heights: list[float] = []
    by_track: dict[int, list[np.ndarray]] = defaultdict(list)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for det in detector.track(frame):
            detections += 1
            heights.append(det.bbox[3] - det.bbox[1])
            crop = head_crop(frame, det.bbox, min_px=1)
            if crop is None or crop.size == 0:
                continue
            usable += sum(1 for f in engine.detect(crop) if min(f.box[2], f.box[3]) >= cfg.min_face_px)
            big = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
            found = faces_in(big)
            if found:
                face = max(found, key=lambda f: f.box[2] * f.box[3])
                face_widths.append(face.box[2] / UPSCALE)
                by_track[det.track_id].append(engine.embed(big, face))

    if not detections:
        sys.exit("No se detectó ninguna persona en el video.")
    heights_a = np.array(heights)
    print(f"Personas detectadas: {detections} (altura mediana {np.median(heights_a):.0f} px)")
    print(f"Caras encontradas (forzando la búsqueda): {len(face_widths)} de {detections} "
          f"({len(face_widths) / detections:.0%})")
    if face_widths:
        widths = np.array(face_widths)
        print(f"Ancho de las caras: mediana {np.median(widths):.0f} px, p90 {np.percentile(widths, 90):.0f} px")
    print(f"Caras que la app usaría (≥ {cfg.min_face_px} px): {usable} de {detections} ({usable / detections:.0%})")

    tracks = {t: np.array(v) for t, v in by_track.items() if len(v) >= MIN_FACES_PER_TRACK}
    same, other = [], []
    for t, vectors in tracks.items():
        sims = vectors @ vectors.T
        same += list(sims[np.triu_indices(len(sims), 1)])
    ids = list(tracks)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            other += list((tracks[a] @ tracks[b].T).ravel())
    if same and other:
        same_a, other_a = np.array(same), np.array(other)
        confusion = float(np.mean(other_a >= cfg.match_threshold))
        print(f"Huellas: misma persona {np.median(same_a):.2f} (mediana) · personas distintas {np.median(other_a):.2f} "
              f"(p95 {np.percentile(other_a, 95):.2f}) · confusión con el umbral {cfg.match_threshold}: {confusion:.1%}")
    else:
        confusion = None

    share = usable / detections
    print()
    if share >= 0.3 and (confusion is None or confusion < 0.01):
        print("VEREDICTO: viable. Hay caras de tamaño suficiente; conviene igual medir con personas registradas.")
    elif share >= 0.05:
        print("VEREDICTO: parcial. Solo algunas caras alcanzan el tamaño (personas cerca de la cámara); "
              "la identificación va a funcionar de a ratos.")
    else:
        print("VEREDICTO: no viable. Las caras son demasiado chicas (hacen falta ≥ 40 px, idealmente 60-80 px "
              "de ancho y la cámara de frente a la altura de la cara). La ocupación anónima sí funciona.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/video.mp4")
