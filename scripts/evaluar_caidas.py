r"""Mide la detección de caídas (y las falsas alarmas) con el UR Fall Detection Dataset, usando el mismo detector,
modelo de pose y reglas que la app. Cada video se extiende 3 s con su último cuadro (la persona queda en el piso).

Resultado de referencia (2026-10): 13/20 caídas detectadas (65 %); 2/20 falsas alarmas, ambas de personas
acostadas en un sillón (indistinguible de estar tirado). Diseño 01-10: 7/10; control no mirado 11-20: 6/10.

Uso (los videos traen profundidad + RGB lado a lado; se recorta la mitad RGB con el ffmpeg de tools/):
    for i in 01 02 ... 20: bajar http://fenix.ur.edu.pl/~mkepski/ds/data/fall-$i-cam0.mp4 y adl-$i-cam0.mp4
    tools\ffmpeg.exe -i fall-01-cam0.mp4 -vf "crop=320:240:320:0" -c:v mjpeg -q:v 3 rgb\fall-01-cam0.avi
    docker compose cp rgb app:/tmp/urfall/rgb
    Get-Content scripts\evaluar_caidas.py -Raw | docker compose exec -T app python - /tmp/urfall/rgb
"""
import sys
from collections import Counter
from pathlib import Path

import cv2

from app.config import DetectorConfig, PostureConfig
from app.detector import UltralyticsPersonDetector, UltralyticsPoseEstimator
from app.posture import PostureMonitor, attach_keypoints, read_posture

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/urfall/rgb")
FPS, HOLD_LAST_S = 30, 3.0
cfg = PostureConfig()
detector = UltralyticsPersonDetector(DetectorConfig())
pose = UltralyticsPoseEstimator(detector.device)


def frames(path):
    cap = cv2.VideoCapture(str(path))
    last = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        last = f                                 # ya recortado a la cámara RGB
        yield last
    if last is None:
        raise RuntimeError(f"No se pudo leer ningún frame de {path}")
    for _ in range(int(HOLD_LAST_S * FPS)):     # la persona queda como terminó el video
        yield last


def evaluate(path):
    detector.reset()
    monitor = PostureMonitor(cfg)
    raised, lying_frames, seen_frames, t = [], 0, 0, 0.0
    for i, image in enumerate(frames(path)):
        t = i / FPS
        dets = attach_keypoints(detector.track(image), pose.estimate(image))
        readings = {d.track_id: read_posture(d.keypoints, d.bbox) for d in dets if d.keypoints}
        seen_frames += bool(readings)
        lying_frames += any(r.lying for r in readings.values())
        raised += [(c.alert.kind, round(t, 1)) for c in monitor.update(t, readings) if c.action == "raise"]
    return raised, lying_frames, seen_frames


results = {}
for path in sorted(ROOT.glob("*.avi")):
    results[path.stem] = evaluate(path)

for kind in ("fall", "adl"):
    print(f"\n{'CAÍDAS' if kind == 'fall' else 'ACTIVIDADES COTIDIANAS (sin caída)'}")
    for name, (raised, lying, seen) in results.items():
        if name.startswith(kind):
            kinds = Counter(k for k, _ in raised)
            first_fall = next((t for k, t in raised if k == "fall"), None)
            print(f"  {name}: alertas {dict(kinds) or '-'}"
                  + (f" · caída avisada a los {first_fall}s" if first_fall is not None else "")
                  + f" · frames 'en el piso' {lying}/{seen} con pose")

def summary(label, names_filter):
    falls = [n for n in results if n.startswith("fall") and names_filter(n)]
    adls = [n for n in results if n.startswith("adl") and names_filter(n)]
    hit = sum(any(k == "fall" for k, _ in results[n][0]) for n in falls)
    false_fall = sum(any(k == "fall" for k, _ in results[n][0]) for n in adls)
    other = Counter(k for n in adls for k, _ in results[n][0] if k != "fall")
    print(f"{label}: caídas detectadas {hit}/{len(falls)} · falsas alarmas de caída {false_fall}/{len(adls)}"
          f" · otras alertas en actividades cotidianas: {dict(other) or 'ninguna'}")


number = lambda n: int(n.split("-")[1])
print()
summary("Conjunto de diseño (01-10)", lambda n: number(n) <= 10)
summary("Conjunto de control (11-20, no mirado al diseñar)", lambda n: number(n) > 10)
