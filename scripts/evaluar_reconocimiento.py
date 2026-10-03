r"""Mide el reconocimiento facial con LFW (Labeled Faces in the Wild) usando el mismo motor, galería y umbrales que
la app: cuántas fotos de personas registradas reconoce, cuántas confunde y cuántos desconocidos acepta por error.
Sirve para recalibrar `identification.match_threshold` (por ejemplo, si cambian las cámaras o la iluminación).

Uso (todo dentro del contenedor; LFW pesa ~180 MB y es solo para medir, no se sube al repo):
    curl -L -o lfw.tgz https://ndownloader.figshare.com/files/5976018
    docker compose cp lfw.tgz app:/tmp/lfw.tgz
    docker compose exec app sh -c "cd /tmp && tar -xzf lfw.tgz"
    Get-Content scripts\evaluar_reconocimiento.py -Raw | docker compose exec -T app python - /tmp/lfw
"""
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from app.config import IdentificationConfig
from app.faces import FaceEngine
from app.identity import Gallery

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/lfw")
ENROLLED, ENROLL_IMAGES, IMPOSTORS = 25, 2, 300
cfg = IdentificationConfig()
engine = FaceEngine()
random.seed(42)


def embedding(path):
    img = cv2.imread(str(path))
    faces = engine.detect(img)
    if not faces:
        return None
    face = max(faces, key=lambda f: f.box[2] * f.box[3])  # la cara principal de la foto
    return engine.embed(img, face)


people = sorted((p for p in ROOT.iterdir() if p.is_dir()), key=lambda p: p.name)
rich = [p for p in people if len(list(p.glob("*.jpg"))) >= 6]
enrolled = random.sample(rich, ENROLLED)
others = [p for p in people if p not in enrolled]
impostor_paths = [sorted(p.glob("*.jpg"))[0] for p in random.sample(others, IMPOSTORS)]

gallery_vectors, probes, no_face = {}, [], 0
for pid, person in enumerate(enrolled, start=1):
    images = sorted(person.glob("*.jpg"))
    vecs = [v for v in (embedding(p) for p in images[:ENROLL_IMAGES]) if v is not None]
    gallery_vectors[pid] = np.array(vecs)
    probes += [(pid, p) for p in images[ENROLL_IMAGES:]]
gallery = Gallery(gallery_vectors)
names = {pid: p.name for pid, p in enumerate(enrolled, start=1)}
print(f"Registradas {len(enrolled)} personas ({ENROLL_IMAGES} fotos c/u). Pruebas genuinas: {len(probes)}. "
      f"Impostores (no registrados): {len(impostor_paths)}")

genuine, impostor = [], []   # (verdad, match) para el barrido
for truth, path in probes:
    v = embedding(path)
    if v is None:
        no_face += 1
        continue
    genuine.append((truth, v))
for path in impostor_paths:
    v = embedding(path)
    if v is None:
        no_face += 1
        continue
    impostor.append(v)
print(f"Fotos sin cara detectada: {no_face}")


def evaluate(threshold, margin):
    outcome = Counter()
    confusions = []
    for truth, v in genuine:
        m = gallery.match(v, threshold, margin)
        if m.person_id == truth:
            outcome["correcta"] += 1
        elif m.person_id is None:
            outcome["no reconocida"] += 1
        else:
            outcome["CONFUNDIDA"] += 1
            confusions.append((names[truth], names[m.person_id], round(m.score, 3)))
    false_accepts = [(names[m.person_id], round(m.score, 3))
                     for m in (gallery.match(v, threshold, margin) for v in impostor) if m.person_id is not None]
    return outcome, confusions, false_accepts


g_scores = np.array([max(float(np.max(gallery_vectors[t] @ v)), 0) for t, v in genuine])
i_scores = np.array([max(float(np.max(vs @ v)) for vs in gallery_vectors.values()) for v in impostor])
print("\nSimilitud con la persona correcta (genuinas): "
      f"p5={np.percentile(g_scores, 5):.2f} p25={np.percentile(g_scores, 25):.2f} mediana={np.median(g_scores):.2f}")
print("Similitud máxima de un desconocido con alguien registrado: "
      f"mediana={np.median(i_scores):.2f} p95={np.percentile(i_scores, 95):.2f} p99={np.percentile(i_scores, 99):.2f} máx={i_scores.max():.2f}")

print(f"\n{'umbral':>7} {'margen':>7} | {'correctas':>10} {'no recon.':>10} {'CONFUND.':>9} | {'falsos aceptados':>17}")
for threshold in (0.30, 0.35, 0.40, 0.45, 0.50):
    o, c, fa = evaluate(threshold, cfg.min_margin)
    n = len(genuine)
    print(f"{threshold:7.2f} {cfg.min_margin:7.2f} | {o['correcta']/n:9.1%} {o['no reconocida']/n:10.1%} "
          f"{o['CONFUNDIDA']:5d} ({o['CONFUNDIDA']/n:.1%}) | {len(fa):5d} de {len(impostor)} ({len(fa)/len(impostor):.1%})")

o, c, fa = evaluate(cfg.match_threshold, cfg.min_margin)
print(f"\nCon la configuración de la app (umbral {cfg.match_threshold}, margen {cfg.min_margin}):")
print("  confusiones:", c[:10] or "ninguna")
print("  desconocidos aceptados:", fa[:10] or "ninguno")
