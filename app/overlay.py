"""Dibujo sobre el video: cajas, etiquetas y barra de estado.

El texto se dibuja con Pillow (cv2.putText no soporta acentos ni ñ) y cada etiqueta renderizada se guarda en
caché, así que dibujar el mismo texto en frames sucesivos cuesta casi nada.
"""
from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.domain import Detection
from app.formatting import format_duration
from app.occupancy import ZoneStatus
from app.zones import Point, Zone

Color = tuple[int, int, int]  # BGR

# Paleta para distinguir tracks (BGR), con buen contraste sobre video.
TRACK_COLORS: tuple[Color, ...] = (
    (255, 178, 29), (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (211, 188, 0), (209, 99, 255), (128, 128, 255), (255, 149, 200), (82, 132, 255),
)
WHITE: Color = (255, 255, 255)
BLACK: Color = (0, 0, 0)


def _font_path(bold: bool) -> str | None:
    # Matplotlib (dependencia de Ultralytics) trae DejaVu Sans, con soporte Unicode completo.
    spec = importlib.util.find_spec("matplotlib")
    if spec is None or spec.origin is None:
        return None
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path(spec.origin).parent / "mpl-data" / "fonts" / "ttf" / name
    return str(path) if path.exists() else None


@lru_cache(maxsize=8)
def _font(size: int, bold: bool) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = _font_path(bold)
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size)


@lru_cache(maxsize=1024)
def _label_patch(text: str, size: int, fg: Color, bg: Color, bg_alpha: int, bold: bool) -> np.ndarray:
    """Etiqueta renderizada como imagen BGRA (texto sobre fondo semitransparente)."""
    font = _font(size, bold)
    left, top, right, bottom = font.getbbox(text)
    pad_x, pad_y = max(4, size // 3), max(2, size // 5)
    w, h = right - left + 2 * pad_x, bottom - top + 2 * pad_y
    img = Image.new("RGBA", (w, h), (bg[2], bg[1], bg[0], bg_alpha))
    ImageDraw.Draw(img).text((pad_x - left, pad_y - top), text, font=font, fill=(fg[2], fg[1], fg[0], 255))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2BGRA)


def draw_label(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    size: int = 15,
    fg: Color = WHITE,
    bg: Color = BLACK,
    bg_alpha: int = 170,
    bold: bool = False,
) -> tuple[int, int]:
    """Dibuja `text` con esquina superior izquierda en `org` (corrida hacia adentro si no entra en el cuadro).

    Devuelve (ancho, alto) de la etiqueta.
    """
    patch = _label_patch(text, size, fg, bg, bg_alpha, bold)
    ph, pw = patch.shape[:2]
    fh, fw = frame.shape[:2]
    x = min(max(org[0], 0), max(fw - pw, 0))
    y = min(max(org[1], 0), max(fh - ph, 0))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + pw, fw), min(y + ph, fh)
    if x0 >= x1 or y0 >= y1:
        return pw, ph
    crop = patch[y0 - y : y1 - y, x0 - x : x1 - x]
    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y0:y1, x0:x1]
    roi[:] = (crop[:, :, :3] * alpha + roi * (1.0 - alpha)).astype(np.uint8)
    return pw, ph


ZONE_FILL_ALPHA = 0.22
POSTURE_SHORT = {"arms_raised": "brazos arriba", "trunk_flexion": "tronco flexionado", "fall": "en el piso"}
# Líneas del esqueleto (índices COCO): hombros, brazos, torso y piernas.
SKELETON = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))
KEYPOINT_MIN_CONF = 0.5
ALERT_COLOR: Color = (73, 81, 248)   # rojo (#f85149), el mismo de la web
MAX_BANNER_ALERTS = 3


def track_color(track_id: int) -> Color:
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


def hex_to_bgr(color: str) -> Color:
    value = color.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def _zone_label(status: ZoneStatus, now: float) -> str:
    count = len(status.present)
    prefix = "⚠ " if status.zone.kind == "restricted" else ""
    if count:
        return f"{prefix}{status.zone.name} · {count} {'persona' if count == 1 else 'personas'}"
    empty_for = format_duration(now - status.empty_since) if status.empty_since is not None else ""
    return f"{prefix}{status.zone.name} · vacía {empty_for}".rstrip()


def draw_zones(
    frame: np.ndarray, statuses: list[ZoneStatus], now: float, alert_zone_ids: set[int] | None = None
) -> None:
    """Zonas semitransparentes con nombre y ocupación. Las que tienen una alerta activa, en rojo y con borde grueso."""
    if not statuses:
        return
    alert_zone_ids = alert_zone_ids or set()
    h, w = frame.shape[:2]
    layer = frame.copy()
    shapes = []
    for status in statuses:
        pts = np.array([[round(x * w), round(y * h)] for x, y in status.zone.polygon], dtype=np.int32)
        in_alert = status.zone.id in alert_zone_ids
        color = ALERT_COLOR if in_alert else hex_to_bgr(status.zone.color)
        cv2.fillPoly(layer, [pts], color)
        shapes.append((pts, color, status, in_alert))
    cv2.addWeighted(layer, ZONE_FILL_ALPHA, frame, 1 - ZONE_FILL_ALPHA, 0, dst=frame)

    for pts, color, status, in_alert in shapes:
        cv2.polylines(frame, [pts], True, color, 4 if in_alert else 2, cv2.LINE_AA)
        text = _zone_label(status, now)
        label_h = _label_patch(text, 14, BLACK, color, 230, True).shape[0]
        top_x, top_y = pts[np.argmin(pts[:, 1])]  # vértice más alto: la etiqueta va justo arriba
        y = top_y - label_h if top_y - label_h >= 0 else top_y
        draw_label(frame, text, (int(top_x), int(y)), size=14, fg=BLACK, bg=color, bg_alpha=230, bold=True)


def resize_for_stream(image: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    """Achica el frame para mandarlo al navegador. Devuelve la imagen y el factor de escala aplicado."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image.copy(), 1.0
    scale = max_width / w
    return cv2.resize(image, (max_width, round(h * scale)), interpolation=cv2.INTER_AREA), scale


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    scale: float,
    anchors: dict[int, Point] | None = None,
    track_zones: dict[int, list[tuple[Zone, float]]] | None = None,
    alert_track_ids: set[int] | None = None,
    names: dict[int, str] | None = None,
    postures: dict[int, list[tuple[str, float]]] | None = None,
) -> None:
    """Caja e ID de cada persona (y su nombre, si está identificada); si está en zonas, cuáles y hace cuánto;
    y el punto que decide la zona. Quien tiene una alerta activa (permanencia, zona restringida) va en rojo.
    """
    h, w = frame.shape[:2]
    for det in detections:
        in_alert = det.track_id in (alert_track_ids or set())
        color = ALERT_COLOR if in_alert else track_color(det.track_id)
        x1, y1, x2, y2 = (round(v * scale) for v in det.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if in_alert else 2, cv2.LINE_AA)

        parts = [f"ID {det.track_id}"]
        if names and det.track_id in names:
            parts.append(names[det.track_id])
        parts += [f"{zone.name} {format_duration(dwell)}" for zone, dwell in (track_zones or {}).get(det.track_id, [])]
        parts += [f"⚠ {POSTURE_SHORT[kind]} {format_duration(held)}" for kind, held in (postures or {}).get(det.track_id, [])]
        text = " · ".join(parts)
        label_h = _label_patch(text, 14, BLACK, color, 230, True).shape[0]
        y = y1 - label_h if y1 - label_h >= 0 else y1  # arriba de la caja si entra; si no, adentro
        draw_label(frame, text, (x1, y), size=14, fg=BLACK, bg=color, bg_alpha=230, bold=True)

        if anchors and det.track_id in anchors:
            ax, ay = anchors[det.track_id]
            center = (round(ax * w), round(ay * h))
            cv2.circle(frame, center, 6, BLACK, -1, cv2.LINE_AA)
            cv2.circle(frame, center, 4, color, -1, cv2.LINE_AA)


def draw_skeletons(frame: np.ndarray, detections: list[Detection], scale: float) -> None:
    """Esqueleto fino de cada persona con pose (lo que usa el análisis de posturas)."""
    for det in detections:
        if not det.keypoints:
            continue
        color = track_color(det.track_id)
        pts = [(round(x * scale), round(y * scale), c) for x, y, c in det.keypoints]
        for a, b in SKELETON:
            if pts[a][2] >= KEYPOINT_MIN_CONF and pts[b][2] >= KEYPOINT_MIN_CONF:
                cv2.line(frame, pts[a][:2], pts[b][:2], color, 2, cv2.LINE_AA)
        for x, y, c in pts[5:]:
            if c >= KEYPOINT_MIN_CONF:
                cv2.circle(frame, (x, y), 3, color, -1, cv2.LINE_AA)


def draw_status_bar(frame: np.ndarray, text: str) -> None:
    """Abajo a la derecha: arriba a la izquierda se pisaría con las etiquetas de zonas que empiezan en la esquina."""
    w, h = _label_patch(text, 13, WHITE, BLACK, 150, False).shape[1::-1]
    draw_label(frame, text, (frame.shape[1] - w - 8, frame.shape[0] - h - 8), size=13, bg_alpha=150)


def draw_identification_badge(frame: np.ndarray) -> None:
    """Aviso visible mientras se procesan caras: quien está frente a la cámara tiene que poder saberlo."""
    text = "● Identificación facial activa"
    width = _label_patch(text, 14, WHITE, ALERT_COLOR, 220, True).shape[1]
    draw_label(frame, text, (frame.shape[1] - width - 8, 8), size=14, fg=WHITE, bg=ALERT_COLOR, bg_alpha=220, bold=True)


def draw_alert_banner(frame: np.ndarray, messages: list[str]) -> None:
    """Alertas activas apiladas abajo a la izquierda; `messages` va de la más vieja a la más nueva."""
    if not messages:
        return
    lines = [f"⚠ {m}" for m in messages[-MAX_BANNER_ALERTS:]]
    if len(messages) > MAX_BANNER_ALERTS:
        lines.insert(0, f"+ {len(messages) - MAX_BANNER_ALERTS} alerta(s) más")
    y = frame.shape[0] - 8
    for line in reversed(lines):  # de abajo hacia arriba: la más nueva queda pegada al borde
        y -= _label_patch(line, 15, WHITE, ALERT_COLOR, 235, True).shape[0]
        draw_label(frame, line, (8, y), size=15, fg=WHITE, bg=ALERT_COLOR, bg_alpha=235, bold=True)
        y -= 4


def placeholder_frame(width: int, height: int, title: str, subtitle: str) -> np.ndarray:
    """Imagen que se muestra mientras no hay video (fuente caída o cargando el modelo)."""
    frame = np.full((height, width, 3), 32, dtype=np.uint8)
    w, h = _label_patch(title, 22, WHITE, BLACK, 0, True).shape[1::-1]
    draw_label(frame, title, ((width - w) // 2, height // 2 - h - 4), size=22, bg_alpha=0, bold=True)
    w2, _ = _label_patch(subtitle, 15, (180, 180, 180), BLACK, 0, False).shape[1::-1]
    draw_label(frame, subtitle, ((width - w2) // 2, height // 2 + 8), size=15, fg=(180, 180, 180), bg_alpha=0)
    return frame
