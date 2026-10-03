"""Bucle principal: frame → detección/tracking → zonas y ocupación → alertas → registro → overlay → navegador.

Corre en un hilo propio, que es el único que usa el modelo y los motores de ocupación y alertas. La web solo
lee resultados ya calculados (el último JPEG en `FrameHub` y una foto inmutable del estado en `state`) y le
pide cambios al pipeline con banderas que este revisa entre frame y frame. Lo que hay que guardar se encola
en `Recorder`, que escribe en SQLite desde otro hilo sin frenar el video.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from app.alerts import Alert, AlertChange, AlertEngine
from app.capture import Frame, FrameSource
from app.config import AppConfig
from app.detector import PersonDetector
from app.domain import Detection
from app.faces import ENROLL_TIMEOUT_S, EnrollmentError, IdentityChange, Recognizer
from app.occupancy import OccupancyEngine, ZoneEvent, ZoneStatus
from app.overlay import (
    draw_alert_banner,
    draw_detections,
    draw_identification_badge,
    draw_status_bar,
    draw_zones,
    placeholder_frame,
    resize_for_stream,
)
from app.recorder import Recorder
from app.storage import Storage
from app.zones import Point, anchor_point

log = logging.getLogger(__name__)

NO_SIGNAL_AFTER_S = 1.5        # sin frames durante este tiempo se muestra "sin señal"
PLACEHOLDER_INTERVAL_S = 0.5
ERROR_LOG_INTERVAL_S = 10.0
HEARTBEAT_S = 5.0              # cada cuánto se marca en la base que el sistema sigue mirando (pérdida máx. ante un corte)
RECENT_ALERTS = 10


class FrameHub:
    """Último JPEG anotado. Se codifica una vez y lo comparten todos los clientes del stream."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._jpeg: bytes | None = None

    def publish(self, jpeg: bytes) -> None:
        with self._lock:
            self._seq += 1
            self._jpeg = jpeg

    def latest(self) -> tuple[int, bytes | None]:
        with self._lock:
            return self._seq, self._jpeg


class FpsMeter:
    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._times.append(time.monotonic())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


@dataclass
class _LoopState:
    last_frame_at: float           # monotonic
    last_frame_ts: float = 0.0     # epoch del último frame (fin de la sesión si se corta el video)
    last_seq: int = 0
    generation: int = -1
    last_placeholder_at: float = 0.0
    next_encode_at: float = 0.0
    next_heartbeat_at: float = 0.0


def _alert_dict(alert: Alert, resolved_ts: float | None = None) -> dict[str, Any]:
    return {
        "id": alert.uid,
        "kind": alert.kind,
        "zone_id": alert.zone_id,
        "zone_name": alert.zone_name,
        "track_id": alert.track_id,
        "ts": alert.ts,
        "message": alert.message,
        "resolved_ts": resolved_ts,
    }


class Pipeline:
    def __init__(
        self,
        cfg: AppConfig,
        source: FrameSource,
        detector_factory: Callable[[], PersonDetector],
        storage: Storage,
        recorder: Recorder | None = None,
        recognizer: Recognizer | None = None,
    ) -> None:
        self._cfg = cfg
        self._source = source
        self._detector_factory = detector_factory
        self._storage = storage
        self._recorder = recorder
        self._stop = threading.Event()
        self._reload_zones = threading.Event()
        self._commands: queue.Queue[Callable[[], None]] = queue.Queue()   # pedidos de la API, en orden
        self._recognizer = recognizer or Recognizer(cfg.identification)
        self._identification = cfg.identification.enabled
        self._person_names = storage.person_names()
        self._recognizer.set_gallery(storage.load_embeddings())
        self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self.hub = FrameHub()
        self._detector: PersonDetector | None = None
        zones = storage.list_zones(cfg.camera.id)
        self._zone_names = {z.id: z.name for z in zones}  # incluye zonas ya borradas, para logs y registro
        self._engine = OccupancyEngine(
            zones,
            enter_debounce_s=cfg.occupancy.enter_debounce_s,
            exit_grace_s=cfg.occupancy.exit_grace_s,
            start_ts=time.time(),
        )
        self._alerts = AlertEngine()
        self._recent_alerts: deque[dict[str, Any]] = deque(maxlen=RECENT_ALERTS)
        self._monitoring_since: float | None = None   # desde cuándo hay video sin cortes
        self._fps = FpsMeter()
        self._last_error_log = 0.0
        self._state: dict[str, Any] = self._build_state(phase="starting", people=0)

    @property
    def state(self) -> dict[str, Any]:
        return self._state  # se reemplaza entero, nunca se modifica: leerlo desde otro hilo es seguro

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def request_zones_reload(self) -> None:
        """Lo llama la API al cambiar zonas; el pipeline las relee antes del próximo frame."""
        self._reload_zones.set()

    # ---------- Identificación: pedidos desde la API (se ejecutan en el hilo del pipeline) ----------

    @property
    def identification_enabled(self) -> bool:
        return self._identification

    def set_identification(self, enabled: bool) -> None:
        def command() -> None:
            self._identification = enabled
            if not enabled:
                self._recognizer.clear()  # todos vuelven a ser anónimos en el acto
            log.info("Identificación de personas %s", "ACTIVADA" if enabled else "desactivada")

        self._commands.put(command)

    def reload_persons(self, forget_person_id: int | None = None) -> None:
        def command() -> None:
            if forget_person_id is not None:
                self._recognizer.forget_person(forget_person_id)
            self._person_names = self._storage.person_names()
            self._recognizer.set_gallery(self._storage.load_embeddings())

        self._commands.put(command)

    def enroll(self) -> np.ndarray:
        """Captura huellas de la (única) cara frente a la cámara. Bloquea unos segundos; lanza EnrollmentError."""
        future: Future = Future()
        self._commands.put(lambda: self._recognizer.start_enrollment(future, time.monotonic()))
        try:
            return future.result(timeout=ENROLL_TIMEOUT_S + 4)
        except FutureTimeout:
            future.cancel()
            raise EnrollmentError("El registro tardó demasiado; probá de nuevo.") from None

    def match_existing(self, samples: np.ndarray) -> tuple[int | None, float]:
        return self._recognizer.match_existing(samples)

    def _run_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            command()

    # ---------- Bucle ----------

    def _run(self) -> None:
        self._publish_placeholder("Cargando el modelo de detección…", "La primera vez puede tardar unos segundos")
        try:
            self._detector = self._detector_factory()
        except Exception as exc:  # sin detector no hay demo: se informa en la UI y en el log
            log.exception("No se pudo cargar el detector")
            self._state = self._build_state(phase="error", people=0, error=str(exc))
            self._publish_placeholder("No se pudo cargar el modelo", str(exc)[:120])
            return

        loop = _LoopState(last_frame_at=time.monotonic())
        while not self._stop.is_set():
            try:
                self._step(loop)
            except Exception:  # un error puntual no puede congelar el video: se registra y se sigue
                self._log_error_throttled("Error inesperado en el pipeline; sigue con el próximo frame")
                self._stop.wait(0.1)
        self._shutdown()

    def _step(self, loop: _LoopState) -> None:
        self._run_commands()
        self._apply_zone_reload()
        frame = self._source.wait_frame(loop.last_seq, timeout=0.5)
        now = time.monotonic()
        if frame is None:
            self._recognizer.step_enrollment(None, now)
            self._step_without_video(loop, now)
            return

        loop.last_seq, loop.last_frame_at, loop.last_frame_ts = frame.seq, now, frame.ts
        if frame.generation != loop.generation:
            loop.generation = frame.generation
            self._detector.reset()
        self._keep_monitoring(frame.ts, loop, now)

        try:
            detections = self._detector.track(frame.image)
        except Exception:
            self._log_error_throttled("Falló la detección en un frame")
            return
        self._fps.tick()

        anchors = self._anchors(detections, frame)
        self._process_events(frame.ts, self._engine.update(frame.ts, anchors))
        if self._identification:
            for change in self._recognizer.identify(frame.image, detections, frame.ts):
                self._on_identity_change(change, frame.ts)
        self._recognizer.step_enrollment(frame.image, now)
        statuses = self._engine.snapshot()

        if now >= loop.next_encode_at:
            loop.next_encode_at = now + 1.0 / max(1, self._cfg.stream.fps)
            self._publish_annotated(frame, detections, anchors, statuses)
        self._state = self._build_state(phase="running", people=len(detections), statuses=statuses)

    def _step_without_video(self, loop: _LoopState, now: float) -> None:
        ts = time.time()
        if now - loop.last_frame_at > NO_SIGNAL_AFTER_S:
            self._end_monitoring(loop.last_frame_ts)
            self._state = self._build_state(phase="no_signal", people=0)
            if now - loop.last_placeholder_at > PLACEHOLDER_INTERVAL_S:
                loop.last_placeholder_at = now
                self._publish_placeholder("Sin señal de video — reconectando…", self._source.display_source)
        # Sin video no se sabe si hay alguien: las visitas abiertas se cierran al vencer la gracia.
        self._process_events(ts, self._engine.update(ts, {}))

    def _shutdown(self) -> None:
        ts = time.time()
        self._process_events(ts, self._engine.close_all())
        self._handle_alert_changes(self._alerts.resolve_all(ts))
        self._end_monitoring(ts)

    def _anchors(self, detections: list[Detection], frame: Frame) -> dict[int, Point]:
        h, w = frame.image.shape[:2]
        mode = self._cfg.camera.anchor
        return {d.track_id: anchor_point(d.bbox, w, h, mode) for d in detections}

    def _apply_zone_reload(self) -> None:
        if not self._reload_zones.is_set():
            return
        self._reload_zones.clear()
        zones = self._storage.list_zones(self._cfg.camera.id)
        ts = time.time()
        self._zone_names.update({z.id: z.name for z in zones})
        self._process_events(ts, self._engine.set_zones(zones, ts))
        log.info("Zonas recargadas: %s", ", ".join(z.name for z in zones) or "ninguna")

    # ---------- Sesiones de monitoreo (períodos con video) ----------

    def _keep_monitoring(self, ts: float, loop: _LoopState, now: float) -> None:
        if self._monitoring_since is None:
            self._monitoring_since = ts
            loop.next_heartbeat_at = now + HEARTBEAT_S
            if self._recorder:
                self._recorder.session_start(ts)
            log.info("Comienza el monitoreo (hay video)")
        elif now >= loop.next_heartbeat_at:
            loop.next_heartbeat_at = now + HEARTBEAT_S
            if self._recorder:
                self._recorder.session_heartbeat(ts)

    def _end_monitoring(self, ts: float) -> None:
        if self._monitoring_since is None:
            return
        self._monitoring_since = None
        if self._recorder:
            self._recorder.session_end(ts)
        log.info("Se interrumpe el monitoreo (sin video)")

    # ---------- Eventos y alertas ----------

    def _process_events(self, ts: float, events: list[ZoneEvent]) -> None:
        """Toda la salida de los motores pasa por acá: log, registro y evaluación de alertas."""
        self._log_events(events)
        for event in events:
            if event.kind == "rebind":  # el tracker cambió el ID: el nombre pasa, en duda hasta ver la cara
                self._recognizer.transfer(event.previous_track_id, event.track_id, ts)
        persons = self._recognizer.assignments()
        if self._recorder and events:
            self._recorder.zone_events(events, self._zone_names, persons)
        changes = self._alerts.evaluate(
            ts, self._engine.snapshot(), events, self._monitoring_since, self._track_names()
        )
        self._handle_alert_changes(changes, persons)

    def _track_names(self) -> dict[int, str]:
        return {
            track_id: self._person_names[person_id]
            for track_id, person_id in self._recognizer.assignments().items()
            if person_id in self._person_names
        }

    def _on_identity_change(self, change: IdentityChange, ts: float) -> None:
        track_id, person_id = change.track_id, change.person_id
        if person_id is None:
            # Otra persona heredó el track: la visita anterior se cierra cuando se vio por última vez a quien
            # tenía el nombre y empieza una nueva, así el tiempo de cada uno queda bien atribuido.
            log.info("Se retira el nombre de la persona %d: su cara ya no coincide (era otra persona)", track_id)
            self._process_events(ts, self._engine.split_track(track_id, change.end_ts, change.start_ts))
            return
        log.info("Identificada: persona %d es %s", track_id, self._person_names.get(person_id, f"#{person_id}"))
        if not self._recorder:
            return
        for status in self._engine.snapshot():  # sus visitas abiertas pasan a tener nombre desde el comienzo
            for occupant in status.occupants:
                if occupant.track_id == track_id:
                    self._recorder.visit_person(occupant.visit_id, person_id)

    def _log_events(self, events: list[ZoneEvent]) -> None:
        for event in events:
            zone = self._zone_names.get(event.zone_id, f"zona {event.zone_id}")
            if event.kind == "enter":
                log.info("Entrada: persona %d → %s", event.track_id, zone)
            elif event.kind == "exit":
                log.info("Salida: persona %d ← %s (%.1f s)", event.track_id, zone, event.duration_s)
            else:
                log.info("Reasignación en %s: persona %d sigue como %d", zone, event.previous_track_id, event.track_id)

    def _handle_alert_changes(self, changes: list[AlertChange], persons: dict[int, int] | None = None) -> None:
        for change in changes:
            if change.action == "raise":
                log.warning("ALERTA: %s", change.alert.message)
            else:
                log.info("Alerta resuelta: %s", change.alert.message)
                self._recent_alerts.appendleft(_alert_dict(change.alert, resolved_ts=change.ts))
        if self._recorder and changes:
            self._recorder.alert_changes(changes, persons)

    # ---------- Salidas: video y estado ----------

    def _publish_annotated(
        self,
        frame: Frame,
        detections: list[Detection],
        anchors: dict[int, Point],
        statuses: list[ZoneStatus],
    ) -> None:
        active = self._alerts.active
        view, scale = resize_for_stream(frame.image, self._cfg.stream.max_width)
        draw_zones(view, statuses, frame.ts, {a.zone_id for a in active})
        draw_detections(
            view, detections, scale, anchors, self._engine.track_zones(),
            {a.track_id for a in active if a.track_id is not None},
            self._track_names(),
        )
        draw_status_bar(view, f"{self._fps.fps:4.1f} fps · {self._detector.device} · {self._detector.model_name}")
        if self._identification:
            draw_identification_badge(view)
        draw_alert_banner(view, [a.message for a in active])
        self._publish_jpeg(view)

    def _publish_placeholder(self, title: str, subtitle: str) -> None:
        width = self._cfg.stream.max_width
        self._publish_jpeg(placeholder_frame(width, width * 9 // 16, title, subtitle))

    def _publish_jpeg(self, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._cfg.stream.jpeg_quality])
        if ok:
            self.hub.publish(buf.tobytes())

    def _build_state(
        self,
        phase: str,
        people: int,
        error: str | None = None,
        statuses: list[ZoneStatus] | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        status = self._source.status()
        if statuses is None:
            statuses = self._engine.snapshot()
        names = self._track_names()
        return {
            "ts": now,
            "phase": phase,
            "error": error,
            "fps": round(self._fps.fps, 1) if phase == "running" else 0.0,
            "people": people,
            "device": getattr(self._detector, "device", None),
            "model": getattr(self._detector, "model_name", None),
            "camera": {
                "id": self._cfg.camera.id,
                "name": self._cfg.camera.name,
                "source": self._source.display_source,
                "anchor": self._cfg.camera.anchor,
                "connected": status.connected,
                "reconnects": status.reconnects,
                "last_error": status.last_error,
                "width": status.width,
                "height": status.height,
            },
            "zones": [
                {
                    "id": s.zone.id,
                    "name": s.zone.name,
                    "kind": s.zone.kind,
                    "color": s.zone.color,
                    "count": len(s.present),
                    "occupants": [
                        {"track_id": o.track_id, "name": names.get(o.track_id), "dwell_s": round(o.dwell_s, 1)}
                        for o in s.present
                    ],
                    "empty_for_s": round(now - s.empty_since, 1) if s.empty_since is not None else None,
                }
                for s in statuses
            ],
            "alerts": {
                "active": [_alert_dict(a) for a in self._alerts.active],
                "recent": list(self._recent_alerts),
            },
            "identification": {
                "enabled": self._identification,
                "persons": len(self._person_names),
                "identified_now": len(names),
            },
        }

    def _log_error_throttled(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_error_log > ERROR_LOG_INTERVAL_S:
            self._last_error_log = now
            log.exception(message)
