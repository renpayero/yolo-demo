"""Lectura de la fuente de video en un hilo propio.

Conserva solo el último frame (los viejos se descartan para no acumular latencia) y, si la fuente se cae,
reintenta con backoff exponencial sin detener la aplicación.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

OPEN_TIMEOUT_MS = 5000
READ_TIMEOUT_MS = 5000
BACKOFF_START_S = 1.0
# Reintentar contra una fuente caída es barato; un tope bajo hace que el video vuelva rápido
# (por ejemplo, si se levanta Docker antes que el puente de la webcam).
BACKOFF_MAX_S = 5.0


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    ts: float          # epoch UTC del momento de captura
    seq: int           # correlativo, crece con cada frame
    generation: int    # cambia cada vez que se (re)abre la fuente


@dataclass(frozen=True)
class SourceStatus:
    connected: bool
    reconnects: int
    last_error: str | None
    width: int
    height: int


def mask_credentials(url: str) -> str:
    """Oculta la contraseña de una URL tipo rtsp://usuario:clave@host para logs y UI."""
    return re.sub(r"//([^:/@]+):([^@/]+)@", r"//\1:***@", url)


def _is_local_file(source: str) -> bool:
    return "://" not in source and not source.isdigit()


class FrameSource:
    def __init__(self, source: str) -> None:
        self._source = source
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._latest: Frame | None = None
        self._seq = 0
        self._generation = 0
        self._connected = False
        self._reconnects = 0
        self._last_error: str | None = None
        self._size = (0, 0)

    @property
    def display_source(self) -> str:
        return mask_credentials(self._source)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=OPEN_TIMEOUT_MS / 1000 + 2)

    def wait_frame(self, after_seq: int, timeout: float) -> Frame | None:
        """Devuelve el último frame si es más nuevo que `after_seq`; si no llega ninguno en `timeout`, None."""
        with self._cond:
            self._cond.wait_for(
                lambda: self._stop.is_set() or (self._latest is not None and self._latest.seq > after_seq),
                timeout=timeout,
            )
            latest = self._latest
        return latest if latest is not None and latest.seq > after_seq else None

    def latest(self) -> Frame | None:
        with self._cond:
            return self._latest

    def status(self) -> SourceStatus:
        with self._cond:
            return SourceStatus(self._connected, self._reconnects, self._last_error, *self._size)

    def _open(self) -> cv2.VideoCapture | None:
        if self._source.isdigit():  # índice de webcam local (solo fuera de Docker)
            cap = cv2.VideoCapture(int(self._source))
        else:
            params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS]
            cap = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG, params)
        if cap.isOpened():
            return cap
        cap.release()
        return None

    def _run(self) -> None:
        backoff = BACKOFF_START_S
        is_file = _is_local_file(self._source)
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                self._set_disconnected("No se pudo abrir la fuente de video")
                log.warning("No se pudo abrir %s; reintento en %.0f s", self.display_source, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX_S)
                continue

            backoff = BACKOFF_START_S
            first_open = self._generation == 0
            self._set_connected(count_reconnect=not is_file)
            if first_open or not is_file:
                log.info("Fuente de video abierta: %s", self.display_source)
            # Un archivo se lee al ritmo de sus fps (si no, se consumiría en segundos) y vuelve a empezar al final.
            interval = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 25.0) if is_file else 0.0
            try:
                self._read_loop(cap, interval)
            finally:
                cap.release()

            if self._stop.is_set():
                break
            if is_file:
                continue
            self._set_disconnected("La fuente dejó de enviar video")
            log.warning("Se perdió la fuente %s; reintento en %.0f s", self.display_source, backoff)
            self._stop.wait(backoff)

    def _read_loop(self, cap: cv2.VideoCapture, interval: float) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            ok, image = cap.read()
            if not ok or image is None:
                return
            with self._cond:
                self._seq += 1
                self._latest = Frame(image, time.time(), self._seq, self._generation)
                self._size = (image.shape[1], image.shape[0])
                self._cond.notify_all()
            if interval:
                self._stop.wait(max(0.0, interval - (time.monotonic() - started)))

    def _set_connected(self, count_reconnect: bool) -> None:
        with self._cond:
            if self._generation > 0 and count_reconnect:
                self._reconnects += 1
            self._generation += 1
            self._connected = True
            self._last_error = None

    def _set_disconnected(self, error: str) -> None:
        with self._cond:
            self._connected = False
            self._last_error = error
