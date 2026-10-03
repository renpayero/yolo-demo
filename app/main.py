"""Servidor web de la demo: UI, video en vivo (MJPEG), API y WebSocket de estado."""
from __future__ import annotations

import asyncio
import logging
import os
import signal
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.capture import FrameSource
from app.config import load_config
from app.dashboard_api import router as dashboard_router
from app.detector import UltralyticsPersonDetector
from app.overlay import resize_for_stream
from app.persons_api import router as persons_router
from app.pipeline import FrameHub, Pipeline
from app.recorder import Recorder
from app.storage import Storage
from app.zones_api import router as zones_router

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "/srv/config.yaml"))
DB_PATH = Path(os.environ.get("DB_PATH", "/data/vision.db"))
STATIC_DIR = Path(__file__).parent / "static"
STATE_PUSH_INTERVAL_S = 0.25
NO_CACHE = {"Cache-Control": "no-store"}

log = logging.getLogger(__name__)


def _notify_on_shutdown_signal(event: asyncio.Event) -> None:
    """Marca `event` apenas llega la señal de apagado, antes de que uvicorn espere a las conexiones abiertas.

    El stream de video y el WebSocket son infinitos: si no terminan solos, uvicorn los cancela a la fuerza
    (y lo registra como error). Se encadena con el handler de uvicorn, que sigue haciendo lo suyo.
    """
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous = signal.getsignal(sig)

        def handler(signum: int, frame: object, previous: object = previous) -> None:
            loop.call_soon_threadsafe(event.set)
            if callable(previous):
                previous(signum, frame)

        signal.signal(sig, handler)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app.state.shutting_down = asyncio.Event()
    _notify_on_shutdown_signal(app.state.shutting_down)
    cfg = load_config(CONFIG_PATH)
    storage = Storage(DB_PATH)
    report = storage.recover_unclosed(cfg.camera.id)
    if report.sessions or report.visits or report.alerts:
        log.warning(
            "La app no se cerró bien la vez anterior: se cerraron %d sesión(es), %d visita(s) y %d alerta(s) abiertas",
            report.sessions, report.visits, report.alerts,
        )
    recorder = Recorder(storage.path, cfg.camera.id)
    source = FrameSource(cfg.camera.source)
    pipeline = Pipeline(cfg, source, lambda: UltralyticsPersonDetector(cfg.detector), storage, recorder)
    recorder.start()
    source.start()
    pipeline.start()
    app.state.cfg, app.state.storage, app.state.source, app.state.pipeline = cfg, storage, source, pipeline
    log.info("Demo lista en http://localhost:8000 (fuente: %s)", source.display_source)
    yield
    pipeline.stop()   # cierra visitas, alertas y la sesión...
    recorder.stop()   # ...y recién entonces se termina de escribir todo
    source.stop()


app = FastAPI(title="Visión en planta — demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(zones_router)
app.include_router(dashboard_router)
app.include_router(persons_router)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


@app.get("/api/health")
def health(request: Request) -> dict:
    state = request.app.state.pipeline.state
    return {"ok": state["phase"] != "error", "phase": state["phase"], "camera_connected": state["camera"]["connected"]}


@app.get("/api/state")
def live_state(request: Request) -> dict:
    return request.app.state.pipeline.state


async def _mjpeg(hub: FrameHub, fps: int, shutting_down: asyncio.Event) -> AsyncIterator[bytes]:
    last_seq = -1
    poll = 1.0 / (2 * max(1, fps))
    while not shutting_down.is_set():
        seq, jpeg = hub.latest()
        if jpeg is not None and seq != last_seq:
            last_seq = seq
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%b\r\n" % (len(jpeg), jpeg)
        await asyncio.sleep(poll)


@app.get("/stream.mjpg", include_in_schema=False)
def stream(request: Request) -> StreamingResponse:
    pipeline: Pipeline = request.app.state.pipeline
    return StreamingResponse(
        _mjpeg(pipeline.hub, request.app.state.cfg.stream.fps, request.app.state.shutting_down),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=NO_CACHE,
    )


@app.get("/api/snapshot.jpg")
def snapshot(request: Request) -> Response:
    """Frame crudo (sin dibujos), del mismo tamaño que el stream. Lo usa el editor de zonas."""
    frame = request.app.state.source.latest()
    if frame is None:
        raise HTTPException(status_code=503, detail="Todavía no hay video de la cámara")
    view, _ = resize_for_stream(frame.image, request.app.state.cfg.stream.max_width)
    ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo codificar la imagen")
    return Response(buf.tobytes(), media_type="image/jpeg", headers=NO_CACHE)


@app.websocket("/ws")
async def ws_state(websocket: WebSocket) -> None:
    await websocket.accept()
    pipeline: Pipeline = websocket.app.state.pipeline
    shutting_down: asyncio.Event = websocket.app.state.shutting_down
    try:
        while not shutting_down.is_set():
            await websocket.send_json(pipeline.state)
            await asyncio.sleep(STATE_PUSH_INTERVAL_S)
        await websocket.close(code=1012)  # "service restart": el navegador reconecta solo
    except (WebSocketDisconnect, RuntimeError):
        pass
