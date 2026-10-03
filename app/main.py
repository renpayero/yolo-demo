"""Servidor web de la demo: UI, video en vivo (MJPEG), API y WebSocket de estado."""
from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.capture import FrameSource
from app.config import load_config
from app.detector import UltralyticsPersonDetector
from app.overlay import resize_for_stream
from app.pipeline import FrameHub, Pipeline
from app.storage import Storage
from app.zones_api import router as zones_router

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "/srv/config.yaml"))
DB_PATH = Path(os.environ.get("DB_PATH", "/data/vision.db"))
STATIC_DIR = Path(__file__).parent / "static"
STATE_PUSH_INTERVAL_S = 0.25
NO_CACHE = {"Cache-Control": "no-store"}

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(CONFIG_PATH)
    storage = Storage(DB_PATH)
    source = FrameSource(cfg.camera.source)
    pipeline = Pipeline(cfg, source, lambda: UltralyticsPersonDetector(cfg.detector), storage)
    source.start()
    pipeline.start()
    app.state.cfg, app.state.storage, app.state.source, app.state.pipeline = cfg, storage, source, pipeline
    log.info("Demo lista en http://localhost:8000 (fuente: %s)", source.display_source)
    yield
    pipeline.stop()
    source.stop()


app = FastAPI(title="Visión en planta — demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(zones_router)


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


async def _mjpeg(hub: FrameHub, fps: int) -> AsyncIterator[bytes]:
    last_seq = -1
    poll = 1.0 / (2 * max(1, fps))
    while True:
        seq, jpeg = hub.latest()
        if jpeg is not None and seq != last_seq:
            last_seq = seq
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%b\r\n" % (len(jpeg), jpeg)
        await asyncio.sleep(poll)


@app.get("/stream.mjpg", include_in_schema=False)
def stream(request: Request) -> StreamingResponse:
    pipeline: Pipeline = request.app.state.pipeline
    return StreamingResponse(
        _mjpeg(pipeline.hub, request.app.state.cfg.stream.fps),
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
    try:
        while True:
            await websocket.send_json(pipeline.state)
            await asyncio.sleep(STATE_PUSH_INTERVAL_S)
    except (WebSocketDisconnect, RuntimeError):
        pass
