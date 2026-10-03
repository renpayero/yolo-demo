import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.capture import FrameSource, mask_credentials


@pytest.fixture
def video_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 50, (64, 48))
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    writer.release()
    return path


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_reads_file_and_loops(video_file):
    source = FrameSource(str(video_file))
    source.start()
    try:
        first = source.wait_frame(0, timeout=5)
        assert first is not None
        assert first.image.shape == (48, 64, 3)
        assert source.status().connected
        # 10 frames a 50 fps: en ~1 s tiene que haber dado varias vueltas (cada vuelta, nueva generación).
        assert wait_until(lambda: source.latest().generation >= 3)
        assert source.status().reconnects == 0  # dar la vuelta a un archivo no es una reconexión
    finally:
        source.stop()


def test_wait_frame_only_returns_newer_frames(video_file):
    source = FrameSource(str(video_file))
    source.start()
    try:
        frame = source.wait_frame(0, timeout=5)
        newer = source.wait_frame(frame.seq, timeout=5)
        assert newer is not None and newer.seq > frame.seq
    finally:
        source.stop()


def test_unreachable_source_does_not_crash():
    source = FrameSource("/no/existe.mp4")
    source.start()
    try:
        assert wait_until(lambda: source.status().last_error is not None)
        status = source.status()
        assert not status.connected
        assert source.wait_frame(0, timeout=0.2) is None
        assert source._thread.is_alive()  # sigue reintentando
    finally:
        source.stop()
    assert not source._thread.is_alive()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("rtsp://admin:secreto@10.0.0.5:554/stream1", "rtsp://admin:***@10.0.0.5:554/stream1"),
        ("rtsp://mediamtx:8554/webcam", "rtsp://mediamtx:8554/webcam"),
        ("/media/video.mp4", "/media/video.mp4"),
    ],
)
def test_mask_credentials(url, expected):
    assert mask_credentials(url) == expected
