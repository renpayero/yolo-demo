from pathlib import Path

import pytest

from app.config import load_config


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_file_uses_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("CAMERA_SOURCE", raising=False)
    cfg = load_config(tmp_path / "no-existe.yaml")
    assert cfg.camera.source == "rtsp://mediamtx:8554/webcam"
    assert cfg.detector.model == "auto"
    assert cfg.stream.fps == 15


def test_partial_sections_keep_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("CAMERA_SOURCE", raising=False)
    cfg = load_config(write(tmp_path, "camera:\n  name: Lijado\ndetector:\n  confidence: 0.6\n"))
    assert cfg.camera.name == "Lijado"
    assert cfg.camera.id == "cam1"
    assert cfg.detector.confidence == 0.6
    assert cfg.detector.image_size == 640


def test_env_overrides_source(tmp_path, monkeypatch):
    monkeypatch.setenv("CAMERA_SOURCE", "rtsp://mediamtx:8554/demo")
    cfg = load_config(write(tmp_path, "camera:\n  source: rtsp://otra/cosa\n"))
    assert cfg.camera.source == "rtsp://mediamtx:8554/demo"


def test_empty_env_does_not_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CAMERA_SOURCE", "")  # compose pasa "" cuando no se define
    cfg = load_config(write(tmp_path, "camera:\n  source: rtsp://cam/1\n"))
    assert cfg.camera.source == "rtsp://cam/1"


def test_invalid_anchor_is_reported(tmp_path):
    with pytest.raises(ValueError, match="anchor"):
        load_config(write(tmp_path, "camera:\n  anchor: cabeza\n"))


def test_unknown_key_is_reported(tmp_path):
    with pytest.raises(ValueError, match="clave.*'camera'.*sorce"):
        load_config(write(tmp_path, "camera:\n  sorce: rtsp://x\n"))
