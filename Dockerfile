# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

# Índice de PyTorch: CUDA por defecto; compose.cpu.yaml lo cambia a /whl/cpu.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.14.1 torchvision==0.29.1 --index-url ${TORCH_INDEX}

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Va después de los pip install: cambiar estas variables no obliga a reinstalar PyTorch.
ENV YOLO_OFFLINE=1 \
    YOLO_AUTOINSTALL=0 \
    YOLO_CONFIG_DIR=/tmp \
    OPENCV_LOG_LEVEL=ERROR \
    OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|fflags;nobuffer"

# Pesos de los modelos dentro de la imagen: el contenedor no necesita internet para arrancar.
# Detección de personas (s con GPU, n con CPU) y pose para el análisis de posturas.
ADD --chmod=644 https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt /models/yolo26s.pt
ADD --chmod=644 https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt /models/yolo26n.pt
ADD --chmod=644 https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt /models/yolo26s-pose.pt
ADD --chmod=644 https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt /models/yolo26n-pose.pt

# Identificación facial (opcional, apagada por defecto): OpenCV Zoo, licencias MIT (YuNet) y Apache-2.0 (SFace).
ADD --chmod=644 --checksum=sha256:ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0 \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx \
    /models/face_detection_yunet_2026may.onnx
ADD --chmod=644 --checksum=sha256:0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79 \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx \
    /models/face_recognition_sface_2021dec.onnx

COPY config.yaml ./config.yaml
COPY tests ./tests
COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "3"]
