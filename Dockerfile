# ---- builder: install deps into a venv with uv -----------------------------
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=0
WORKDIR /app
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH"
# CPU-only torch first (from the CPU index), so the rest don't drag CUDA wheels.
RUN uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN uv pip install \
      fastapi "uvicorn[standard]" python-multipart transformers sentencepiece \
      pillow numpy opencv-python-headless rembg onnxruntime httpx tqdm

# ---- runtime: slim Debian + venv + code + baked model ----------------------
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/models TRANSFORMERS_CACHE=/models/transformers
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgomp1 libjpeg62-turbo zlib1g libglib2.0-0 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY service ./service
COPY scripts ./scripts
# Bake the public SigLIP model so first request isn't a download (no PII).
RUN python -c "from transformers import AutoProcessor, AutoModel; \
AutoProcessor.from_pretrained('google/siglip-base-patch16-224'); \
AutoModel.from_pretrained('google/siglip-base-patch16-224')"
EXPOSE 8088
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8088/health || exit 1
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8088"]
