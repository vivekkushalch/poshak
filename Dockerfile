# poshak image-search service — CPU build.
# Swap base + torch wheel for nvidia/cuda when GPU is available.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    TORCH_HOME=/models/torch \
    TRANSFORMERS_CACHE=/models/transformers

WORKDIR /app

# System deps: libgomp for torch; libjpeg/zlib for Pillow; libgl/libglib
# for opencv-python-headless still wants a couple of base libs.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgomp1 \
        libjpeg62-turbo \
        zlib1g \
        libglib2.0-0 \
        ca-certificates \
        curl \
 && rm -rf /var/lib/apt/lists/*

# CPU-only torch + torchvision installed together so their ABIs match.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY service ./service
COPY scripts ./scripts
COPY sql ./sql

# Warm the prototype's exact model into the image so the first request
# isn't a ~400MB HuggingFace download. Must match service/config.MODEL_HF_ID.
RUN python -c "from transformers import AutoProcessor, AutoModel; \
AutoProcessor.from_pretrained('google/siglip-base-patch16-224'); \
AutoModel.from_pretrained('google/siglip-base-patch16-224')"

EXPOSE 8088

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8088/health || exit 1

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8088"]
