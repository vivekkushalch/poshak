# syntax=docker/dockerfile:1
# ---- builder: resolve the locked environment with uv ------------------------
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
# deps first so code edits don't invalidate the (large) torch layer
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra s3 --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra s3 --no-editable

# ---- runtime -----------------------------------------------------------------
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" HF_HOME=/models U2NET_HOME=/models/u2net
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --uid 1000 --create-home poshak \
 && mkdir -p /data /corpus /models && chown poshak:poshak /data /corpus /models
COPY --from=builder /app/.venv /app/.venv
USER poshak
# Bake the public SigLIP weights so the first request isn't a 800 MB download.
RUN python -c "from transformers import AutoModel, AutoProcessor; \
m='google/siglip-base-patch16-224'; AutoProcessor.from_pretrained(m); AutoModel.from_pretrained(m)"
EXPOSE 8088
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8088/health/live || exit 1
CMD ["poshak", "serve"]
