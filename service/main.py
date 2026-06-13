"""FastAPI app.

Endpoints:
  GET  /health       — model + index status
  POST /embed        — single-image embed (used by indexers)
  POST /embed-batch  — batched embed (used by indexers)
  POST /search       — full pipeline (rembg + dual cloth / single fabric,
                       SigLIP-coarse → top-40 → SIFT rerank → fuse)
  POST /reload       — re-read stock_image_vectors into STORE (after reindex)

Run:
    uvicorn service.main:app --host 0.0.0.0 --port 8088
"""

import asyncio
import gc
from contextlib import asynccontextmanager
from typing import List

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from fastapi.responses import JSONResponse

from . import config
from .debug import dbg, timed
from .index_store import STORE
from .ingest import INGEST_POOL, forget as _forget, ingest_async
from .loader import load_from_mysql
from .model import device, siglip_embed
from .search import fabric_search, unified_cloth_search


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Best-effort: a missing DB on first boot shouldn't crash the
    # container (so /embed still works for the indexer to populate it).
    try:
        stats = load_from_mysql()
        print(f"[startup] index loaded: {stats}")
    except Exception as e:
        print(f"[startup] index load skipped: {e}")
    yield


app = FastAPI(title="poshak image-search service", version="0.1.0", lifespan=lifespan)

_request_count = 0


def _maybe_gc() -> None:
    global _request_count
    _request_count += 1
    if _request_count % config.GC_EVERY_N_REQUESTS == 0:
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def _request_count_force_gc() -> None:
    """Bump the counter without the cadence check — /search already
    called gc.collect() itself; this just keeps cuda cache flushing
    aligned with the periodic cadence."""
    global _request_count
    _request_count += 1
    if _request_count % config.GC_EVERY_N_REQUESTS == 0 and device == "cuda":
        torch.cuda.empty_cache()


def _decode_rgb(data: bytes) -> np.ndarray:
    """Bytes → uint8 RGB ndarray. Matches the prototype's load_image
    (cv2.imread then BGR→RGB), but reads from memory instead of disk."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="cannot decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _entry_to_result(e: dict) -> dict:
    """Shape an index entry into a JSON-safe result dict for HTTP.

    IMPORTANT: `filename` here is the name PHP feeds into
    `<img src="uploads/...">` on the web host. For cloth_bg entries
    `e["filename"]` is the rembg PNG name (which lives in /corpus/data/
    on AWS — not in the web host's uploads/). Always promote
    `json_filename` (the original JPEG name) so the browser request
    resolves against the web host's filesystem."""
    return {
        "sku_id": e.get("sku_id"),
        "designno": e.get("designno"),
        "item": e.get("item"),
        "filename": e.get("json_filename") or e.get("filename"),
        "json_filename": e.get("json_filename") or e.get("filename"),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "model_tag": config.MODEL_TAG,
        "embed_dim": config.EMBED_DIM,
        "index": STORE.stats(),
        "index_ready": STORE.is_ready(),
    }


@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    data = await file.read()
    try:
        img = _decode_rgb(data)
        vec = siglip_embed(img).tolist()
    finally:
        del data
        _maybe_gc()
    return {"vector": vec, "model_tag": config.MODEL_TAG}


@app.post("/embed-batch")
async def embed_batch(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="no files")
    results = []
    try:
        for f in files:
            data = await f.read()
            img = _decode_rgb(data)
            results.append({
                "filename": f.filename,
                "vector": siglip_embed(img).tolist(),
            })
            del data
    finally:
        _maybe_gc()
    return {"results": results, "model_tag": config.MODEL_TAG}


@app.post("/search")
async def search(
    image: UploadFile = File(...),
    category: str = Form(...),
    top_k: int = Form(5),
):
    """The multistep pipeline:

        cloth   → rembg → SigLIP coarse (BG idx + Orig idx)
                → SIFT rerank each → fuse → SKU rank
        fabric  → SigLIP coarse (Fabric idx)
                → SIFT rerank → SKU rank
    """
    if not STORE.is_ready():
        dbg("/search REJECTED: index not loaded")
        raise HTTPException(
            status_code=503,
            detail="index not loaded yet — run reindex.py first",
        )
    if category not in ("cloth", "fabric"):
        dbg(f"/search REJECTED: bad category={category!r}")
        raise HTTPException(
            status_code=400, detail="category must be 'cloth' or 'fabric'"
        )

    # Async-read the upload so the event loop isn't blocked on the
    # network buffer.
    data = await image.read()
    dbg(f"/search RECEIVED: category={category} top_k={top_k} bytes={len(data)} "
        f"filename={image.filename!r}")

    # Heavy work (decode + pipeline) runs in a worker thread so the
    # event loop stays free to answer /health during a long search.
    def _run_blocking():
        img = _decode_rgb(data)
        dbg(f"/search decoded image: shape={img.shape} dtype={img.dtype}")
        if category == "cloth":
            return unified_cloth_search(img, STORE, top_k=top_k)
        return fabric_search(img, STORE, top_k=top_k)

    try:
        with timed("/search TOTAL"):
            loop = asyncio.get_event_loop()
            # Default executor (starlette's) — leaves PIPELINE_POOL
            # entirely free for pipelines A and B to run truly parallel.
            ranked = await loop.run_in_executor(None, _run_blocking)
    finally:
        del data
        # Always gc after a search — pipelines hold transient SIFT
        # buffers + image arrays that otherwise wait for the periodic
        # gc cadence. On a 4GB cap this matters.
        gc.collect()
        _request_count_force_gc()
    dbg(f"/search RESPONDING: {len(ranked)} results")

    # Flatten best_entry → JSON-safe shape for the PHP client.
    out = []
    for r in ranked:
        best = r.get("best_entry") or {}
        out.append({
            **{k: v for k, v in r.items() if k != "best_entry"},
            "best_entry": _entry_to_result(best),
        })
    return {
        "status": "success",
        "method": "SIFT",
        "category": category,
        "total_found": len(out),
        "results": out,
        "model_tag": config.MODEL_TAG,
    }


@app.post("/reload")
def reload():
    """Re-read stock_image_vectors → STORE. Call after a reindex."""
    try:
        stats = load_from_mysql()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"reload failed: {e}")
    return {"status": "reloaded", "index": stats}


def _safe_photo_name(photo_name: str) -> str:
    """Reject paths that try to break out of CORPUS_DIR (../, slashes).
    photo_name is just a basename; anything else is a misconfigured
    caller."""
    if not photo_name or "/" in photo_name or "\\" in photo_name or ".." in photo_name:
        raise HTTPException(status_code=400, detail="invalid photo_name")
    return photo_name


@app.post("/ingest-photo")
async def ingest_photo(
    image: UploadFile = File(...),
    stock_id: int = Form(...),
    photo_name: str = Form(...),
):
    """Save the photo to /corpus/uploads/, return 202 immediately, do
    the embed + DB insert + STORE update in a background thread.

    PHP-side flow: after admin's `move_uploaded_file` + `INSERT INTO
    stock`, PHP forwards the same bytes here. The fire-and-forget
    return means admin save UX stays snappy."""
    photo_name = _safe_photo_name(photo_name)

    data = await image.read()
    config.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.CORPUS_DIR / photo_name
    out_path.write_bytes(data)
    dbg(f"/ingest-photo SAVED: stock_id={stock_id} photo={photo_name} "
        f"({len(data)} bytes)")

    INGEST_POOL.submit(ingest_async, stock_id, photo_name)

    return JSONResponse(
        {"status": "accepted", "photo_name": photo_name},
        status_code=202,
    )


@app.post("/forget-photo")
def forget_photo(photo_name: str = Form(...)):
    """Remove file + DB row + STORE entries for this photo. Sync.

    Called by PHP when admin deletes a stock entry (which unlinks the
    file on the web host). 404 is NOT returned if the photo wasn't indexed
    (e.g., wasn't cloth/fabric) — `db_deleted: 0` in the response is
    the signal."""
    photo_name = _safe_photo_name(photo_name)
    dbg(f"/forget-photo RECEIVED: {photo_name}")
    result = _forget(photo_name)
    return {"status": "ok", "photo_name": photo_name, **result}
