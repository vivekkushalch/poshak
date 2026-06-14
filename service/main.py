"""FastAPI app on the local SQLite + precomputed-descriptor engine."""

import gc

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from contextlib import asynccontextmanager

from . import config
from .auth import require_api_key
from .debug import dbg
from .search import fabric_search, unified_cloth_search
from .store import STORE


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        stats = STORE.load(str(config.SQLITE_PATH), config.MODEL_TAG)
        print(f"[startup] index loaded: {stats}")
    except Exception as e:
        print(f"[startup] index load skipped: {e}")
    yield


app = FastAPI(title="poshak image-search", version="0.2.0", lifespan=lifespan)


def _decode_rgb(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="cannot decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _entry_to_result(e: dict) -> dict:
    return {
        "sku_id": e.get("sku_id"), "designno": e.get("designno"),
        "item": e.get("item"),
        "filename": e.get("json_filename") or e.get("filename"),
        "json_filename": e.get("json_filename") or e.get("filename"),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_tag": config.MODEL_TAG,
            "embed_dim": config.EMBED_DIM, "index": STORE.stats(),
            "index_ready": STORE.is_ready()}


@app.get("/config")
def get_config():
    return {"image_base_url": config.IMAGE_BASE_URL,
            "uploads_prefix": "uploads", "rmbg_prefix": "rmbg"}


@app.post("/search", dependencies=[Depends(require_api_key)])
async def search(image: UploadFile = File(...), category: str = Form(...),
                 top_k: int = Form(5)):
    if not STORE.is_ready():
        raise HTTPException(status_code=503, detail="index not loaded yet")
    if category not in ("cloth", "fabric"):
        raise HTTPException(status_code=400, detail="category must be cloth or fabric")
    data = await image.read()
    try:
        img = _decode_rgb(data)
        if category == "cloth":
            ranked = unified_cloth_search(img, STORE, top_k=top_k)
        else:
            ranked = fabric_search(img, STORE, top_k=top_k)
    finally:
        del data
        gc.collect()
    out = []
    for r in ranked:
        best = r.get("best_entry") or {}
        out.append({**{k: v for k, v in r.items() if k != "best_entry"},
                    "best_entry": _entry_to_result(best)})
    return {"status": "success", "method": "SIFT", "category": category,
            "total_found": len(out), "results": out, "model_tag": config.MODEL_TAG}
