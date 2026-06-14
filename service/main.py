"""FastAPI app on the local SQLite + precomputed-descriptor engine."""

import gc
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from . import config
from .auth import require_api_key
from .debug import dbg
from .manage import add_image, replace_image, forget_image
from .search import fabric_search, unified_cloth_search
from .store import STORE, list_skus


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        stats = STORE.load(str(config.SQLITE_PATH), config.MODEL_TAG)
        print(f"[startup] index loaded: {stats}")
    except Exception as e:
        print(f"[startup] index load skipped: {e}")
    yield


app = FastAPI(title="poshak image-search", version="0.2.0", lifespan=lifespan)

_STATIC = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


def _page(name: str) -> HTMLResponse:
    return HTMLResponse((_STATIC / name).read_text(encoding="utf-8"))


@app.get("/ui", response_class=HTMLResponse)
def ui_page():
    return _page("ui.html")


@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    return _page("admin.html")


def _safe_name(name: str) -> str:
    if not name or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="invalid photo_name")
    return name


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


@app.post("/ingest", dependencies=[Depends(require_api_key)])
async def ingest(image: UploadFile = File(...), sku_id: int = Form(...),
                 designno: str = Form(""), item: str = Form(...),
                 photo_name: str = Form(...)):
    photo_name = _safe_name(photo_name)
    data = await image.read()
    res = add_image(sku_id=sku_id, designno=designno, item=item,
                    photo_name=photo_name, data=data)
    return JSONResponse(res, status_code=202)


@app.post("/replace", dependencies=[Depends(require_api_key)])
async def replace(image: UploadFile = File(...), photo_name: str = Form(...),
                  kind: str = Form(...)):
    photo_name = _safe_name(photo_name)
    if kind not in ("original", "rmbg"):
        raise HTTPException(status_code=400, detail="kind must be original or rmbg")
    data = await image.read()
    return replace_image(photo_name, kind=kind, data=data)


@app.post("/forget", dependencies=[Depends(require_api_key)])
def forget(photo_name: str = Form(...)):
    photo_name = _safe_name(photo_name)
    return {"status": "ok", "photo_name": photo_name, **forget_image(photo_name)}


@app.get("/skus", dependencies=[Depends(require_api_key)])
def skus(page: int = 1, page_size: int = 50):
    return list_skus(str(config.SQLITE_PATH), config.MODEL_TAG, page, page_size)


@app.get("/img/original/{name}")
def img_original(name: str):
    name = _safe_name(name)
    path = config.CORPUS_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(path))


@app.get("/img/rmbg/{name}")
def img_rmbg(name: str):
    name = _safe_name(name)
    path = config.RMBG_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(path))
