"""Local-first add / replace / forget. Recomputes embeddings + descriptors at
768, writes SQLite, reloads STORE under a lock, appends the change ledger."""

import os
import threading
from datetime import UTC, datetime

import cv2
import numpy as np

from . import config, ledger
from .bg_remove import remove_bg_for_index
from .descriptors import compute_descriptors, load_descriptors, save_descriptors
from .image_io import to_working_res
from .model import siglip_embed
from .store import (
    STORE,
    blob_to_vec,
    connect,
    delete_by_photo,
    init_db,
    resolve_category,
    upsert_image,
)

_LOCK = threading.Lock()


# --- seams (overridden in tests) -------------------------------------------
def _embed(img):
    return siglip_embed(img)


def _desc(img):
    return compute_descriptors(img)


def _bg(img, engine):
    return remove_bg_for_index(img, engine=engine)


def _load_original(photo_name) -> np.ndarray:
    img = cv2.imread(str(config.CORPUS_DIR / photo_name))
    if img is None:
        raise FileNotFoundError(f"original not on disk: {photo_name}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _decode_full(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _already_indexed(photo_name) -> bool:
    with connect(str(config.SQLITE_PATH)) as c:
        r = c.execute(
            "SELECT orig_desc FROM images WHERE photo_name=? AND model_tag=?",
            (photo_name, config.MODEL_TAG),
        ).fetchone()
    return bool(r and r[0] and os.path.exists(r[0]))


def _reload_store():
    init_db(str(config.SQLITE_PATH))
    load_descriptors.cache_clear()
    STORE.load(str(config.SQLITE_PATH), config.MODEL_TAG)


def _build_descriptors_for(photo_name, full_img, category, engine):
    """Returns (rembg_name, orig_vec, rembg_vec, orig_desc, rembg_desc, bg_engine).
    full_img is the full-size original RGB."""
    config.DESC_DIR.mkdir(parents=True, exist_ok=True)
    orig768 = to_working_res(full_img)
    orig_vec = _embed(orig768)
    odes, on = _desc(orig768)
    orig_desc = str(config.DESC_DIR / f"{photo_name}.orig.npz")
    save_descriptors(orig_desc, odes, on)

    rembg_name = rembg_vec = rembg_desc = bg_engine = None
    if category == "cloth":
        config.RMBG_DIR.mkdir(parents=True, exist_ok=True)
        base = os.path.splitext(photo_name)[0]
        rembg_name = f"{base}.png"
        rpath = config.RMBG_DIR / rembg_name
        bg_full, bg_engine = _bg(full_img, engine)
        cv2.imwrite(str(rpath), cv2.cvtColor(bg_full, cv2.COLOR_RGB2BGR))
        ledger.record("replace", "rmbg", rembg_name, _now())
        rimg768 = to_working_res(bg_full)
        rembg_vec = _embed(rimg768)
        rdes, rn = _desc(rimg768)
        rembg_desc = str(config.DESC_DIR / f"{photo_name}.rmbg.npz")
        save_descriptors(rembg_desc, rdes, rn)
    return rembg_name, orig_vec, rembg_vec, orig_desc, rembg_desc, bg_engine


def index_one(
    sku_id, designno, item, photo_name, *, engine="auto", force=False, reload=True
) -> dict:
    """Index a single ALREADY-STAGED original (in CORPUS_DIR). Shared by
    /ingest, the job worker, and seed."""
    category = resolve_category(str(config.SQLITE_PATH), item)
    if not force and _already_indexed(photo_name):
        return {"status": "skipped", "reason": "already indexed", "photo_name": photo_name}
    with _LOCK:
        full = _load_original(photo_name)
        rembg_name, ov, rv, od, rd, bg_engine = _build_descriptors_for(
            photo_name, full, category, engine
        )
        upsert_image(
            str(config.SQLITE_PATH),
            sku_id,
            designno,
            item,
            category,
            photo_name,
            rembg_name,
            ov,
            rv,
            od,
            rd,
            config.MODEL_TAG,
            bg_engine,
        )
        if reload:
            _reload_store()
    return {
        "status": "indexed",
        "photo_name": photo_name,
        "category": category,
        "bg_engine": bg_engine,
    }


def add_image(sku_id, designno, item, photo_name, data: bytes) -> dict:
    """Stage bytes to disk then index synchronously (single-call path)."""
    config.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    (config.CORPUS_DIR / photo_name).write_bytes(data)
    ledger.record("add", "original", photo_name, _now())
    return index_one(sku_id, designno, item, photo_name, reload=True)


def replace_image(photo_name, kind: str, data: bytes) -> dict:
    """kind: 'original' (re-derive everything incl. rmbg) or 'rmbg' (rmbg only)."""
    with _LOCK:
        with connect(str(config.SQLITE_PATH)) as c:
            row = c.execute(
                "SELECT sku_id, designno, item, category FROM images "
                "WHERE photo_name=? AND model_tag=?",
                (photo_name, config.MODEL_TAG),
            ).fetchone()
        if row is None:
            return {"status": "not_found", "photo_name": photo_name}
        sku_id, designno, item, category = row[0], row[1], row[2], row[3]

        if kind == "original":
            (config.CORPUS_DIR / photo_name).write_bytes(data)
            ledger.record("replace", "original", photo_name, _now())
            full = _decode_full(data)
            rembg_name, ov, rv, od, rd, bg_engine = _build_descriptors_for(
                photo_name, full, category, "auto"
            )
            upsert_image(
                str(config.SQLITE_PATH),
                sku_id,
                designno,
                item,
                category,
                photo_name,
                rembg_name,
                ov,
                rv,
                od,
                rd,
                config.MODEL_TAG,
                bg_engine,
            )
        else:  # replace just the rmbg image
            base = os.path.splitext(photo_name)[0]
            rembg_name = f"{base}.png"
            (config.RMBG_DIR / rembg_name).write_bytes(data)
            ledger.record("replace", "rmbg", rembg_name, _now())
            rimg768 = to_working_res(_decode_full(data))
            rv = _embed(rimg768)
            rdes, rn = _desc(rimg768)
            rd = str(config.DESC_DIR / f"{photo_name}.rmbg.npz")
            save_descriptors(rd, rdes, rn)
            with connect(str(config.SQLITE_PATH)) as c:
                r = c.execute(
                    "SELECT orig_vec, orig_desc FROM images WHERE photo_name=? AND model_tag=?",
                    (photo_name, config.MODEL_TAG),
                ).fetchone()
            upsert_image(
                str(config.SQLITE_PATH),
                sku_id,
                designno,
                item,
                category,
                photo_name,
                rembg_name,
                blob_to_vec(r[0]),
                rv,
                r[1],
                rd,
                config.MODEL_TAG,
                "manual",
            )
        _reload_store()
    return {"status": "replaced", "photo_name": photo_name, "kind": kind}


def forget_image(photo_name: str) -> dict:
    base = os.path.splitext(photo_name)[0]
    with _LOCK:
        files_removed = 0
        for p in (
            config.CORPUS_DIR / photo_name,
            config.RMBG_DIR / f"{base}.png",
            config.DESC_DIR / f"{photo_name}.orig.npz",
            config.DESC_DIR / f"{photo_name}.rmbg.npz",
        ):
            if p.exists():
                p.unlink()
                files_removed += 1
        db_deleted = delete_by_photo(str(config.SQLITE_PATH), photo_name, config.MODEL_TAG)
        ledger.record("delete", "original", photo_name, _now())
        _reload_store()
    return {"files_removed": files_removed, "db_deleted": db_deleted}
