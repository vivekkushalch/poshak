"""Live ingest/forget for admin add+delete of stock photos.

Flow when admin adds a photo:
  1. PHP saves the file to the web host /uploads/, INSERTs into `stock`,
     gets back stock_id.
  2. PHP cURL-POSTs the same bytes to /ingest-photo on AWS.
  3. This module saves the file to /corpus/uploads/ (AWS mirror) and
     returns 202 Accepted — PHP doesn't wait for embedding.
  4. INGEST_POOL background thread:
        - look up the SKU's item → category
        - rembg + save PNG (cloth only)
        - SigLIP embed orig + rembg
        - UPSERT stock_image_vectors
        - append to STORE in-place (under a lock) so the photo is
          searchable on the very next /search.

Forget runs synchronously — it's just unlink + DELETE + list-filter,
fast enough that backgrounding adds complexity for no gain.
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

from . import config
from .bg_removal import remove_bg_rembg
from .db import get_connection
from .debug import dbg
from .image_io import load_image
from .index_store import STORE
from .model import siglip_embed


# Two workers is plenty — admin saves come in human time, not as a
# burst. More workers would let burst saves OOM during rembg.
INGEST_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest")

# Serializes STORE writes (append, regenerate matrices) against
# concurrent /search reads. Holds for ~1ms per ingest at small scale.
_STORE_LOCK = threading.Lock()


SELECT_SKU_SQL = """
    SELECT s.id        AS stock_id,
           s.designno  AS designno,
           si.name     AS item
      FROM stock s
 LEFT JOIN stockitem si ON si.id = s.itemname
     WHERE s.id = %s
"""

UPSERT_VEC_SQL = """
INSERT INTO stock_image_vectors
    (stock_id, category, photo_name, rembg_name,
     orig_vec, rembg_vec, model_tag)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    stock_id   = VALUES(stock_id),
    category   = VALUES(category),
    rembg_name = VALUES(rembg_name),
    orig_vec   = VALUES(orig_vec),
    rembg_vec  = VALUES(rembg_vec)
"""

DELETE_VEC_SQL = """
DELETE FROM stock_image_vectors
 WHERE photo_name = %s AND model_tag = %s
"""


def _categorize(item: str):
    item = (item or "").strip()
    if item in config.CLOTH_ITEMS:
        return "cloth"
    if item in config.FABRIC_ITEMS:
        return "fabric"
    return None


def ingest_async(stock_id: int, photo_name: str) -> None:
    """Background worker — runs after /ingest-photo has already saved
    the file and returned 202 to the caller."""
    dbg(f"ingest_async: stock_id={stock_id} photo={photo_name}")

    with closing(get_connection()) as conn:
        with conn.cursor() as cur:
            cur.execute(SELECT_SKU_SQL, (stock_id,))
            row = cur.fetchone()
    if not row:
        dbg(f"ingest_async: stock_id={stock_id} not found in stock; skip")
        return

    category = _categorize(row["item"])
    if category is None:
        dbg(f"ingest_async: item={row['item']!r} not in CLOTH/FABRIC; skip")
        return

    orig_path = config.CORPUS_DIR / photo_name
    if not orig_path.exists():
        dbg(f"ingest_async: orig not on disk: {orig_path}; skip")
        return

    orig_img = load_image(str(orig_path))
    if orig_img is None:
        dbg(f"ingest_async: failed to decode {orig_path}; skip")
        return

    orig_vec = siglip_embed(orig_img)

    rembg_name = None
    rembg_vec = None
    if category == "cloth":
        try:
            rembg_img = remove_bg_rembg(orig_img)
            base = os.path.splitext(photo_name)[0]
            rembg_path = config.DATA_DIR / f"{base}.png"
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(rembg_path), cv2.cvtColor(rembg_img, cv2.COLOR_RGB2BGR))
            rembg_name = rembg_path.name
            rembg_vec = siglip_embed(rembg_img)
        except Exception as e:
            dbg(f"ingest_async: rembg failed for {photo_name}: {e}; "
                "indexing orig only")

    with closing(get_connection()) as conn:
        with conn.cursor() as cur:
            cur.execute(UPSERT_VEC_SQL, (
                stock_id, category, photo_name, rembg_name,
                json.dumps(orig_vec.tolist()),
                json.dumps(rembg_vec.tolist()) if rembg_vec is not None else None,
                config.MODEL_TAG,
            ))
        conn.commit()

    designno = row.get("designno") or ""
    item_name = row.get("item") or ""
    orig_entry = {
        "sku_id": stock_id, "designno": designno, "item": item_name,
        "filename": photo_name, "json_filename": photo_name,
        "path": str(orig_path),
        "img": None, "embed": orig_vec, "trace": None,
    }
    with _STORE_LOCK:
        # Drop any prior entries for this photo first (re-upload of the
        # same filename has to overwrite, not duplicate).
        STORE.cloth_orig = [e for e in STORE.cloth_orig if e["filename"] != photo_name]
        STORE.cloth_bg   = [e for e in STORE.cloth_bg   if e["json_filename"] != photo_name]
        STORE.fabric     = [e for e in STORE.fabric     if e["filename"] != photo_name]

        if category == "fabric":
            STORE.fabric.append(orig_entry)
        else:
            STORE.cloth_orig.append(orig_entry)
            if rembg_vec is not None and rembg_name is not None:
                STORE.cloth_bg.append({
                    "sku_id": stock_id, "designno": designno, "item": item_name,
                    "filename": rembg_name, "json_filename": photo_name,
                    "path": str(config.DATA_DIR / rembg_name),
                    "img": None, "embed": rembg_vec, "trace": None,
                })
        STORE.prepare()

    dbg(f"ingest_async: DONE stock_id={stock_id} photo={photo_name} "
        f"category={category} rembg={'yes' if rembg_vec is not None else 'no'}")


def forget(photo_name: str) -> dict:
    """Synchronous removal of a photo from disk + DB + STORE.

    Returns a small dict of what was removed (useful for the endpoint
    response and for diagnosing inconsistencies)."""
    dbg(f"forget: photo_name={photo_name}")
    base = os.path.splitext(photo_name)[0]

    files_removed = 0
    for path in (config.CORPUS_DIR / photo_name,
                 config.DATA_DIR / f"{base}.png"):
        if path.exists():
            try:
                path.unlink()
                files_removed += 1
            except Exception as e:
                dbg(f"forget: unlink {path} failed: {e}")

    with closing(get_connection()) as conn:
        with conn.cursor() as cur:
            cur.execute(DELETE_VEC_SQL, (photo_name, config.MODEL_TAG))
            db_deleted = cur.rowcount
        conn.commit()

    with _STORE_LOCK:
        before = (len(STORE.cloth_orig), len(STORE.cloth_bg), len(STORE.fabric))
        STORE.cloth_orig = [e for e in STORE.cloth_orig if e["filename"] != photo_name]
        STORE.cloth_bg   = [e for e in STORE.cloth_bg   if e["json_filename"] != photo_name]
        STORE.fabric     = [e for e in STORE.fabric     if e["filename"] != photo_name]
        after = (len(STORE.cloth_orig), len(STORE.cloth_bg), len(STORE.fabric))
        STORE.prepare()
    store_removed = sum(b - a for b, a in zip(before, after))

    dbg(f"forget: DONE photo={photo_name} files={files_removed} "
        f"db_rows={db_deleted} store_entries={store_removed}")
    return {"files_removed": files_removed,
            "db_deleted": db_deleted,
            "store_removed": store_removed}
