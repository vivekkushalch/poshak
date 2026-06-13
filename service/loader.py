"""Boot-time loader: MySQL stock_image_vectors → in-memory STORE.

Read once at FastAPI startup, also exposed via /reload. Per-photo dicts
match the shape the search pipeline expects (see index_store.py).

The query is filtered by model_tag = current — rows produced under any
other model are ignored. This is the guard from schema.sql.
"""

import json
from contextlib import closing

import numpy as np

from . import config
from .db import get_connection
from .index_store import STORE


SELECT_VECTORS_SQL = """
    SELECT v.stock_id,
           v.category,
           v.photo_name,
           v.rembg_name,
           v.orig_vec,
           v.rembg_vec,
           si.name      AS item,
           s.designno   AS designno
      FROM stock_image_vectors v
INNER JOIN stock s         ON s.id  = v.stock_id
 LEFT JOIN stockitem si    ON si.id = s.itemname
     WHERE v.model_tag = %s
"""


def _vec_from_json(raw) -> np.ndarray:
    """JSON column may come back as a str or a parsed list depending on
    PyMySQL version + server. Handle both."""
    if isinstance(raw, str):
        raw = json.loads(raw)
    return np.asarray(raw, dtype=np.float32)


def load_from_mysql() -> dict:
    """Populates STORE in place. Returns a stats dict."""
    cloth_bg: list[dict] = []
    cloth_orig: list[dict] = []
    fabric: list[dict] = []

    with closing(get_connection()) as conn:
        with conn.cursor() as cur:
            cur.execute(SELECT_VECTORS_SQL, (config.MODEL_TAG,))
            for row in cur.fetchall():
                stock_id = row["stock_id"]
                category = row["category"]
                photo = row["photo_name"]
                rembg = row["rembg_name"]
                item = row["item"] or ""
                designno = row["designno"] or ""

                orig_vec = _vec_from_json(row["orig_vec"])
                orig_entry = {
                    "sku_id": stock_id,
                    "designno": designno,
                    "item": item,
                    "filename": photo,
                    "json_filename": photo,
                    "path": str(config.CORPUS_DIR / photo),
                    "img": None,
                    "embed": orig_vec,
                    "trace": None,
                }

                if category == "fabric":
                    fabric.append(orig_entry)
                else:
                    cloth_orig.append(orig_entry)
                    if rembg and row["rembg_vec"] is not None:
                        rembg_vec = _vec_from_json(row["rembg_vec"])
                        cloth_bg.append({
                            "sku_id": stock_id,
                            "designno": designno,
                            "item": item,
                            "filename": rembg,
                            "json_filename": photo,
                            "path": str(config.DATA_DIR / rembg),
                            "img": None,
                            "embed": rembg_vec,
                            "trace": None,
                        })

    STORE.cloth_bg = cloth_bg
    STORE.cloth_orig = cloth_orig
    STORE.fabric = fabric
    STORE.prepare()

    return STORE.stats()
