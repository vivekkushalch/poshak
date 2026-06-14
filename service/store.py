"""Local SQLite index + in-RAM store. Replaces the MySQL db.py/loader.py.

Embeddings load into RAM as stacked matrices (port of spipeline.py
prepare_matrix L2233). Descriptor paths stay as strings; pixels are never
held. Entry dict shape matches the prototype's index entries."""

import sqlite3
from typing import Optional

import numpy as np

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
  id         INTEGER PRIMARY KEY,
  sku_id     INTEGER NOT NULL,
  designno   TEXT,
  item       TEXT,
  category   TEXT NOT NULL,
  photo_name TEXT NOT NULL,
  rembg_name TEXT,
  orig_vec   BLOB NOT NULL,
  rembg_vec  BLOB,
  orig_desc  TEXT,
  rembg_desc TEXT,
  model_tag  TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(photo_name, model_tag)
);
CREATE INDEX IF NOT EXISTS idx_model ON images(model_tag);
"""


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def vec_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def upsert_image(db_path, sku_id, designno, item, category, photo_name,
                 rembg_name, orig_vec, rembg_vec, orig_desc, rembg_desc,
                 model_tag) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO images
               (sku_id, designno, item, category, photo_name, rembg_name,
                orig_vec, rembg_vec, orig_desc, rembg_desc, model_tag)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(photo_name, model_tag) DO UPDATE SET
                 sku_id=excluded.sku_id, designno=excluded.designno,
                 item=excluded.item, category=excluded.category,
                 rembg_name=excluded.rembg_name, orig_vec=excluded.orig_vec,
                 rembg_vec=excluded.rembg_vec, orig_desc=excluded.orig_desc,
                 rembg_desc=excluded.rembg_desc""",
            (sku_id, designno, item, category, photo_name, rembg_name,
             vec_to_blob(orig_vec), vec_to_blob(rembg_vec) if rembg_vec is not None else None,
             orig_desc, rembg_desc, model_tag),
        )
        conn.commit()


def delete_by_photo(db_path: str, photo_name: str, model_tag: str) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            "DELETE FROM images WHERE photo_name=? AND model_tag=?",
            (photo_name, model_tag))
        conn.commit()
        return cur.rowcount


def _stack(entries: list[dict]) -> Optional[np.ndarray]:
    if not entries:
        return None
    return np.stack([e["embed"] for e in entries])


class IndexStore:
    def __init__(self) -> None:
        self.cloth_bg: list[dict] = []
        self.cloth_orig: list[dict] = []
        self.fabric: list[dict] = []
        self.cloth_bg_embeds: Optional[np.ndarray] = None
        self.cloth_orig_embeds: Optional[np.ndarray] = None
        self.fabric_embeds: Optional[np.ndarray] = None

    def load(self, db_path: str, model_tag: str) -> dict:
        cloth_bg, cloth_orig, fabric = [], [], []
        with connect(db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM images WHERE model_tag=?", (model_tag,)).fetchall()
        for r in rows:
            orig = {
                "sku_id": r["sku_id"], "designno": r["designno"] or "",
                "item": r["item"] or "", "filename": r["photo_name"],
                "json_filename": r["photo_name"],
                "desc_path": r["orig_desc"], "embed": blob_to_vec(r["orig_vec"]),
            }
            if r["category"] == "fabric":
                fabric.append(orig)
            else:
                cloth_orig.append(orig)
                if r["rembg_vec"] is not None and r["rembg_desc"]:
                    cloth_bg.append({
                        "sku_id": r["sku_id"], "designno": r["designno"] or "",
                        "item": r["item"] or "", "filename": r["rembg_name"],
                        "json_filename": r["photo_name"],
                        "desc_path": r["rembg_desc"],
                        "embed": blob_to_vec(r["rembg_vec"]),
                    })
        self.cloth_bg, self.cloth_orig, self.fabric = cloth_bg, cloth_orig, fabric
        self.cloth_bg_embeds = _stack(cloth_bg)
        self.cloth_orig_embeds = _stack(cloth_orig)
        self.fabric_embeds = _stack(fabric)
        return self.stats()

    def is_ready(self) -> bool:
        return any(len(x) > 0 for x in (self.cloth_bg, self.cloth_orig, self.fabric))

    def stats(self) -> dict:
        return {"cloth_bg": len(self.cloth_bg), "cloth_orig": len(self.cloth_orig),
                "fabric": len(self.fabric), "model_tag": config.MODEL_TAG}


def list_skus(db_path: str, model_tag: str, page: int, page_size: int) -> dict:
    offset = (max(1, page) - 1) * page_size
    with connect(db_path) as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM images WHERE model_tag=?", (model_tag,)).fetchone()[0]
        rows = conn.execute(
            "SELECT sku_id, item, designno, photo_name, rembg_name FROM images "
            "WHERE model_tag=? ORDER BY id LIMIT ? OFFSET ?",
            (model_tag, page_size, offset)).fetchall()
    items = [dict(r) for r in rows]
    return {"total": total, "page": page, "page_size": page_size, "items": items}


STORE = IndexStore()
