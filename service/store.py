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
  bg_engine  TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(photo_name, model_tag)
);
CREATE INDEX IF NOT EXISTS idx_model ON images(model_tag);
CREATE TABLE IF NOT EXISTS index_jobs (
  batch_id    TEXT PRIMARY KEY,
  status      TEXT NOT NULL DEFAULT 'open',
  total       INTEGER NOT NULL DEFAULT 0,
  done        INTEGER NOT NULL DEFAULT 0,
  failed      INTEGER NOT NULL DEFAULT 0,
  skipped     INTEGER NOT NULL DEFAULT 0,
  api_count   INTEGER NOT NULL DEFAULT 0,
  local_count INTEGER NOT NULL DEFAULT 0,
  created_at  TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS index_job_items (
  id         INTEGER PRIMARY KEY,
  batch_id   TEXT NOT NULL,
  photo_name TEXT NOT NULL,
  sku_id     INTEGER NOT NULL,
  designno   TEXT,
  item       TEXT,
  bg_engine  TEXT,
  force      INTEGER NOT NULL DEFAULT 0,
  status     TEXT NOT NULL DEFAULT 'queued',
  error      TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(batch_id, photo_name)
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(images)").fetchall()]
        if "bg_engine" not in cols:
            conn.execute("ALTER TABLE images ADD COLUMN bg_engine TEXT")
        conn.commit()


def vec_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def upsert_image(db_path, sku_id, designno, item, category, photo_name,
                 rembg_name, orig_vec, rembg_vec, orig_desc, rembg_desc,
                 model_tag, bg_engine=None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO images
               (sku_id, designno, item, category, photo_name, rembg_name,
                orig_vec, rembg_vec, orig_desc, rembg_desc, model_tag, bg_engine)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(photo_name, model_tag) DO UPDATE SET
                 sku_id=excluded.sku_id, designno=excluded.designno,
                 item=excluded.item, category=excluded.category,
                 rembg_name=excluded.rembg_name, orig_vec=excluded.orig_vec,
                 rembg_vec=excluded.rembg_vec, orig_desc=excluded.orig_desc,
                 rembg_desc=excluded.rembg_desc, bg_engine=excluded.bg_engine""",
            (sku_id, designno, item, category, photo_name, rembg_name,
             vec_to_blob(orig_vec), vec_to_blob(rembg_vec) if rembg_vec is not None else None,
             orig_desc, rembg_desc, model_tag, bg_engine),
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


def create_or_touch_job(db_path: str, batch_id: str) -> None:
    with connect(db_path) as c:
        c.execute(
            "INSERT INTO index_jobs (batch_id, status) VALUES (?, 'running') "
            "ON CONFLICT(batch_id) DO UPDATE SET updated_at=datetime('now')",
            (batch_id,))
        c.commit()


def add_job_item(db_path, batch_id, photo_name, sku_id, designno, item,
                 engine, force) -> None:
    with connect(db_path) as c:
        c.execute(
            """INSERT INTO index_job_items
               (batch_id, photo_name, sku_id, designno, item, bg_engine, force, status)
               VALUES (?,?,?,?,?,?,?, 'queued')
               ON CONFLICT(batch_id, photo_name) DO UPDATE SET
                 status='queued', sku_id=excluded.sku_id, item=excluded.item,
                 designno=excluded.designno, bg_engine=excluded.bg_engine,
                 force=excluded.force, error=NULL, updated_at=datetime('now')""",
            (batch_id, photo_name, sku_id, designno, item, engine,
             1 if force else 0))
        c.execute(
            "UPDATE index_jobs SET total=(SELECT COUNT(*) FROM index_job_items "
            "WHERE batch_id=?), updated_at=datetime('now') WHERE batch_id=?",
            (batch_id, batch_id))
        c.commit()


def set_item_status(db_path, batch_id, photo_name, status, error=None) -> None:
    with connect(db_path) as c:
        c.execute(
            "UPDATE index_job_items SET status=?, error=?, "
            "updated_at=datetime('now') WHERE batch_id=? AND photo_name=?",
            (status, error, batch_id, photo_name))
        c.commit()


def bump_engine(db_path, batch_id, engine_used) -> None:
    col = "api_count" if engine_used == "api" else "local_count"
    with connect(db_path) as c:
        c.execute(f"UPDATE index_jobs SET {col}={col}+1 WHERE batch_id=?",
                  (batch_id,))
        c.commit()


def refresh_job_counts(db_path, batch_id) -> None:
    with connect(db_path) as c:
        row = c.execute(
            "SELECT SUM(status='done'), SUM(status='failed'), "
            "SUM(status='skipped'), COUNT(*) FROM index_job_items WHERE batch_id=?",
            (batch_id,)).fetchone()
        done, failed, skipped, total = (row[0] or 0, row[1] or 0,
                                        row[2] or 0, row[3] or 0)
        status = "done" if total > 0 and (done + failed + skipped) >= total \
            else "running"
        c.execute(
            "UPDATE index_jobs SET done=?, failed=?, skipped=?, status=?, "
            "updated_at=datetime('now') WHERE batch_id=?",
            (done, failed, skipped, status, batch_id))
        c.commit()


def list_pending_items(db_path: str) -> list[dict]:
    with connect(db_path) as c:
        rows = c.execute(
            "SELECT * FROM index_job_items WHERE status IN ('queued','running') "
            "ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def get_job(db_path: str, batch_id: str):
    with connect(db_path) as c:
        job = c.execute("SELECT * FROM index_jobs WHERE batch_id=?",
                        (batch_id,)).fetchone()
        if job is None:
            return None
        errs = c.execute(
            "SELECT photo_name, error FROM index_job_items "
            "WHERE batch_id=? AND status='failed' ORDER BY id", (batch_id,)
        ).fetchall()
    d = dict(job)
    d["errors"] = [dict(e) for e in errs]
    return d


def list_jobs(db_path: str) -> list[dict]:
    with connect(db_path) as c:
        rows = c.execute(
            "SELECT * FROM index_jobs ORDER BY created_at DESC, batch_id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


STORE = IndexStore()
