"""SQLite repositories. One short-lived connection per call; WAL mode lets the
search path's reads proceed while the ingest worker writes."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from ...domain.models import Category, ImageRecord, JobItem
from .migrations import migrate


class SqliteDatabase:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            migrate(conn)


def _vec_to_blob(vec: np.ndarray | None) -> bytes | None:
    return None if vec is None else np.asarray(vec, dtype=np.float32).tobytes()


def _blob_to_vec(blob: bytes | None) -> np.ndarray | None:
    return None if blob is None else np.frombuffer(blob, dtype=np.float32)


def _record(r: sqlite3.Row) -> ImageRecord:
    return ImageRecord(
        sku_id=r["sku_id"],
        designno=r["designno"] or "",
        item=r["item"] or "",
        category=r["category"],
        photo_name=r["photo_name"],
        orig_vec=_blob_to_vec(r["orig_vec"]),
        orig_desc=r["orig_desc"],
        model_tag=r["model_tag"],
        rembg_name=r["rembg_name"],
        rembg_vec=_blob_to_vec(r["rembg_vec"]),
        rembg_desc=r["rembg_desc"],
        bg_engine=r["bg_engine"],
    )


class SqliteImageRepository:
    def __init__(self, db: SqliteDatabase) -> None:
        self.db = db

    def upsert(self, rec: ImageRecord) -> None:
        with self.db.connect() as c:
            c.execute(
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
                (
                    rec.sku_id, rec.designno, rec.item, rec.category, rec.photo_name,
                    rec.rembg_name, _vec_to_blob(rec.orig_vec), _vec_to_blob(rec.rembg_vec),
                    rec.orig_desc, rec.rembg_desc, rec.model_tag, rec.bg_engine,
                ),
            )  # fmt: skip

    def get(self, photo_name: str, model_tag: str) -> ImageRecord | None:
        with self.db.connect() as c:
            r = c.execute(
                "SELECT * FROM images WHERE photo_name=? AND model_tag=?", (photo_name, model_tag)
            ).fetchone()
        return None if r is None else _record(r)

    def delete(self, photo_name: str, model_tag: str) -> int:
        with self.db.connect() as c:
            return c.execute(
                "DELETE FROM images WHERE photo_name=? AND model_tag=?", (photo_name, model_tag)
            ).rowcount

    def all(self, model_tag: str) -> list[ImageRecord]:
        with self.db.connect() as c:
            rows = c.execute(
                "SELECT * FROM images WHERE model_tag=? ORDER BY id", (model_tag,)
            ).fetchall()
        return [_record(r) for r in rows]

    def page(self, model_tag: str, page: int, page_size: int) -> tuple[int, list[dict[str, Any]]]:
        offset = (max(1, page) - 1) * page_size
        with self.db.connect() as c:
            total = c.execute(
                "SELECT COUNT(*) FROM images WHERE model_tag=?", (model_tag,)
            ).fetchone()[0]
            rows = c.execute(
                "SELECT sku_id, item, designno, photo_name, rembg_name FROM images "
                "WHERE model_tag=? ORDER BY id LIMIT ? OFFSET ?",
                (model_tag, page_size, offset),
            ).fetchall()
        return total, [dict(r) for r in rows]


class SqliteJobRepository:
    def __init__(self, db: SqliteDatabase) -> None:
        self.db = db

    def add_item(self, item: JobItem) -> None:
        with self.db.connect() as c:
            c.execute(
                "INSERT INTO index_jobs (batch_id, status) VALUES (?, 'running') "
                "ON CONFLICT(batch_id) DO UPDATE SET updated_at=datetime('now')",
                (item.batch_id,),
            )
            c.execute(
                """INSERT INTO index_job_items
                   (batch_id, photo_name, sku_id, designno, item, bg_engine, force, status)
                   VALUES (?,?,?,?,?,?,?, 'queued')
                   ON CONFLICT(batch_id, photo_name) DO UPDATE SET
                     status='queued', sku_id=excluded.sku_id, item=excluded.item,
                     designno=excluded.designno, bg_engine=excluded.bg_engine,
                     force=excluded.force, error=NULL, updated_at=datetime('now')""",
                (
                    item.batch_id, item.photo_name, item.sku_id, item.designno, item.item,
                    item.bg_engine, int(item.force),
                ),
            )  # fmt: skip
            c.execute(
                "UPDATE index_jobs SET total=(SELECT COUNT(*) FROM index_job_items "
                "WHERE batch_id=?), updated_at=datetime('now') WHERE batch_id=?",
                (item.batch_id, item.batch_id),
            )

    def set_status(
        self, batch_id: str, photo_name: str, status: str, error: str | None = None
    ) -> None:
        with self.db.connect() as c:
            c.execute(
                "UPDATE index_job_items SET status=?, error=?, updated_at=datetime('now') "
                "WHERE batch_id=? AND photo_name=?",
                (status, error, batch_id, photo_name),
            )

    def bump_engine(self, batch_id: str, engine: str) -> None:
        col = "api_count" if engine == "api" else "local_count"  # fixed set: safe to inline
        with self.db.connect() as c:
            c.execute(f"UPDATE index_jobs SET {col}={col}+1 WHERE batch_id=?", (batch_id,))

    def refresh_counts(self, batch_id: str) -> None:
        with self.db.connect() as c:
            row = c.execute(
                "SELECT SUM(status='done'), SUM(status='failed'), SUM(status='skipped'), "
                "COUNT(*) FROM index_job_items WHERE batch_id=?",
                (batch_id,),
            ).fetchone()
            done, failed, skipped, total = (row[0] or 0, row[1] or 0, row[2] or 0, row[3] or 0)
            status = "done" if total > 0 and done + failed + skipped >= total else "running"
            c.execute(
                "UPDATE index_jobs SET done=?, failed=?, skipped=?, status=?, "
                "updated_at=datetime('now') WHERE batch_id=?",
                (done, failed, skipped, status, batch_id),
            )

    def pending(self) -> list[JobItem]:
        with self.db.connect() as c:
            rows = c.execute(
                "SELECT * FROM index_job_items WHERE status IN ('queued','running') ORDER BY id"
            ).fetchall()
        return [
            JobItem(
                batch_id=r["batch_id"], photo_name=r["photo_name"], sku_id=r["sku_id"],
                designno=r["designno"] or "", item=r["item"] or "",
                bg_engine=r["bg_engine"] or "auto", force=bool(r["force"]),
            )
            for r in rows
        ]  # fmt: skip

    def get(self, batch_id: str) -> dict[str, Any] | None:
        with self.db.connect() as c:
            job = c.execute("SELECT * FROM index_jobs WHERE batch_id=?", (batch_id,)).fetchone()
            if job is None:
                return None
            errors = c.execute(
                "SELECT photo_name, error FROM index_job_items "
                "WHERE batch_id=? AND status='failed' ORDER BY id",
                (batch_id,),
            ).fetchall()
        return {**dict(job), "errors": [dict(e) for e in errors]}

    def list(self) -> list[dict[str, Any]]:
        with self.db.connect() as c:
            rows = c.execute(
                "SELECT * FROM index_jobs ORDER BY created_at DESC, batch_id DESC"
            ).fetchall()
        return [dict(r) for r in rows]


class SqliteCategoryRepository:
    def __init__(self, db: SqliteDatabase) -> None:
        self.db = db

    def get(self, item: str) -> Category | None:
        with self.db.connect() as c:
            row = c.execute(
                "SELECT category FROM item_categories WHERE item = ? COLLATE NOCASE",
                (item.strip(),),
            ).fetchone()
        return None if row is None else row["category"]

    def add(self, item: str, category: Category, source: str) -> None:
        with self.db.connect() as c:
            c.execute(
                "INSERT OR IGNORE INTO item_categories (item, category, source) VALUES (?, ?, ?)",
                (item.strip(), category, source),
            )

    def set_manual(self, item: str, category: Category) -> None:
        with self.db.connect() as c:
            c.execute(
                "INSERT INTO item_categories (item, category, source) VALUES (?, ?, 'manual') "
                "ON CONFLICT(item) DO UPDATE SET category=excluded.category, source='manual'",
                (item.strip(), category),
            )

    def list(self) -> list[dict[str, Any]]:
        with self.db.connect() as c:
            rows = c.execute(
                "SELECT item, category, source FROM item_categories ORDER BY item"
            ).fetchall()
        return [dict(r) for r in rows]
