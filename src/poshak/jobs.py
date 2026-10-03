"""Single-worker, SQLite-backed, resumable index queue.

/ingest stages a file then calls submit(); one background worker drains the
queue, runs the shared manage.index_one (reload=False), checkpoints each item
to SQLite, and reloads STORE when the queue idles or every RELOAD_EVERY items.
Job/item rows live in the bind-mounted DB so the worker resumes on restart."""

import contextlib
import queue
import threading

from . import config
from .descriptors import load_descriptors
from .manage import index_one
from .store import (
    STORE,
    add_job_item,
    bump_engine,
    create_or_touch_job,
    list_pending_items,
    refresh_job_counts,
    set_item_status,
)

RELOAD_EVERY = 50

_Q: "queue.Queue" = queue.Queue()
_started = False
_start_lock = threading.Lock()
_since_reload = 0


def _reload_store():
    load_descriptors.cache_clear()
    STORE.load(str(config.SQLITE_PATH), config.MODEL_TAG)


def submit(batch_id, photo_name, sku_id, designno, item, engine="auto", force=False) -> None:
    db = str(config.SQLITE_PATH)
    create_or_touch_job(db, batch_id)
    add_job_item(db, batch_id, photo_name, sku_id, designno, item, engine, force)
    _Q.put((batch_id, photo_name, sku_id, designno, item, engine, force))


def _process_one(batch_id, photo_name, sku_id, designno, item, engine, force):
    db = str(config.SQLITE_PATH)
    set_item_status(db, batch_id, photo_name, "running")
    try:
        res = index_one(
            sku_id, designno, item, photo_name, engine=engine, force=force, reload=False
        )
        status = "skipped" if res.get("status") == "skipped" else "done"
        set_item_status(db, batch_id, photo_name, status)
        if res.get("bg_engine") in ("api", "local"):
            bump_engine(db, batch_id, res["bg_engine"])
    except Exception as e:  # noqa: BLE001 - record, never kill the worker
        set_item_status(db, batch_id, photo_name, "failed", str(e))
    finally:
        refresh_job_counts(db, batch_id)


def _run():
    global _since_reload
    while True:
        job = _Q.get()
        try:
            _process_one(*job)
        finally:
            _since_reload += 1
            if _Q.empty() or _since_reload >= RELOAD_EVERY:
                with contextlib.suppress(Exception):
                    _reload_store()
                _since_reload = 0
            _Q.task_done()


def start() -> None:
    global _started
    with _start_lock:
        if not _started:
            threading.Thread(target=_run, daemon=True, name="index-worker").start()
            _started = True


def resume() -> None:
    for it in list_pending_items(str(config.SQLITE_PATH)):
        _Q.put(
            (
                it["batch_id"],
                it["photo_name"],
                it["sku_id"],
                it["designno"],
                it["item"],
                it["bg_engine"] or "auto",
                bool(it["force"]),
            )
        )
