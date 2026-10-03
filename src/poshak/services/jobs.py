"""Background ingest queue: one worker thread, every item checkpointed in SQLite.

POST /ingest returns 202 straight after submit(). The worker indexes items in
order, records each outcome, and reloads the live index when the queue drains
or every RELOAD_EVERY items. resume() re-queues unfinished items after a restart.

ponytail: in-process queue, so one node only. A multi-node deployment moves the
queue to Redis or SQS behind the same submit/process interface."""

import queue
import threading

from ..core.logging import log
from ..domain.models import JobItem
from ..domain.ports import JobRepository
from .index_state import IndexState
from .indexer import IndexerService

RELOAD_EVERY = 50
_STOP = object()


class JobService:
    def __init__(self, indexer: IndexerService, jobs: JobRepository, state: IndexState) -> None:
        self._indexer = indexer
        self._jobs = jobs
        self._state = state
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._since_reload = 0

    def submit(self, item: JobItem) -> None:
        self._jobs.add_item(item)
        self._queue.put(item)

    def resume(self) -> int:
        pending = self._jobs.pending()
        for item in pending:
            self._queue.put(item)
        return len(pending)

    def process(self, item: JobItem) -> None:
        """Index one item and record the outcome. Never raises."""
        self._jobs.set_status(item.batch_id, item.photo_name, "running")
        try:
            res = self._indexer.index(
                item.photo_name, item.sku_id, item.designno, item.item,
                engine=item.bg_engine, force=item.force, reload=False,
            )  # fmt: skip
            status = "skipped" if res["status"] == "skipped" else "done"
            self._jobs.set_status(item.batch_id, item.photo_name, status)
            if res.get("bg_engine") in ("api", "local"):
                self._jobs.bump_engine(item.batch_id, res["bg_engine"])
        except Exception as e:  # noqa: BLE001 - record the failure, keep the worker alive
            log.warning("ingest failed for %s: %s", item.photo_name, e)
            self._jobs.set_status(item.batch_id, item.photo_name, "failed", str(e))
        finally:
            self._jobs.refresh_counts(item.batch_id)

    def _maybe_reload(self) -> None:
        self._since_reload += 1
        if self._queue.empty() or self._since_reload >= RELOAD_EVERY:
            try:
                self._state.reload()
            except Exception:  # noqa: BLE001
                log.exception("index reload failed")
            self._since_reload = 0

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                self.process(item)
                self._maybe_reload()
            finally:
                self._queue.task_done()

    def run_until_empty(self) -> None:
        """Process the queue on the calling thread (tests, CLI)."""
        while not self._queue.empty():
            item = self._queue.get_nowait()
            self.process(item)
            self._queue.task_done()
        self._state.reload()

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="ingest-worker", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            self._queue.put(_STOP)
            self._thread.join(timeout)
            self._thread = None
