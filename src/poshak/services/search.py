"""Search use case: decode the query, run the engine on the live index snapshot."""

import gc
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..domain.errors import IndexNotReady, Overloaded
from ..domain.models import Category
from ..domain.ports import BackgroundRemover, Embedder
from ..engine import pipeline
from ..engine.image_ops import decode_rgb, to_working_res
from .index_state import IndexState


class SearchService:
    def __init__(
        self,
        state: IndexState,
        embedder: Embedder,
        query_remover: BackgroundRemover,
        *,
        working_max_h: int = 768,
        max_concurrent: int = 2,
        sift_workers: int = 4,
        queue_timeout_s: float = 10.0,
    ) -> None:
        self._state = state
        self._embedder = embedder
        self._query_remover = query_remover
        self._max_h = working_max_h
        # Each search holds decoded images, a cutout and ~80 descriptor sets; the
        # semaphore caps how many run at once so a burst can't exhaust RAM.
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._queue_timeout_s = queue_timeout_s
        self._pool = ThreadPoolExecutor(sift_workers, thread_name_prefix="sift")

    def search(self, data: bytes, category: Category, top_k: int = 5) -> list[dict[str, Any]]:
        if not self._slots.acquire(timeout=self._queue_timeout_s):
            raise Overloaded("too many concurrent searches")
        try:
            snap = self._state.current
            if not snap.ready:
                raise IndexNotReady("index not loaded yet")
            img = decode_rgb(data)
            query = to_working_res(img, self._max_h)
            load = self._state.load_descriptors
            if category == "cloth":
                # background removal runs on the full-size query for the best mask
                cutout, _ = self._query_remover.remove(img, "local")
                return pipeline.cloth_search(
                    query, to_working_res(cutout, self._max_h), self._embedder.embed,
                    snap.cutouts, snap.originals, load, self._pool, top_k,
                )  # fmt: skip
            return pipeline.fabric_search(
                query, self._embedder.embed, snap.fabric, load, self._pool, top_k
            )
        finally:
            self._slots.release()
            gc.collect()  # transient image and descriptor buffers matter on a 2 GB box

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
