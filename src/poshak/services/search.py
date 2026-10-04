"""Search use case: decode the query, run the engine on the live index snapshot."""

import gc
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..domain.errors import IndexNotReady, Overloaded
from ..domain.models import Category
from ..domain.ports import BackgroundRemover, Embedder
from ..engine import pipeline, sift, v2
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
        ranker: str = "legacy",
        v2_max_views: int = 1000,
    ) -> None:
        self._state = state
        self._embedder = embedder
        self._query_remover = query_remover
        self._max_h = working_max_h
        # Each search holds decoded images, a cutout and ~80 descriptor sets; the
        # semaphore caps how many run at once so a burst can't exhaust RAM.
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._queue_timeout_s = queue_timeout_s
        self._ranker = ranker
        self._v2_max_views = v2_max_views
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
            if self.active_ranker(category) == "v2":
                return self._search_v2(query, category, top_k)
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

    def active_ranker(self, category: Category) -> str:
        """The ranker that will serve this category now: v2 when it is enabled, its coarse
        banks are built and the catalog is small enough, otherwise legacy."""
        if self._ranker != "v2":
            return "legacy"
        snap = self._state.current
        part, bank = (
            (snap.originals, snap.originals_bank)
            if category == "cloth"
            else (snap.fabric, snap.fabric_bank)
        )
        usable = bank is not None and 0 < len(part) <= self._v2_max_views
        return "v2" if usable else "legacy"

    def _search_v2(self, query, category: Category, top_k: int) -> list[dict[str, Any]]:
        snap = self._state.current
        part, bank = (
            (snap.originals, snap.originals_bank)
            if category == "cloth"
            else (snap.fabric, snap.fabric_bank)
        )
        des, _ = sift.compute_descriptors_by_response(query)
        ranked = v2.rank(
            des, lambda: self._embedder.embed(query), part, bank,
            self._state.load_descriptors, top_k,
        )  # fmt: skip
        return [_v2_result(r, category) for r in ranked]

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def _tier(matches: int) -> str:
    """Same thresholds as the legacy confidence tiers, applied to mutual matches."""
    return (
        "HIGH" if matches >= 30 else "MED" if matches >= 15 else "LOW" if matches >= 5 else "WEAK"
    )


def _v2_result(r: v2.Ranked, category: Category) -> dict[str, Any]:
    """A v2 hit in the API's result shape. There is no cutout pass, so the cutout fields
    of a cloth result are empty."""
    result = {
        "sku_id": r.sku_id,
        "unified_score": r.score,
        "total_sift_matches": r.matches,
        "best_entry": r.entry,
    }
    if category == "fabric":
        return {**result, "tier": _tier(r.matches)}
    return {
        **result,
        "bg_sift": 0,
        "orig_sift": r.matches,
        "bg_tier": "NONE",
        "orig_tier": _tier(r.matches),
        "bg_score": 0.0,
        "orig_score": r.score,
        "both_found": False,
    }
