"""The live in-RAM search index and the descriptor cache.

Readers take `state.current` once per request and use that snapshot throughout.
`reload()` builds a complete new snapshot off to the side, then swaps the
reference in one assignment, so a search never sees a half-built index."""

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..core.logging import log, timed
from ..domain.models import IndexEntry
from ..domain.ports import BlobStore, ImageRepository, VectorIndex
from ..engine import sift
from ..engine.pipeline import Partition

IndexFactory = Callable[[Sequence[np.ndarray]], VectorIndex]
DESCRIPTOR_CACHE_SIZE = 512  # ~80 candidates per cloth search; repeats hit RAM, not disk/S3


@dataclass(frozen=True, slots=True)
class IndexSnapshot:
    cutouts: Partition  # cloth, background removed
    originals: Partition  # cloth, as photographed
    fabric: Partition

    @property
    def ready(self) -> bool:
        return any(len(p) for p in (self.cutouts, self.originals, self.fabric))

    def stats(self) -> dict[str, int]:
        return {
            "cloth_bg": len(self.cutouts),
            "cloth_orig": len(self.originals),
            "fabric": len(self.fabric),
        }


class IndexState:
    def __init__(
        self,
        images: ImageRepository,
        descriptors: BlobStore,
        model_tag: str,
        index_factory: IndexFactory,
    ) -> None:
        self._images = images
        self._descriptors = descriptors
        self._model_tag = model_tag
        self._make_index = index_factory
        self._reload_lock = threading.Lock()
        self._current = self._snapshot([], [], [])
        self.load_descriptors = lru_cache(maxsize=DESCRIPTOR_CACHE_SIZE)(self._load_descriptors)

    @property
    def current(self) -> IndexSnapshot:
        return self._current

    def _load_descriptors(self, key: str) -> tuple[np.ndarray, int]:
        return sift.loads(self._descriptors.get(key))

    def _partition(self, entries: list[IndexEntry]) -> Partition:
        return Partition(tuple(entries), self._make_index([e.embed for e in entries]))

    def _snapshot(self, cut, orig, fabric) -> IndexSnapshot:
        return IndexSnapshot(self._partition(cut), self._partition(orig), self._partition(fabric))

    def reload(self) -> dict[str, int]:
        with self._reload_lock, timed("index_reload"):
            cut: list[IndexEntry] = []
            orig: list[IndexEntry] = []
            fabric: list[IndexEntry] = []
            for rec in self._images.all(self._model_tag):
                original, cutout = rec.views()
                if rec.category == "fabric":
                    fabric.append(original)
                    continue
                orig.append(original)
                if cutout is not None:
                    cut.append(cutout)
            snapshot = self._snapshot(cut, orig, fabric)
            self.load_descriptors.cache_clear()  # replaced photos reuse descriptor keys
            self._current = snapshot
        log.info("index loaded: %s", snapshot.stats())
        return snapshot.stats()
