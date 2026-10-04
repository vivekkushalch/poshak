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
from ..domain.errors import NotFound
from ..domain.models import IndexEntry
from ..domain.ports import BlobStore, ImageRepository, VectorIndex
from ..engine import sift
from ..engine.pipeline import Partition
from ..engine.v2 import COARSE_KEYPOINTS, CoarseBank, build_coarse_bank

IndexFactory = Callable[[Sequence[np.ndarray]], VectorIndex]
DESCRIPTOR_CACHE_SIZE = 512  # ~80 candidates per cloth search; repeats hit RAM, not disk/S3


@dataclass(frozen=True, slots=True)
class IndexSnapshot:
    cutouts: Partition  # cloth, background removed
    originals: Partition  # cloth, as photographed
    fabric: Partition
    # RANKER=v2 only: strongest keypoints of every original-view and fabric photo
    originals_bank: CoarseBank | None = None
    fabric_bank: CoarseBank | None = None
    descriptors_ordered: tuple[int, int] = (0, 0)  # (response-ordered, total) in those banks

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
        *,
        build_banks: bool = False,
    ) -> None:
        self._images = images
        self._descriptors = descriptors
        self._model_tag = model_tag
        self._make_index = index_factory
        self._build_banks = build_banks
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

    def _bank(self, entries: list[IndexEntry]) -> tuple[CoarseBank, int]:
        """The coarse bank of one partition, and how many of its descriptor files are
        response-ordered (older files are not; `poshak refresh-descriptors` rewrites them)."""
        descs, ordered = [], 0
        for e in entries:
            try:
                des, _, by_response = sift.loads_ex(self._descriptors.get(e.desc_key))
            except NotFound:
                log.warning("descriptors missing for %s", e.filename)
                des, by_response = np.zeros((0, 128), np.float32), True
            descs.append(des)
            ordered += by_response
        return build_coarse_bank(descs, COARSE_KEYPOINTS), ordered

    def _snapshot(self, cut, orig, fabric) -> IndexSnapshot:
        snap = IndexSnapshot(self._partition(cut), self._partition(orig), self._partition(fabric))
        if not self._build_banks or not (orig or fabric):
            return snap
        orig_bank, orig_ordered = self._bank(orig)
        fabric_bank, fabric_ordered = self._bank(fabric)
        total = len(orig) + len(fabric)
        if orig_ordered + fabric_ordered < total:
            log.warning(
                "%d of %d descriptor files are not response-ordered; the v2 ranker is less "
                "accurate until you run `poshak refresh-descriptors`",
                total - orig_ordered - fabric_ordered,
                total,
            )
        return IndexSnapshot(
            snap.cutouts, snap.originals, snap.fabric, orig_bank, fabric_bank,
            (orig_ordered + fabric_ordered, total),
        )  # fmt: skip

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
