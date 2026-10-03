"""Exact cosine search over an in-RAM matrix. One matrix product per query.

ponytail: O(N*d) per query; fine to ~1e5-1e6 rows on CPU. Swap in an ANN
implementation of the VectorIndex port when the catalog outgrows it."""

import numpy as np


class NumpyIndex:
    def __init__(self, vectors: np.ndarray | None = None) -> None:
        self._m = vectors if vectors is not None and len(vectors) else None

    @classmethod
    def from_rows(cls, rows: list[np.ndarray]) -> "NumpyIndex":
        return cls(np.stack(rows) if rows else None)

    def __len__(self) -> int:
        return 0 if self._m is None else len(self._m)

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._m is None:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        sims = self._m @ q
        # reversed stable argsort: identical tie order to the prototype
        return np.argsort(sims)[::-1][:k], sims
