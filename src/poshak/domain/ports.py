"""Ports: the interfaces services depend on. Infrastructure provides implementations.

Adding a backend (S3, Postgres, FAISS, another embedder) means writing one class
that satisfies the matching Protocol and wiring it in container.py."""

from typing import Any, Protocol

import numpy as np

from .models import BgEngine, Category, ImageRecord, JobItem


class BlobStore(Protocol):
    """Flat key -> bytes store for one namespace (originals, cutouts or descriptors)."""

    def get(self, key: str) -> bytes: ...  # raises NotFound
    def put(self, key: str, data: bytes) -> None: ...
    def delete(self, key: str) -> bool: ...  # True if something was removed
    def exists(self, key: str) -> bool: ...


class Embedder(Protocol):
    dim: int

    def embed(self, img_rgb: np.ndarray) -> np.ndarray: ...  # L2-normalised float32[dim]


class BackgroundRemover(Protocol):
    """Returns the cutout composited on flat gray (128) and the engine that produced it."""

    def remove(self, img_rgb: np.ndarray, engine: BgEngine = "auto") -> tuple[np.ndarray, str]: ...


class VectorIndex(Protocol):
    def __len__(self) -> int: ...

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(row indices best-first, cosine similarity of every row)."""
        ...


class ImageRepository(Protocol):
    def upsert(self, rec: ImageRecord) -> None: ...
    def get(self, photo_name: str, model_tag: str) -> ImageRecord | None: ...
    def delete(self, photo_name: str, model_tag: str) -> int: ...
    def all(self, model_tag: str) -> list[ImageRecord]: ...
    def page(
        self, model_tag: str, page: int, page_size: int
    ) -> tuple[int, list[dict[str, Any]]]: ...


class JobRepository(Protocol):
    def add_item(self, item: JobItem) -> None: ...
    def set_status(
        self, batch_id: str, photo_name: str, status: str, error: str | None = None
    ) -> None: ...
    def bump_engine(self, batch_id: str, engine: str) -> None: ...
    def refresh_counts(self, batch_id: str) -> None: ...
    def pending(self) -> list[JobItem]: ...
    def get(self, batch_id: str) -> dict[str, Any] | None: ...
    def list(self) -> list[dict[str, Any]]: ...


class CategoryRepository(Protocol):
    def resolve(self, item: str) -> Category: ...
    def list(self) -> list[dict[str, Any]]: ...
    def set_manual(self, item: str, category: Category) -> None: ...
