"""Indexing use cases: add, replace and forget catalog photos.

One code path serves the HTTP API, the background job worker and `poshak seed`.
Writes are serialised by one lock; the live index reloads after each change
unless the caller batches reloads (the job worker does)."""

import threading
from pathlib import PurePath
from typing import Any

from ..domain.errors import NotFound
from ..domain.models import BgEngine, Category, ImageKind, ImageRecord
from ..domain.ports import BackgroundRemover, BlobStore, Embedder, ImageRepository
from ..engine import sift
from ..engine.image_ops import decode_rgb, encode_png, to_working_res
from .catalog import CatalogService
from .index_state import IndexState


def cutout_name(photo_name: str) -> str:
    return f"{PurePath(photo_name).stem}.png"


def descriptor_keys(photo_name: str) -> tuple[str, str]:
    return f"{photo_name}.orig.npz", f"{photo_name}.rmbg.npz"


class IndexerService:
    def __init__(
        self,
        *,
        images: ImageRepository,
        catalog: CatalogService,
        originals: BlobStore,
        cutouts: BlobStore,
        descriptors: BlobStore,
        embedder: Embedder,
        remover: BackgroundRemover,
        state: IndexState,
        model_tag: str,
        working_max_h: int = 768,
    ) -> None:
        self._images = images
        self._catalog = catalog
        self._originals = originals
        self._cutouts = cutouts
        self._descriptors = descriptors
        self._embedder = embedder
        self._remover = remover
        self._state = state
        self._model_tag = model_tag
        self._max_h = working_max_h
        self._lock = threading.Lock()

    # --- features ---------------------------------------------------------------
    def _features(self, img_rgb, key: str):
        """Embed + SIFT-describe one view at working resolution; store its descriptors."""
        view = to_working_res(img_rgb, self._max_h)
        vec = self._embedder.embed(view)
        des, n_kp = sift.compute_descriptors(view)
        self._descriptors.put(key, sift.dumps(des, n_kp))
        return vec

    def _build(
        self, photo_name, full, *, sku_id, designno, item, category: Category,
        engine: BgEngine, reuse_cutout: bool,
    ) -> ImageRecord:  # fmt: skip
        orig_key, cut_key = descriptor_keys(photo_name)
        rec = ImageRecord(
            sku_id=sku_id, designno=designno or "", item=item or "", category=category,
            photo_name=photo_name, orig_vec=self._features(full, orig_key), orig_desc=orig_key,
            model_tag=self._model_tag,
        )  # fmt: skip
        if category == "cloth":
            name = cutout_name(photo_name)
            if reuse_cutout and self._cutouts.exists(name):
                cut, rec.bg_engine = decode_rgb(self._cutouts.get(name)), None
            else:
                cut, rec.bg_engine = self._remover.remove(full, engine)
                self._cutouts.put(name, encode_png(cut))
            rec.rembg_name, rec.rembg_desc = name, cut_key
            rec.rembg_vec = self._features(cut, cut_key)
        return rec

    # --- use cases --------------------------------------------------------------
    def stage(self, photo_name: str, data: bytes) -> None:
        self._originals.put(photo_name, data)

    def is_indexed(self, photo_name: str) -> bool:
        rec = self._images.get(photo_name, self._model_tag)
        return rec is not None and self._descriptors.exists(rec.orig_desc)

    def index(
        self,
        photo_name: str,
        sku_id: int,
        designno: str,
        item: str,
        *,
        engine: BgEngine = "auto",
        force: bool = False,
        reuse_cutout: bool = False,
        reload: bool = True,
    ) -> dict[str, Any]:
        """Index a photo already staged in the originals store. Idempotent unless force."""
        category = self._catalog.resolve(item)
        if not force and self.is_indexed(photo_name):
            return {"status": "skipped", "reason": "already indexed", "photo_name": photo_name}
        with self._lock:
            full = decode_rgb(self._originals.get(photo_name))
            rec = self._build(
                photo_name, full, sku_id=sku_id, designno=designno, item=item,
                category=category, engine=engine, reuse_cutout=reuse_cutout,
            )  # fmt: skip
            self._images.upsert(rec)
        if reload:
            self._state.reload()
        return {
            "status": "indexed",
            "photo_name": photo_name,
            "category": category,
            "bg_engine": rec.bg_engine,
        }

    def replace(self, photo_name: str, kind: ImageKind, data: bytes) -> dict[str, Any]:
        """'original': re-derive everything; 'rmbg': swap in a hand-made cutout."""
        with self._lock:
            rec = self._images.get(photo_name, self._model_tag)
            if rec is None:
                raise NotFound(f"not indexed: {photo_name}")
            if kind == "original":
                self._originals.put(photo_name, data)
                rec = self._build(
                    photo_name, decode_rgb(data), sku_id=rec.sku_id, designno=rec.designno,
                    item=rec.item, category=rec.category, engine="auto", reuse_cutout=False,
                )  # fmt: skip
            else:
                name, (_, cut_key) = cutout_name(photo_name), descriptor_keys(photo_name)
                self._cutouts.put(name, data)
                rec.rembg_name, rec.rembg_desc, rec.bg_engine = name, cut_key, "manual"
                rec.rembg_vec = self._features(decode_rgb(data), cut_key)
            self._images.upsert(rec)
            self._state.reload()
        return {"status": "replaced", "photo_name": photo_name, "kind": kind}

    def forget(self, photo_name: str) -> dict[str, int]:
        orig_key, cut_key = descriptor_keys(photo_name)
        with self._lock:
            removed = sum(
                store.delete(key)
                for store, key in (
                    (self._originals, photo_name),
                    (self._cutouts, cutout_name(photo_name)),
                    (self._descriptors, orig_key),
                    (self._descriptors, cut_key),
                )
            )
            deleted = self._images.delete(photo_name, self._model_tag)
            self._state.reload()
        return {"files_removed": removed, "db_deleted": deleted}
