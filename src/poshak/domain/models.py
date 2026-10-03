"""Domain types shared by services, engine and infrastructure. No framework imports."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Category = Literal["cloth", "fabric"]
BgEngine = Literal["auto", "api", "local"]
ImageKind = Literal["original", "rmbg"]
JobStatus = Literal["queued", "running", "done", "failed", "skipped"]


@dataclass(frozen=True, slots=True, eq=False)
class IndexEntry:
    """One searchable image view (an original photo, or its background-removed cutout)."""

    sku_id: int
    designno: str
    item: str
    filename: str  # the view's own file: original photo or cutout PNG
    json_filename: str  # always the original photo, which is what clients display
    desc_key: str  # descriptor blob key
    embed: np.ndarray = field(repr=False)


@dataclass(slots=True, eq=False)
class ImageRecord:
    """One catalog photo as stored: both views' vectors and descriptor keys."""

    sku_id: int
    designno: str
    item: str
    category: Category
    photo_name: str
    orig_vec: np.ndarray = field(repr=False)
    orig_desc: str
    model_tag: str
    rembg_name: str | None = None
    rembg_vec: np.ndarray | None = field(default=None, repr=False)
    rembg_desc: str | None = None
    bg_engine: str | None = None

    def views(self) -> tuple[IndexEntry, IndexEntry | None]:
        """(original view, cutout view or None)."""
        orig = IndexEntry(
            self.sku_id,
            self.designno,
            self.item,
            self.photo_name,
            self.photo_name,
            self.orig_desc,
            self.orig_vec,
        )
        if self.rembg_vec is None or not self.rembg_desc or not self.rembg_name:
            return orig, None
        cut = IndexEntry(
            self.sku_id,
            self.designno,
            self.item,
            self.rembg_name,
            self.photo_name,
            self.rembg_desc,
            self.rembg_vec,
        )
        return orig, cut


@dataclass(frozen=True, slots=True)
class JobItem:
    batch_id: str
    photo_name: str
    sku_id: int
    designno: str
    item: str
    bg_engine: BgEngine = "auto"
    force: bool = False
