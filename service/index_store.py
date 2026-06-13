"""In-memory index store — port of the prototype's three numpy lists
plus its `prepare_matrix` precomputation.

The prototype keeps three module-level globals after `build_index`:

    cloth_bg_index    = build_index(cloth_skus, BG_DIR,     "CLOTH_BG_REMOVED")
    cloth_orig_index  = build_index(cloth_skus, CLOTH_DIR,  "CLOTH_ORIGINAL")
    fabric_index      = build_index(fabric_skus, FABRIC_DIR,"FABRIC_ORIGINAL")

then a tiny precomputation:

    def prepare_matrix(index):
        if not index: return None, []
        return np.stack([e['embed'] for e in index]), index

so cosine for stage 1 of search is one `embeds @ q` dot product.

This module owns those globals and the loader that populates them.
Step 1 ships an empty store; Step 2 will populate it from MySQL.
"""

from typing import Optional

import numpy as np

from . import config


# Per-photo dict shape (matches the prototype's index entries):
#   {
#       'sku_id':        int,            # = stock.id
#       'designno':      str,
#       'item':          str,            # 'Sherwani' / 'Fabric' / ...
#       'filename':      str,            # actual file on disk
#       'json_filename': str,            # filename as listed in JSON
#       'path':          str,            # absolute path
#       'img':           np.ndarray|None, # lazy (None until ensure_image)
#       'embed':         np.ndarray,     # 768-d L2-normalized
#       'trace':         np.ndarray|None,# highpass binary, lazy
#   }


class IndexStore:
    """Holds the three indexes + their precomputed embed matrices."""

    def __init__(self) -> None:
        self.cloth_bg: list[dict] = []
        self.cloth_orig: list[dict] = []
        self.fabric: list[dict] = []

        self.cloth_bg_embeds: Optional[np.ndarray] = None
        self.cloth_orig_embeds: Optional[np.ndarray] = None
        self.fabric_embeds: Optional[np.ndarray] = None

    def prepare(self) -> None:
        """Port of `prepare_matrix`. Call after any (re)load."""
        self.cloth_bg_embeds = _stack(self.cloth_bg)
        self.cloth_orig_embeds = _stack(self.cloth_orig)
        self.fabric_embeds = _stack(self.fabric)

    def is_ready(self) -> bool:
        return any(
            x is not None and len(x) > 0
            for x in (self.cloth_bg, self.cloth_orig, self.fabric)
        )

    def stats(self) -> dict:
        return {
            "cloth_bg": len(self.cloth_bg),
            "cloth_orig": len(self.cloth_orig),
            "fabric": len(self.fabric),
            "model_tag": config.MODEL_TAG,
        }


def _stack(index: list[dict]) -> Optional[np.ndarray]:
    if not index:
        return None
    return np.stack([e["embed"] for e in index])


# Module-level singleton — the FastAPI app and the search pipeline both
# read from this instance.
STORE = IndexStore()
