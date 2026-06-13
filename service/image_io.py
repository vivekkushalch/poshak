"""Image I/O helpers — port of the prototype's `load_image`,
`find_actual_file`, `ensure_image`.

Matches the notebook's behavior exactly:
  - load_image returns a uint8 RGB ndarray (cv2 BGR → RGB)
  - find_actual_file does graceful extension fuzzing so a `.jpg` listed
    in MySQL still resolves when the on-disk file is `.JPEG` or `.png`
  - ensure_image is the lazy loader used inside the search loop
"""

import glob
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .highpass import highpass_trace


def load_image(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _resize_max(img: np.ndarray, max_dim: int) -> np.ndarray:
    """Cap the longer side at max_dim. SIFT is scale-invariant, so
    downsizing candidates for SIFT doesn't hurt match counts much
    but drops decoded RAM 4-9x for typical phone photos."""
    h, w = img.shape[:2]
    longer = max(h, w)
    if longer <= max_dim:
        return img
    scale = max_dim / float(longer)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# Cap candidate / query images used in SIFT to this many px on the
# longer side. 800 keeps fine-detail features (embroidery, weave) while
# making each decoded image ~2MB instead of 10-15MB.
SIFT_MAX_DIM = 800


def find_actual_file(source_dir: Path, filename: str) -> Optional[str]:
    """Port of the prototype's `find_actual_file`. Walks extension
    variants so a CSV typo in `stock.photos` still resolves."""
    source_dir = Path(source_dir)
    if not source_dir.exists():
        return None
    exact = source_dir / filename
    if exact.exists():
        return str(exact)
    base = os.path.splitext(filename)[0]
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"):
        cand = source_dir / (base + ext)
        if cand.exists():
            return str(cand)
    matches = glob.glob(str(source_dir / f"{base}.*"))
    return matches[0] if matches else None


def ensure_image(entry: dict) -> Optional[np.ndarray]:
    """Load a candidate image for SIFT. Returns full-resolution RGB
    (matches the prototype — downsizing destroys SIFT distinctiveness
    at scale).

    Self-healing: if the stored path doesn't exist (e.g., stock.photos
    CSV said "abc.jpg" but disk has "abc.JPEG" — Linux is case-sensitive
    so the exact path fails), retry with extension fuzzing and cache
    the corrected path back on the entry so next time is fast.

    We deliberately do NOT cache the decoded image — that grew RAM
    unboundedly across searches. The highpass trace IS cached (~50KB).
    """
    img = load_image(entry["path"])
    if img is None:
        p = Path(entry["path"])
        resolved = find_actual_file(p.parent, p.name)
        if resolved is None:
            return None
        entry["path"] = resolved
        img = load_image(resolved)
        if img is None:
            return None
    if entry.get("trace") is None:
        entry["trace"] = highpass_trace(img)
    return img
