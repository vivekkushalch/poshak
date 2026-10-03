"""Image I/O helpers — port of the prototype's `load_image` and
`find_actual_file`.

Matches the notebook's behavior exactly:
  - load_image returns a uint8 RGB ndarray (cv2 BGR → RGB)
  - find_actual_file does graceful extension fuzzing so a `.jpg` listed
    in MySQL still resolves when the on-disk file is `.JPEG` or `.png`
  - to_working_res scales by HEIGHT to max_h (default config.WORKING_MAX_H),
    proportional, INTER_AREA — only shrinks, never crops or distorts
"""

import glob
import os
from pathlib import Path

import cv2
import numpy as np

from . import config


def load_image(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def find_actual_file(source_dir: Path, filename: str) -> str | None:
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


# Port of the research prototype's load_image (max_h=768): scale by HEIGHT, INTER_AREA,
# only when taller than max_h. Proportional — never crops or distorts.
def to_working_res(img: np.ndarray, max_h: int | None = None) -> np.ndarray:
    max_h = config.WORKING_MAX_H if max_h is None else max_h
    h, w = img.shape[:2]
    if h <= max_h:
        return img
    scale = max_h / float(h)
    new_w = max(1, int(w * scale))
    return cv2.resize(img, (new_w, max_h), interpolation=cv2.INTER_AREA)
