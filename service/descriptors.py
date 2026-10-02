"""Precomputed SIFT descriptors.

Candidate descriptors never change, so we compute them once (offline / on
ingest) and cache them. A search loads only the ~80 selected candidates'
descriptors from disk. SIFT params match the research prototype's sift_match_ratio."""

from functools import lru_cache

import cv2
import numpy as np


def _sift() -> "cv2.SIFT":
    return cv2.SIFT_create(nfeatures=1200, contrastThreshold=0.02, edgeThreshold=10)


def compute_descriptors(img_rgb: np.ndarray) -> tuple[np.ndarray, int]:
    """Returns (descriptors[N,128] float32, keypoint_count). img must already
    be at working resolution (768) — callers resize first."""
    kp, des = _sift().detectAndCompute(img_rgb, None)
    if des is None:
        return np.zeros((0, 128), dtype=np.float32), 0
    return des.astype(np.float32), len(kp)


def save_descriptors(path: str, des: np.ndarray, n_kp: int) -> None:
    np.savez_compressed(path, des=des, n_kp=np.int32(n_kp))


@lru_cache(maxsize=512)
def load_descriptors(path: str) -> tuple[np.ndarray, int]:
    with np.load(path) as z:
        return z["des"].astype(np.float32), int(z["n_kp"])
