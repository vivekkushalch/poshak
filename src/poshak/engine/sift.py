"""SIFT keypoints and Lowe-ratio matching, split for precomputed candidates.

Parameters and match math are the research prototype's sift_match_ratio verbatim;
they are tuned together with engine/scoring.py, so never change one alone."""

import io

import cv2
import numpy as np

LOWE_RATIO = 0.75
_EMPTY = np.zeros((0, 128), dtype=np.float32)


def _sift() -> "cv2.SIFT":
    return cv2.SIFT_create(nfeatures=1200, contrastThreshold=0.02, edgeThreshold=10)


def compute_descriptors(img_rgb: np.ndarray) -> tuple[np.ndarray, int]:
    """(descriptors[N,128] float32, keypoint count). Image must be at working res."""
    kp, des = _sift().detectAndCompute(img_rgb, None)
    if des is None:
        return _EMPTY, 0
    return des.astype(np.float32), len(kp)


def detect(img_rgb: np.ndarray) -> tuple[int, np.ndarray]:
    """Query-side form: (keypoint count, descriptors)."""
    des, n = compute_descriptors(img_rgb)
    return n, des


def match_descriptors(des_q, n_kp_q, des_c, n_kp_c) -> tuple[float, int, int]:
    """(ratio, good match count, n_kp_q); ratio = good / max(n_kp_q, n_kp_c)."""
    if des_q is None or des_c is None or len(des_q) < 2 or len(des_c) < 2:
        return 0.0, 0, 0
    good = 0
    for m_n in cv2.BFMatcher().knnMatch(des_q, des_c, k=2):
        if len(m_n) == 2 and m_n[0].distance < LOWE_RATIO * m_n[1].distance:
            good += 1
    denom = max(n_kp_q, n_kp_c)
    return float(good / denom if denom else 0.0), good, n_kp_q


def dumps(des: np.ndarray, n_kp: int) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, des=des, n_kp=np.int32(n_kp))
    return buf.getvalue()


def loads(data: bytes) -> tuple[np.ndarray, int]:
    with np.load(io.BytesIO(data)) as z:
        return z["des"].astype(np.float32), int(z["n_kp"])
