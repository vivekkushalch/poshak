"""SIFT geometric verification — split form of spipeline.py sift_match_ratio
(L2029). detect() runs once on the query; match_descriptors() consumes
precomputed candidate descriptors. Match math is identical to the prototype."""

import cv2
import numpy as np


def _sift() -> "cv2.SIFT":
    return cv2.SIFT_create(nfeatures=1200, contrastThreshold=0.02, edgeThreshold=10)


def detect(img_rgb: np.ndarray) -> tuple[int, np.ndarray]:
    """Returns (keypoint_count, descriptors[N,128] float32)."""
    kp, des = _sift().detectAndCompute(img_rgb, None)
    if des is None:
        return 0, np.zeros((0, 128), dtype=np.float32)
    return len(kp), des.astype(np.float32)


def match_descriptors(des_q, n_kp_q, des_c, n_kp_c) -> tuple[float, int, int]:
    """Returns (ratio, good_match_count, n_kp_q). Lowe 0.75, ratio =
    good / max(n_kp_q, n_kp_c) — exactly the prototype's formula."""
    if des_q is None or des_c is None or len(des_q) < 2 or len(des_c) < 2:
        return 0.0, 0, 0
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_q, des_c, k=2)
    good = 0
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good += 1
    denom = max(n_kp_q, n_kp_c)
    ratio = good / denom if denom else 0.0
    return float(ratio), good, n_kp_q
