"""SIFT geometric verification — port of the prototype's
`sift_match_ratio`.

Primary ranking signal of the whole search pipeline. Returns the
good-match COUNT (Lowe ratio 0.75); SigLIP cosine is only a 0.05
tiebreaker downstream.
"""

import cv2
import numpy as np


def sift_match_ratio(
    img1: np.ndarray,
    img2: np.ndarray,
    nfeatures: int = 1200,
) -> tuple[float, int, int]:
    """Returns (ratio, good_match_count, kp1_count).

    `ratio = len(good) / max(len(kp1), len(kp2))`.
    `good` = Lowe ratio test, `m.distance < 0.75 * n.distance`.

    Constants (nfeatures=1200, contrastThreshold=0.02, edgeThreshold=10)
    are the prototype's — do not retune without re-calibrating the
    confidence tiers in scoring.py.
    """
    sift = cv2.SIFT_create(
        nfeatures=nfeatures, contrastThreshold=0.02, edgeThreshold=10
    )
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0.0, 0, 0
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)
    ratio = len(good) / max(len(kp1), len(kp2)) if kp1 and kp2 else 0.0
    return float(ratio), len(good), len(kp1)
