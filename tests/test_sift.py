import numpy as np

from poshak import sift


def _legacy_sift_match_ratio(img1, img2):
    """Inlined copy of the prototype's sift_match_ratio
    used ONLY as a parity oracle in the test."""
    import cv2

    s = cv2.SIFT_create(nfeatures=1200, contrastThreshold=0.02, edgeThreshold=10)
    kp1, des1 = s.detectAndCompute(img1, None)
    kp2, des2 = s.detectAndCompute(img2, None)
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0.0, 0, 0
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [
        m for m_n in matches if len(m_n) == 2 for m, n in [m_n] if m.distance < 0.75 * n.distance
    ]
    ratio = len(good) / max(len(kp1), len(kp2)) if kp1 and kp2 else 0.0
    return float(ratio), len(good), len(kp1)


def test_detect_then_match_equals_legacy(textured_rgb):
    img2 = np.roll(textured_rgb, 8, axis=1)  # shifted copy -> real matches
    n1, d1 = sift.detect(textured_rgb)
    n2, d2 = sift.detect(img2)
    ratio, good, total_kp = sift.match_descriptors(d1, n1, d2, n2)
    exp_ratio, exp_good, exp_total = _legacy_sift_match_ratio(textured_rgb, img2)
    assert good == exp_good
    assert total_kp == exp_total
    assert abs(ratio - exp_ratio) < 1e-9


def test_empty_descriptors_safe():
    empty = np.zeros((0, 128), np.float32)
    assert sift.match_descriptors(empty, 0, empty, 0) == (0.0, 0, 0)
