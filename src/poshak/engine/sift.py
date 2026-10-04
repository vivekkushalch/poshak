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


def compute_descriptors_by_response(img_rgb: np.ndarray) -> tuple[np.ndarray, int]:
    """The same keypoints as compute_descriptors, strongest first. RANKER=v2 stores and
    queries this order so that the first K rows are a view's K best keypoints."""
    kp, des = _sift().detectAndCompute(img_rgb, None)
    if des is None:
        return _EMPTY, 0
    order = np.argsort([-k.response for k in kp], kind="stable")
    return des[order].astype(np.float32), len(kp)


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


def dumps(des: np.ndarray, n_kp: int, *, by_response: bool = False) -> bytes:
    """Serialise one view's descriptors. SIFT descriptors are integers in 0..255, so
    response-ordered (v2) files store uint8; the legacy format is unchanged."""
    buf = io.BytesIO()
    if by_response:
        np.savez_compressed(
            buf, des=des.astype(np.uint8), n_kp=np.int32(n_kp), by_response=np.bool_(True)
        )
    else:
        np.savez_compressed(buf, des=des, n_kp=np.int32(n_kp))
    return buf.getvalue()


def loads_ex(data: bytes) -> tuple[np.ndarray, int, bool]:
    """(descriptors, keypoint count, whether the rows are strongest-first)."""
    with np.load(io.BytesIO(data)) as z:
        ordered = bool(z["by_response"]) if "by_response" in z.files else False
        return z["des"].astype(np.float32), int(z["n_kp"]), ordered


def loads(data: bytes) -> tuple[np.ndarray, int]:
    des, n_kp, _ = loads_ex(data)
    return des, n_kp
