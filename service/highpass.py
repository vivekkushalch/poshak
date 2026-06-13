"""Highpass texture trace + IoU — port of the prototype's
`highpass_trace` and `compute_iou`.

The texture trace is stored on every index entry. IoU between query and
candidate is reported in search results but is NOT used in the final
ranking (it's a visualization / diagnostic only — keep parity with the
prototype here).
"""

import cv2
import numpy as np


def highpass_trace(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (25, 25), 5)
    texture = gray - blur
    texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(texture, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def compute_iou(bin1: np.ndarray, bin2: np.ndarray) -> float:
    h = max(bin1.shape[0], bin2.shape[0])
    w = max(bin1.shape[1], bin2.shape[1])
    b1 = cv2.resize(bin1, (w, h), interpolation=cv2.INTER_NEAREST)
    b2 = cv2.resize(bin2, (w, h), interpolation=cv2.INTER_NEAREST)
    inter = np.logical_and(b1 > 0, b2 > 0).sum()
    union = np.logical_or(b1 > 0, b2 > 0).sum()
    return float(inter / (union + 1e-8))
