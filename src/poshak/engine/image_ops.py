"""Image decode/encode and the working-resolution resize.

The resize is a port of the research prototype's load_image(max_h=768): scale by
height with INTER_AREA, only when taller than max_h, never crop or distort."""

import cv2
import numpy as np

from ..domain.errors import InvalidImage

WORKING_MAX_H = 768
GRAY = 128


def decode_rgb(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImage("cannot decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encode_png(img_rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise InvalidImage("cannot encode PNG")
    return buf.tobytes()


def to_working_res(img: np.ndarray, max_h: int = WORKING_MAX_H) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= max_h:
        return img
    scale = max_h / float(h)
    return cv2.resize(img, (max(1, int(w * scale)), max_h), interpolation=cv2.INTER_AREA)


def composite_on_gray(rgba: np.ndarray) -> np.ndarray:
    """RGBA cutout -> RGB over flat gray. RGB input passes through."""
    if rgba.ndim == 3 and rgba.shape[2] == 4:
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        fg = rgba[:, :, :3].astype(np.float32)
        return (fg * alpha + GRAY * (1 - alpha)).astype(np.uint8)
    return rgba[:, :, :3]
