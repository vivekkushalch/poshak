"""Background removal — port of the prototype's `remove_bg_rembg`.

Cloth-only. rembg + U2Net default. Output is composited over flat gray
(128,128,128) so the foreground sits on a neutral, predictable background
that doesn't push the SigLIP embedding around.

Parity note: the prototype (spipeline.py) never fed rembg a full-res
image. Every image first passed through `load_image(path, max_h=768)`,
which caps HEIGHT to 768px (INTER_AREA, only when taller). We reproduce
that cap here so the bg-removed embeddings match the notebook exactly.
"""

import cv2
import numpy as np
from PIL import Image
from rembg import remove

from .debug import dbg, timed


# Verbatim from the prototype's load_image default (max_h=768). Bg
# removal runs on images capped to this height — not full resolution.
REMBG_MAX_H = 768


def _cap_height(img_rgb: np.ndarray, max_h: int = REMBG_MAX_H) -> np.ndarray:
    """Port of the prototype's load_image resize step: scale by HEIGHT
    to max_h with INTER_AREA, only when the image is taller than max_h."""
    h, w = img_rgb.shape[:2]
    if h > max_h:
        scale = max_h / h
        img_rgb = cv2.resize(
            img_rgb, (int(w * scale), max_h), interpolation=cv2.INTER_AREA
        )
    return img_rgb


def remove_bg_rembg(img_rgb: np.ndarray) -> np.ndarray:
    """Verbatim port of the prototype:

        def remove_bg_rembg(img_rgb):
            pil_img = Image.fromarray(img_rgb)
            output = remove(pil_img)
            arr = np.array(output)
            bg = np.full((arr.shape[0], arr.shape[1], 3), 128, dtype=np.uint8)
            if arr.shape[2] == 4:
                alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
                fg = arr[:, :, :3].astype(np.float32)
                bg = (fg * alpha + bg.astype(np.float32) * (1 - alpha)).astype(np.uint8)
            else:
                bg = arr[:, :, :3]
            return bg

    Returns the original image unchanged if rembg fails (matches prototype's
    try/except — keeps the search pipeline alive even when bg removal trips).
    """
    try:
        with timed("rembg_remove_bg"):
            # Match the prototype: it removed background on images already
            # capped to 768px height by load_image, never on full-res.
            pil_img = Image.fromarray(_cap_height(img_rgb))
            output = remove(pil_img)
            arr = np.array(output)
            bg = np.full((arr.shape[0], arr.shape[1], 3), 128, dtype=np.uint8)
            if arr.shape[2] == 4:
                alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
                fg = arr[:, :, :3].astype(np.float32)
                bg = (fg * alpha + bg.astype(np.float32) * (1 - alpha)).astype(np.uint8)
            else:
                bg = arr[:, :, :3]
            return bg
    except Exception as e:
        dbg(f"rembg FAILED: {e}")
        return img_rgb
