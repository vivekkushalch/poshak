"""Background removal — exact port of spipeline.py remove_bg_rembg (L2004).
Runs on the FULL-SIZE image (best mask). Callers apply the 768 cap afterward
via image_io.to_working_res before embedding / descriptor extraction."""

import numpy as np
from PIL import Image
from rembg import remove

from .debug import dbg, timed


def remove_bg_rembg(img_rgb: np.ndarray) -> np.ndarray:
    try:
        with timed("rembg_remove_bg"):
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
    except Exception as e:
        dbg(f"rembg FAILED: {e}")
        return img_rgb
