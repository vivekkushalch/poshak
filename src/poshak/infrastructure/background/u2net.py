"""Local background removal with rembg's U2Net. Port of the prototype's remove_bg_rembg.

Runs on the full-size image (best mask); callers downscale afterward. On any
failure it returns the input unchanged, so a bad mask never fails a search."""

import numpy as np
from PIL import Image

from ...core.logging import log, timed
from ...engine.image_ops import composite_on_gray


class U2NetRemover:
    def remove(self, img_rgb: np.ndarray, engine: str = "local") -> tuple[np.ndarray, str]:
        try:
            from rembg import remove  # heavy import, deferred until first use

            with timed("u2net_remove"):
                out = np.array(remove(Image.fromarray(img_rgb)))
            return composite_on_gray(out), "local"
        except Exception as e:  # noqa: BLE001 - degrade to the unmasked image
            log.warning("u2net background removal failed: %s", e)
            return img_rgb, "local"
