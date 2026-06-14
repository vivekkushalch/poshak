"""Indexing-time background removal.

Primary: the external birefnet API (config.REMBG_API_URL). Per image:
REMBG_API_TIMEOUT seconds, REMBG_API_RETRIES retries, then per-image fallback
to the local U2Net rembg. Returns (cutout-on-gray RGB, engine_used).

NOT used on the search path — query bg-removal stays local (bg_removal.py)."""

import io

import cv2
import numpy as np
from PIL import Image

from . import config
from .bg_removal import remove_bg_rembg  # local U2Net fallback
from .debug import dbg, timed
from .image_io import to_working_res


def _composite_on_gray(rgba: np.ndarray) -> np.ndarray:
    """RGBA (or RGB) -> RGB composited over flat gray (128,128,128)."""
    if rgba.ndim == 3 and rgba.shape[2] == 4:
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        fg = rgba[:, :, :3].astype(np.float32)
        bg = np.full(fg.shape, 128, dtype=np.float32)
        return (fg * alpha + bg * (1 - alpha)).astype(np.uint8)
    return rgba[:, :, :3]


def _encode(img_rgb: np.ndarray) -> bytes:
    src = img_rgb
    if config.REMBG_API_MAX_H:
        src = to_working_res(img_rgb, max_h=config.REMBG_API_MAX_H)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(src, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _api_health_ok() -> bool:
    import httpx
    try:
        r = httpx.get(f"{config.REMBG_API_URL}/health", timeout=10)
        return r.status_code == 200
    except Exception as e:
        dbg(f"rembg api health check failed: {e}")
        return False


def _api_post_remove(data: bytes):
    import httpx
    return httpx.post(
        f"{config.REMBG_API_URL}/remove",
        files={"file": ("image.png", data, "image/png")},
        timeout=config.REMBG_API_TIMEOUT,
    )


def _api_remove(img_rgb: np.ndarray) -> np.ndarray:
    """One API attempt (+ retries). Raises on failure; caller handles fallback."""
    data = _encode(img_rgb)
    last = None
    for attempt in range(config.REMBG_API_RETRIES + 1):
        try:
            with timed("rembg_api_remove"):
                resp = _api_post_remove(data)
            if resp.status_code != 200:
                raise RuntimeError(f"api status {resp.status_code}")
            png = np.array(Image.open(io.BytesIO(resp.content)).convert("RGBA"))
            return _composite_on_gray(png)
        except Exception as e:  # noqa: BLE001 - retry then fall back
            last = e
            dbg(f"rembg api attempt {attempt} failed: {e}")
    raise last if last else RuntimeError("rembg api failed")


def remove_bg_for_index(img_rgb: np.ndarray, engine: str = "auto"):
    """Returns (rgb_on_gray, engine_used in {'api','local'}).
    engine: 'auto' (api->local), 'api' (api with per-image local fallback on
    error), or 'local' (skip api entirely)."""
    engine = engine or config.REMBG_DEFAULT_ENGINE
    if engine == "local":
        return remove_bg_rembg(img_rgb), "local"
    if engine == "auto" and not _api_health_ok():
        return remove_bg_rembg(img_rgb), "local"
    try:
        return _api_remove(img_rgb), "api"
    except Exception as e:  # noqa: BLE001
        dbg(f"rembg api failed, falling back to local: {e}")
        return remove_bg_rembg(img_rgb), "local"
