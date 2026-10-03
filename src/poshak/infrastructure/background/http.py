"""Background removal through an HTTP service (e.g. BiRefNet): GET /health, POST /remove."""

import io

import httpx
import numpy as np
from PIL import Image

from ...core.logging import log, timed
from ...engine.image_ops import composite_on_gray, encode_png, to_working_res


class HttpRemover:
    def __init__(self, url: str, timeout: float = 360, retries: int = 1, max_h: int = 0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.max_h = max_h  # optional downscale before upload to cut inference time

    def healthy(self) -> bool:
        try:
            return httpx.get(f"{self.url}/health", timeout=10).status_code == 200
        except Exception as e:  # noqa: BLE001 - any failure means "not healthy"
            log.info("background API health check failed: %s", e)
            return False

    def _post(self, data: bytes) -> httpx.Response:
        return httpx.post(
            f"{self.url}/remove",
            files={"file": ("image.png", data, "image/png")},
            timeout=self.timeout,
        )

    def remove(self, img_rgb: np.ndarray, engine: str = "api") -> tuple[np.ndarray, str]:
        """One call plus `retries` retries. Raises on failure; FallbackRemover handles it."""
        src = to_working_res(img_rgb, self.max_h) if self.max_h else img_rgb
        data = encode_png(src)
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with timed("http_remove"):
                    resp = self._post(data)
                if resp.status_code != 200:
                    raise RuntimeError(f"background API returned {resp.status_code}")
                png = np.array(Image.open(io.BytesIO(resp.content)).convert("RGBA"))
                return composite_on_gray(png), "api"
            except Exception as e:  # noqa: BLE001 - retry, then let the caller fall back
                last = e
                log.info("background API attempt %d failed: %s", attempt, e)
        raise last or RuntimeError("background API failed")
