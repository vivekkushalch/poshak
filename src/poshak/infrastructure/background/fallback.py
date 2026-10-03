"""Try the remote remover first, fall back to local U2Net per image.

engine="local" skips the remote; "auto" also skips it when its health check
fails; "api" always tries it. Any remote error falls back to local."""

import numpy as np

from ...core.logging import log
from ...domain.models import BgEngine
from .http import HttpRemover
from .u2net import U2NetRemover


class FallbackRemover:
    def __init__(
        self, primary: HttpRemover | None, fallback: U2NetRemover, default_engine: BgEngine = "auto"
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.default_engine = default_engine

    def remove(self, img_rgb: np.ndarray, engine: BgEngine | None = None) -> tuple[np.ndarray, str]:
        engine = engine or self.default_engine
        if self.primary is None or engine == "local":
            return self.fallback.remove(img_rgb)
        if engine == "auto" and not self.primary.healthy():
            return self.fallback.remove(img_rgb)
        try:
            return self.primary.remove(img_rgb)
        except Exception as e:  # noqa: BLE001
            log.warning("background API failed, using local: %s", e)
            return self.fallback.remove(img_rgb)
