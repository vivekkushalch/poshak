import io

import numpy as np
import pytest
from PIL import Image

from poshak.infrastructure.background import u2net
from poshak.infrastructure.background.fallback import FallbackRemover
from poshak.infrastructure.background.http import HttpRemover

IMG = np.full((8, 8, 3), 200, np.uint8)


class FakeRemote:
    def __init__(self, healthy=True, fail=False):
        self._healthy, self.fail, self.calls = healthy, fail, 0

    def healthy(self):
        return self._healthy

    def remove(self, img, engine="api"):
        self.calls += 1
        if self.fail:
            raise RuntimeError("down")
        return img, "api"


class FakeLocal:
    def __init__(self):
        self.calls = 0

    def remove(self, img, engine="local"):
        self.calls += 1
        return img, "local"


def test_no_remote_configured_uses_local():
    assert FallbackRemover(None, FakeLocal()).remove(IMG, "api")[1] == "local"


def test_local_engine_never_touches_remote():
    remote = FakeRemote()
    assert FallbackRemover(remote, FakeLocal()).remove(IMG, "local")[1] == "local"
    assert remote.calls == 0


def test_auto_skips_unhealthy_remote():
    remote = FakeRemote(healthy=False)
    assert FallbackRemover(remote, FakeLocal()).remove(IMG, "auto")[1] == "local"
    assert remote.calls == 0


def test_healthy_remote_is_used():
    assert FallbackRemover(FakeRemote(), FakeLocal()).remove(IMG, "auto")[1] == "api"


def test_remote_failure_falls_back_per_image():
    local = FakeLocal()
    _, engine = FallbackRemover(FakeRemote(fail=True), local).remove(IMG, "api")
    assert engine == "local" and local.calls == 1


def test_default_engine_applies_when_none_given():
    remote = FakeRemote()
    assert FallbackRemover(remote, FakeLocal(), default_engine="local").remove(IMG)[1] == "local"
    assert remote.calls == 0


def _png(alpha):
    rgba = np.dstack([np.full((8, 8, 3), 200, np.uint8), np.full((8, 8), alpha, np.uint8)])
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


class _Resp:
    def __init__(self, status_code, content=b""):
        self.status_code, self.content = status_code, content


def test_http_remover_composites_transparent_pixels_to_gray(monkeypatch):
    r = HttpRemover("http://bg.test", retries=0)
    monkeypatch.setattr(r, "_post", lambda data: _Resp(200, _png(alpha=0)))
    out, engine = r.remove(IMG)
    assert engine == "api" and out.shape == (8, 8, 3) and np.all(out == 128)


def test_http_remover_retries_then_raises(monkeypatch):
    r = HttpRemover("http://bg.test", retries=2)
    calls = []
    monkeypatch.setattr(r, "_post", lambda data: calls.append(1) or _Resp(500))
    with pytest.raises(RuntimeError):
        r.remove(IMG)
    assert len(calls) == 3


def test_u2net_keeps_full_resolution(monkeypatch):
    import rembg

    def fake_remove(pil_img):
        arr = np.array(pil_img)
        return Image.fromarray(np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)]))

    monkeypatch.setattr(rembg, "remove", fake_remove)
    img = np.full((1200, 900, 3), 70, np.uint8)
    out, engine = u2net.U2NetRemover().remove(img)
    assert engine == "local" and out.shape == (1200, 900, 3)


def test_u2net_failure_returns_input(monkeypatch):
    import rembg

    def boom(pil_img):
        raise RuntimeError("model missing")

    monkeypatch.setattr(rembg, "remove", boom)
    out, _ = u2net.U2NetRemover().remove(IMG)
    assert out is IMG
