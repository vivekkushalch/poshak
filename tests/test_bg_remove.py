import io
import numpy as np
from PIL import Image
from service import bg_remove


def _png_rgba(h=8, w=8, alpha=255, rgb=200):
    arr = np.dstack([
        np.full((h, w), rgb, np.uint8),
        np.full((h, w), rgb, np.uint8),
        np.full((h, w), rgb, np.uint8),
        np.full((h, w), alpha, np.uint8),
    ])
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


class _Resp:
    def __init__(self, status_code, content=b"", json_data=None):
        self.status_code = status_code
        self.content = content
        self._json = json_data or {}
    def json(self):
        return self._json


def test_api_success_composites_alpha_over_gray(monkeypatch):
    # alpha=0 everywhere -> fully transparent -> output is pure gray 128
    monkeypatch.setattr(bg_remove, "_api_health_ok", lambda: True)
    monkeypatch.setattr(bg_remove, "_api_post_remove",
                        lambda data: _Resp(200, _png_rgba(alpha=0)))
    img = np.full((8, 8, 3), 200, np.uint8)
    out, engine = bg_remove.remove_bg_for_index(img, engine="api")
    assert engine == "api"
    assert out.shape == (8, 8, 3)
    assert np.all(out == 128)


def test_api_failure_falls_back_to_local(monkeypatch):
    monkeypatch.setattr(bg_remove, "_api_health_ok", lambda: True)
    def boom(data):
        raise RuntimeError("api down")
    monkeypatch.setattr(bg_remove, "_api_post_remove", boom)
    called = {"local": 0}
    monkeypatch.setattr(bg_remove, "remove_bg_rembg",
                        lambda im: called.__setitem__("local", 1) or im)
    img = np.full((4, 4, 3), 70, np.uint8)
    out, engine = bg_remove.remove_bg_for_index(img, engine="auto")
    assert engine == "local" and called["local"] == 1
    assert out.shape == (4, 4, 3)


def test_engine_local_skips_api(monkeypatch):
    def boom(data):
        raise AssertionError("API must not be called when engine=local")
    monkeypatch.setattr(bg_remove, "_api_post_remove", boom)
    monkeypatch.setattr(bg_remove, "remove_bg_rembg", lambda im: im)
    img = np.full((4, 4, 3), 70, np.uint8)
    out, engine = bg_remove.remove_bg_for_index(img, engine="local")
    assert engine == "local"


def test_auto_uses_local_when_health_down(monkeypatch):
    monkeypatch.setattr(bg_remove, "_api_health_ok", lambda: False)
    def boom(data):
        raise AssertionError("must not POST /remove when health is down")
    monkeypatch.setattr(bg_remove, "_api_post_remove", boom)
    monkeypatch.setattr(bg_remove, "remove_bg_rembg", lambda im: im)
    img = np.full((4, 4, 3), 70, np.uint8)
    out, engine = bg_remove.remove_bg_for_index(img, engine="auto")
    assert engine == "local"
