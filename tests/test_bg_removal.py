import numpy as np

from poshak import bg_removal


def test_no_internal_resize(monkeypatch):
    # Fake rembg.remove: return an RGBA where alpha=255 everywhere -> output
    # must equal the RGB of the *full-size* input (no downscaling).
    def fake_remove(pil_img):
        from PIL import Image

        arr = np.array(pil_img)
        rgba = np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)])
        return Image.fromarray(rgba)

    monkeypatch.setattr(bg_removal, "remove", fake_remove)
    img = np.full((1200, 900, 3), 70, dtype=np.uint8)
    out = bg_removal.remove_bg_rembg(img)
    assert out.shape == (1200, 900, 3)  # SAME size as input (full-res)


def test_gray_composite_when_no_alpha(monkeypatch):
    def fake_remove(pil_img):
        from PIL import Image

        return Image.fromarray(np.array(pil_img))  # RGB, no alpha

    monkeypatch.setattr(bg_removal, "remove", fake_remove)
    img = np.full((10, 10, 3), 200, dtype=np.uint8)
    out = bg_removal.remove_bg_rembg(img)
    assert out.shape == (10, 10, 3)
