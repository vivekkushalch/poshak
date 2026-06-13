import numpy as np
from service.image_io import to_working_res


def test_shrinks_tall_image_preserving_aspect():
    img = np.zeros((1536, 1024, 3), dtype=np.uint8)  # h=1536, w=1024
    out = to_working_res(img, max_h=768)
    assert out.shape[0] == 768                 # height capped
    assert out.shape[1] == 512                 # width scaled by same factor (1024*0.5)
    assert out.dtype == np.uint8


def test_leaves_small_image_untouched():
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    out = to_working_res(img, max_h=768)
    assert out.shape == (400, 600, 3)
    assert out is img                          # no copy when no resize needed
