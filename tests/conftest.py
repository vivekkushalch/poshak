import numpy as np
import pytest


@pytest.fixture
def textured_rgb():
    """A deterministic textured RGB image SIFT can find keypoints in.
    Checkerboard + gradient so detectAndCompute returns real descriptors."""
    rng = np.random.RandomState(0)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for y in range(0, 256, 16):
        for x in range(0, 256, 16):
            if (x // 16 + y // 16) % 2 == 0:
                img[y:y + 16, x:x + 16] = rng.randint(60, 200, size=3)
    return img
