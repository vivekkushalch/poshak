import numpy as np
from service import descriptors


def test_compute_returns_descriptors(textured_rgb):
    des, n_kp = descriptors.compute_descriptors(textured_rgb)
    assert n_kp > 0
    assert des.shape[0] == n_kp
    assert des.shape[1] == 128                      # SIFT descriptor length


def test_save_load_roundtrip(tmp_path, textured_rgb):
    des, n_kp = descriptors.compute_descriptors(textured_rgb)
    p = tmp_path / "x.npz"
    descriptors.save_descriptors(str(p), des, n_kp)
    des2, n_kp2 = descriptors.load_descriptors(str(p))
    assert n_kp2 == n_kp
    assert np.array_equal(des2, des)


def test_load_is_lru_cached(tmp_path, textured_rgb):
    des, n_kp = descriptors.compute_descriptors(textured_rgb)
    p = tmp_path / "y.npz"
    descriptors.save_descriptors(str(p), des, n_kp)
    descriptors.load_descriptors.cache_clear()
    a = descriptors.load_descriptors(str(p))
    b = descriptors.load_descriptors(str(p))
    assert a[0] is b[0]                              # same object -> cache hit
