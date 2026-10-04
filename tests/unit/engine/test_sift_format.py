"""Descriptor serialisation: the legacy format is unchanged, v2 adds an ordering marker."""

import io

import numpy as np

from poshak.engine import sift
from tests.golden.fixtures import _image


def test_by_response_returns_the_same_keypoints_strongest_first():
    img = _image(5)
    kp, des = sift._sift().detectAndCompute(img, None)
    strongest_first = np.argsort([-k.response for k in kp], kind="stable")

    got, n_kp = sift.compute_descriptors_by_response(img)
    assert n_kp == len(kp)
    assert np.array_equal(got, des[strongest_first])
    legacy, legacy_n = sift.compute_descriptors(img)
    assert legacy_n == n_kp
    assert np.array_equal(np.sort(got, axis=0), np.sort(legacy, axis=0))  # same set of rows


def test_a_blank_image_has_no_descriptors():
    des, n_kp = sift.compute_descriptors_by_response(np.full((64, 64, 3), 128, np.uint8))
    assert n_kp == 0 and des.shape == (0, 128)


def test_legacy_files_are_unchanged_and_v2_files_are_flagged():
    des, n_kp = sift.compute_descriptors(_image(6))
    legacy = sift.dumps(des, n_kp)
    with np.load(io.BytesIO(legacy)) as z:
        assert sorted(z.files) == ["des", "n_kp"]  # byte-for-byte the original layout
    assert sift.loads_ex(legacy)[2] is False

    ordered, n = sift.compute_descriptors_by_response(_image(6))
    v2_file = sift.dumps(ordered, n, by_response=True)
    got, got_n, flag = sift.loads_ex(v2_file)
    assert flag is True and got_n == n and got.dtype == np.float32
    assert np.array_equal(got, ordered)  # uint8 storage loses nothing
    assert len(v2_file) < len(legacy)


def test_loads_reads_both_formats():
    ordered, n = sift.compute_descriptors_by_response(_image(7))
    for blob in (sift.dumps(ordered, n), sift.dumps(ordered, n, by_response=True)):
        des, n_kp = sift.loads(blob)
        assert n_kp == n and np.array_equal(des, ordered)
