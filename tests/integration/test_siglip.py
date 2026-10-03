import numpy as np
import pytest

from poshak.infrastructure.embedding.siglip import SiglipEmbedder


def test_constructing_does_not_load_model():
    assert SiglipEmbedder("google/siglip-base-patch16-224")._model is None


@pytest.mark.slow
def test_embed_is_768_and_normalized(textured_rgb):
    e = SiglipEmbedder("google/siglip-base-patch16-224")
    v = e.embed(textured_rgb)
    assert v.shape == (768,) and v.dtype == np.float32
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-3
    assert getattr(e._model, "text_model", None) is None
