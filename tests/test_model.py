import numpy as np
import pytest


def test_import_does_not_load_model():
    # Importing must NOT instantiate the model (keeps unit tests fast).
    import importlib

    model = importlib.reload(importlib.import_module("poshak.model"))
    assert model._model is None


@pytest.mark.slow
def test_embed_is_768_normalized_and_text_tower_freed(textured_rgb):
    from poshak import model

    v = model.siglip_embed(textured_rgb)
    assert v.shape == (768,)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-3  # L2-normalized
    assert getattr(model._model, "text_model", None) is None
