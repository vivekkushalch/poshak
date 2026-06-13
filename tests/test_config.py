import importlib


def test_model_tag_is_v2_768():
    cfg = importlib.import_module("service.config")
    assert cfg.MODEL_TAG == "siglip-base-patch16-224-v2-768"
    assert cfg.EMBED_DIM == 768
    assert cfg.WORKING_MAX_H == 768


def test_paths_and_env(monkeypatch):
    monkeypatch.setenv("IMAGE_BASE_URL", "https://example.test")
    cfg = importlib.reload(importlib.import_module("service.config"))
    assert str(cfg.SQLITE_PATH).endswith("index.sqlite")
    assert cfg.IMAGE_BASE_URL == "https://example.test"
    assert cfg.CLOTH_ITEMS == {
        "Sherwani", "Indowestern", "Bandhgala", "Tuxedo", "Jacket", "Bomber Jacket"
    }
