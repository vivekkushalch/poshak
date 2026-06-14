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


def test_rembg_api_defaults(monkeypatch):
    for k in ("REMBG_API_URL", "REMBG_API_TIMEOUT", "REMBG_API_RETRIES",
              "REMBG_DEFAULT_ENGINE"):
        monkeypatch.delenv(k, raising=False)
    import importlib
    from service import config as cfg
    importlib.reload(cfg)
    assert cfg.REMBG_API_URL == "https://rembg.example.com"
    assert cfg.REMBG_API_TIMEOUT == 360
    assert cfg.REMBG_API_RETRIES == 1
    assert cfg.REMBG_DEFAULT_ENGINE == "auto"
    assert not hasattr(cfg, "S3_BUCKET")
