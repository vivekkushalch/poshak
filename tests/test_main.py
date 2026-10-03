from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from poshak import config, main, store


def _client(monkeypatch):
    monkeypatch.setattr(config, "DISABLE_AUTH", True)
    return TestClient(main.app)


def test_health_open(monkeypatch):
    r = _client(monkeypatch).get("/health")
    assert r.status_code == 200
    assert r.json()["model_tag"] == config.MODEL_TAG


def test_config_returns_image_base(monkeypatch):
    monkeypatch.setattr(config, "IMAGE_BASE_URL", "https://x.test")
    r = _client(monkeypatch).get("/config")
    assert r.json()["image_base_url"] == "https://x.test"


def test_search_rejects_bad_category(monkeypatch):
    monkeypatch.setattr(store.STORE, "cloth_orig", [{}])  # make is_ready() true
    c = _client(monkeypatch)
    r = c.post(
        "/search", data={"category": "bogus"}, files={"image": ("q.jpg", b"x", "image/jpeg")}
    )
    assert r.status_code == 400


def test_search_calls_fabric_pipeline(monkeypatch):
    monkeypatch.setattr(store.STORE, "fabric", [{}])
    monkeypatch.setattr(main, "_decode_rgb", lambda b: np.zeros((10, 10, 3), np.uint8))
    monkeypatch.setattr(
        main,
        "fabric_search",
        lambda img, s, top_k: [
            {
                "sku_id": 7,
                "unified_score": 0.9,
                "total_sift_matches": 5,
                "tier": "LOW",
                "best_entry": {
                    "sku_id": 7,
                    "designno": "",
                    "item": "Fabric",
                    "filename": "f.jpg",
                    "json_filename": "f.jpg",
                },
            }
        ],
    )
    c = _client(monkeypatch)
    r = c.post(
        "/search",
        data={"category": "fabric", "top_k": 5},
        files={"image": ("q.jpg", b"x", "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["results"][0]["sku_id"] == 7
    assert body["results"][0]["best_entry"]["filename"] == "f.jpg"


def test_get_categories_lists_seed(monkeypatch, tmp_path):
    db = tmp_path / "m.sqlite"
    store.init_db(str(db))
    monkeypatch.setattr(config, "SQLITE_PATH", Path(db))
    r = _client(monkeypatch).get("/categories")
    assert r.status_code == 200
    items = {row["item"] for row in r.json()["categories"]}
    assert "Sherwani" in items


def test_post_categories_sets_manual(monkeypatch, tmp_path):
    db = tmp_path / "m2.sqlite"
    store.init_db(str(db))
    monkeypatch.setattr(config, "SQLITE_PATH", Path(db))
    r = _client(monkeypatch).post("/categories", data={"item": "Kurta", "category": "fabric"})
    assert r.status_code == 200
    assert store.resolve_category(str(db), "Kurta") == "fabric"


def test_post_categories_rejects_bad_category(monkeypatch, tmp_path):
    db = tmp_path / "m3.sqlite"
    store.init_db(str(db))
    monkeypatch.setattr(config, "SQLITE_PATH", Path(db))
    r = _client(monkeypatch).post("/categories", data={"item": "Kurta", "category": "bogus"})
    assert r.status_code == 400
