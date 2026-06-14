import numpy as np
from fastapi.testclient import TestClient
from service import main, config
from service.store import init_db, upsert_image


def _seed(tmp_path, monkeypatch, n=120):
    db = str(tmp_path / "i.sqlite"); init_db(db)
    monkeypatch.setattr(config, "SQLITE_PATH", db)
    monkeypatch.setattr(config, "DISABLE_AUTH", True)
    v = np.ones(768, np.float32)
    for i in range(n):
        upsert_image(db, i, "D", "Sherwani", "cloth", f"p{i}.jpg", f"p{i}.png",
                     v, v, f"{i}.o.npz", f"{i}.r.npz", config.MODEL_TAG)
    return db


def test_skus_paginates(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, n=120)
    c = TestClient(main.app)
    r = c.get("/skus?page=1&page_size=50")
    body = r.json()
    assert body["total"] == 120
    assert len(body["items"]) == 50
    assert body["page"] == 1
    first = body["items"][0]
    assert {"sku_id", "item", "designno", "photo_name", "rembg_name"} <= set(first)


def test_img_serves_local_file(tmp_path, monkeypatch):
    up = tmp_path / "uploads"; up.mkdir()
    (up / "p.jpg").write_bytes(b"JPEGDATA")
    monkeypatch.setattr(config, "CORPUS_DIR", up)
    monkeypatch.setattr(config, "DISABLE_AUTH", True)
    c = TestClient(main.app)
    r = c.get("/img/original/p.jpg")
    assert r.status_code == 200 and r.content == b"JPEGDATA"
