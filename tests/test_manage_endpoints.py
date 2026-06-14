from fastapi.testclient import TestClient
from service import main, config


def test_ingest_calls_add(monkeypatch):
    monkeypatch.setattr(config, "DISABLE_AUTH", True)
    captured = {}
    monkeypatch.setattr(main, "add_image",
                        lambda **kw: captured.update(kw) or {"status": "indexed"})
    c = TestClient(main.app)
    r = c.post("/ingest", data={"sku_id": 5, "designno": "D", "item": "Fabric",
                                "photo_name": "z.jpg"},
               files={"image": ("z.jpg", b"x", "image/jpeg")})
    assert r.status_code == 202
    assert captured["photo_name"] == "z.jpg" and captured["sku_id"] == 5


def test_forget_calls_forget(monkeypatch):
    monkeypatch.setattr(config, "DISABLE_AUTH", True)
    monkeypatch.setattr(main, "forget_image",
                        lambda name: {"files_removed": 1, "db_deleted": 1})
    c = TestClient(main.app)
    r = c.post("/forget", data={"photo_name": "z.jpg"})
    assert r.status_code == 200 and r.json()["db_deleted"] == 1


def test_replace_rejects_bad_kind(monkeypatch):
    monkeypatch.setattr(config, "DISABLE_AUTH", True)
    c = TestClient(main.app)
    r = c.post("/replace", data={"photo_name": "z.jpg", "kind": "bogus"},
               files={"image": ("z.jpg", b"x", "image/jpeg")})
    assert r.status_code == 400
