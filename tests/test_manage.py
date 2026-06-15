import os
import numpy as np
from service import manage, store
from service.store import init_db, connect


def _wire(tmp_path, monkeypatch):
    db = str(tmp_path / "i.sqlite"); init_db(db)
    for d in ("uploads", "rmbg", "desc"):
        (tmp_path / d).mkdir()
    monkeypatch.setattr(manage.config, "SQLITE_PATH", db)
    monkeypatch.setattr(manage.config, "CORPUS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(manage.config, "RMBG_DIR", tmp_path / "rmbg")
    monkeypatch.setattr(manage.config, "DESC_DIR", tmp_path / "desc")
    monkeypatch.setattr(manage.config, "MODEL_TAG", "m")
    monkeypatch.setattr(manage, "_embed", lambda img: np.ones(768, np.float32))
    monkeypatch.setattr(manage, "_desc", lambda img: (np.zeros((4, 128), np.float32), 4))
    monkeypatch.setattr(manage, "_bg", lambda img, engine: (img, "api"))
    monkeypatch.setattr(manage, "_load_original", lambda name: np.zeros((40, 40, 3), np.uint8))
    monkeypatch.setattr(manage, "_now", lambda: "2026-06-14T00:00:00")
    return db


def test_add_fabric_indexes_and_loads_store(tmp_path, monkeypatch):
    db = _wire(tmp_path, monkeypatch)
    manage.add_image(sku_id=2, designno="D", item="Fabric",
                     photo_name="b.jpg", data=b"x")
    with connect(db) as c:
        assert c.execute("SELECT category FROM images WHERE photo_name='b.jpg'"
                         ).fetchone()[0] == "fabric"
    assert (tmp_path / "uploads" / "b.jpg").exists()
    assert len(store.STORE.fabric) == 1


def test_add_cloth_makes_rmbg_and_records_engine(tmp_path, monkeypatch):
    db = _wire(tmp_path, monkeypatch)
    res = manage.add_image(1, "D", "Sherwani", "a.jpg", b"x")
    assert res["bg_engine"] == "api"
    assert (tmp_path / "rmbg" / "a.png").exists()
    with connect(db) as c:
        row = c.execute("SELECT rembg_name, bg_engine FROM images "
                        "WHERE photo_name='a.jpg'").fetchone()
        assert row[0] == "a.png" and row[1] == "api"
    assert len(store.STORE.cloth_bg) == 1


def test_index_one_skips_when_already_done(tmp_path, monkeypatch):
    db = _wire(tmp_path, monkeypatch)
    manage.add_image(1, "D", "Sherwani", "a.jpg", b"x")
    calls = {"n": 0}
    monkeypatch.setattr(manage, "_embed",
                        lambda img: calls.__setitem__("n", calls["n"] + 1) or np.ones(768, np.float32))
    res = manage.index_one(1, "D", "Sherwani", "a.jpg")
    assert res["status"] == "skipped" and calls["n"] == 0


def test_index_one_indexes_unknown_item(tmp_path, monkeypatch):
    db = _wire(tmp_path, monkeypatch)
    (tmp_path / "uploads" / "k.jpg").write_bytes(b"x")
    # 'Kurta' is in neither config set — must be indexed (cloth), not skipped.
    res = manage.index_one(1, "D", "Kurta", "k.jpg")
    assert res["status"] == "indexed"
    assert res["category"] == "cloth"
    with connect(db) as c:
        assert c.execute("SELECT category FROM images WHERE photo_name='k.jpg'"
                         ).fetchone()[0] == "cloth"


def test_index_one_force_reindexes(tmp_path, monkeypatch):
    db = _wire(tmp_path, monkeypatch)
    manage.add_image(1, "D", "Sherwani", "a.jpg", b"x")
    res = manage.index_one(1, "D", "Sherwani", "a.jpg", force=True)
    assert res["status"] == "indexed"


def test_forget_removes_everything(tmp_path, monkeypatch):
    db = _wire(tmp_path, monkeypatch)
    manage.add_image(1, "D", "Sherwani", "a.jpg", b"x")
    res = manage.forget_image("a.jpg")
    assert res["db_deleted"] == 1
    assert not (tmp_path / "uploads" / "a.jpg").exists()
    assert len(store.STORE.cloth_orig) == 0
