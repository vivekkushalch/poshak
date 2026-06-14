import numpy as np
from service import manage, store
from service.store import init_db, connect


def _wire(tmp_path, monkeypatch):
    db = str(tmp_path / "i.sqlite"); init_db(db)
    for d in ("uploads", "rmbg", "desc", "data"):
        (tmp_path / d).mkdir()
    monkeypatch.setattr(manage.config, "SQLITE_PATH", db)
    monkeypatch.setattr(manage.config, "CORPUS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(manage.config, "RMBG_DIR", tmp_path / "rmbg")
    monkeypatch.setattr(manage.config, "DESC_DIR", tmp_path / "desc")
    monkeypatch.setattr(manage.config, "MODEL_TAG", "m")
    monkeypatch.setattr(manage, "_embed", lambda img: np.ones(768, np.float32))
    monkeypatch.setattr(manage, "_desc", lambda img: (np.zeros((4, 128), np.float32), 4))
    monkeypatch.setattr(manage, "_rembg", lambda img: img)
    monkeypatch.setattr(manage, "_decode_full", lambda b: np.zeros((40, 40, 3), np.uint8))
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


def test_add_cloth_makes_rmbg(tmp_path, monkeypatch):
    db = _wire(tmp_path, monkeypatch)
    manage.add_image(sku_id=1, designno="D", item="Sherwani",
                     photo_name="a.jpg", data=b"x")
    assert (tmp_path / "rmbg" / "a.png").exists()
    with connect(db) as c:
        row = c.execute("SELECT rembg_name FROM images WHERE photo_name='a.jpg'"
                        ).fetchone()
        assert row[0] == "a.png"
    assert len(store.STORE.cloth_orig) == 1
    assert len(store.STORE.cloth_bg) == 1


def test_forget_removes_everything(tmp_path, monkeypatch):
    db = _wire(tmp_path, monkeypatch)
    manage.add_image(1, "D", "Sherwani", "a.jpg", b"x")
    res = manage.forget_image("a.jpg")
    assert res["db_deleted"] == 1
    assert not (tmp_path / "uploads" / "a.jpg").exists()
    assert len(store.STORE.cloth_orig) == 0


def test_replace_original_invalidates_desc_cache(tmp_path, monkeypatch):
    db = _wire(tmp_path, monkeypatch)
    manage.add_image(1, "D", "Sherwani", "a.jpg", b"x")
    cleared = {"n": 0}
    from service import descriptors
    monkeypatch.setattr(descriptors.load_descriptors, "cache_clear",
                        lambda: cleared.__setitem__("n", cleared["n"] + 1))
    manage.replace_image("a.jpg", kind="original", data=b"y")
    assert cleared["n"] >= 1
