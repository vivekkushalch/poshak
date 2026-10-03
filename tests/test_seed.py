import numpy as np
from scripts import seed

from poshak.store import connect, init_db


def test_resumes_without_redoing_completed(tmp_path, monkeypatch):
    db = str(tmp_path / "i.sqlite")
    init_db(db)
    up = tmp_path / "uploads"
    up.mkdir()
    rm = tmp_path / "rmbg"
    rm.mkdir()
    desc = tmp_path / "desc"
    desc.mkdir()
    (up / "a.jpg").write_bytes(b"x")  # original staged locally
    monkeypatch.setattr(seed.config, "SQLITE_PATH", db)
    monkeypatch.setattr(seed.config, "CORPUS_DIR", up)
    monkeypatch.setattr(seed.config, "RMBG_DIR", rm)
    monkeypatch.setattr(seed.config, "DESC_DIR", desc)
    monkeypatch.setattr(seed.config, "MODEL_TAG", "m")

    calls = {"embed": 0, "rembg": 0}
    monkeypatch.setattr(seed, "decode_768", lambda b: np.zeros((32, 32, 3), np.uint8))
    monkeypatch.setattr(seed, "decode_full", lambda b: np.zeros((40, 40, 3), np.uint8))
    monkeypatch.setattr(
        seed,
        "embed",
        lambda img: calls.__setitem__("embed", calls["embed"] + 1) or np.ones(768, np.float32),
    )
    monkeypatch.setattr(seed, "compute_desc", lambda img: (np.zeros((3, 128), np.float32), 3))
    monkeypatch.setattr(
        seed,
        "rembg_fullsize",
        lambda img: calls.__setitem__("rembg", calls["rembg"] + 1) or (img, "local"),
    )

    skus = [{"stock_id": 1, "designno": "", "item": "Fabric", "images_on_server": ["a.jpg"]}]
    seed.run(skus)
    seed.run(skus)

    assert calls["embed"] == 1
    with connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1


def test_seed_indexes_unknown_item(tmp_path, monkeypatch):
    db = str(tmp_path / "i.sqlite")
    init_db(db)
    up = tmp_path / "uploads"
    up.mkdir()
    (tmp_path / "rmbg").mkdir()
    (tmp_path / "desc").mkdir()
    (up / "k.jpg").write_bytes(b"x")
    monkeypatch.setattr(seed.config, "SQLITE_PATH", db)
    monkeypatch.setattr(seed.config, "CORPUS_DIR", up)
    monkeypatch.setattr(seed.config, "RMBG_DIR", tmp_path / "rmbg")
    monkeypatch.setattr(seed.config, "DESC_DIR", tmp_path / "desc")
    monkeypatch.setattr(seed.config, "MODEL_TAG", "m")
    monkeypatch.setattr(seed, "decode_768", lambda b: np.zeros((32, 32, 3), np.uint8))
    monkeypatch.setattr(seed, "decode_full", lambda b: np.zeros((40, 40, 3), np.uint8))
    monkeypatch.setattr(seed, "embed", lambda img: np.ones(768, np.float32))
    monkeypatch.setattr(seed, "compute_desc", lambda img: (np.zeros((3, 128), np.float32), 3))
    monkeypatch.setattr(seed, "rembg_fullsize", lambda img: (img, "local"))

    # 'Kurta' is in neither config set — previously skipped, now indexed (cloth).
    skus = [{"stock_id": 1, "designno": "", "item": "Kurta", "images_on_server": ["k.jpg"]}]
    stats = seed.run(skus)
    assert stats["done"] == 1
    with connect(db) as c:
        assert (
            c.execute("SELECT category FROM images WHERE photo_name='k.jpg'").fetchone()[0]
            == "cloth"
        )


def test_missing_local_file_is_counted(tmp_path, monkeypatch):
    db = str(tmp_path / "i.sqlite")
    init_db(db)
    for d in ("uploads", "rmbg", "desc"):
        (tmp_path / d).mkdir()
    monkeypatch.setattr(seed.config, "SQLITE_PATH", db)
    monkeypatch.setattr(seed.config, "CORPUS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(seed.config, "RMBG_DIR", tmp_path / "rmbg")
    monkeypatch.setattr(seed.config, "DESC_DIR", tmp_path / "desc")
    monkeypatch.setattr(seed.config, "MODEL_TAG", "m")
    skus = [{"stock_id": 1, "designno": "", "item": "Fabric", "images_on_server": ["missing.jpg"]}]
    stats = seed.run(skus)
    assert stats["missing"] == 1 and stats["done"] == 0
