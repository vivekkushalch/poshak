import numpy as np
from service.store import init_db, connect
from scripts import seed


def test_resumes_without_redoing_completed(tmp_path, monkeypatch):
    db = str(tmp_path / "i.sqlite"); init_db(db)
    up = tmp_path / "uploads"; up.mkdir()
    rm = tmp_path / "rmbg"; rm.mkdir()
    desc = tmp_path / "desc"; desc.mkdir()
    monkeypatch.setattr(seed.config, "SQLITE_PATH", db)
    monkeypatch.setattr(seed.config, "CORPUS_DIR", up)
    monkeypatch.setattr(seed.config, "RMBG_DIR", rm)
    monkeypatch.setattr(seed.config, "DESC_DIR", desc)
    monkeypatch.setattr(seed.config, "MODEL_TAG", "m")

    calls = {"s3": 0, "embed": 0, "rembg": 0}
    monkeypatch.setattr(seed, "s3_get", lambda k: (calls.__setitem__("s3", calls["s3"] + 1) or b"x"))
    monkeypatch.setattr(seed, "decode_768", lambda b: np.zeros((32, 32, 3), np.uint8))
    monkeypatch.setattr(seed, "embed", lambda img: (calls.__setitem__("embed", calls["embed"] + 1) or np.ones(768, np.float32)))
    monkeypatch.setattr(seed, "compute_desc", lambda img: (np.zeros((3, 128), np.float32), 3))
    monkeypatch.setattr(seed, "rembg_fullsize", lambda img: (calls.__setitem__("rembg", calls["rembg"] + 1) or img))

    skus = [{"stock_id": 1, "designno": "", "item": "Fabric",
             "images_on_server": ["a.jpg"]}]
    seed.run(skus)
    seed.run(skus)   # second pass must be a no-op for the completed item

    assert calls["embed"] == 1          # embedded once, not twice
    with connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1
