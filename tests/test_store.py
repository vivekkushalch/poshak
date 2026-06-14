import numpy as np
from service.store import IndexStore, init_db, upsert_image, vec_to_blob, blob_to_vec
from service.store import (connect, create_or_touch_job, add_job_item,
                           set_item_status, refresh_job_counts, bump_engine,
                           get_job, list_jobs, list_pending_items)


def test_vec_blob_roundtrip():
    v = np.random.rand(768).astype(np.float32)
    assert np.array_equal(blob_to_vec(vec_to_blob(v)), v)


def test_load_routes_cloth_and_fabric(tmp_path):
    db = str(tmp_path / "t.sqlite")
    init_db(db)
    v = np.ones(768, dtype=np.float32)
    upsert_image(db, sku_id=1, designno="D1", item="Sherwani", category="cloth",
                 photo_name="a.jpg", rembg_name="a.png",
                 orig_vec=v, rembg_vec=v, orig_desc="a.npz", rembg_desc="ab.npz",
                 model_tag="m")
    upsert_image(db, sku_id=2, designno="D2", item="Fabric", category="fabric",
                 photo_name="b.jpg", rembg_name=None,
                 orig_vec=v, rembg_vec=None, orig_desc="b.npz", rembg_desc=None,
                 model_tag="m")
    store = IndexStore()
    store.load(db, model_tag="m")
    assert len(store.cloth_orig) == 1
    assert len(store.cloth_bg) == 1
    assert len(store.fabric) == 1
    assert store.cloth_orig_embeds.shape == (1, 768)
    assert store.is_ready()


def test_load_filters_by_model_tag(tmp_path):
    db = str(tmp_path / "t2.sqlite")
    init_db(db)
    v = np.ones(768, dtype=np.float32)
    upsert_image(db, 1, "D", "Fabric", "fabric", "b.jpg", None, v, None,
                 "b.npz", None, "OLD_TAG")
    store = IndexStore()
    store.load(db, model_tag="NEW_TAG")
    assert not store.is_ready()


def test_upsert_records_bg_engine(tmp_path):
    db = str(tmp_path / "i.sqlite"); init_db(db)
    upsert_image(db, 1, "D", "Sherwani", "cloth", "a.jpg", "a.png",
                 np.ones(4, np.float32), np.ones(4, np.float32),
                 "od", "rd", "m", bg_engine="api")
    with connect(db) as c:
        assert c.execute("SELECT bg_engine FROM images WHERE photo_name='a.jpg'"
                         ).fetchone()[0] == "api"


def test_job_lifecycle(tmp_path):
    db = str(tmp_path / "i.sqlite"); init_db(db)
    create_or_touch_job(db, "B1")
    add_job_item(db, "B1", "a.jpg", 1, "D", "Sherwani", "auto", False)
    add_job_item(db, "B1", "b.jpg", 1, "D", "Sherwani", "auto", False)
    assert len(list_pending_items(db)) == 2

    set_item_status(db, "B1", "a.jpg", "done")
    bump_engine(db, "B1", "api")
    set_item_status(db, "B1", "b.jpg", "failed", "boom")
    refresh_job_counts(db, "B1")

    job = get_job(db, "B1")
    assert job["total"] == 2 and job["done"] == 1 and job["failed"] == 1
    assert job["api_count"] == 1 and job["status"] == "done"
    assert job["errors"][0]["photo_name"] == "b.jpg"
    assert len(list_pending_items(db)) == 0
    assert list_jobs(db)[0]["batch_id"] == "B1"


def test_init_db_migrates_legacy_images_table(tmp_path):
    db = str(tmp_path / "legacy.sqlite")
    with connect(db) as c:  # legacy table WITHOUT bg_engine
        c.executescript("""CREATE TABLE images (
          id INTEGER PRIMARY KEY, sku_id INTEGER NOT NULL, designno TEXT,
          item TEXT, category TEXT NOT NULL, photo_name TEXT NOT NULL,
          rembg_name TEXT, orig_vec BLOB NOT NULL, rembg_vec BLOB,
          orig_desc TEXT, rembg_desc TEXT, model_tag TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          UNIQUE(photo_name, model_tag));""")
        c.commit()
    init_db(db)  # must add bg_engine without error
    cols = [r[1] for r in connect(db).execute("PRAGMA table_info(images)").fetchall()]
    assert "bg_engine" in cols
