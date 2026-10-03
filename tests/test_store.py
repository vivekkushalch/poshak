import numpy as np

from poshak import config as _config
from poshak.store import (
    IndexStore,
    add_job_item,
    blob_to_vec,
    bump_engine,
    connect,
    create_or_touch_job,
    get_job,
    infer_category,
    init_db,
    list_item_categories,
    list_jobs,
    list_pending_items,
    refresh_job_counts,
    resolve_category,
    set_item_category,
    set_item_status,
    upsert_image,
    vec_to_blob,
)


def test_vec_blob_roundtrip():
    v = np.random.rand(768).astype(np.float32)
    assert np.array_equal(blob_to_vec(vec_to_blob(v)), v)


def test_load_routes_cloth_and_fabric(tmp_path):
    db = str(tmp_path / "t.sqlite")
    init_db(db)
    v = np.ones(768, dtype=np.float32)
    upsert_image(
        db,
        sku_id=1,
        designno="D1",
        item="Sherwani",
        category="cloth",
        photo_name="a.jpg",
        rembg_name="a.png",
        orig_vec=v,
        rembg_vec=v,
        orig_desc="a.npz",
        rembg_desc="ab.npz",
        model_tag="m",
    )
    upsert_image(
        db,
        sku_id=2,
        designno="D2",
        item="Fabric",
        category="fabric",
        photo_name="b.jpg",
        rembg_name=None,
        orig_vec=v,
        rembg_vec=None,
        orig_desc="b.npz",
        rembg_desc=None,
        model_tag="m",
    )
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
    upsert_image(db, 1, "D", "Fabric", "fabric", "b.jpg", None, v, None, "b.npz", None, "OLD_TAG")
    store = IndexStore()
    store.load(db, model_tag="NEW_TAG")
    assert not store.is_ready()


def test_upsert_records_bg_engine(tmp_path):
    db = str(tmp_path / "i.sqlite")
    init_db(db)
    upsert_image(
        db,
        1,
        "D",
        "Sherwani",
        "cloth",
        "a.jpg",
        "a.png",
        np.ones(4, np.float32),
        np.ones(4, np.float32),
        "od",
        "rd",
        "m",
        bg_engine="api",
    )
    with connect(db) as c:
        assert (
            c.execute("SELECT bg_engine FROM images WHERE photo_name='a.jpg'").fetchone()[0]
            == "api"
        )


def test_job_lifecycle(tmp_path):
    db = str(tmp_path / "i.sqlite")
    init_db(db)
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


def test_infer_category_fabric_substring():
    assert infer_category("Fabric") == "fabric"
    assert infer_category("Silk Fabric") == "fabric"
    assert infer_category("COTTON FABRIC") == "fabric"
    assert infer_category("  fabric  ") == "fabric"


def test_infer_category_defaults_to_cloth():
    assert infer_category("Sherwani") == "cloth"
    assert infer_category("Bomber Jacket") == "cloth"
    assert infer_category("") == "cloth"
    assert infer_category(None) == "cloth"


def test_init_db_seeds_config_categories(tmp_path):
    db = str(tmp_path / "seed.sqlite")
    init_db(db)
    with connect(db) as c:
        rows = {
            r["item"]: r["category"]
            for r in c.execute("SELECT item, category FROM item_categories").fetchall()
        }
    for it in _config.CLOTH_ITEMS:
        assert rows[it] == "cloth"
    for it in _config.FABRIC_ITEMS:
        assert rows[it] == "fabric"


def test_init_db_seed_does_not_clobber_existing(tmp_path):
    db = str(tmp_path / "seed2.sqlite")
    init_db(db)
    cloth_item = sorted(_config.CLOTH_ITEMS)[0]
    with connect(db) as c:
        c.execute(
            "UPDATE item_categories SET category='fabric', source='manual' WHERE item=?",
            (cloth_item,),
        )
        c.commit()
    init_db(db)  # re-run: INSERT OR IGNORE must not overwrite the manual row
    with connect(db) as c:
        row = c.execute(
            "SELECT category, source FROM item_categories WHERE item=?", (cloth_item,)
        ).fetchone()
    assert row["category"] == "fabric" and row["source"] == "manual"


def test_init_db_preserves_existing_images(tmp_path):
    db = str(tmp_path / "mig.sqlite")
    init_db(db)
    v = np.ones(768, dtype=np.float32)
    upsert_image(db, 1, "D", "Sherwani", "cloth", "a.jpg", "a.png", v, v, "a.npz", "ab.npz", "m")
    init_db(db)  # simulate a restart migration over a populated volume DB
    with connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1
        assert (
            c.execute("SELECT category FROM images WHERE photo_name='a.jpg'").fetchone()[0]
            == "cloth"
        )


def test_resolve_category_lookup_hit(tmp_path):
    db = str(tmp_path / "r.sqlite")
    init_db(db)
    assert resolve_category(db, "Sherwani") == "cloth"


def test_resolve_category_case_insensitive(tmp_path):
    db = str(tmp_path / "r2.sqlite")
    init_db(db)
    assert resolve_category(db, "sherwani") == "cloth"
    assert resolve_category(db, "FABRIC") == "fabric"


def test_resolve_category_infers_and_persists(tmp_path):
    db = str(tmp_path / "r3.sqlite")
    init_db(db)
    assert resolve_category(db, "Kurta") == "cloth"
    with connect(db) as c:
        row = c.execute(
            "SELECT category, source FROM item_categories WHERE item='Kurta'"
        ).fetchone()
    assert row["category"] == "cloth" and row["source"] == "inferred"
    assert resolve_category(db, "Kurta") == "cloth"


def test_resolve_category_infers_fabric(tmp_path):
    db = str(tmp_path / "r4.sqlite")
    init_db(db)
    assert resolve_category(db, "Linen Fabric") == "fabric"
    with connect(db) as c:
        assert (
            c.execute("SELECT source FROM item_categories WHERE item='Linen Fabric'").fetchone()[
                "source"
            ]
            == "inferred"
        )


def test_set_item_category_upserts_manual(tmp_path):
    db = str(tmp_path / "e.sqlite")
    init_db(db)
    set_item_category(db, "Kurta", "fabric")
    with connect(db) as c:
        row = c.execute(
            "SELECT category, source FROM item_categories WHERE item='Kurta'"
        ).fetchone()
    assert row["category"] == "fabric" and row["source"] == "manual"
    set_item_category(db, "Kurta", "cloth")
    with connect(db) as c:
        assert (
            c.execute("SELECT category FROM item_categories WHERE item='Kurta'").fetchone()[0]
            == "cloth"
        )


def test_list_item_categories_returns_rows(tmp_path):
    db = str(tmp_path / "e2.sqlite")
    init_db(db)
    items = {r["item"] for r in list_item_categories(db)}
    assert "Sherwani" in items and "Fabric" in items


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
