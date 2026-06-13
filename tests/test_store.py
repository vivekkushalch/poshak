import numpy as np
from service.store import IndexStore, init_db, upsert_image, vec_to_blob, blob_to_vec


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
