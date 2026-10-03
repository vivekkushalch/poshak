import numpy as np

from poshak import descriptors, search
from poshak.store import IndexStore


def _entry(sku, img, path, tmp):
    des, n = descriptors.compute_descriptors(img)
    p = str(tmp / path)
    descriptors.save_descriptors(p, des, n)
    return {
        "sku_id": sku,
        "designno": "",
        "item": "Fabric",
        "filename": f"{sku}.jpg",
        "json_filename": f"{sku}.jpg",
        "desc_path": p,
        "embed": np.eye(768, dtype=np.float32)[sku % 768],
    }


def test_pipeline_ranks_matching_sku_first(tmp_path, textured_rgb, monkeypatch):
    # query embeds to one-hot index 1; candidate sku=1 is the same texture.
    monkeypatch.setattr(search, "siglip_embed", lambda img: np.eye(768, dtype=np.float32)[1])
    other = np.roll(textured_rgb, 64, axis=0)
    e1 = _entry(1, textured_rgb, "c1.npz", tmp_path)  # true match
    e2 = _entry(2, other, "c2.npz", tmp_path)
    embeds = np.stack([e1["embed"], e2["embed"]])
    results = search.search_pipeline_sift_first(textured_rgb, embeds, [e1, e2], top_k_img=40)
    assert results[0]["entry"]["sku_id"] == 1  # most SIFT matches first
    assert "highpass_iou" not in results[0]  # highpass removed


def test_fabric_search_shape(tmp_path, textured_rgb, monkeypatch):
    monkeypatch.setattr(search, "siglip_embed", lambda img: np.eye(768, dtype=np.float32)[1])
    e1 = _entry(1, textured_rgb, "f1.npz", tmp_path)
    store = IndexStore()
    store.fabric = [e1]
    store.fabric_embeds = np.stack([e1["embed"]])
    out = search.fabric_search(textured_rgb, store, top_k=5)
    assert out[0]["sku_id"] == 1
    assert "best_entry" in out[0]
