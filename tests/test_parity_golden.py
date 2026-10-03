"""Search parity golden test. Pins ranked output for a synthetic catalog so the
restructure can prove it returns exactly what the original code returned.
Regenerate (only when ranking is meant to change): GOLDEN_WRITE=1 pytest this file."""

import json
import os
from pathlib import Path

import numpy as np

from service import descriptors, search
from service.store import IndexStore
from tests.golden.fixtures import catalog, fake_embed, fake_remove_bg, queries, summarize

GOLDEN = Path(__file__).parent / "golden" / "search_golden.json"


def _entry(row, img, name, desc_path):
    des, n = descriptors.compute_descriptors(img)
    descriptors.save_descriptors(desc_path, des, n)
    return {"sku_id": row["sku_id"], "designno": "", "item": row["category"],
            "filename": name, "json_filename": row["photo"],
            "desc_path": desc_path, "embed": fake_embed(img)}


def _run(tmp_path, monkeypatch) -> dict:
    monkeypatch.setattr(search, "siglip_embed", fake_embed)
    monkeypatch.setattr(search, "remove_bg_rembg", fake_remove_bg)
    store = IndexStore()
    for row in catalog():
        orig = _entry(row, row["image"], row["photo"], str(tmp_path / f"{row['photo']}.orig.npz"))
        if row["category"] == "fabric":
            store.fabric.append(orig)
        else:
            store.cloth_orig.append(orig)
            png = row["photo"].replace(".jpg", ".png")
            store.cloth_bg.append(_entry(row, fake_remove_bg(row["image"]), png,
                                         str(tmp_path / f"{row['photo']}.rmbg.npz")))
    store.cloth_bg_embeds = np.stack([e["embed"] for e in store.cloth_bg])
    store.cloth_orig_embeds = np.stack([e["embed"] for e in store.cloth_orig])
    store.fabric_embeds = np.stack([e["embed"] for e in store.fabric])

    out = {}
    for name, (category, img) in queries().items():
        if category == "cloth":
            res = search.unified_cloth_search(img, store, top_k=5)
        else:
            res = search.fabric_search(img, store, top_k=5)
        out[name] = summarize(category, res)
    return out


def test_search_matches_golden(tmp_path, monkeypatch):
    got = _run(tmp_path, monkeypatch)
    if os.environ.get("GOLDEN_WRITE") == "1":
        GOLDEN.write_text(json.dumps(got, indent=1), encoding="utf-8")
    assert got == json.loads(GOLDEN.read_text(encoding="utf-8"))
