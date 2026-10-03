import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from poshak.domain.models import IndexEntry
from poshak.engine import pipeline, sift
from poshak.infrastructure.vector.numpy_index import NumpyIndex
from tests.golden.fixtures import catalog, fake_embed, fake_remove_bg, queries, summarize

GOLDEN = Path(__file__).parents[2] / "golden" / "search_golden.json"


def _partitions():
    descs: dict[str, tuple] = {}
    views: dict[str, list[IndexEntry]] = {"cut": [], "orig": [], "fabric": []}

    def view(row, img, filename, key, bucket):
        descs[key] = sift.compute_descriptors(img)
        views[bucket].append(
            IndexEntry(row["sku_id"], "", row["category"], filename, row["photo"], key,
                       fake_embed(img))
        )  # fmt: skip

    for row in catalog():
        if row["category"] == "fabric":
            view(row, row["image"], row["photo"], f"{row['photo']}.orig.npz", "fabric")
        else:
            view(row, row["image"], row["photo"], f"{row['photo']}.orig.npz", "orig")
            png = row["photo"].replace(".jpg", ".png")
            view(row, fake_remove_bg(row["image"]), png, f"{row['photo']}.rmbg.npz", "cut")

    parts = {
        name: pipeline.Partition(tuple(es), NumpyIndex.from_rows([e.embed for e in es]))
        for name, es in views.items()
    }
    return parts, descs.__getitem__


def _run(pool):
    parts, load = _partitions()
    out = {}
    for name, (category, img) in queries().items():
        if category == "cloth":
            res = pipeline.cloth_search(img, fake_remove_bg(img), fake_embed, parts["cut"],
                                        parts["orig"], load, pool, top_k=5)  # fmt: skip
        else:
            res = pipeline.fabric_search(img, fake_embed, parts["fabric"], load, pool, top_k=5)
        out[name] = summarize(category, res)
    return out


def test_engine_matches_golden():
    """The golden file pins rankings recorded from the original implementation.
    Regenerate only for an intended ranking change: GOLDEN_WRITE=1 pytest <this file>."""
    got = _run(None)
    if os.environ.get("GOLDEN_WRITE") == "1":
        GOLDEN.write_text(json.dumps(got, indent=1), encoding="utf-8")
    assert got == json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_parallel_verify_equals_sequential():
    with ThreadPoolExecutor(4) as pool:
        assert _run(pool) == _run(None)


def test_empty_partition_returns_nothing():
    empty = pipeline.Partition((), NumpyIndex.from_rows([]))
    _, load = _partitions()
    img = catalog()[0]["image"]
    assert pipeline.fabric_search(img, fake_embed, empty, load) == []


@pytest.mark.parametrize("n_kp", [0, 1])
def test_too_few_descriptors_match_nothing(n_kp):
    des, _ = sift.compute_descriptors(catalog()[0]["image"])
    assert sift.match_descriptors(des[:n_kp], n_kp, des, len(des)) == (0.0, 0, 0)


def test_descriptor_bytes_roundtrip():
    des, n = sift.compute_descriptors(catalog()[0]["image"])
    des2, n2 = sift.loads(sift.dumps(des, n))
    assert n2 == n and (des2 == des).all()
