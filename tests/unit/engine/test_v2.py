"""The progressive ranker's building blocks and its decision logic."""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from poshak.domain.models import IndexEntry
from poshak.engine import sift, v2
from poshak.engine.pipeline import Partition
from poshak.infrastructure.vector.numpy_index import NumpyIndex
from tests.golden.fixtures import catalog, fake_embed, queries

GOLDEN = Path(__file__).parents[2] / "golden" / "search_golden_v2.json"


def descriptors(rng, n: int) -> np.ndarray:
    """Integer-valued rows in SIFT's 0..255 range."""
    return rng.randint(0, 256, size=(n, 128)).astype(np.float32)


def noisy_copy(rng, des: np.ndarray) -> np.ndarray:
    return np.clip(des + rng.randint(-2, 3, des.shape), 0, 255).astype(np.float32)


def reference_mutual(q: np.ndarray, c: np.ndarray) -> int:
    """Independent implementation with OpenCV's brute-force matcher."""
    bf = cv2.BFMatcher()
    forward = bf.knnMatch(q, c, k=2)
    back = bf.match(c, q)  # nearest query descriptor of every candidate descriptor
    return sum(
        1
        for i, m in enumerate(forward)
        if len(m) == 2
        and m[0].distance < v2.RATIO * m[1].distance
        and back[m[0].trainIdx].trainIdx == i
    )


def reference_ratio(q: np.ndarray, c: np.ndarray) -> int:
    if len(q) < 2 or len(c) < 2:
        return 0
    forward = cv2.BFMatcher().knnMatch(q, c, k=2)
    return sum(1 for m in forward if m[0].distance < v2.RATIO * m[1].distance)


def test_mutual_count_matches_an_independent_opencv_reference():
    rng = np.random.RandomState(1)
    q = descriptors(rng, 300)
    for planted in (0, 25, 120, 300):
        c = np.vstack([noisy_copy(rng, q[:planted]), descriptors(rng, 200)])
        assert v2.mutual_count(q, c) == reference_mutual(q, c)
    assert v2.mutual_count(q, noisy_copy(rng, q[:120])) > 100  # the planted matches are found


def test_mutual_count_is_zero_when_either_side_is_too_small():
    rng = np.random.RandomState(2)
    assert v2.mutual_count(descriptors(rng, 1), descriptors(rng, 50)) == 0
    assert v2.mutual_count(descriptors(rng, 50), descriptors(rng, 0)) == 0


def test_coarse_counts_equal_the_reference_on_each_views_strongest_keypoints():
    rng = np.random.RandomState(3)
    q = descriptors(rng, 500)
    views = [
        np.vstack([noisy_copy(rng, q[:150]), descriptors(rng, 250)]),  # 400 rows, 150 planted
        descriptors(rng, 600),
        descriptors(rng, 1),  # too small to match
        np.vstack([noisy_copy(rng, q[:20]), descriptors(rng, 30)]),  # shorter than the bank width
    ]
    got = v2.coarse_counts(q, v2.build_coarse_bank(views))
    k = v2.COARSE_KEYPOINTS
    assert got.tolist() == [reference_ratio(q[:k], c[:k]) for c in views]


def make_partition(rng, n_views: int, skus: list[int], dim: int = 8):
    """Views with random descriptors and embeddings; returns what rank() needs."""
    descs, entries, vecs = {}, [], []
    for i in range(n_views):
        key = f"d{i}"
        descs[key] = descriptors(rng, 400)
        vec = rng.randn(dim).astype(np.float32)
        vecs.append(vec / np.linalg.norm(vec))
        entries.append(IndexEntry(skus[i], "", "", f"v{i}.jpg", f"v{i}.jpg", key, vecs[-1]))
    part = Partition(tuple(entries), NumpyIndex.from_rows(vecs))
    bank = v2.build_coarse_bank([descs[e.desc_key] for e in entries])
    return part, bank, lambda key: (descs[key], len(descs[key])), descs, vecs


def test_a_decisive_leader_stops_before_the_embedding_is_computed():
    rng = np.random.RandomState(4)
    skus = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    part, bank, load, descs, _ = make_partition(rng, 12, skus)
    query = noisy_copy(rng, descs["d7"])

    def never():
        raise AssertionError("the embedding must not be computed for an easy query")

    ranked = v2.rank(query, never, part, bank, load, top_k=3)
    assert ranked[0].sku_id == 4 and ranked[0].entry.filename == "v7.jpg"
    assert ranked[0].matches >= v2.STOP_MATCHES
    assert len(ranked) == 3


def test_an_unmatched_query_widens_and_computes_the_embedding_exactly_once():
    rng = np.random.RandomState(5)
    skus = list(range(1, 41))
    part, bank, load, _, vecs = make_partition(rng, 40, skus)
    calls = []

    def embed():
        calls.append(1)
        return vecs[13]  # the query "looks like" view 13

    ranked = v2.rank(descriptors(rng, 400), embed, part, bank, load, top_k=5)
    assert len(calls) == 1
    assert ranked[0].sku_id == 14  # no geometric evidence, so the embedding decides
    assert [r.sku_id for r in ranked] == sorted(
        {r.sku_id for r in ranked}, key=[r.sku_id for r in ranked].index
    )  # one row per SKU


def test_each_sku_is_ranked_by_its_best_photo():
    rng = np.random.RandomState(6)
    part, bank, load, descs, _ = make_partition(rng, 6, [1, 1, 2, 2, 3, 3])
    ranked = v2.rank(noisy_copy(rng, descs["d3"]), lambda: np.zeros(8), part, bank, load, top_k=5)
    assert ranked[0].sku_id == 2 and ranked[0].entry.filename == "v3.jpg"
    assert len({r.sku_id for r in ranked}) == len(ranked)


def test_an_empty_partition_returns_nothing():
    part = Partition((), NumpyIndex.from_rows([]))
    assert v2.rank(np.zeros((5, 128), np.float32), lambda: np.zeros(8), part,
                   v2.build_coarse_bank([]), lambda k: (np.zeros((0, 128)), 0)) == []  # fmt: skip


def golden_run() -> dict:
    """Rank the golden catalog's queries with the real descriptors and the fake embedder."""
    entries: dict[str, list[IndexEntry]] = {"cloth": [], "fabric": []}
    descs: dict[str, np.ndarray] = {}
    for row in catalog():
        des, _ = sift.compute_descriptors_by_response(row["image"])
        key = f"{row['photo']}.orig.npz"
        descs[key] = des
        entries[row["category"]].append(
            IndexEntry(
                row["sku_id"], "", "", row["photo"], row["photo"], key, fake_embed(row["image"])
            )  # fmt: skip
        )
    out = {}
    for name, (category, img) in queries().items():
        es = entries[category]
        part = Partition(tuple(es), NumpyIndex.from_rows([e.embed for e in es]))
        bank = v2.build_coarse_bank([descs[e.desc_key] for e in es])
        q, _ = sift.compute_descriptors_by_response(img)
        ranked = v2.rank(q, lambda img=img: fake_embed(img), part, bank, lambda k: (descs[k], 0), 5)
        out[name] = [
            {"sku_id": r.sku_id, "matches": r.matches, "score": round(r.score, 6),
             "best": r.entry.json_filename} for r in ranked
        ]  # fmt: skip
    return out


def test_v2_ranking_matches_golden():
    """Pins the v2 ranking on the synthetic catalog. It was recorded when v2 was added;
    regenerate only for an intended ranking change: GOLDEN_WRITE=1 pytest <this file>."""
    got = golden_run()
    if os.environ.get("GOLDEN_WRITE") == "1":
        GOLDEN.write_text(json.dumps(got, indent=1), encoding="utf-8")
    assert got == json.loads(GOLDEN.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("query", "sku"),
    [("cloth_noisy", 2), ("cloth_crop", 4), ("fabric_noisy", 8), ("fabric_crop", 9)],
)
def test_golden_queries_find_their_sku(query, sku):
    assert golden_run()[query][0]["sku_id"] == sku
