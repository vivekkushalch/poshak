"""The search algorithm: embedding recall, SIFT verification, per-SKU fusion.

Port of the research prototype's search_pipeline_sift_first, aggregate_by_sku and
unified_cloth_search_sift_priority. Pure: images, vectors and descriptors come in
through arguments and callables, so this module does no I/O.

SIFT good-match count is the primary ranking signal; embedding cosine is a coarse
recall filter and a small tiebreaker inside the confidence score."""

from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..domain.models import IndexEntry
from ..domain.ports import VectorIndex
from . import sift
from .scoring import confidence_and_tier, unified_cloth_score

RECALL_K = 40  # candidates verified per pipeline; query cost is fixed by this, not catalog size
SKUS_PER_PIPELINE = 10

Embed = Callable[[np.ndarray], np.ndarray]
LoadDescriptors = Callable[[str], tuple[np.ndarray, int]]
Hit = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Partition:
    """One searchable set of image views and the vector index over their embeddings."""

    entries: Sequence[IndexEntry]
    index: VectorIndex

    def __len__(self) -> int:
        return len(self.entries)


def recall_verify(
    query_img: np.ndarray,
    q_embed: np.ndarray,
    part: Partition,
    load_desc: LoadDescriptors,
    pool: Executor | None = None,
    k: int = RECALL_K,
) -> list[Hit]:
    """Stage 1: top-k by cosine. Stage 2: SIFT-verify each candidate (in parallel
    when a pool is given; cv2 releases the GIL). Best SIFT match count first."""
    if len(part) == 0:
        return []
    top_idx, sims = part.index.search(q_embed, k)
    n_kp_q, des_q = sift.detect(query_img)

    def verify(i: int) -> Hit:
        entry = part.entries[int(i)]
        des_c, n_kp_c = load_desc(entry.desc_key)
        ratio, good, total_kp = sift.match_descriptors(des_q, n_kp_q, des_c, n_kp_c)
        sim = float(sims[i])
        confidence, tier = confidence_and_tier(good, ratio, sim)
        return {
            "entry": entry,
            "siglip_sim": sim,
            "sift_ratio": ratio,
            "sift_matches": good,
            "total_kp": total_kp,
            "score": confidence,
            "confidence_tier": tier,
        }

    # pool.map keeps input order, so the stable sort below breaks ties exactly as
    # the sequential loop would
    hits = list(pool.map(verify, top_idx)) if pool else [verify(i) for i in top_idx]
    hits.sort(key=lambda h: (h["sift_matches"], h["score"]), reverse=True)
    return hits


def aggregate_by_sku(hits: list[Hit], top_k_sku: int = SKUS_PER_PIPELINE) -> list[tuple[int, Hit]]:
    """Best-scoring image wins for each SKU."""
    best: dict[int, Hit] = {}
    for h in hits:
        sku_id = h["entry"].sku_id
        cur = best.get(sku_id)
        if cur is None or h["score"] > cur["best_score"]:
            best[sku_id] = {
                "best_score": h["score"],
                "best_entry": h["entry"],
                "siglip": h["siglip_sim"],
                "sift": h["sift_ratio"],
                "sift_matches": h["sift_matches"],
                "tier": h["confidence_tier"],
            }
    return sorted(best.items(), key=lambda x: x[1]["best_score"], reverse=True)[:top_k_sku]


def cloth_search(
    query: np.ndarray,
    query_cutout: np.ndarray,
    embed: Embed,
    cutouts: Partition,
    originals: Partition,
    load_desc: LoadDescriptors,
    pool: Executor | None = None,
    top_k: int = 5,
) -> list[Hit]:
    """Dual-view search: cutout query vs cutout index (A), original vs original (B),
    fused per SKU. Both images must already be at working resolution. A and B run
    one after the other to bound peak memory; verification inside each is parallel."""
    bg_sku = aggregate_by_sku(
        recall_verify(query_cutout, embed(query_cutout), cutouts, load_desc, pool)
    )
    orig_sku = aggregate_by_sku(recall_verify(query, embed(query), originals, load_desc, pool))

    fused: dict[int, dict[str, Any]] = {}
    for sku_id, d in bg_sku:
        fused[sku_id] = {
            "bg_score": d["best_score"], "bg_sift_matches": d["sift_matches"],
            "bg_tier": d["tier"], "bg_entry": d["best_entry"],
            "orig_score": 0, "orig_sift_matches": 0, "orig_tier": "NONE", "orig_entry": None,
        }  # fmt: skip
    for sku_id, d in orig_sku:
        if sku_id in fused:
            fused[sku_id].update(
                orig_score=d["best_score"], orig_sift_matches=d["sift_matches"],
                orig_tier=d["tier"], orig_entry=d["best_entry"],
            )  # fmt: skip
        else:
            fused[sku_id] = {
                "bg_score": 0, "bg_sift_matches": 0, "bg_tier": "NONE", "bg_entry": None,
                "orig_score": d["best_score"], "orig_sift_matches": d["sift_matches"],
                "orig_tier": d["tier"], "orig_entry": d["best_entry"],
            }  # fmt: skip

    results = []
    for sku_id, c in fused.items():
        unified, both = unified_cloth_score(
            c["bg_sift_matches"], c["orig_sift_matches"], c["bg_score"], c["orig_score"]
        )
        results.append({
            "sku_id": sku_id,
            "unified_score": unified,
            "total_sift_matches": c["bg_sift_matches"] + c["orig_sift_matches"],
            "bg_sift": c["bg_sift_matches"],
            "orig_sift": c["orig_sift_matches"],
            "bg_tier": c["bg_tier"],
            "orig_tier": c["orig_tier"],
            "bg_score": c["bg_score"],
            "orig_score": c["orig_score"],
            "both_found": both > 0.5,
            "best_entry": c["orig_entry"] or c["bg_entry"],
        })  # fmt: skip
    results.sort(key=lambda x: (x["total_sift_matches"], x["unified_score"]), reverse=True)
    return results[:top_k]


def fabric_search(
    query: np.ndarray,
    embed: Embed,
    part: Partition,
    load_desc: LoadDescriptors,
    pool: Executor | None = None,
    top_k: int = 5,
) -> list[Hit]:
    """Single-view search for flat textiles. Query must be at working resolution."""
    hits = recall_verify(query, embed(query), part, load_desc, pool)
    return [
        {
            "sku_id": sku_id,
            "unified_score": d["best_score"],
            "total_sift_matches": d["sift_matches"],
            "tier": d["tier"],
            "best_entry": d["best_entry"],
        }
        for sku_id, d in aggregate_by_sku(hits, top_k_sku=top_k)
    ]
