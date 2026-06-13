"""Search pipeline — verbatim port of the prototype's
`search_pipeline_sift_first`, `aggregate_by_sku`, and
`unified_cloth_search_sift_priority`, parallelized for CPU speed.

Pipeline shape:

    query → (cloth only) rembg → SigLIP embed
          → top-40 by cosine vs each index
          → SIFT rerank (good-match count, Lowe 0.75)   ← parallel across candidates
          → per-candidate tiered confidence
          → aggregate_by_sku (best image per SKU)
          → CLOTH: fuse BG + Orig pipelines per SKU      ← A and B run in parallel
          → sort by (total_sift, unified_score) desc
"""

from typing import Optional

import numpy as np

from .bg_removal import remove_bg_rembg
from .debug import dbg, timed
from .highpass import compute_iou, highpass_trace
from .image_io import ensure_image
from .index_store import IndexStore
from .model import siglip_embed
from .pools import PIPELINE_POOL, SIFT_POOL
from .scoring import confidence_and_tier, unified_cloth_score
from .sift import sift_match_ratio


def _score_one_candidate(
    rank: int,
    idx: int,
    sims: np.ndarray,
    entries: list[dict],
    query_img: np.ndarray,
    query_trace: Optional[np.ndarray],
) -> Optional[dict]:
    """Worker for stage 2 — runs once per candidate, in a SIFT_POOL
    thread. Returns the result dict or None if the image couldn't load."""
    entry = entries[idx]
    cand_img = ensure_image(entry)
    if cand_img is None:
        return None

    sift_ratio, good_matches, total_kp = sift_match_ratio(query_img, cand_img)
    confidence, tier = confidence_and_tier(
        good_matches, sift_ratio, float(sims[idx])
    )

    entry_trace = entry.get("trace")
    if entry_trace is None:
        entry_trace = highpass_trace(cand_img)
        entry["trace"] = entry_trace
    hp_iou = compute_iou(query_trace, entry_trace) if query_trace is not None else 0.0

    return {
        "entry": entry,
        "siglip_sim": float(sims[idx]),
        "sift_ratio": sift_ratio,
        "sift_matches": good_matches,
        "total_kp": total_kp,
        "highpass_iou": hp_iou,
        "score": confidence,
        "confidence_tier": tier,
    }


def search_pipeline_sift_first(
    query_img: np.ndarray,
    embeds: Optional[np.ndarray],
    entries: list[dict],
    top_k_img: int = 40,
) -> list[dict]:
    """One SigLIP-coarse → SIFT-rerank pass over a single index.

    Stage 1: SigLIP cosine → top-40 candidates (single-threaded; fast).
    Stage 2: SIFT good-match count, run in PARALLEL across candidates
             via SIFT_POOL. cv2 SIFT releases the GIL during compute,
             so threads give real speedup.
    """
    if embeds is None or len(entries) == 0:
        dbg("  pipeline: SKIPPED (empty index)")
        return []

    dbg(f"  pipeline: stage 1 — SigLIP coarse over {len(entries)} entries")
    with timed("    stage1_total"):
        q_embed = siglip_embed(query_img)
        with timed("    cosine_topk"):
            sims = embeds @ q_embed
            top_idx = np.argsort(sims)[::-1][:top_k_img]
    top_sims = [round(float(sims[i]), 3) for i in top_idx[:5]]
    dbg(f"  pipeline: stage 1 top-5 similarities = {top_sims}")

    # Precompute the query's highpass trace once — used by every
    # candidate's IoU calc. Same as the prototype, just hoisted out
    # of the inner loop where it was recomputed per candidate.
    query_trace = highpass_trace(query_img)

    dbg(f"  pipeline: stage 2 — SIFT rerank on top-{len(top_idx)} "
        f"(parallel across {SIFT_POOL._max_workers} workers)")
    futures = [
        SIFT_POOL.submit(
            _score_one_candidate, rank, int(idx), sims, entries,
            query_img, query_trace,
        )
        for rank, idx in enumerate(top_idx)
    ]
    with timed("    stage2_total"):
        raw = [f.result() for f in futures]
    results = [r for r in raw if r is not None]
    skipped = len(raw) - len(results)
    dbg(f"  pipeline: stage 2 done — {len(results)} loaded, {skipped} skipped")

    results.sort(key=lambda x: (x["sift_matches"], x["score"]), reverse=True)
    if results:
        top5 = [(r["entry"]["sku_id"], r["sift_matches"], r["confidence_tier"])
                for r in results[:5]]
        dbg(f"  pipeline: top-5 by SIFT (sku,matches,tier) = {top5}")
    return results


def aggregate_by_sku(image_results: list[dict], top_k_sku: int = 10):
    """Collapse image-level results to best-per-SKU. Returns a list of
    `(sku_id, data)` tuples, sorted by best_score desc."""
    dbg(f"  aggregate_by_sku: collapsing {len(image_results)} image results")
    sku_scores: dict = {}
    for r in image_results:
        sku_id = r["entry"]["sku_id"]
        if sku_id not in sku_scores:
            sku_scores[sku_id] = {
                "best_score": r["score"],
                "best_entry": r["entry"],
                "siglip": r["siglip_sim"],
                "sift": r["sift_ratio"],
                "sift_matches": r["sift_matches"],
                "tier": r["confidence_tier"],
                "hp": r["highpass_iou"],
            }
        else:
            if r["score"] > sku_scores[sku_id]["best_score"]:
                sku_scores[sku_id].update({
                    "best_score": r["score"],
                    "best_entry": r["entry"],
                    "siglip": r["siglip_sim"],
                    "sift": r["sift_ratio"],
                    "sift_matches": r["sift_matches"],
                    "tier": r["confidence_tier"],
                    "hp": r["highpass_iou"],
                })
    out = sorted(
        sku_scores.items(), key=lambda x: x[1]["best_score"], reverse=True
    )[:top_k_sku]
    dbg(f"  aggregate_by_sku: {len(sku_scores)} unique SKUs → top {len(out)}")
    return out


def unified_cloth_search(
    query_img: np.ndarray, store: IndexStore, top_k: int = 5
) -> list[dict]:
    """Cloth search — dual pipeline (BG + Orig) → fusion.

    Pipelines A and B have no data dependency on each other, so they
    run concurrently via PIPELINE_POOL. Inside each, stage 2 fans out
    to SIFT_POOL. Total CPU usage capped by SIFT_POOL.max_workers.
    """
    dbg(f"cloth pipeline: starting (bg_idx={len(store.cloth_bg)}, "
        f"orig_idx={len(store.cloth_orig)}, top_k={top_k})")

    def _pipeline_a():
        dbg("cloth pipeline A: bg-remove query, then search BG index")
        with timed("  pipelineA_total"):
            query_bg = remove_bg_rembg(query_img)
            bg_results = search_pipeline_sift_first(
                query_bg, store.cloth_bg_embeds, store.cloth_bg, top_k_img=40
            )
            return aggregate_by_sku(bg_results, top_k_sku=10)

    def _pipeline_b():
        dbg("cloth pipeline B: original query vs Orig index")
        with timed("  pipelineB_total"):
            orig_results = search_pipeline_sift_first(
                query_img, store.cloth_orig_embeds, store.cloth_orig, top_k_img=40
            )
            return aggregate_by_sku(orig_results, top_k_sku=10)

    # Run pipelines sequentially: A's rembg + B's candidate decode
    # together would double peak RAM and trip OOM on a 4GB cap. Stage 2
    # SIFT inside each pipeline is still parallel via SIFT_POOL.
    with timed("cloth dual-pipeline (sequential)"):
        bg_sku = _pipeline_a()
        orig_sku = _pipeline_b()

    # Union the candidates across both pipelines.
    all_candidates: dict = {}
    for sku_id, data in bg_sku:
        all_candidates[sku_id] = {
            "bg_score": data["best_score"],
            "bg_sift_matches": data["sift_matches"],
            "bg_tier": data["tier"],
            "bg_entry": data["best_entry"],
            "orig_score": 0,
            "orig_sift_matches": 0,
            "orig_tier": "NONE",
            "orig_entry": None,
        }
    for sku_id, data in orig_sku:
        if sku_id in all_candidates:
            all_candidates[sku_id]["orig_score"] = data["best_score"]
            all_candidates[sku_id]["orig_sift_matches"] = data["sift_matches"]
            all_candidates[sku_id]["orig_tier"] = data["tier"]
            all_candidates[sku_id]["orig_entry"] = data["best_entry"]
        else:
            all_candidates[sku_id] = {
                "bg_score": 0,
                "bg_sift_matches": 0,
                "bg_tier": "NONE",
                "bg_entry": None,
                "orig_score": data["best_score"],
                "orig_sift_matches": data["sift_matches"],
                "orig_tier": data["tier"],
                "orig_entry": data["best_entry"],
            }

    verified = []
    for sku_id, cand in all_candidates.items():
        unified_score, both_found = unified_cloth_score(
            cand["bg_sift_matches"],
            cand["orig_sift_matches"],
            cand["bg_score"],
            cand["orig_score"],
        )
        total_sift = cand["bg_sift_matches"] + cand["orig_sift_matches"]
        best_entry = cand["orig_entry"] if cand["orig_entry"] else cand["bg_entry"]
        verified.append({
            "sku_id": sku_id,
            "unified_score": unified_score,
            "total_sift_matches": total_sift,
            "bg_sift": cand["bg_sift_matches"],
            "orig_sift": cand["orig_sift_matches"],
            "bg_tier": cand["bg_tier"],
            "orig_tier": cand["orig_tier"],
            "bg_score": cand["bg_score"],
            "orig_score": cand["orig_score"],
            "both_found": both_found > 0.5,
            "best_entry": best_entry,
        })

    verified.sort(
        key=lambda x: (x["total_sift_matches"], x["unified_score"]), reverse=True
    )
    dbg(f"cloth fusion: union={len(all_candidates)} SKUs → returning top {top_k}")
    if verified[:top_k]:
        top_summary = [(v["sku_id"], v["total_sift_matches"],
                        round(v["unified_score"], 3),
                        "BOTH" if v["both_found"] else "ONE")
                       for v in verified[:top_k]]
        dbg(f"cloth fusion: top results = {top_summary}")
    return verified[:top_k]


def fabric_search(
    query_img: np.ndarray, store: IndexStore, top_k: int = 5
) -> list[dict]:
    """Fabric search — single pipeline, no bg removal. Stage 2 SIFT
    rerank is still parallel via SIFT_POOL."""
    dbg(f"fabric pipeline: starting (idx={len(store.fabric)}, top_k={top_k})")
    with timed("  fabric_total"):
        results = search_pipeline_sift_first(
            query_img, store.fabric_embeds, store.fabric, top_k_img=40
        )
        sku_ranked = aggregate_by_sku(results, top_k_sku=top_k)
    out = []
    for sku_id, data in sku_ranked:
        out.append({
            "sku_id": sku_id,
            "unified_score": data["best_score"],
            "total_sift_matches": data["sift_matches"],
            "tier": data["tier"],
            "best_entry": data["best_entry"],
        })
    return out
