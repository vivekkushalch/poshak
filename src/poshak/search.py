"""Search pipeline — port of the research prototype's search_pipeline_sift_first,
aggregate_by_sku, unified_cloth_search_sift_priority (L2351).
Candidates use PRECOMPUTED descriptors; query is detected once. Highpass IoU
removed (unused in ranking). Resolution: 768 (callers pass working-res images)."""

import numpy as np

from .bg_removal import remove_bg_rembg
from .descriptors import load_descriptors
from .image_io import to_working_res
from .model import siglip_embed
from .scoring import confidence_and_tier, unified_cloth_score
from .sift import detect, match_descriptors


def search_pipeline_sift_first(query_img, embeds, entries, top_k_img=40):
    """query_img must be working-res (768). Stage 1 cosine top-40; stage 2
    SIFT match query descriptors vs each candidate's cached descriptors."""
    if embeds is None or len(entries) == 0:
        return []
    q_embed = siglip_embed(query_img)
    sims = embeds @ q_embed
    top_idx = np.argsort(sims)[::-1][:top_k_img]

    n_kp_q, des_q = detect(query_img)
    results = []
    for idx in top_idx:
        entry = entries[int(idx)]
        des_c, n_kp_c = load_descriptors(entry["desc_path"])
        sift_ratio, good, total_kp = match_descriptors(des_q, n_kp_q, des_c, n_kp_c)
        confidence, tier = confidence_and_tier(good, sift_ratio, float(sims[idx]))
        results.append(
            {
                "entry": entry,
                "siglip_sim": float(sims[idx]),
                "sift_ratio": sift_ratio,
                "sift_matches": good,
                "total_kp": total_kp,
                "score": confidence,
                "confidence_tier": tier,
            }
        )
    results.sort(key=lambda x: (x["sift_matches"], x["score"]), reverse=True)
    return results


def aggregate_by_sku(image_results, top_k_sku=10):
    sku_scores: dict = {}
    for r in image_results:
        sku_id = r["entry"]["sku_id"]
        cur = sku_scores.get(sku_id)
        if cur is None or r["score"] > cur["best_score"]:
            sku_scores[sku_id] = {
                "best_score": r["score"],
                "best_entry": r["entry"],
                "siglip": r["siglip_sim"],
                "sift": r["sift_ratio"],
                "sift_matches": r["sift_matches"],
                "tier": r["confidence_tier"],
            }
    return sorted(sku_scores.items(), key=lambda x: x[1]["best_score"], reverse=True)[:top_k_sku]


def unified_cloth_search(query_img, store, top_k=5):
    """query_img is the FULL-SIZE decoded query. Pipeline B uses the 768
    original; pipeline A rembg's the full-size query, then 768."""
    q768 = to_working_res(query_img)
    # Pipeline A — bg-removed (rembg full-size, then 768)
    query_bg = to_working_res(remove_bg_rembg(query_img))
    bg_results = search_pipeline_sift_first(query_bg, store.cloth_bg_embeds, store.cloth_bg, 40)
    bg_sku = aggregate_by_sku(bg_results, top_k_sku=10)
    # Pipeline B — original
    orig_results = search_pipeline_sift_first(q768, store.cloth_orig_embeds, store.cloth_orig, 40)
    orig_sku = aggregate_by_sku(orig_results, top_k_sku=10)

    all_c: dict = {}
    for sku_id, d in bg_sku:
        all_c[sku_id] = {
            "bg_score": d["best_score"],
            "bg_sift_matches": d["sift_matches"],
            "bg_tier": d["tier"],
            "bg_entry": d["best_entry"],
            "orig_score": 0,
            "orig_sift_matches": 0,
            "orig_tier": "NONE",
            "orig_entry": None,
        }
    for sku_id, d in orig_sku:
        if sku_id in all_c:
            all_c[sku_id].update(
                orig_score=d["best_score"],
                orig_sift_matches=d["sift_matches"],
                orig_tier=d["tier"],
                orig_entry=d["best_entry"],
            )
        else:
            all_c[sku_id] = {
                "bg_score": 0,
                "bg_sift_matches": 0,
                "bg_tier": "NONE",
                "bg_entry": None,
                "orig_score": d["best_score"],
                "orig_sift_matches": d["sift_matches"],
                "orig_tier": d["tier"],
                "orig_entry": d["best_entry"],
            }

    verified = []
    for sku_id, c in all_c.items():
        unified_score, both = unified_cloth_score(
            c["bg_sift_matches"], c["orig_sift_matches"], c["bg_score"], c["orig_score"]
        )
        total_sift = c["bg_sift_matches"] + c["orig_sift_matches"]
        best_entry = c["orig_entry"] if c["orig_entry"] else c["bg_entry"]
        verified.append(
            {
                "sku_id": sku_id,
                "unified_score": unified_score,
                "total_sift_matches": total_sift,
                "bg_sift": c["bg_sift_matches"],
                "orig_sift": c["orig_sift_matches"],
                "bg_tier": c["bg_tier"],
                "orig_tier": c["orig_tier"],
                "bg_score": c["bg_score"],
                "orig_score": c["orig_score"],
                "both_found": both > 0.5,
                "best_entry": best_entry,
            }
        )
    verified.sort(key=lambda x: (x["total_sift_matches"], x["unified_score"]), reverse=True)
    return verified[:top_k]


def fabric_search(query_img, store, top_k=5):
    q768 = to_working_res(query_img)
    results = search_pipeline_sift_first(q768, store.fabric_embeds, store.fabric, 40)
    ranked = aggregate_by_sku(results, top_k_sku=top_k)
    return [
        {
            "sku_id": sku_id,
            "unified_score": d["best_score"],
            "total_sift_matches": d["sift_matches"],
            "tier": d["tier"],
            "best_entry": d["best_entry"],
        }
        for sku_id, d in ranked
    ]
