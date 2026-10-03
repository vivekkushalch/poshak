"""Scoring constants and formulas — verbatim from the prototype's
final SIFT-FIRST iteration.

Two layers:
  1. Per-candidate confidence (tiered by SIFT good-match count, with
     SigLIP cosine as a small tiebreaker).
  2. Unified cloth fusion across BG + Orig pipelines (per SKU).

If you change a constant here, retune across the test set. The numbers
below are tuned to the catalog (Sherwani / Indowestern / etc.) — they
are not generic.
"""


def confidence_and_tier(good_matches: int, sift_ratio: float, siglip_sim: float):
    """Per-candidate confidence + tier.

    good >= 30: HIGH  = 0.90 + 0.10 * min(good/100, 1)
    good >= 15: MED   = 0.70 + 0.20 * ((good-15)/15)
    good >=  5: LOW   = 0.40 + 0.30 * ((good-5)/10)
    else:       WEAK  = 0.10 + 0.30 * sift_ratio
    confidence += 0.05 * siglip_sim   # tiebreaker
    """
    if good_matches >= 30:
        confidence = 0.90 + 0.10 * min(good_matches / 100.0, 1.0)
        tier = "HIGH"
    elif good_matches >= 15:
        confidence = 0.70 + 0.20 * ((good_matches - 15) / 15.0)
        tier = "MED"
    elif good_matches >= 5:
        confidence = 0.40 + 0.30 * ((good_matches - 5) / 10.0)
        tier = "LOW"
    else:
        confidence = 0.10 + 0.30 * sift_ratio
        tier = "WEAK"
    confidence += 0.05 * float(siglip_sim)
    return confidence, tier


def unified_cloth_score(
    bg_sift: int,
    orig_sift: int,
    bg_score: float,
    orig_score: float,
) -> tuple[float, float]:
    """Cloth fusion across BG-removed pipeline + Original pipeline.

        total_sift    = bg_sift + orig_sift
        both_found    = 1.0 if BOTH pipelines hit this SKU else 0.5
        unified_score = 0.50 * min(total_sift/60, 1.0)
                      + 0.25 * max(bg_score, orig_score)
                      + 0.20 * (bg_score + orig_score) / 2
                      + 0.05 * both_found

    Returns (unified_score, both_found).
    """
    total_sift = bg_sift + orig_sift
    both_found = 1.0 if (bg_score > 0 and orig_score > 0) else 0.5
    unified_score = (
        0.50 * min(total_sift / 60.0, 1.0)
        + 0.25 * max(bg_score, orig_score)
        + 0.20 * (bg_score + orig_score) / 2.0
        + 0.05 * both_found
    )
    return unified_score, both_found
