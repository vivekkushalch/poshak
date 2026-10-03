# 1. Rank by SIFT matches, use embeddings only to shortlist

## Context

Catalog items differ mostly in embroidery and print. Cut and colour repeat across
hundreds of SKUs, and a global embedding describes cut and colour well and fine
detail poorly. The research prototype went through several ranking schemes and
settled on this one, its "SIFT-first" iteration.

## Decision

Recall the top 40 candidates by cosine similarity, then count Lowe-ratio SIFT
matches between the query and each candidate. Sort by good-match count first. The
cosine score enters only as a small term in the confidence score.

SIFT parameters (`nfeatures=1200`, `contrastThreshold=0.02`, `edgeThreshold=10`,
ratio 0.75) and the confidence tiers in `engine/scoring.py` were tuned together on
the prototype's test set. Changing one means re-tuning the others.

## Consequences

- Verification cost is fixed at 40 candidates per pass, whatever the catalog size.
- Descriptors are computed at ingest and stored, so a search loads 40 small arrays
  instead of decoding 40 images.
- A true match outside the top 40 by cosine can't be recovered. Raising the recall
  size trades latency for recall.
