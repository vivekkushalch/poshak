# 6. Progressive ranking for small catalogs (RANKER=v2)

## Context

The legacy ranker embeds the query, verifies the embedding's top 40 with SIFT, then
repeats on a background-removed copy. On a real shop catalog (408 SKUs, 12 MP phone
photos, i5-7200U) a cloth search took about 7 s, and three problems showed up:

- Verification cost is 40 full matches per pass (about 30 ms each).
- The cutout pass never changed an answer on that data. The remover returned an almost
  blank image for 24% of customer-style queries.
- Ranking by raw ratio-test match counts lets one high-texture photo beat the right one
  (a distractor with 415 matches against the truth's 78). And for 7% of heavily
  degraded queries the right photo was not in the embedding's top 40 at all.

## Decision

Add an opt-in ranker, `RANKER=v2` (default stays `legacy`):

1. Coarse pass: match the query's strongest 300 keypoints against the strongest 300 of
   every view. No embedding needed.
2. Fully match a shortlist, counting a match only if it passes the ratio test and is a
   mutual nearest neighbour. Stop when the leader has at least 16 matches and 1.5 times
   the runner-up's. Otherwise widen (coarse 10 plus embedding 10, then coarse 20 plus
   embedding 30), computing the embedding lazily and blending its cosine into the score.
3. No cutout pass.

It applies only while the catalog has at most `V2_MAX_VIEWS` (1000) photos per category.
Above that it falls back to legacy, because the coarse pass costs a fixed amount per view
and v2 was only evaluated on a few hundred.

Descriptors are stored strongest-first (a new `by_response` file marker, uint8), so the
first 300 rows are a view's best keypoints. `poshak refresh-descriptors` rewrites existing
files without re-embedding. Legacy reads both formats.

## How it was chosen

`bench/lab.py` caches embeddings, cutouts and SIFT features and replays ranker variants in
seconds. Queries are split by SKU into tuning and test halves; every parameter was chosen
on tuning, and the test half was read once. The lab reproduces the production pipeline's
ranks on 100/100 queries (99/100 for the full pipeline).

Rejected on the tuning half: similarity-transform RANSAC (worst of the geometric checks),
homography RANSAC (tied with mutual matching on the client set, worse on DeepFashion),
coarse-only recall (76% shortlist recall on heavy queries against 93% for the embedding),
and embedding-only shortlists (84% at 20 candidates). The union of coarse and embedding
recall beats either alone.

The fusion weight is a compromise: the client catalog prefers geometry, DeepFashion
(224 px studio shots) prefers the embedding. The weight was set on both tuning halves and
checked on both test halves.

## Consequences

- Faster: no background removal, no embedding for easy queries, 5 to 15 full matches
  instead of 80.
- More accurate on the client test half and DeepFashion test half (see the README), but
  the embedding alone still beats v2 on DeepFashion cross-view photos.
- Indexing is unchanged (cutouts are still produced; the admin UI and the legacy fallback
  use them). Skipping them for v2-only deployments would roughly halve indexing time.
- The constants in `engine/v2.py` were tuned together. Change them only with the lab.
