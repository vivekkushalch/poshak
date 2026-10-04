"""Progressive ranking for small catalogs (RANKER=v2).

The legacy ranker embeds the query, takes the embedding's top 40 and counts SIFT ratio-test
matches against each, then repeats the whole thing on a background-removed copy. On a few
hundred photos that spends nearly all its time on 40 full matches plus a second pass, and
raw match counts are inflated by the repeating motifs of embroidery.

This ranker does less, and stops as soon as the answer is clear:

  1. Match the query's strongest 300 keypoints against the strongest 300 of every view in
     the catalog (a "coarse" pass; it needs no embedding).
  2. Fully match a small shortlist. A match counts only if the two descriptors are each
     other's nearest neighbour as well as passing Lowe's ratio test, which removes the
     matches that make one high-texture photo beat every other.
  3. If the leading SKU has at least STOP_MATCHES matches and STOP_MARGIN times the
     runner-up's, stop. Otherwise widen the shortlist, now also using the embedding's top
     views, and blend the embedding's cosine into the score.

There is no background-removal pass: on the evaluation catalog it never changed an answer
and the remover returned a blank image for about a quarter of customer-style queries.

Every constant below was chosen on the tuning half of a real catalog (bench/lab.py) and
checked once on the untouched test half. See docs/adr/0006-progressive-ranking.md. Change
them only with an evaluation to back it."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..domain.models import IndexEntry
from .pipeline import Partition

COARSE_KEYPOINTS = 300  # strongest keypoints per view used by the coarse pass
RATIO = 0.75  # Lowe ratio test, same as the legacy matcher

# (coarse shortlist size, embedding shortlist size) per stage; a stage with an embedding
# size is the first to need the query embedding, so easy queries never compute it
PLAN: tuple[tuple[int, int], ...] = ((5, 0), (10, 10), (20, 30))
STOP_MATCHES = 16  # the leader needs at least this many mutual matches...
STOP_MARGIN = 1.5  # ...and this multiple of the runner-up SKU's
FUSION_WEIGHT = 1.5  # score = m / (m + FUSION_SQUASH) + FUSION_WEIGHT * cosine
FUSION_SQUASH = 20.0  # (once the embedding has been computed)

Embed = Callable[[], np.ndarray]
LoadDescriptors = Callable[[str], tuple[np.ndarray, int]]


@dataclass(frozen=True, slots=True)
class CoarseBank:
    """The strongest keypoints of every view in a partition, for the coarse pass.

    des[i, :n[i]] are view i's descriptors (float32, strongest first); the rest is padding.
    Memory is views * 300 * 128 * 4 bytes, about 150 KB per view."""

    des: np.ndarray  # (views, COARSE_KEYPOINTS, 128)
    sq: np.ndarray  # (views, COARSE_KEYPOINTS) squared norms
    n: np.ndarray  # (views,) valid rows

    def __len__(self) -> int:
        return len(self.n)


def build_coarse_bank(descriptors: Sequence[np.ndarray], k: int = COARSE_KEYPOINTS) -> CoarseBank:
    des = np.zeros((len(descriptors), k, 128), dtype=np.float32)
    n = np.zeros(len(descriptors), dtype=np.int32)
    for i, d in enumerate(descriptors):
        n[i] = min(len(d), k)
        des[i, : n[i]] = d[: n[i]]
    return CoarseBank(des, (des * des).sum(axis=2), n)


def _nearest_two(q: np.ndarray, c: np.ndarray, qq: np.ndarray, cc: np.ndarray):
    """Per query descriptor: nearest candidate, the squared distances to its nearest and
    second-nearest candidates, and per candidate descriptor its nearest query descriptor.
    One matrix product; exact for SIFT's integer descriptors (every sum is below 2**24)."""
    d2 = qq[:, None] + cc[None, :] - 2.0 * (q @ c.T)
    rows = np.arange(len(q))
    back = d2.argmin(axis=0)
    nearest = d2.argmin(axis=1)
    first = d2[rows, nearest].copy()
    d2[rows, nearest] = np.inf
    return nearest, first, d2.min(axis=1), back


def mutual_count(q: np.ndarray, c: np.ndarray) -> int:
    """Matches that pass the ratio test and are mutual nearest neighbours."""
    if len(q) < 2 or len(c) < 2:
        return 0
    nearest, first, second, back = _nearest_two(q, c, (q * q).sum(1), (c * c).sum(1))
    keep = np.nonzero(first < RATIO * RATIO * second)[0]
    return int((back[nearest[keep]] == keep).sum())


def coarse_counts(q: np.ndarray, bank: CoarseBank) -> np.ndarray:
    """Ratio-test match count of the query's strongest keypoints against every view."""
    q = q[:COARSE_KEYPOINTS]
    out = np.zeros(len(bank), dtype=np.int32)
    if len(q) < 2:
        return out
    qq = (q * q).sum(1)
    rows = np.arange(len(q))
    for i in range(len(bank)):
        m = int(bank.n[i])
        if m < 2:
            continue
        d2 = qq[:, None] + bank.sq[i, :m][None, :] - 2.0 * (q @ bank.des[i, :m].T)
        nearest = d2.argmin(axis=1)
        first = d2[rows, nearest].copy()
        d2[rows, nearest] = np.inf
        out[i] = int((first < RATIO * RATIO * d2.min(axis=1)).sum())
    return out


@dataclass(frozen=True, slots=True)
class Ranked:
    sku_id: int
    entry: IndexEntry  # the SKU's best-scoring photo
    matches: int  # mutual matches of that photo
    score: float


def rank(
    query: np.ndarray,
    embed_query: Embed,
    part: Partition,
    bank: CoarseBank,
    load_desc: LoadDescriptors,
    top_k: int = 5,
) -> list[Ranked]:
    """Rank the SKUs of one partition. `query` holds the query's descriptors strongest
    first; `embed_query` is called at most once, and only if the cheap stages are not
    decisive."""
    if len(part) == 0:
        return []
    coarse_order = np.argsort(-coarse_counts(query, bank), kind="stable")
    k_max = max(k for _, k in PLAN)
    embedded: tuple[np.ndarray, np.ndarray] | None = None  # (best-first views, cosine of all)
    matches: dict[int, int] = {}
    seen: list[int] = []
    ranked: list[Ranked] = []
    for n, k in PLAN:
        if k and embedded is None:
            embedded = part.index.search(embed_query(), k_max)
        wanted = coarse_order[:n].tolist() + (embedded[0][:k].tolist() if k and embedded else [])
        for i in dict.fromkeys(wanted):
            if i not in matches:
                matches[i] = mutual_count(query, load_desc(part.entries[i].desc_key)[0])
                seen.append(i)

        best: dict[int, tuple[float, float, int]] = {}  # sku -> (score, cosine, view)
        most: dict[int, int] = {}  # sku -> most matches of any of its photos
        for i in seen:
            m, sku = matches[i], part.entries[i].sku_id
            cosine = float(embedded[1][i]) if embedded else 0.0
            score = m / (m + FUSION_SQUASH) + FUSION_WEIGHT * cosine if embedded else float(m)
            if sku not in best or (score, cosine) > best[sku][:2]:
                best[sku] = (score, cosine, i)
            most[sku] = max(most.get(sku, 0), m)
        order = sorted(best.items(), key=lambda kv: kv[1][:2], reverse=True)
        ranked = [
            Ranked(sku, part.entries[i], matches[i], score) for sku, (score, _, i) in order[:top_k]
        ]
        lead, runner_up = (sorted(most.values(), reverse=True) + [0, 0])[:2]
        if lead >= STOP_MATCHES and lead >= STOP_MARGIN * max(runner_up, 1):
            break
    return ranked
