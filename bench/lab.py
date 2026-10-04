"""Offline lab for ranker experiments on a real catalog.

The real evaluation (bench/evaluate.py) costs about 14 s per query, which is too slow
to tune anything. The lab caches the expensive parts once and scores ranker variants
from precomputed pair tables in seconds.

    uv run python bench/lab.py prep  --catalog catalog.json --photos ./photos --work ./work
    uv run python bench/lab.py pairs --cap 1200
    uv run python bench/lab.py eval  --cap 1200 --split tuning

`--work` is an index built by evaluate.py (gallery embeddings and cutouts come from it).
Everything the lab writes goes to --lab (default: <work>/../lab).

Discipline: queries are split into tuning and test halves by SKU hash. Choose parameters on
the tuning half only. `eval --split test` is for the final, frozen configuration.

SIFT features are extracted once at a high keypoint cap and sorted by response, so any
smaller cap is a slice (OpenCV keeps the strongest keypoints by response)."""

import argparse
import hashlib
import json
import math
import pickle
import random
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from poshak.engine.image_ops import decode_rgb, to_working_res
from poshak.engine.pipeline import aggregate_by_sku
from poshak.engine.scoring import confidence_and_tier, unified_cloth_score

LEVELS = {"mild": 0, "heavy": 1}
MODEL_TAG = "siglip-base-patch16-224-v2-768"
FEATURE_CAP = 4000
RECALL_K = 40
GEOM_FEATURES = ("sim4", "sim8", "sim16", "hom5", "hom10")
FEATURES = ("ratio", "mutual", *GEOM_FEATURES, "coarse300")


class Lab:
    def __init__(self, lab: Path) -> None:
        self.dir = lab
        for sub in ("img", "q", "feat", "pairs"):
            (lab / sub).mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        return self.dir.joinpath(*parts)


def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def half_of(sku: int) -> str:
    """Stable tuning/test split by SKU, so a SKU never appears in both."""
    return "tuning" if int(hashlib.md5(str(sku).encode()).hexdigest(), 16) % 2 == 0 else "test"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


# --------------------------------------------------------------------------- prep


def _vec(blob: bytes | None) -> np.ndarray | None:
    return None if blob is None else np.frombuffer(blob, dtype=np.float32)


def load_gallery(work: Path, photos: Path) -> list[dict]:
    db = sqlite3.connect(work / "data" / "index.sqlite")
    db.row_factory = sqlite3.Row
    rows = db.execute("SELECT * FROM images WHERE model_tag=? ORDER BY id", (MODEL_TAG,)).fetchall()
    out = []
    for r in rows:
        out.append(
            {
                "photo": r["photo_name"],
                "sku": r["sku_id"],
                "item": r["item"],
                "category": r["category"],
                "rembg": r["rembg_name"],
                "orig_vec": np.frombuffer(r["orig_vec"], dtype=np.float32),
                "cut_vec": _vec(r["rembg_vec"]),
                "path": str(photos / r["photo_name"]),
                "cut_path": None if not r["rembg_name"] else str(work / "rmbg" / r["rembg_name"]),
            }
        )  # fmt: skip
    return out


def build_queries(
    lab: Lab,
    gallery: list[dict],
    catalog: Path,
    photos: Path,
    work: Path,
    n_per_half: int,
    seed: int,
) -> list[dict]:
    """Known re-shots (cloth and fabric) and held-out second photos (cloth)."""
    import evaluate  # bench/evaluate.py, reused for the degradation and the held-out split

    skus = evaluate.load_catalog(catalog, photos)
    in_gallery = {g["photo"] for g in gallery}
    queries: list[dict] = []

    # known re-shots: first the SKUs the earlier real-pipeline run used (for the step-0 check),
    # then a seeded fill per half
    prior = set()
    ckpt = work / "queries.jsonl"
    if ckpt.exists():
        prior = {
            json.loads(x)["sku"] for x in ckpt.read_text("utf-8").splitlines() if '"known"' in x
        }
    rng = random.Random(seed)
    cloth = [g for g in gallery if g["category"] == "cloth"]
    chosen = {g["sku"]: g for g in cloth if g["sku"] in prior}
    for half in ("tuning", "test"):
        pool = [g for g in cloth if half_of(g["sku"]) == half and g["sku"] not in chosen]
        have = sum(1 for g in chosen.values() if half_of(g["sku"]) == half)
        for g in rng.sample(pool, max(0, min(n_per_half - have, len(pool)))):
            chosen[g["sku"]] = g
    fabric = [g for g in gallery if g["category"] == "fabric"]
    for g in [*chosen.values(), *fabric]:
        for level, idx in LEVELS.items():
            raw = Path(g["path"]).read_bytes()
            data = evaluate.degrade(
                decode_rgb(raw), np.random.RandomState([0, g["sku"], idx]), level
            )
            qid = f"known-{level}-{g['sku']}"
            lab.path("q", f"{qid}.jpg").write_bytes(data)
            queries.append({"qid": qid, "kind": "known", "level": level, "sku": g["sku"],
                            "category": g["category"], "half": half_of(g["sku"])})  # fmt: skip

    # held-out second photos of SKUs that have two; their other photo is in the gallery
    gallery_skus = {g["sku"] for g in gallery}
    for sku, info in sorted(skus.items()):
        if sku not in gallery_skus:
            continue
        held = [p for p in info["photos"] if p not in in_gallery]
        if not held:
            continue
        category = next(g["category"] for g in gallery if g["sku"] == sku)
        qid = f"held-asis-{sku}"
        lab.path("q", f"{qid}.jpg").write_bytes(Path(photos / held[0]).read_bytes())
        queries.append({"qid": qid, "kind": "held", "level": "asis", "sku": sku,
                        "category": category, "half": half_of(sku)})  # fmt: skip
    return queries


def prep(args: argparse.Namespace) -> None:
    from poshak.infrastructure.background.u2net import U2NetRemover
    from poshak.infrastructure.embedding.siglip import SiglipEmbedder

    lab = Lab(args.lab)
    q_file = lab.path("queries.json")
    gallery = load_gallery(args.work, args.photos)
    with open(lab.path("gallery.pkl"), "wb") as f:
        pickle.dump(gallery, f)
    print(f"gallery: {len(gallery)} photos "
          f"({sum(g['category'] == 'cloth' for g in gallery)} cloth, "
          f"{sum(g['category'] == 'fabric' for g in gallery)} fabric)", flush=True)  # fmt: skip
    if q_file.exists():
        queries = json.loads(q_file.read_text("utf-8"))
    else:
        queries = build_queries(
            lab, gallery, args.catalog, args.photos, args.work, args.per_half, args.seed
        )
        q_file.write_text(json.dumps(queries), encoding="utf-8")
    print(
        "queries:",
        dict(Counter((q["kind"], q["level"], q["category"], q["half"]) for q in queries)),
        flush=True,
    )

    embedder, remover = SiglipEmbedder("google/siglip-base-patch16-224"), U2NetRemover()

    def save_img(name: str, img: np.ndarray) -> None:
        np.save(lab.path("img", f"{name}.npy"), img)

    # gallery images at working resolution
    for i, g in enumerate(tqdm(gallery, desc="gallery images")):
        if not lab.path("img", f"g-orig-{i}.npy").exists():
            save_img(f"g-orig-{i}", to_working_res(decode_rgb(Path(g["path"]).read_bytes())))
        if g["cut_path"] and not lab.path("img", f"g-cut-{i}.npy").exists():
            save_img(f"g-cut-{i}", to_working_res(decode_rgb(Path(g["cut_path"]).read_bytes())))

    # query images, cutouts and embeddings
    emb_orig, emb_cut = {}, {}
    emb_file = lab.path("qemb.npz")
    if emb_file.exists():
        with np.load(emb_file, allow_pickle=True) as z:
            emb_orig, emb_cut = dict(z["orig"].item()), dict(z["cut"].item())
    for q in tqdm(queries, desc="query images + embeddings"):
        qid = q["qid"]
        if lab.path("img", f"q-orig-{qid}.npy").exists() and qid in emb_orig:
            continue
        img = decode_rgb(lab.path("q", f"{qid}.jpg").read_bytes())
        q768 = to_working_res(img)
        save_img(f"q-orig-{qid}", q768)
        emb_orig[qid] = embedder.embed(q768)
        if q["category"] == "cloth":
            cut, _ = remover.remove(img, "local")
            c768 = to_working_res(cut)
            save_img(f"q-cut-{qid}", c768)
            emb_cut[qid] = embedder.embed(c768)
        if len(emb_orig) % 25 == 0:
            np.savez(
                emb_file, orig=np.array(emb_orig, dtype=object), cut=np.array(emb_cut, dtype=object)
            )
    np.savez(emb_file, orig=np.array(emb_orig, dtype=object), cut=np.array(emb_cut, dtype=object))

    # SIFT features at the high cap, sorted by response
    sift = cv2.SIFT_create(nfeatures=FEATURE_CAP, contrastThreshold=0.02, edgeThreshold=10)
    names = [f"g-orig-{i}" for i in range(len(gallery))]
    names += [f"g-cut-{i}" for i, g in enumerate(gallery) if g["cut_path"]]
    names += [f"q-orig-{q['qid']}" for q in queries]
    names += [f"q-cut-{q['qid']}" for q in queries if q["category"] == "cloth"]
    for name in tqdm(names, desc="sift features"):
        out = lab.path("feat", f"{name}.npz")
        if out.exists():
            continue
        kp, des = sift.detectAndCompute(np.load(lab.path("img", f"{name}.npy")), None)
        if des is None:  # a blank image (for example a cutout that erased the garment)
            xy, des, kp = np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32), []
        else:
            order = np.argsort([-k.response for k in kp], kind="stable")
            xy = np.array([kp[i].pt for i in order], dtype=np.float32)
            des = des[order]
        np.savez(out, xy=xy, des=des.astype(np.uint8), n=np.int32(len(kp)))
    print("prep done", flush=True)


# ------------------------------------------------------------------------ matching


def load_features(lab: Lab, name: str, cap: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(lab.path("feat", f"{name}.npz")) as z:
        return z["xy"][:cap], z["des"][:cap]


def prepare(des: np.ndarray, root: bool) -> np.ndarray:
    d = des.astype(np.float32)
    if root:
        d = np.sqrt(d / (d.sum(axis=1, keepdims=True) + 1e-9))
    return d


def knn2(q: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each query descriptor: nearest candidate, and the squared distances to its nearest
    and second nearest, via one matrix product. Also the nearest query per candidate (for the
    mutual check). Exact for the integer-valued SIFT descriptors (sums stay below 2**24)."""
    d2 = (q * q).sum(1)[:, None] + (c * c).sum(1)[None, :] - 2.0 * (q @ c.T)
    rows = np.arange(len(q))
    back = d2.argmin(0)
    i1 = d2.argmin(1)
    m1 = d2[rows, i1].copy()
    d2[rows, i1] = np.inf
    m2 = d2.min(1)
    return i1, m1, m2, back


def ratio_matches(
    qd: np.ndarray, cd: np.ndarray, ratio: float = 0.75
) -> tuple[np.ndarray, np.ndarray]:
    """(query idx, candidate idx) of the matches that pass Lowe's ratio test, plus a mask of
    those that are also mutual nearest neighbours."""
    if len(qd) < 2 or len(cd) < 2:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, bool)
    i1, m1, m2, back = knn2(qd, cd)
    keep = np.nonzero(m1 < ratio * ratio * m2)[0]
    mutual = back[i1[keep]] == keep
    return keep, i1[keep], mutual


def inlier_count(src: np.ndarray, dst: np.ndarray, kind: str, thr: float) -> int:
    if len(src) < (3 if kind == "sim" else 4):
        return 0
    if kind == "sim":
        _, m = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=thr, maxIters=1500, confidence=0.995
        )
    else:
        _, m = cv2.findHomography(src, dst, cv2.RANSAC, thr, maxIters=1500, confidence=0.995)
    return 0 if m is None else int(m.sum())


def pair_features(qxy, qd, cxy, cd, coarse) -> list[float]:
    qi, ci, mutual = ratio_matches(qd, cd)
    out = [float(len(qi)), float(mutual.sum())]
    src, dst = qxy[qi], cxy[ci]
    for kind, thr in (("sim", 4), ("sim", 8), ("sim", 16), ("hom", 5), ("hom", 10)):
        out.append(float(inlier_count(src, dst, kind, thr)))
    cq, cc, _ = ratio_matches(qd[:coarse], cd[:coarse])
    out.append(float(len(cq)))
    return out


def pairs(args: argparse.Namespace) -> None:
    """Pair tables for one feature config: for every query and view, the top-K candidates by
    embedding cosine and the match features of each (query, candidate) pair."""
    cv2.setRNGSeed(0)
    lab = Lab(args.lab)
    gallery = load_pkl(lab.path("gallery.pkl"))
    queries = json.loads(lab.path("queries.json").read_text("utf-8"))
    with np.load(lab.path("qemb.npz"), allow_pickle=True) as z:
        emb = {"orig": z["orig"].item(), "cut": z["cut"].item()}
    if args.half:
        queries = [q for q in queries if q["half"] == args.half]
    if args.prior:  # a small batch: only the queries the earlier real-pipeline run answered
        queries = [q for q in queries if q["qid"] in prior_qids(args.work)]
    tag = f"cap{args.cap}{'-root' if args.root else ''}" + (f"-{args.half}" if args.half else "")
    tag += "-prior" if args.prior else ""
    out_file = lab.path("pairs", f"{tag}.pkl")

    zero = np.zeros(768, np.float32)
    gal_vec = {
        v: np.stack([zero if g[f"{v}_vec"] is None else g[f"{v}_vec"] for g in gallery])
        for v in ("orig", "cut")
    }
    cache: dict[str, tuple] = {}

    def gfeat(view: str, i: int):
        key = f"g-{view}-{i}"
        if key not in cache:
            xy, des = load_features(lab, key, args.cap)
            cache[key] = (xy, prepare(des, args.root))
        return cache[key]

    # shuffled so that a partial table is representative of every group, saved as it grows
    random.Random(0).shuffle(queries)
    table: dict = load_pkl(out_file) if out_file.exists() else {"rows": {}}
    table.update(cap=args.cap, root=args.root)
    t0 = time.perf_counter()
    for done, q in enumerate(tqdm(queries, desc=f"pairs {tag}")):
        for view in ("orig", "cut"):
            if view == "cut" and q["category"] != "cloth":
                continue
            if (q["qid"], view) in table["rows"]:
                continue
            idx = np.array(
                [
                    i
                    for i, g in enumerate(gallery)
                    if g["category"] == q["category"]
                    and (view == "orig" or g["cut_vec"] is not None)
                ]
            )
            sims = gal_vec[view][idx] @ emb[view][q["qid"]]
            top = np.argsort(-sims)[:RECALL_K]
            qxy, qd0 = load_features(lab, f"q-{view}-{q['qid']}", args.cap)
            qd = prepare(qd0, args.root)
            rows = [pair_features(qxy, qd, *gfeat(view, int(idx[j])), args.coarse) for j in top]
            feats = np.array(rows, dtype=np.float32)
            table["rows"][(q["qid"], view)] = {
                "cand": idx[top],
                "cos": sims[top].astype(np.float32),
                "feats": feats,
            }
        if done % 25 == 24:
            save_table(table, out_file)
    table["seconds"] = time.perf_counter() - t0
    save_table(table, out_file)
    print(f"wrote {out_file} in {table['seconds']:.0f}s", flush=True)


# --------------------------------------------------------------------------- eval


class Entry:
    def __init__(self, sku_id: int) -> None:
        self.sku_id = sku_id


def rank_skus(skus: np.ndarray, score: np.ndarray, cos: np.ndarray) -> list[int]:
    """SKU order, best photo per SKU, ties broken by embedding cosine."""
    best: dict[int, tuple[float, float]] = {}
    for s, sc, c in zip(skus.tolist(), score.tolist(), cos.tolist(), strict=True):
        if s not in best or (sc, c) > best[s]:
            best[s] = (sc, c)
    return [s for s, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)]


def production_hits(row: dict, skus: np.ndarray, cap: int) -> list[dict]:
    """Candidates as the production code builds them (pipeline.recall_verify), best count first."""
    hits = []
    for j in np.argsort(-row["feats"][:, 0], kind="stable"):
        good, cos = int(row["feats"][j, 0]), float(row["cos"][j])
        conf, tier = confidence_and_tier(good, good / cap, cos)
        hits.append(
            {
                "entry": Entry(int(skus[row["cand"][j]])),
                "siglip_sim": cos,
                "sift_ratio": good / cap,
                "sift_matches": good,
                "score": conf,
                "confidence_tier": tier,
            }
        )
    return hits


def legacy_orig(row: dict, skus: np.ndarray, cap: int) -> list[int]:
    """The production original-only ranking, using the production aggregation code."""
    return [sku for sku, _ in aggregate_by_sku(production_hits(row, skus, cap), top_k_sku=10)]


def legacy_full(orig: dict, cut: dict, skus: np.ndarray, cap: int) -> list[int]:
    """The production dual-view fusion (pipeline.cloth_search), from the same tables."""
    fresh = {"bg_score": 0, "bg": 0, "orig_score": 0, "orig": 0}
    fused: dict[int, dict] = {}
    for sku, d in aggregate_by_sku(production_hits(cut, skus, cap)):
        fused[sku] = {**fresh, "bg_score": d["best_score"], "bg": d["sift_matches"]}
    for sku, d in aggregate_by_sku(production_hits(orig, skus, cap)):
        fused.setdefault(sku, dict(fresh)).update(
            orig_score=d["best_score"], orig=d["sift_matches"]
        )
    res = []
    for sku, c in fused.items():
        uni, _ = unified_cloth_score(c["bg"], c["orig"], c["bg_score"], c["orig_score"])
        res.append((c["bg"] + c["orig"], uni, sku))
    res.sort(reverse=True)
    return [r[2] for r in res]


def save_table(table: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(table, f)
    tmp.replace(path)  # readers never see a half-written table


def load_tables(lab: Lab, tag: str):
    return load_pkl(lab.path("pairs", f"{tag}.pkl"))


def fmt(hits: list[bool], n: int) -> str:
    k = sum(hits)
    lo, hi = wilson(k, n)
    return f"{k:3d}/{n:<3d} {k / n:5.1%} [{lo:4.0%}-{hi:4.0%}]" if n else "   -"


def group_key(q: dict) -> str:
    return f"{q['kind']}/{q['level']}/{q['category']}"


def evaluate_variants(lab: Lab, tag: str, split: str, variants: dict) -> dict:
    table = load_tables(lab, tag)
    gallery = load_pkl(lab.path("gallery.pkl"))
    skus = np.array([g["sku"] for g in gallery])
    queries = [q for q in json.loads(lab.path("queries.json").read_text("utf-8"))
               if q["half"] == split and (q["qid"], "orig") in table["rows"]]  # fmt: skip
    results: dict[str, dict[str, list[tuple[str, int | None]]]] = {}
    for name, fn in variants.items():
        per: dict[str, list] = {}
        for q in queries:
            orig = table["rows"][(q["qid"], "orig")]
            cut = table["rows"].get((q["qid"], "cut"))
            ranking = fn(orig, cut, skus, table["cap"])
            rank = ranking.index(q["sku"]) + 1 if q["sku"] in ranking else None
            per.setdefault(group_key(q), []).append((q["qid"], rank))
        results[name] = per
    return results


def report(results: dict, baseline: str | None = None) -> None:
    groups = sorted({g for per in results.values() for g in per})
    for g in groups:
        n = len(next(iter(results.values()))[g])
        print(f"\n== {g}  (n={n})")
        print(
            f"  {'variant':34s} {'R@1':>22s} {'R@5':>6s} {'R@10':>6s}"
            + ("   vs baseline (gain/loss)" if baseline else "")
        )
        base = dict(results[baseline][g]) if baseline else {}
        for name, per in results.items():
            ranks = [r for _, r in per[g]]
            r1 = [r == 1 for r in ranks]
            r5 = sum(1 for r in ranks if r is not None and r <= 5) / n
            r10 = sum(1 for r in ranks if r is not None and r <= 10) / n
            line = f"  {name:34s} {fmt(r1, n):>22s} {r5:6.1%} {r10:6.1%}"
            if baseline and name != baseline:
                gain = sum(1 for q, r in per[g] if r == 1 and base[q] != 1)
                loss = sum(1 for q, r in per[g] if r != 1 and base[q] == 1)
                line += f"   +{gain}/-{loss}"
            print(line)


def standard_variants(feature_names: tuple[str, ...] = FEATURES) -> dict:
    """The variants that need only the pair table for one feature config."""

    def single(col: int, view_first: bool = True):
        def fn(orig, cut, skus, cap):
            return rank_skus(skus[orig["cand"]], orig["feats"][:, col], orig["cos"])

        return fn

    def two_pass(col: int, how: str = "sum"):
        def fn(orig, cut, skus, cap):
            if cut is None:
                return rank_skus(skus[orig["cand"]], orig["feats"][:, col], orig["cos"])
            best: dict[int, float] = {}
            for row in (orig, cut):
                per: dict[int, float] = {}
                for j, s in enumerate(skus[row["cand"]].tolist()):
                    per[s] = max(per.get(s, 0.0), float(row["feats"][j, col]))
                for s, v in per.items():
                    best[s] = best.get(s, 0.0) + v if how == "sum" else max(best.get(s, 0.0), v)
            return sorted(best, key=lambda s: best[s], reverse=True)

        return fn

    v = {
        "legacy original-only": lambda o, c, s, cap: legacy_orig(o, s, cap),
        "legacy full (production)": lambda o, c, s, cap: (
            legacy_full(o, c, s, cap) if c is not None else legacy_orig(o, s, cap)
        ),
    }
    for name in feature_names:
        col = FEATURES.index(name)
        v[f"orig: {name}"] = single(col)
    for name in ("ratio", "mutual", "sim8", "hom10"):
        v[f"two-pass sum: {name}"] = two_pass(FEATURES.index(name))
    return v


def cascade_variants(fine: str) -> dict:
    """Coarse match on the strongest keypoints for every candidate, full match plus geometry
    only on the best N. Candidates outside the best N score zero."""
    col, coarse = FEATURES.index(fine), FEATURES.index("coarse300")

    def make(n: int):
        def fn(orig, cut, skus, cap):
            keep = np.argsort(-orig["feats"][:, coarse], kind="stable")[:n]
            score = np.zeros(len(orig["cos"]), np.float32)
            score[keep] = orig["feats"][keep, col]
            return rank_skus(skus[orig["cand"]], score, orig["cos"])

        return fn

    return {f"cascade N={n}: coarse300 -> {fine}": make(n) for n in (3, 5, 8, 12, 20)}


def embedding_topk_variants(geom: str) -> dict:
    """Verify only the embedding's best K candidates (the table is in cosine order)."""
    col = FEATURES.index(geom)

    def make(k: int):
        def fn(orig, cut, skus, cap):
            return rank_skus(skus[orig["cand"][:k]], orig["feats"][:k, col], orig["cos"][:k])

        return fn

    return {f"emb top {k}: {geom}": make(k) for k in (5, 10, 15, 20, 30, 40)}


def fusion_variants(geom: str) -> dict:
    """Geometry score squashed to [0, 1), plus w times the embedding cosine."""
    col = FEATURES.index(geom)

    def make(w: float):
        def fn(orig, cut, skus, cap):
            g = orig["feats"][:, col]
            return rank_skus(skus[orig["cand"]], g / (g + 20.0) + w * orig["cos"], orig["cos"])

        return fn

    return {f"fusion w={w}: {geom}": make(w) for w in (0.0, 0.25, 0.5, 1.0, 2.0)}


def early_exit_report(lab: Lab, tag: str, split: str, geom: str) -> None:
    """For cloth queries: how often can the original pass alone decide, and does deciding
    early ever change the answer? Compared against always running both passes."""
    table = load_tables(lab, tag)
    gallery = load_pkl(lab.path("gallery.pkl"))
    skus = np.array([g["sku"] for g in gallery])
    col = FEATURES.index(geom)
    qs = [
        q
        for q in json.loads(lab.path("queries.json").read_text("utf-8"))
        if q["half"] == split and q["category"] == "cloth" and (q["qid"], "cut") in table["rows"]
    ]
    two = standard_variants()["two-pass sum: " + geom]

    def lead(orig):
        per: dict[int, float] = {}
        for j, s in enumerate(skus[orig["cand"]].tolist()):
            per[s] = max(per.get(s, 0.0), float(orig["feats"][j, col]))
        top = sorted(per.values(), reverse=True) + [0.0, 0.0]
        return top[0], top[1]

    data = []
    for q in qs:
        orig, cut = table["rows"][(q["qid"], "orig")], table["rows"][(q["qid"], "cut")]
        o_rank = rank_skus(skus[orig["cand"]], orig["feats"][:, col], orig["cos"])
        t_rank = two(orig, cut, skus, table["cap"])
        data.append((q, lead(orig), o_rank, t_rank))
    n = len(data)
    print(f"\n== early exit on {n} cloth queries ({split}), geometry = {geom}")
    print(f"  always two passes:  R@1 {sum(d[3][:1] == [d[0]['sku']] for d in data) / n:.1%}")
    print(f"  always orig only:   R@1 {sum(d[2][:1] == [d[0]['sku']] for d in data) / n:.1%}")
    print(f"  {'T':>4s} {'M':>4s} {'skip':>6s} {'R@1':>6s} {'changed vs two-pass':>20s}")
    for t in (8, 12, 16, 24, 32, 48):
        for m in (1.5, 2.0, 3.0, 5.0):
            skip = hit = changed = 0
            for q, (lead1, lead2), o_rank, t_rank in data:
                decisive = lead1 >= t and lead1 >= m * max(lead2, 1.0)
                rank = o_rank if decisive else t_rank
                skip += decisive
                hit += rank[:1] == [q["sku"]]
                changed += decisive and o_rank[:1] != t_rank[:1]
            print(f"  {t:4d} {m:4.1f} {skip / n:6.1%} {hit / n:6.1%} {changed:>20d}")


def evaluate(args: argparse.Namespace) -> None:
    lab = Lab(args.lab)
    tag = f"cap{args.cap}{'-root' if args.root else ''}" + (
        f"-{args.table_half}" if args.table_half else ""
    )
    tag += "-prior" if args.prior else ""
    if args.split == "test" and not args.final:
        sys.exit("the test half is for the final frozen configuration; pass --final to read it")
    if args.family == "early":
        early_exit_report(lab, tag, args.split, args.geom)
        return
    families = {
        "standard": standard_variants(),
        "cascade": {**standard_variants(), **cascade_variants(args.geom)},
        "fusion": {**standard_variants(), **fusion_variants(args.geom)},
        "embk": {**standard_variants(), **embedding_topk_variants(args.geom)},
    }
    results = evaluate_variants(lab, tag, args.split, families[args.family])
    report(results, baseline="legacy original-only")


def query_id(row: dict) -> str:
    """Lab id of a query from the real-pipeline checkpoint (evaluate.py's queries.jsonl)."""
    if row["kind"] == "known":
        return f"known-{row['variant']}-{row['sku']}"
    return f"held-asis-{row['sku']}"


def prior_qids(work: Path) -> set[str]:
    rows = (work / "queries.jsonl").read_text("utf-8").splitlines()
    return {query_id(json.loads(x)) for x in rows}


def cascade_live(args: argparse.Namespace) -> None:
    """Small-catalog recall experiment, run live (candidates differ per config, so a pair table
    cannot be reused): coarse-match every original-view gallery photo on its strongest
    keypoints, shortlist the best N (optionally unioned with the embedding's top K), then run
    the full match and geometric check on the shortlist only."""
    cv2.setRNGSeed(0)
    lab = Lab(args.lab)
    gallery = load_pkl(lab.path("gallery.pkl"))
    skus = np.array([g["sku"] for g in gallery])
    with np.load(lab.path("qemb.npz"), allow_pickle=True) as z:
        emb = z["orig"].item()
    queries = [
        q
        for q in json.loads(lab.path("queries.json").read_text("utf-8"))
        if q["half"] == args.split and (args.kind == "all" or q["kind"] == args.kind)
    ]
    random.Random(0).shuffle(queries)
    queries = queries[: args.limit] if args.limit else queries
    gal = {}
    for i in range(len(gallery)):
        xy, des = load_features(lab, f"g-orig-{i}", args.cap)
        gal[i] = (xy, prepare(des, False))
    gvec = np.stack([g["orig_vec"] for g in gallery])
    cats = np.array([g["category"] for g in gallery])
    cache: dict[tuple[str, int], list[float]] = {}
    configs = [(n, k) for n in (5, 10, 20) for k in (0, 10)] + [(0, 40)]
    stats = {c: {} for c in configs}
    t_coarse = []
    for q in tqdm(queries, desc="cascade live"):
        t0 = time.perf_counter()
        qxy, qd0 = load_features(lab, f"q-orig-{q['qid']}", args.cap)
        qd = prepare(qd0, False)
        idx = np.nonzero(cats == q["category"])[0]
        coarse = np.array(
            [len(ratio_matches(qd[: args.coarse], gal[i][1][: args.coarse])[0]) for i in idx]
        )
        t_coarse.append(time.perf_counter() - t0)
        by_coarse = idx[np.argsort(-coarse, kind="stable")]
        cos = gvec[idx] @ emb[q["qid"]]
        by_emb = idx[np.argsort(-cos)]
        cos_of = dict(zip(idx.tolist(), cos.tolist(), strict=True))
        for n, k in configs:
            chosen = list(dict.fromkeys([*by_coarse[:n].tolist(), *by_emb[:k].tolist()]))
            for c in chosen:
                if (q["qid"], c) not in cache:
                    cache[(q["qid"], c)] = pair_features(qxy, qd, *gal[c], args.coarse)
            f = np.array([cache[(q["qid"], c)] for c in chosen], dtype=np.float32)
            ranks = {}
            for name in ("ratio", "mutual", "hom10"):
                col = FEATURES.index(name)
                order = rank_skus(skus[chosen], f[:, col], np.array([cos_of[c] for c in chosen]))
                ranks[name] = order.index(q["sku"]) + 1 if q["sku"] in order else None
            key = f"{q['kind']}/{q['level']}/{q['category']}"
            st = stats[(n, k)].setdefault(
                key, {"n": 0, "in_set": 0, "fine": 0, "ratio": 0, "mutual": 0, "hom10": 0}
            )
            st["n"] += 1
            st["in_set"] += q["sku"] in set(skus[chosen].tolist())
            st["fine"] += len(chosen)
            for name in ("ratio", "mutual", "hom10"):
                st[name] += ranks[name] == 1
    print(f"\ncoarse pass over the whole gallery: {np.mean(t_coarse) * 1000:.0f} ms per query")
    head = f"  {'shortlist':26s} {'n':>4s} {'true in set':>11s} {'fine/q':>7s}"
    head += f" {'R@1 ratio':>10s} {'mutual':>7s} {'hom10':>7s}"
    for key in sorted({k for st in stats.values() for k in st}):
        print(f"\n== {key}")
        print(head)
        for (n, k), st in stats.items():
            if key not in st:
                continue
            r, m = st[key], st[key]["n"]
            label = f"emb top {k} (baseline)" if not n else f"coarse top {n}"
            label += f" + emb top {k}" if n and k else ""
            line = f"  {label:26s} {m:4d} {r['in_set'] / m:11.1%} {r['fine'] / m:7.1f}"
            print(line + f" {r['ratio'] / m:10.1%} {r['mutual'] / m:7.1%} {r['hom10'] / m:7.1%}")


def shortlist(args: argparse.Namespace) -> None:
    """Save, per query, the coarse-match order over the whole gallery and the full match
    features of the coarse top candidates that the embedding's top 40 does not already cover.
    Together with the pair table this is enough to replay any progressive policy offline."""
    cv2.setRNGSeed(0)
    lab = Lab(args.lab)
    gallery = load_pkl(lab.path("gallery.pkl"))
    table = load_tables(lab, f"cap{args.cap}-{args.split}")
    with np.load(lab.path("qemb.npz"), allow_pickle=True) as z:
        emb = z["orig"].item()
    queries = [
        q
        for q in json.loads(lab.path("queries.json").read_text("utf-8"))
        if q["half"] == args.split and (q["qid"], "orig") in table["rows"]
    ]
    random.Random(1).shuffle(queries)
    queries = queries[: args.limit] if args.limit else queries
    out_file = lab.path("pairs", f"short-{args.split}.pkl")
    out = load_pkl(out_file) if out_file.exists() else {}
    gal = {}
    for i in range(len(gallery)):
        xy, des = load_features(lab, f"g-orig-{i}", args.cap)
        gal[i] = (xy, prepare(des, False))
    gvec = np.stack([g["orig_vec"] for g in gallery])
    cats = np.array([g["category"] for g in gallery])
    for done, q in enumerate(tqdm(queries, desc="shortlist")):
        if q["qid"] in out:
            continue
        qxy, qd0 = load_features(lab, f"q-orig-{q['qid']}", args.cap)
        qd = prepare(qd0, False)
        idx = np.nonzero(cats == q["category"])[0]
        c = args.coarse
        counts = np.array([len(ratio_matches(qd[:c], gal[i][1][:c])[0]) for i in idx])
        order = np.argsort(-counts, kind="stable")
        have = set(table["rows"][(q["qid"], "orig")]["cand"].tolist())
        extra = {}
        for i in idx[order[: args.top]].tolist():
            if i not in have:
                feats = pair_features(qxy, qd, *gal[i], c)
                extra[i] = (np.array(feats, np.float32), float(gvec[i] @ emb[q["qid"]]))
        out[q["qid"]] = {"order": idx[order[:40]], "counts": counts[order[:40]], "extra": extra}
        if done % 25 == 24:
            save_table(out, out_file)
    save_table(out, out_file)
    print(f"saved {len(out)} shortlists to {out_file}", flush=True)


def progressive(args: argparse.Namespace) -> None:
    """Replay progressive verification policies: verify a small shortlist, stop when the leader
    is decisive, otherwise widen. Reports accuracy and the number of full matches used."""
    lab = Lab(args.lab)
    gallery = load_pkl(lab.path("gallery.pkl"))
    skus = np.array([g["sku"] for g in gallery])
    table = load_tables(lab, f"cap{args.cap}-{args.split}")
    short = load_pkl(lab.path("pairs", f"short-{args.split}.pkl"))
    col = FEATURES.index(args.geom)
    qs = [
        q
        for q in json.loads(lab.path("queries.json").read_text("utf-8"))
        if q["qid"] in short and (q["qid"], "orig") in table["rows"] and q["category"] in args.cats
    ]

    def run(q, plan, t, m, w=0.0, squash=20.0):
        row, sh = table["rows"][(q["qid"], "orig")], short[q["qid"]]
        feats = {int(c): (row["feats"][j], float(row["cos"][j])) for j, c in enumerate(row["cand"])}
        feats.update({c: v for c, v in sh["extra"].items()})
        seen: list[int] = []
        for stage, (n, k) in enumerate(plan):
            pool = [*sh["order"][:n].tolist(), *row["cand"][:k].tolist()]
            seen += [c for c in dict.fromkeys(pool) if c not in seen]
            sc = np.array([feats[x][0][col] for x in seen])
            cosv = np.array([feats[x][1] for x in seen])
            # blend the embedding in only once it has been computed (some stage used it)
            score = sc / (sc + squash) + w * cosv if any(k for _, k in plan[: stage + 1]) else sc
            ranking = rank_skus(skus[seen], score, cosv)
            best: dict[int, float] = {}
            for c, v in zip(seen, sc.tolist(), strict=True):
                best[int(skus[c])] = max(best.get(int(skus[c]), 0.0), v)
            top = sorted(best.values(), reverse=True) + [0.0, 0.0]
            if (top[0] >= t and top[0] >= m * max(top[1], 1.0)) or stage == len(plan) - 1:
                used = plan[: stage + 1]
                return ranking, len(seen), stage, any(n for n, _ in used), any(k for _, k in used)
        raise AssertionError

    plans = {
        "baseline: emb 40": [(0, 40)],
        "fixed: coarse10+emb10": [(10, 10)],
        "fixed: coarse10+emb20": [(10, 20)],
        "fixed: coarse10+emb30": [(10, 30)],
        "progressive A: 5+5, 10+10, emb 40": [(5, 5), (10, 10), (0, 40)],
        "progressive B: 5+5, 10+20, emb 40": [(5, 5), (10, 20), (0, 40)],
        "progressive C: 5+5, 10+10, 20+30": [(5, 5), (10, 10), (20, 30)],
        "progressive D: emb 10, 10+20, emb 40": [(0, 10), (10, 20), (0, 40)],
        "lazy-embed E1: coarse 5, 10+10, emb 40": [(5, 0), (10, 10), (0, 40)],
        "lazy-embed E2: coarse 8, 10+10, 20+30": [(8, 0), (10, 10), (20, 30)],
        "lazy-embed E3: coarse 5, 5+5, 10+20, 20+30": [(5, 0), (5, 5), (10, 20), (20, 30)],
        "lazy-embed E4: coarse 5, 10+10, 20+30": [(5, 0), (10, 10), (20, 30)],
    }
    # per-stage costs measured on the i5-7200U (seconds); the coarse pass scales with gallery size
    t_detect, t_coarse, t_embed, t_full = 0.12, 0.224, 0.40, 0.03
    groups = sorted({group_key(q) for q in qs})
    for t, m in zip(args.thresholds[::2], args.thresholds[1::2], strict=True):
        print(f"\n##### decisive when leader >= {t} {args.geom} inliers and >= {m}x the runner-up")
        for g in groups:
            members = [q for q in qs if group_key(q) == g]
            print(f"\n== {g} (n={len(members)})")
            head = f"  {'policy':44s} {'R@1':>6s} {'full/q':>7s} {'stage1':>7s}"
            head += f" {'emb':>5s} {'est s':>6s}"
            print(head)
            for name, plan in plans.items():
                if args.plan_filter not in name:
                    continue
                res = [run(q, plan, t, m, args.w, args.squash) for q in members]
                r1 = np.mean([r[0][:1] == [q["sku"]] for r, q in zip(res, members, strict=True)])
                full = np.mean([r[1] for r in res])
                stage1 = np.mean([r[2] == 0 for r in res])
                emb = np.mean([r[4] for r in res])
                secs = t_detect + t_coarse * np.mean([r[3] for r in res]) + t_embed * emb
                secs += t_full * full
                print(f"  {name:44s} {r1:6.1%} {full:7.1f} {stage1:7.0%} {emb:5.0%} {secs:6.2f}")
                if args.paired:
                    print("    " + paired_line(members, res, table, skus, args.cap))


def paired_line(members: list[dict], res: list, table: dict, skus: np.ndarray, cap: int) -> str:
    """The production ranker on the same queries, and what the new one gained and lost."""
    old, new = [], []
    for q, r in zip(members, res, strict=True):
        orig = table["rows"][(q["qid"], "orig")]
        cut = table["rows"].get((q["qid"], "cut"))
        ranking = (
            legacy_full(orig, cut, skus, cap) if cut is not None else legacy_orig(orig, skus, cap)
        )
        old.append(ranking[:1] == [q["sku"]])
        new.append(r[0][:1] == [q["sku"]])
    n, gained = len(members), sum(b and not a for a, b in zip(old, new, strict=True))
    lost = sum(a and not b for a, b in zip(old, new, strict=True))
    return (
        f"production {fmt(old, n)} | new {fmt(new, n)} | "
        f"gained {gained}, lost {lost} (net {gained - lost:+d})"
    )


def step0(args: argparse.Namespace) -> None:
    """The lab's legacy rankers must reproduce the earlier real-pipeline ranks."""
    lab = Lab(args.lab)
    rows = [json.loads(x) for x in (args.work / "queries.jsonl").read_text("utf-8").splitlines()]
    table = load_tables(lab, f"cap{args.cap}-prior")
    gallery = load_pkl(lab.path("gallery.pkl"))
    skus = np.array([g["sku"] for g in gallery])
    agree = {"sift_original": [0, 0], "full": [0, 0]}
    for r in rows:
        qid = {"known": f"known-{r['variant']}-{r['sku']}", "heldout": f"held-asis-{r['sku']}"}[
            r["kind"]
        ]
        orig, cut = table["rows"].get((qid, "orig")), table["rows"].get((qid, "cut"))
        if orig is None:
            continue
        full = legacy_full(orig, cut, skus, args.cap) if cut else legacy_orig(orig, skus, args.cap)
        for mode, ranking in (("sift_original", legacy_orig(orig, skus, args.cap)), ("full", full)):
            lab_rank = ranking.index(r["sku"]) + 1 if r["sku"] in ranking else None
            ref = r["modes"][mode]["rank"]
            agree[mode][0] += lab_rank == ref
            agree[mode][1] += 1
    for mode, (ok, n) in agree.items():
        print(f"step 0, {mode}: lab rank equals the real pipeline's on {ok}/{n} queries")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prep", "pairs", "eval", "step0", "cascade", "shortlist", "progressive"):
        p = sub.add_parser(name)
        p.add_argument("--work", type=Path, default=Path("G:/tmp-dataset/work"))
        p.add_argument("--lab", type=Path, default=Path("G:/tmp-dataset/lab"))
        if name == "prep":
            p.add_argument("--catalog", type=Path, required=True)
            p.add_argument("--photos", type=Path, required=True)
            p.add_argument("--per-half", type=int, default=100, help="known cloth SKUs per half")
            p.add_argument("--seed", type=int, default=7)
        else:
            p.add_argument("--cap", type=int, default=1200)
            p.add_argument("--root", action="store_true")
        if name == "pairs":
            p.add_argument("--half", choices=("tuning", "test"))
            p.add_argument("--coarse", type=int, default=300)
            p.add_argument("--prior", action="store_true", help="only the step-0 batch")
        if name in ("shortlist", "progressive"):
            p.add_argument("--split", choices=("tuning", "test"), default="tuning")
            p.add_argument("--geom", choices=FEATURES, default="hom10")
        if name == "shortlist":
            p.add_argument("--coarse", type=int, default=300)
            p.add_argument("--top", type=int, default=20)
            p.add_argument("--limit", type=int, default=0)
        if name == "progressive":
            p.add_argument("--cats", nargs="+", default=["cloth", "fabric"])
            p.add_argument("--paired", action="store_true", help="compare with production")
            p.add_argument("--w", type=float, default=0.0, help="embedding cosine weight")
            p.add_argument("--squash", type=float, default=20.0, help="geometry g/(g+squash)")
            p.add_argument("--plan-filter", default="", help="only plans whose name contains this")
            p.add_argument(
                "--thresholds",
                nargs="+",
                type=float,
                default=[16, 1.5, 24, 2.0, 32, 2.0],
                help="pairs of: leader inliers, margin over the runner-up",
            )
        if name == "cascade":
            p.add_argument("--split", choices=("tuning", "test"), default="tuning")
            p.add_argument("--kind", choices=("all", "known", "held"), default="known")
            p.add_argument("--coarse", type=int, default=300)
            p.add_argument("--limit", type=int, default=0)
        if name == "eval":
            p.add_argument("--split", choices=("tuning", "test"), default="tuning")
            p.add_argument("--table-half", choices=("tuning", "test"))
            p.add_argument("--final", action="store_true")
            p.add_argument("--prior", action="store_true", help="read the step-0 batch table")
            p.add_argument("--family", choices=("standard", "cascade", "fusion", "early", "embk"),
                           default="standard")  # fmt: skip
            p.add_argument("--geom", choices=FEATURES, default="sim8")
    args = ap.parse_args()
    cmds = {
        "prep": prep,
        "pairs": pairs,
        "eval": evaluate,
        "step0": step0,
        "cascade": cascade_live,
        "shortlist": shortlist,
        "progressive": progressive,
    }
    cmds[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
