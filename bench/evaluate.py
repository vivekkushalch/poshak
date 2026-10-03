"""Leave-one-out retrieval accuracy on a real catalog, with the real models.

    uv run python bench/evaluate.py --catalog catalog.json --photos ./photos --workdir ./work

Two kinds of query share one index, because they answer different questions:

  known     a degraded re-shot of a photo that IS in the index (crop, tilt, exposure,
            blur, heavy JPEG, at a "mild" and a "heavy" level). Tests robustness to
            capture conditions for the same garment seen from the same side.
  heldout   for every SKU with two or more photos, one photo is held out of the index
            and used as the query. Catalogs often photograph the front and the back
            of a garment, so this is frequently a cross-side match, which local
            features are not built for. Read it as a limit, not as headline accuracy.

All other SKUs stay in the gallery as distractors. A query is a hit at k when its SKU
is in the top k results.

Three rankers run on each query, so the contribution of every stage is visible:
  embedding      SigLIP cosine only
  sift_original  recall 40 by cosine, verify with SIFT, original photo view only
  full           the production cloth pipeline: original view + background-removed view

Indexing and query results are checkpointed, so an interrupted run resumes. The catalog
format is the one `poshak seed` reads. Nothing in the output names a photo."""

import argparse
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from poshak.container import Container
from poshak.core.config import Settings
from poshak.engine import pipeline
from poshak.engine.image_ops import decode_rgb, to_working_res

KS = (1, 5, 10)
MODES = ("embedding", "sift_original", "full")
LEVELS = {"mild": 0, "heavy": 1}


def load_catalog(path: Path, photos: Path) -> dict[int, dict]:
    """sku_id -> {item, photos[]} restricted to photos that exist on disk."""
    skus = {}
    for sku in json.loads(path.read_text(encoding="utf-8"))["skus"]:
        names = sku.get("images") or sku.get("images_on_server") or sku.get("images_in_db") or []
        present = sorted({n.strip() for n in names if n.strip() and (photos / n.strip()).is_file()})
        if present:
            skus[int(sku.get("sku_id") or sku["stock_id"])] = {
                "item": sku.get("item", ""),
                "photos": present,
            }
    return skus


def split(
    skus: dict[int, dict], seed: int, limit: int | None
) -> tuple[list[tuple[int, str]], list[tuple[int, str, str]]]:
    """(queries as (sku, photo), gallery as (sku, photo, item)).
    One photo is held out per multi-photo SKU.."""
    rng = random.Random(seed)
    ids = sorted(skus)
    if limit:
        ids = sorted(rng.sample(ids, min(limit, len(ids))))
    queries, gallery = [], []
    for sku in ids:
        photos = list(skus[sku]["photos"])
        held = rng.choice(photos) if len(photos) >= 2 else None
        if held:
            queries.append((sku, held))
        gallery += [(sku, p, skus[sku]["item"]) for p in photos if p != held]
    return queries, gallery


def degrade(img_rgb: np.ndarray, rng: np.random.RandomState, level: str) -> bytes:
    """A plausible customer photo of the same garment: crop, tilt, exposure, blur, JPEG."""
    heavy = level == "heavy"
    h, w = img_rgb.shape[:2]
    scale = rng.uniform(0.45, 0.70) if heavy else rng.uniform(0.70, 0.95)
    ch, cw = int(h * scale), int(w * scale * rng.uniform(0.92, 1.0))
    y, x = rng.randint(0, h - ch + 1), rng.randint(0, w - cw + 1)
    img = img_rgb[y : y + ch, x : x + cw]
    tilt = 15 if heavy else 8
    m = cv2.getRotationMatrix2D((cw / 2, ch / 2), rng.uniform(-tilt, tilt), 1.0)
    img = cv2.warpAffine(img, m, (cw, ch), borderMode=cv2.BORDER_REFLECT)
    lo, hi = (0.6, 1.4) if heavy else (0.75, 1.25)
    img = cv2.convertScaleAbs(
        img, alpha=rng.uniform(lo, hi), beta=rng.uniform(-35 if heavy else -20, 35 if heavy else 20)
    )
    if heavy or rng.rand() < 0.5:
        img = cv2.GaussianBlur(
            img, (0, 0), rng.uniform(1.2, 2.4) if heavy else rng.uniform(0.6, 1.6)
        )
    longest = max(img.shape[:2])
    cap = 1000 if heavy else 1600  # chat apps and cameras downscale before sharing
    if longest > cap:
        f = cap / longest
        img = cv2.resize(
            img, (int(img.shape[1] * f), int(img.shape[0] * f)), interpolation=cv2.INTER_AREA
        )
    quality = rng.uniform(25, 50) if heavy else rng.uniform(45, 75)
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )
    assert ok
    return buf.tobytes()


def rank_of(sku: int, ranking: list[int]) -> int | None:
    """1-based position of sku in the ranking, or None if absent."""
    return ranking.index(sku) + 1 if sku in ranking else None


def unique_skus(entries, order) -> list[int]:
    """SKU ids in first-seen order (the best view of each SKU comes first)."""
    seen, out = set(), []
    for i in order:
        sku = entries[int(i)].sku_id
        if sku not in seen:
            seen.add(sku)
            out.append(sku)
    return out


def rankings(c: Container, data: bytes) -> dict[str, tuple[list[int], float]]:
    """Per-mode SKU ranking and seconds spent, for one query."""
    snap, embed, load = c.state.current, c.embedder.embed, c.state.load_descriptors
    img = decode_rgb(data)
    q768 = to_working_res(img, c.settings.working_max_h)
    out = {}

    t0 = time.perf_counter()
    q = embed(q768)
    order, _ = snap.originals.index.search(q, len(snap.originals))
    out["embedding"] = (unique_skus(snap.originals.entries, order), time.perf_counter() - t0)

    t0 = time.perf_counter()
    res = pipeline.fabric_search(q768, embed, snap.originals, load, c.search._pool, top_k=10)
    out["sift_original"] = ([r["sku_id"] for r in res], time.perf_counter() - t0)

    t0 = time.perf_counter()
    res = c.search.search(data, "cloth", top_k=10)
    out["full"] = ([r["sku_id"] for r in res], time.perf_counter() - t0)
    return out


def summarize(rows: list[dict], mode: str, kind: str, variant: str) -> dict:
    sel = [r for r in rows if r["kind"] == kind and r["variant"] == variant]
    ranks = [r["modes"][mode]["rank"] for r in sel]
    n = len(ranks)
    out = {"queries": n}
    for k in KS:
        out[f"recall@{k}"] = (
            round(sum(1 for r in ranks if r is not None and r <= k) / n, 4) if n else None
        )
    out["mrr@10"] = (
        round(sum(1 / r for r in ranks if r is not None and r <= 10) / n, 4) if n else None
    )
    out["seconds_p50"] = (
        round(statistics.median(r["modes"][mode]["seconds"] for r in sel), 2) if n else None
    )
    return out


def cpu_name() -> str:
    if platform.system() == "Windows":
        cmd = ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]
        out = subprocess.run(cmd,
                             capture_output=True, text=True, check=False)  # fmt: skip
        return out.stdout.strip() or platform.processor()
    return platform.processor()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument(
        "--photos", type=Path, required=True, help="folder holding the catalog's photos"
    )
    ap.add_argument(
        "--workdir", type=Path, required=True, help="index, cutouts and checkpoints go here"
    )
    ap.add_argument("--out", type=Path, help="results JSON (default: <workdir>/results.json)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-skus", type=int, help="evaluate a random subset of SKUs (dry runs)")
    ap.add_argument("--max-queries", type=int, help="cap the number of held-out queries")
    ap.add_argument(
        "--known-queries", type=int, default=150, help="degraded re-shots of indexed photos"
    )
    ap.add_argument("--index-only", action="store_true", help="build the index, skip queries")
    args = ap.parse_args()

    skus = load_catalog(args.catalog, args.photos)
    queries, gallery = split(skus, args.seed, args.sample_skus)
    if args.max_queries:
        queries = queries[: args.max_queries]
    held_photos = {p for _, p in queries}
    gallery = [g for g in gallery if g[1] not in held_photos]
    n_gallery_skus = len({g[0] for g in gallery})
    print(f"{len(skus)} SKUs with photos | gallery {len(gallery)} photos in {n_gallery_skus} SKUs "
          f"| {len(queries)} held-out queries", flush=True)  # fmt: skip

    work = args.workdir
    work.mkdir(parents=True, exist_ok=True)
    settings = Settings(disable_auth=True, data_root=work / "data", corpus_dir=args.photos,
                        rmbg_dir=work / "rmbg", log_level="WARNING")  # fmt: skip
    c = Container.build(settings)
    c.startup(start_worker=False)
    leaked = [p for _, p in queries if c.indexer.is_indexed(p)]
    if leaked:
        sys.exit(
            f"{len(leaked)} held-out photos are already indexed in {work}; use a fresh --workdir"
        )

    t0, done = time.perf_counter(), 0
    for sku, photo, item in tqdm(gallery, desc="indexing", unit="photo"):
        res = c.indexer.index(photo, sku, "", item, engine="local", reload=False)
        done += res["status"] == "indexed"
    c.state.reload()
    print(
        f"indexed {done} new photos in {time.perf_counter() - t0:.0f}s | {c.state.current.stats()}",
        flush=True,
    )
    if args.index_only:
        return 0

    rng = random.Random(args.seed + 1)
    cloth = [g for g in gallery if c.catalog.resolve(g[2]) == "cloth"]
    known = rng.sample(cloth, min(args.known_queries, len(cloth)))
    specs = [("known", level, sku, photo) for sku, photo, _ in known for level in LEVELS]
    specs += [("heldout", "asis", sku, photo) for sku, photo in queries]
    print(
        f"queries: {len(known)} known x {len(LEVELS)} levels + {len(queries)} held-out "
        f"= {len(specs)}",
        flush=True,
    )

    ckpt = work / "queries.jsonl"
    rows = (
        [json.loads(line) for line in ckpt.read_text(encoding="utf-8").splitlines()]
        if ckpt.exists()
        else []
    )
    finished = {(r["kind"], r["variant"], r["sku"]) for r in rows}
    with ckpt.open("a", encoding="utf-8") as f:
        for kind, variant, sku, photo in tqdm(specs, desc="querying", unit="query"):
            if (kind, variant, sku) in finished:
                continue
            raw = (args.photos / photo).read_bytes()
            data = (
                raw
                if variant == "asis"
                else degrade(
                    decode_rgb(raw),
                    np.random.RandomState([args.seed, sku, LEVELS[variant]]),
                    variant,
                )
            )
            modes = {
                m: {"rank": rank_of(sku, r), "seconds": s}
                for m, (r, s) in rankings(c, data).items()
            }
            row = {"kind": kind, "variant": variant, "sku": sku, "modes": modes}
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()

    groups = sorted({(r["kind"], r["variant"]) for r in rows})
    results = {
        "machine": {
            "cpu": cpu_name(),
            "logical_cpus": __import__("os").cpu_count(),
            "os": platform.platform(terse=True),
        },
        "setup": {
            "model_tag": settings.model_tag,
            "recall_k": pipeline.RECALL_K,
            "seed": args.seed,
            "gallery_photos": len(gallery),
            "gallery_skus": len({g[0] for g in gallery}),
            "known_queries": len(known),
            "heldout_queries": len(queries),
            "working_max_h": settings.working_max_h,
        },  # fmt: skip
        "results": {f"{k}/{v}": {m: summarize(rows, m, k, v) for m in MODES} for k, v in groups},
    }
    out = args.out or work / "results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["results"], indent=2))
    c.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
