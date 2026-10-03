"""Turn the DeepFashion In-shop Clothes Retrieval set into a catalog bench/evaluate.py can read.

    uv run --with pyarrow python bench/prepare_deepfashion.py --out ./deepfashion

Source: the Hugging Face mirror Marqo/deepfashion-inshop (one 216 MB parquet file, images
resized to 224x224). DeepFashion is a research dataset; check its terms before you
publish anything beyond aggregate metrics, and don't redistribute the images.

A SKU is one DeepFashion item id (id_XXXXXXXX). Its photos are that item's front, side,
back and detail shots. A seeded sample keeps the run affordable on a CPU: items with at
least --min-images photos, at most --max-images photos each."""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "Marqo/deepfashion-inshop"
ITEM = re.compile(r"id_(\d{8})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--items", type=int, default=500, help="how many items to sample")
    ap.add_argument("--min-images", type=int, default=3)
    ap.add_argument("--max-images", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = hf_hub_download(REPO, "data/data-00000-of-00001.parquet", repo_type="dataset")
    pf = pq.ParquetFile(path)
    ids = pf.read(columns=["item_ID"]).column("item_ID").to_pylist()

    by_item: dict[int, list[int]] = defaultdict(list)
    for row, name in enumerate(ids):
        m = ITEM.search(name)
        if m:
            by_item[int(m.group(1))].append(row)
    rng = random.Random(args.seed)
    eligible = sorted(i for i, rows in by_item.items() if len(rows) >= args.min_images)
    chosen = sorted(rng.sample(eligible, min(args.items, len(eligible))))
    wanted: dict[int, int] = {}  # row -> item
    for item in chosen:
        for row in rng.sample(by_item[item], min(args.max_images, len(by_item[item]))):
            wanted[row] = item

    images = args.out / "images"
    images.mkdir(parents=True, exist_ok=True)
    photos: dict[int, list[str]] = defaultdict(list)
    kind: dict[int, str] = {}
    row = 0
    for batch in pf.iter_batches(batch_size=2000, columns=["image", "category2"]):
        for image, category in zip(
            batch.column("image").to_pylist(), batch.column("category2").to_pylist(), strict=True
        ):
            if row in wanted:
                item = wanted[row]
                name = f"{item:08d}_{row}.jpg"
                (images / name).write_bytes(image["bytes"])
                photos[item].append(name)
                kind[item] = category
            row += 1

    catalog = {
        "skus": [
            {"sku_id": i, "item": kind[i], "images": sorted(photos[i])} for i in sorted(photos)
        ]
    }
    (args.out / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    n = sum(len(v) for v in photos.values())
    print(
        f"{len(photos)} items, {n} photos -> {args.out} (from {len(by_item)} items in the full set)"
    )


if __name__ == "__main__":
    main()
