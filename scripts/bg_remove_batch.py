"""Offline background removal for cloth photos.

Reads cloth SKUs from live `stock` (NOT JSON). For each photo:
  - Check if /corpus/data/<base>.png already exists → skip
  - Load original from /corpus/uploads/<photo>
  - Run rembg/U2Net → composite over gray (128,128,128)
  - Save as PNG to /corpus/data/<base>.png

Idempotent: re-runs only process new originals. Run after reindex (or
before — order doesn't matter, since reindex.py backfills rembg_vec
for existing cloth rows once it finds the matching PNG).

Run inside the docker container:
    docker compose exec image-search python -m scripts.bg_remove_batch
    docker compose exec image-search python -m scripts.bg_remove_batch --limit 20
"""

import argparse
import os
from contextlib import closing

import cv2
from tqdm import tqdm

from service import config
from service.bg_removal import remove_bg_rembg
from service.db import get_connection
from service.image_io import find_actual_file, load_image


SELECT_STOCK_SQL = """
    SELECT s.id        AS stock_id,
           s.photos    AS photos,
           si.name     AS item
      FROM stock s
 LEFT JOIN stockitem si ON si.id = s.itemname
     WHERE s.photos IS NOT NULL
       AND s.photos != ''
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N cloth SKUs (smoke test)")
    args = ap.parse_args()

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    with closing(get_connection()) as conn:
        with conn.cursor() as cur:
            cur.execute(SELECT_STOCK_SQL)
            sku_rows = cur.fetchall()

    cloth_rows = [
        r for r in sku_rows
        if (r["item"] or "").strip() in config.CLOTH_ITEMS
    ]
    if args.limit > 0:
        cloth_rows = cloth_rows[: args.limit]

    stats = {
        "cloth_skus_seen": 0,
        "photos_written": 0,
        "photos_skipped_existing": 0,
        "photos_missing_file": 0,
        "photos_failed": 0,
    }

    for sku in tqdm(cloth_rows, desc="rembg"):
        stats["cloth_skus_seen"] += 1
        for raw in (sku["photos"] or "").split(","):
            photo = raw.strip()
            if not photo:
                continue

            base = os.path.splitext(photo)[0]
            out_path = config.DATA_DIR / f"{base}.png"

            if out_path.exists():
                stats["photos_skipped_existing"] += 1
                continue

            orig_path = find_actual_file(config.CORPUS_DIR, photo)
            if orig_path is None:
                stats["photos_missing_file"] += 1
                continue
            img = load_image(orig_path)
            if img is None:
                stats["photos_missing_file"] += 1
                continue

            try:
                bg_removed = remove_bg_rembg(img)
            except Exception as e:
                print(f"  [rembg FAILED] {photo}: {e}")
                stats["photos_failed"] += 1
                continue

            # remove_bg_rembg returns RGB; cv2.imwrite expects BGR.
            bg_bgr = cv2.cvtColor(bg_removed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), bg_bgr)
            stats["photos_written"] += 1

    print("\nDone.")
    for k, v in stats.items():
        print(f"  {k:28s} {v}")
    print(f"\nPNGs in: {config.DATA_DIR}")
    print("Next: docker compose exec image-search python -m scripts.reindex")
    print("      (re-running reindex backfills rembg_vec for cloth rows).")


if __name__ == "__main__":
    main()
