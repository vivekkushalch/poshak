"""CLI reindexer.

Reads the live `stock` + `stockitem` tables (NOT sku-image-map.json),
embeds each photo with the current MODEL_TAG, writes to
`stock_image_vectors`. Resumable — re-running skips photos already
indexed under the same model_tag.

For cloth rows: if a bg-removed PNG exists in /corpus/data, its embed is
written to rembg_vec; otherwise rembg_vec stays NULL and gets backfilled
later (after Phase 3 bg-removal).

Run inside the docker container:
    docker compose exec image-search python -m scripts.reindex
    docker compose exec image-search python -m scripts.reindex --limit 20  # dry-ish smoke
"""

import argparse
import json
import os
from contextlib import closing

from tqdm import tqdm

from service import config
from service.db import get_connection
from service.image_io import find_actual_file, load_image
from service.model import siglip_embed


SELECT_STOCK_SQL = """
    SELECT s.id        AS stock_id,
           s.photos    AS photos,
           si.name     AS item
      FROM stock s
 LEFT JOIN stockitem si ON si.id = s.itemname
     WHERE s.photos IS NOT NULL
       AND s.photos != ''
"""

SELECT_EXISTING_SQL = """
    SELECT photo_name, rembg_vec IS NOT NULL AS has_rembg
      FROM stock_image_vectors
     WHERE model_tag = %s
"""

INSERT_SQL = """
INSERT INTO stock_image_vectors
    (stock_id, category, photo_name, rembg_name,
     orig_vec, rembg_vec, model_tag)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE id = id
"""

UPDATE_REMBG_SQL = """
UPDATE stock_image_vectors
   SET rembg_name = %s, rembg_vec = %s
 WHERE photo_name = %s AND model_tag = %s
"""


def categorize(item: str) -> str | None:
    item = (item or "").strip()
    if item in config.CLOTH_ITEMS:
        return "cloth"
    if item in config.FABRIC_ITEMS:
        return "fabric"
    return None


def find_rembg_file(photo_name: str):
    """Look for <base>.png in DATA_DIR (rembg always outputs PNG)."""
    base = os.path.splitext(photo_name)[0]
    cand = config.DATA_DIR / f"{base}.png"
    return str(cand) if cand.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N SKUs (for smoke tests)")
    args = ap.parse_args()

    with closing(get_connection()) as conn:
        with conn.cursor() as cur:
            cur.execute(SELECT_STOCK_SQL)
            sku_rows = cur.fetchall()
            cur.execute(SELECT_EXISTING_SQL, (config.MODEL_TAG,))
            existing = {r["photo_name"]: bool(r["has_rembg"]) for r in cur.fetchall()}

        if args.limit > 0:
            sku_rows = sku_rows[: args.limit]

        stats = {
            "skus_seen": 0, "skus_routed": 0, "skus_skipped_item": 0,
            "photos_inserted": 0, "photos_rembg_added": 0,
            "photos_skipped_existing": 0, "photos_missing_file": 0,
            "photos_failed": 0,
        }

        for sku in tqdm(sku_rows, desc="reindex"):
            stats["skus_seen"] += 1
            category = categorize(sku["item"])
            if category is None:
                stats["skus_skipped_item"] += 1
                continue
            stats["skus_routed"] += 1

            for raw in (sku["photos"] or "").split(","):
                photo = raw.strip()
                if not photo:
                    continue

                existed = photo in existing
                has_rembg = existing.get(photo, False)

                # Quick exit: row already complete (cloth with rembg or fabric).
                if existed and (category == "fabric" or has_rembg):
                    stats["photos_skipped_existing"] += 1
                    continue

                rembg_path = (
                    find_rembg_file(photo) if category == "cloth" else None
                )

                # Existing cloth row missing rembg, and the file now exists:
                # backfill the rembg columns.
                if existed and category == "cloth" and rembg_path:
                    try:
                        img = load_image(rembg_path)
                        if img is None:
                            stats["photos_missing_file"] += 1
                            continue
                        rembg_vec = siglip_embed(img).tolist()
                    except Exception as e:
                        print(f"  [embed FAILED] {photo} (rembg): {e}")
                        stats["photos_failed"] += 1
                        continue
                    with conn.cursor() as cur:
                        cur.execute(
                            UPDATE_REMBG_SQL,
                            (
                                os.path.basename(rembg_path),
                                json.dumps(rembg_vec),
                                photo,
                                config.MODEL_TAG,
                            ),
                        )
                    conn.commit()
                    stats["photos_rembg_added"] += 1
                    existing[photo] = True
                    continue

                if existed:
                    stats["photos_skipped_existing"] += 1
                    continue

                # New row.
                orig_path = find_actual_file(config.CORPUS_DIR, photo)
                if orig_path is None:
                    stats["photos_missing_file"] += 1
                    continue
                try:
                    orig_img = load_image(orig_path)
                    if orig_img is None:
                        stats["photos_missing_file"] += 1
                        continue
                    orig_vec = siglip_embed(orig_img).tolist()
                except Exception as e:
                    print(f"  [embed FAILED] {photo} (orig): {e}")
                    stats["photos_failed"] += 1
                    continue

                rembg_name = None
                rembg_vec_json = None
                if rembg_path:
                    try:
                        rembg_img = load_image(rembg_path)
                        if rembg_img is not None:
                            rembg_vec = siglip_embed(rembg_img).tolist()
                            rembg_name = os.path.basename(rembg_path)
                            rembg_vec_json = json.dumps(rembg_vec)
                    except Exception as e:
                        print(f"  [embed FAILED] {photo} (rembg): {e}")

                with conn.cursor() as cur:
                    cur.execute(
                        INSERT_SQL,
                        (
                            sku["stock_id"], category, photo, rembg_name,
                            json.dumps(orig_vec), rembg_vec_json,
                            config.MODEL_TAG,
                        ),
                    )
                conn.commit()
                existing[photo] = rembg_name is not None
                stats["photos_inserted"] += 1
                if rembg_name:
                    stats["photos_rembg_added"] += 1

    print("\nDone.")
    for k, v in stats.items():
        print(f"  {k:28s} {v}")
    print(f"\nmodel_tag = {config.MODEL_TAG}")
    print("Restart the service (or POST /reload) to pick up new vectors.")


if __name__ == "__main__":
    main()
