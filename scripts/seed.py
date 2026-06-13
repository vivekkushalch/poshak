"""Idempotent, resumable cold-start build. Run inside the container:
    uv run python -m scripts.seed --json /path/sku-image-map.json
Skips any per-item step already done so a crash at item N resumes at N."""

import argparse
import json
import os

import cv2
import numpy as np

from service import config
from service.store import init_db, connect, upsert_image
from service.image_io import to_working_res
from service.descriptors import compute_descriptors, save_descriptors
from service.bg_removal import remove_bg_rembg
from service.model import siglip_embed
from service.s3 import get_bytes as s3_get


def decode_768(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return to_working_res(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def embed(img):            # seam for tests
    return siglip_embed(img)


def compute_desc(img):     # seam for tests
    return compute_descriptors(img)


def rembg_fullsize(img):   # seam for tests
    return remove_bg_rembg(img)


def _categorize(item: str):
    item = (item or "").strip()
    if item in config.CLOTH_ITEMS:
        return "cloth"
    if item in config.FABRIC_ITEMS:
        return "fabric"
    return None


def _already_done(photo: str) -> bool:
    with connect(str(config.SQLITE_PATH)) as c:
        row = c.execute(
            "SELECT orig_desc FROM images WHERE photo_name=? AND model_tag=?",
            (photo, config.MODEL_TAG)).fetchone()
    return bool(row and row[0] and os.path.exists(row[0]))


def run(skus: list) -> dict:
    init_db(str(config.SQLITE_PATH))
    config.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    config.RMBG_DIR.mkdir(parents=True, exist_ok=True)
    config.DESC_DIR.mkdir(parents=True, exist_ok=True)
    stats = {"done": 0, "skipped": 0, "missing": 0}

    for sku in skus:
        category = _categorize(sku.get("item"))
        if category is None:
            continue
        for photo in (sku.get("images_on_server") or sku.get("images_in_db") or []):
            photo = photo.strip()
            if not photo:
                continue
            if _already_done(photo):
                stats["skipped"] += 1
                continue

            local = config.CORPUS_DIR / photo
            if not local.exists():
                data = s3_get(config.S3_UPLOADS_PREFIX + photo)
                if data is None:
                    stats["missing"] += 1
                    continue
                local.write_bytes(data)
            orig768 = decode_768(local.read_bytes())
            orig_vec = embed(orig768)
            des, n = compute_desc(orig768)
            orig_desc = str(config.DESC_DIR / f"{photo}.orig.npz")
            save_descriptors(orig_desc, des, n)

            rembg_name = rembg_vec = rembg_desc = None
            if category == "cloth":
                base = os.path.splitext(photo)[0]
                rembg_name = f"{base}.png"
                rpath = config.RMBG_DIR / rembg_name
                if not rpath.exists():
                    rdata = s3_get(config.S3_RMBG_PREFIX + rembg_name)
                    if rdata is not None:
                        rpath.write_bytes(rdata)
                    else:
                        full = decode_full(local.read_bytes())
                        bg = rembg_fullsize(full)
                        cv2.imwrite(str(rpath), cv2.cvtColor(bg, cv2.COLOR_RGB2BGR))
                rimg768 = to_working_res(_load_rgb(str(rpath)))
                rembg_vec = embed(rimg768)
                rdes, rn = compute_desc(rimg768)
                rembg_desc = str(config.DESC_DIR / f"{photo}.rmbg.npz")
                save_descriptors(rembg_desc, rdes, rn)

            upsert_image(str(config.SQLITE_PATH), sku["stock_id"],
                         sku.get("designno", ""), sku.get("item", ""), category,
                         photo, rembg_name, orig_vec, rembg_vec,
                         orig_desc, rembg_desc, config.MODEL_TAG)
            stats["done"] += 1
    return stats


def decode_full(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    with open(args.json) as f:
        skus = json.load(f).get("skus", [])
    print(run(skus))


if __name__ == "__main__":
    main()
