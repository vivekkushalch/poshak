"""Deterministic synthetic catalog + queries for the search parity golden test.

Images are random shapes on noise so SIFT finds plenty of distinct keypoints.
The fake embedder and background remover stand in for SigLIP and U2Net so the
test runs in milliseconds and pins only the ranking logic."""

import cv2
import numpy as np


def _image(seed: int, h: int = 400, w: int = 300) -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = rng.randint(90, 140, size=(h, w, 3)).astype(np.uint8)
    for _ in range(40):
        color = tuple(int(c) for c in rng.randint(0, 255, size=3))
        x, y = int(rng.randint(0, w)), int(rng.randint(0, h))
        if rng.rand() < 0.5:
            cv2.rectangle(
                img, (x, y), (x + int(rng.randint(8, 60)), y + int(rng.randint(8, 60))), color, -1
            )
        else:
            cv2.circle(img, (x, y), int(rng.randint(4, 30)), color, -1)
    return img


def fake_embed(img: np.ndarray) -> np.ndarray:
    small = (
        cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (8, 8), interpolation=cv2.INTER_AREA)
        .astype(np.float32)
        .flatten()
    )
    small -= small.mean()
    return small / (np.linalg.norm(small) or 1.0)


def fake_remove_bg(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    bh, bw = h // 10, w // 10
    out[:bh], out[-bh:], out[:, :bw], out[:, -bw:] = 128, 128, 128, 128
    return out


def catalog() -> list[dict]:
    """[{sku_id, category, photo, image}] — 6 cloth SKUs and 4 fabric SKUs, 2 photos each."""
    rows = []
    for sku in range(1, 11):
        category = "cloth" if sku <= 6 else "fabric"
        for k in range(2):
            rows.append(
                {
                    "sku_id": sku,
                    "category": category,
                    "photo": f"{sku}_{k}.jpg",
                    "image": _image(sku * 100 + k),
                }
            )
    return rows


def queries() -> dict[str, tuple[str, np.ndarray]]:
    rng = np.random.RandomState(7)
    base = {r["photo"]: r["image"] for r in catalog()}

    def noisy(img):
        return np.clip(img.astype(np.int16) + rng.randint(-12, 12, img.shape), 0, 255).astype(
            np.uint8
        )

    def crop(img):
        h, w = img.shape[:2]
        return cv2.resize(img[h // 8 :, w // 10 :], (w, h), interpolation=cv2.INTER_AREA)

    return {
        "cloth_noisy": ("cloth", noisy(base["2_0.jpg"])),
        "cloth_crop": ("cloth", crop(base["4_1.jpg"])),
        "cloth_flip": ("cloth", np.ascontiguousarray(base["5_0.jpg"][:, ::-1])),
        "cloth_unrelated": ("cloth", _image(9999)),
        "fabric_noisy": ("fabric", noisy(base["8_1.jpg"])),
        "fabric_crop": ("fabric", crop(base["9_0.jpg"])),
        "fabric_unrelated": ("fabric", _image(8888)),
    }


def summarize(category: str, results: list[dict]) -> list[dict]:
    out = []
    for r in results:
        row = {
            "sku_id": r["sku_id"],
            "total_sift_matches": r["total_sift_matches"],
            "unified_score": round(float(r["unified_score"]), 9),
            "best": (r["best_entry"] or {}).get("json_filename"),
        }
        if category == "cloth":
            row.update(bg_sift=r["bg_sift"], orig_sift=r["orig_sift"], both_found=r["both_found"])
        else:
            row["tier"] = r["tier"]
        out.append(row)
    return out
