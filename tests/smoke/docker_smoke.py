"""Smoke test for a running poshak with the real models: ingest, index, search, health.

    python tests/smoke/docker_smoke.py http://127.0.0.1:8088

The docker workflow runs it inside the freshly built image (`docker exec -i ... python - URL`),
so it needs nothing beyond what the image already has. Authentication must be off. It uses
fabric photos: that pipeline needs SigLIP and SIFT but not background removal, which keeps
the check fast and free of model downloads at run time."""

import sys
import time

import cv2
import httpx
import numpy as np

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8088"
BATCH = "smoke"


def photo(seed: int, h: int = 360, w: int = 480) -> np.ndarray:
    """Random shapes on noise: plenty of distinct keypoints, different for every seed."""
    rng = np.random.RandomState(seed)
    img = rng.randint(90, 140, size=(h, w, 3)).astype(np.uint8)
    for _ in range(60):
        color = tuple(int(c) for c in rng.randint(0, 255, size=3))
        x, y = int(rng.randint(0, w)), int(rng.randint(0, h))
        if rng.rand() < 0.5:
            cv2.rectangle(
                img, (x, y), (x + int(rng.randint(10, 70)), y + int(rng.randint(10, 70))), color, -1
            )
        else:
            cv2.circle(img, (x, y), int(rng.randint(5, 35)), color, -1)
    return img


def jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


def main() -> None:
    with httpx.Client(base_url=BASE, timeout=120) as c:
        assert c.get("/health/live").json() == {"status": "ok"}

        for sku in (1, 2, 3):
            name = f"smoke_{sku}.jpg"
            r = c.post(
                "/ingest",
                data={"sku_id": sku, "item": "Fabric", "photo_name": name, "batch_id": BATCH},
                files={"image": (name, jpeg(photo(sku)), "image/jpeg")},
            )
            assert r.status_code == 202, r.text

        deadline = time.time() + 300
        while (job := c.get(f"/index/jobs/{BATCH}").json())["status"] != "done":
            assert time.time() < deadline, f"indexing did not finish: {job}"
            time.sleep(2)
        assert job["done"] == 3 and job["failed"] == 0, job

        health = c.get("/health").json()
        assert health["index_ready"] and health["index"]["fabric"] == 3, health
        assert c.get("/health/ready").status_code == 200

        # a re-shot of SKU 2: cropped and recompressed
        query = photo(2)[20:-20, 30:-30]
        r = c.post(
            "/search",
            data={"category": "fabric", "top_k": 3},
            files={"image": ("q.jpg", jpeg(query, 60), "image/jpeg")},
        )
        assert r.status_code == 200, r.text
        results = r.json()["results"]
        assert results[0]["sku_id"] == 2, results

        for path in ("/docs", "/admin", "/ui", "/static/admin/app.js", "/static/favicon.svg"):
            assert c.get(path).status_code == 200, path
    print("smoke test passed")


if __name__ == "__main__":
    main()
