"""Stage-by-stage latency of a poshak search, with the real models.

    uv run python bench/latency.py            # writes bench/results.json
    uv run python bench/charts.py             # renders docs/assets/bench-*.svg

Images are synthetic (random shapes on noise), which is fine for timing: cost is
set by image size, the SIFT keypoint cap and the model, not by what is pictured.
Accuracy needs real photos and is measured separately."""

import json
import os
import platform
import statistics
import subprocess
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from poshak.container import Container
from poshak.core.config import Settings
from poshak.engine import sift
from poshak.engine.image_ops import decode_rgb, encode_png, to_working_res
from poshak.infrastructure.background.u2net import U2NetRemover
from poshak.infrastructure.embedding.siglip import SiglipEmbedder
from poshak.infrastructure.vector.numpy_index import NumpyIndex

OUT = Path(__file__).with_name("results.json")
PHOTO_HW = (1600, 1200)  # a typical catalog photo before the 768 px working resize
CATALOG_SIZES = [1_000, 10_000, 100_000, 1_000_000]
RECALL_K = 40
SIFT_THREADS = 4


def photo(seed: int, h: int = PHOTO_HW[0], w: int = PHOTO_HW[1]) -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = rng.randint(90, 140, size=(h, w, 3)).astype(np.uint8)
    for _ in range(160):
        color = tuple(int(c) for c in rng.randint(0, 255, size=3))
        x, y = int(rng.randint(0, w)), int(rng.randint(0, h))
        if rng.rand() < 0.5:
            dx, dy = int(rng.randint(15, 160)), int(rng.randint(15, 160))
            cv2.rectangle(img, (x, y), (x + dx, y + dy), color, -1)
        else:
            cv2.circle(img, (x, y), int(rng.randint(8, 80)), color, -1)
    return img


def timeit(fn: Callable[[], object], repeats: int, warmup: int = 1) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return {
        "p50_ms": round(statistics.median(samples), 1),
        "p95_ms": round(samples[min(len(samples) - 1, int(0.95 * len(samples)))], 1),
        "n": repeats,
    }


def cpu_name() -> str:
    if platform.system() == "Windows":
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
        return out.stdout.strip() or platform.processor()
    return platform.processor()


def stages(embedder: SiglipEmbedder, remover: U2NetRemover) -> dict:
    query = photo(1)
    jpeg = cv2.imencode(".jpg", cv2.cvtColor(query, cv2.COLOR_RGB2BGR))[1].tobytes()
    q768 = to_working_res(query)
    candidates = [sift.compute_descriptors(to_working_res(photo(100 + i))) for i in range(RECALL_K)]
    n_kp, des_q = sift.detect(q768)

    def verify(pool):
        def one(c):
            return sift.match_descriptors(des_q, n_kp, c[0], c[1])

        return list(pool.map(one, candidates)) if pool else [one(c) for c in candidates]

    with ThreadPoolExecutor(SIFT_THREADS) as pool:
        return {
            "decode_resize": timeit(lambda: to_working_res(decode_rgb(jpeg)), 10),
            "background_removal": timeit(lambda: remover.remove(query), 5),
            "embed": timeit(lambda: embedder.embed(q768), 10),
            "sift_detect": timeit(lambda: sift.detect(q768), 10),
            "verify_40_sequential": timeit(lambda: verify(None), 5),
            "verify_40_threads": timeit(lambda: verify(pool), 5),
            "query_keypoints": n_kp,
        }


def recall_scaling(dim: int) -> dict:
    rng = np.random.default_rng(0)
    q = rng.standard_normal(dim).astype(np.float32)
    q /= np.linalg.norm(q)
    out = {}
    for n in CATALOG_SIZES:
        m = rng.standard_normal((n, dim), dtype=np.float32)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        index = NumpyIndex(m)
        out[str(n)] = timeit(lambda index=index: index.search(q, RECALL_K), 10)
        del m, index
    return out


def end_to_end(embedder: SiglipEmbedder, remover: U2NetRemover, n_skus: int = 20) -> dict:
    """Index a small catalog through the real services, then search it."""
    with tempfile.TemporaryDirectory(dir=os.environ.get("BENCH_TMP")) as tmp:
        s = Settings(disable_auth=True, data_root=Path(tmp),
                     corpus_dir=Path(tmp) / "uploads", rmbg_dir=Path(tmp) / "rmbg")  # fmt: skip
        c = Container.build(s, embedder=embedder, index_remover=remover, query_remover=remover)
        c.startup(start_worker=False)
        ingest = {"cloth": [], "fabric": []}
        for sku in range(n_skus):
            for category, item in (("cloth", "Sherwani"), ("fabric", "Fabric")):
                name = f"{category}_{sku}.png"
                c.indexer.stage(name, encode_png(photo(1000 * (category == "fabric") + sku)))
                t0 = time.perf_counter()
                c.indexer.index(name, sku, "", item, engine="local", reload=False)
                ingest[category].append((time.perf_counter() - t0) * 1000)
        c.state.reload()
        queries = [encode_png(photo(sku)) for sku in range(5)]
        result = {
            "catalog_photos": c.state.current.stats(),
            "ingest_cloth_p50_ms": round(statistics.median(ingest["cloth"]), 1),
            "ingest_fabric_p50_ms": round(statistics.median(ingest["fabric"]), 1),
            "search_cloth": timeit(lambda: [c.search.search(q, "cloth") for q in queries], 3),
            "search_fabric": timeit(lambda: [c.search.search(q, "fabric") for q in queries], 3),
        }
        for key in ("search_cloth", "search_fabric"):  # timeit ran 5 queries per sample
            result[key] = {k: round(v / len(queries), 1) if k != "n" else v * len(queries)
                           for k, v in result[key].items()}  # fmt: skip
        c.shutdown()
    return result


def main() -> None:
    embedder = SiglipEmbedder("google/siglip-base-patch16-224")
    remover = U2NetRemover()
    t0 = time.perf_counter()
    results = {
        "machine": {
            "cpu": cpu_name(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
            "os": platform.platform(terse=True),
        },
        "photo_px": list(PHOTO_HW),
        "stages": stages(embedder, remover),
        "recall_vs_catalog": recall_scaling(embedder.dim),
        "end_to_end": end_to_end(embedder, remover),
    }
    results["wall_s"] = round(time.perf_counter() - t0, 1)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
