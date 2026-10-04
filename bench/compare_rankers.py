"""End-to-end comparison of the legacy and v2 rankers through the real services.

    uv run python bench/compare_rankers.py --work ./work --photos ./photos --lab ./lab

Uses the queries bench/lab.py prepared (same images, test half by default) and an index
built by bench/evaluate.py. Both rankers run in the real container with the real models;
what is measured is the wall time of `SearchService.search`, as an API request would see it.

The v2 ranker needs response-ordered descriptors. This script copies the index's data
directory (SQLite and descriptors) to a scratch folder and runs `refresh_descriptors` on the
copy, so your index is left untouched; photos and cutouts are read from the originals."""

import argparse
import json
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from poshak.container import Container
from poshak.core.config import Settings


def cpu_name() -> str:
    if platform.system() == "Windows":
        cmd = ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    return platform.processor()


def rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        return None


def pick(queries: list[dict], per_group: int, seed: int) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for q in queries:
        groups.setdefault((q["kind"], q["level"], q["category"]), []).append(q)
    rng = random.Random(seed)
    return [
        q
        for key in sorted(groups)
        for q in rng.sample(groups[key], min(per_group, len(groups[key])))
    ]


def run(c: Container, queries: list[dict], lab: Path) -> dict[str, tuple[int | None, float]]:
    warm = queries[0]
    c.search.search((lab / "q" / f"{warm['qid']}.jpg").read_bytes(), warm["category"], top_k=10)
    out = {}
    for q in queries:
        data = (lab / "q" / f"{q['qid']}.jpg").read_bytes()
        t0 = time.perf_counter()
        results = c.search.search(data, q["category"], top_k=10)
        seconds = time.perf_counter() - t0
        skus = [r["sku_id"] for r in results]
        out[q["qid"]] = (skus.index(q["sku"]) + 1 if q["sku"] in skus else None, seconds)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, required=True, help="index workdir from evaluate.py")
    ap.add_argument("--photos", type=Path, required=True)
    ap.add_argument("--lab", type=Path, required=True, help="lab dir from bench/lab.py prep")
    ap.add_argument("--split", choices=("tuning", "test"), default="test")
    ap.add_argument("--per-group", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    queries = [q for q in json.loads((args.lab / "queries.json").read_text("utf-8"))
               if q["half"] == args.split]  # fmt: skip
    queries = pick(queries, args.per_group, args.seed)
    print(f"{len(queries)} queries, split={args.split}", flush=True)

    def container(ranker: str, data: Path) -> Container:
        s = Settings(disable_auth=True, data_root=data, corpus_dir=args.photos,
                     rmbg_dir=args.work / "rmbg", log_level="WARNING", ranker=ranker)  # fmt: skip
        c = Container.build(s)
        c.startup(start_worker=False)
        return c

    results, memory = {}, {}
    legacy = container("legacy", args.work / "data")
    results["legacy"] = run(legacy, queries, args.lab)
    memory["legacy"] = rss_mb()
    legacy.shutdown()
    del legacy

    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data"
        shutil.copytree(args.work / "data", data)
        v2 = container("v2", data)
        print("refreshed descriptors:", v2.indexer.refresh_descriptors(), flush=True)
        results["v2"] = run(v2, queries, args.lab)
        memory["v2"] = rss_mb()
        v2.shutdown()

    summary = {
        "machine": cpu_name(),
        "split": args.split,
        "per_group": args.per_group,
        "groups": {},
    }
    for key in sorted({(q["kind"], q["level"], q["category"]) for q in queries}):
        members = [q for q in queries if (q["kind"], q["level"], q["category"]) == key]
        row = {}
        for name in ("legacy", "v2"):
            ranks = [results[name][q["qid"]][0] for q in members]
            secs = sorted(results[name][q["qid"]][1] for q in members)
            row[name] = {
                "r1": sum(r == 1 for r in ranks) / len(members),
                "p50_s": round(statistics.median(secs), 2),
                "p95_s": round(secs[min(len(secs) - 1, int(0.95 * len(secs)))], 2),
            }
        old = [results["legacy"][q["qid"]][0] == 1 for q in members]
        new = [results["v2"][q["qid"]][0] == 1 for q in members]
        row["n"] = len(members)
        row["gained"] = sum(b and not a for a, b in zip(old, new, strict=True))
        row["lost"] = sum(a and not b for a, b in zip(old, new, strict=True))
        summary["groups"]["/".join(key)] = row
    summary["peak_rss_mb"] = memory
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
