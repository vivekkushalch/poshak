"""Bake the demo gallery into the image: index the bundled photos, write sample queries.

Runs once while the Docker image builds, through the same services as production. The photos
are in the bundle (demo/photos, named by their position in manifest.json), so the build needs no
network access to Wikimedia. Settings come from the environment the Dockerfile sets.

    python seed_demo.py --demo-dir /home/poshak/demo --samples /home/poshak/demo-samples"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

# Photos whose re-shots an embedding-only search gets wrong and poshak gets right
# (found with docs/demo/build_demo.py --list).
SAMPLES = [91, 82, 89, 33, 37, 24, 80, 21]
MIN_VERIFIED = 4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--demo-dir", type=Path, required=True)
    ap.add_argument("--samples", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.demo_dir))
    from evaluate import degrade  # the bench's own degradation, so samples match the README figures

    from poshak.container import Container
    from poshak.core.config import get_settings
    from poshak.engine.image_ops import decode_rgb

    photos = sorted((args.demo_dir / "photos").glob("*.jpg"))
    if not photos:
        sys.exit("no photos in demo/photos")
    c = Container.build(get_settings())
    c.startup(start_worker=False)
    for i, path in enumerate(photos):
        c.indexer.stage(path.name, path.read_bytes())
        c.indexer.index(path.name, i + 1, "", "Fabric", reload=False)
    c.state.reload()
    stats = c.state.current.stats()
    ranker = c.search.active_ranker("fabric")
    print(f"indexed {stats['fabric']} photos; ranker serving fabric: {ranker}")

    args.samples.mkdir(parents=True, exist_ok=True)
    kept = []
    for idx in SAMPLES:
        if idx >= len(photos):
            continue
        data = degrade(
            decode_rgb(photos[idx].read_bytes()), np.random.RandomState([0, idx + 1, 0]), "mild"
        )
        top = c.search.search(data, "fabric", top_k=1)
        ok = bool(top) and top[0]["sku_id"] == idx + 1
        print(f"sample from photo {idx}: poshak ranks the right photo first: {ok}")
        if ok:
            name = f"sample-{len(kept) + 1:02d}.jpg"
            (args.samples / name).write_bytes(data)
            kept.append({"file": name, "label": f"Sample {len(kept) + 1}", "category": "fabric"})
    if len(kept) < MIN_VERIFIED:
        sys.exit(
            f"only {len(kept)} samples verified; refusing to ship a demo that fails its own samples"
        )
    (args.samples / "index.json").write_text(json.dumps(kept), encoding="utf-8")
    credits = args.demo_dir / "ATTRIBUTION.md"
    if credits.exists():
        shutil.copy(credits, args.samples / "ATTRIBUTION.md")
    c.shutdown()
    print(f"wrote {len(kept)} samples to {args.samples}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
