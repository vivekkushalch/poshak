"""Command line: `poshak serve`, `poshak seed` and `poshak refresh-descriptors`."""

import argparse
import json
import sys
from pathlib import Path

from .core.logging import configure_logging, log


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    # One worker on purpose: the index lives in process memory. Scale out with
    # more containers behind a load balancer, not more workers per container.
    uvicorn.run("poshak.main:app", host=args.host, port=args.port, workers=1)
    return 0


def _photos(sku: dict) -> list[str]:
    names = sku.get("images") or sku.get("images_on_server") or sku.get("images_in_db") or []
    return [n.strip() for n in names if n and n.strip()]


def _seed(args: argparse.Namespace) -> int:
    """Index every photo listed in a catalog JSON whose original is already stored.

    Format: {"skus": [{"sku_id": 1, "designno": "D1", "item": "Sherwani",
                       "images": ["1_a.jpg", "1_b.jpg"]}]}
    ("stock_id" and "images_on_server" are accepted for older exports.)
    Resumable: already-indexed photos are skipped unless --force."""
    from tqdm import tqdm

    from .container import Container
    from .core.config import get_settings
    from .domain.errors import PoshakError

    skus = json.loads(Path(args.json).read_text(encoding="utf-8")).get("skus", [])
    c = Container.build(get_settings())
    c.startup(start_worker=False)
    work = [(sku, photo) for sku in skus for photo in _photos(sku)]
    stats = {"done": 0, "skipped": 0, "missing": 0, "failed": 0}
    try:
        for sku, photo in tqdm(work, unit="photo", desc="indexing"):
            if not c.originals.exists(photo):
                stats["missing"] += 1
                continue
            try:
                res = c.indexer.index(
                    photo, int(sku.get("sku_id") or sku["stock_id"]), sku.get("designno", ""),
                    sku.get("item", ""), force=args.force, reuse_cutout=True, reload=False,
                )  # fmt: skip
                stats["done" if res["status"] == "indexed" else "skipped"] += 1
            except (PoshakError, ValueError) as e:
                stats["failed"] += 1
                log.warning("failed to index %s: %s", photo, e)
    finally:
        c.shutdown()
    print(json.dumps(stats))
    return 0 if stats["failed"] == 0 else 1


def _refresh_descriptors(args: argparse.Namespace) -> int:
    """Rewrite all stored descriptors in the format the current RANKER needs."""
    from .container import Container
    from .core.config import get_settings

    c = Container.build(get_settings())
    c.startup(start_worker=False)
    try:
        print(json.dumps(c.indexer.refresh_descriptors()))
    finally:
        c.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="poshak", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8088)
    serve.set_defaults(func=_serve)

    seed = sub.add_parser("seed", help="index photos listed in a catalog JSON")
    seed.add_argument("--json", required=True, help="catalog JSON path")
    seed.add_argument("--force", action="store_true", help="re-index already indexed photos")
    seed.set_defaults(func=_seed)

    refresh = sub.add_parser(
        "refresh-descriptors",
        help="rewrite stored SIFT descriptors in the current RANKER's format (no re-embedding)",
    )
    refresh.set_defaults(func=_refresh_descriptors)

    args = parser.parse_args(argv)
    configure_logging("INFO")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
