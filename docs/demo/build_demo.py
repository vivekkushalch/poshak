"""Build the README's search-result figures from freely licensed photos, with the real pipeline.

    uv run python docs/demo/build_demo.py --out docs/assets/demo

Downloads the photos listed in manifest.json (CC0, public domain or CC BY, from Wikimedia
Commons; credits in ATTRIBUTION.md), indexes them with poshak using the real SigLIP model,
then searches with re-shots of each photo (cropped, tilted, darkened, blurred and
re-compressed, as bench/evaluate.py does). Nothing here is staged: the figures show what
`SearchService.search` returned, next to what an embedding-only search returns.

The query is another rendering of the same photo, not a second photograph of the same object,
and the figures say so. The printed totals count every re-shot, not only the ones shown."""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "bench"))

from evaluate import LEVELS, degrade  # noqa: E402

from poshak.container import Container  # noqa: E402
from poshak.core.config import Settings  # noqa: E402
from poshak.engine import sift, v2  # noqa: E402
from poshak.engine.image_ops import decode_rgb, to_working_res  # noqa: E402

AGENT = {"User-Agent": "poshak-demo-builder/1.0 (https://github.com/vivekkushalch/poshak)"}
GREEN, RED, INK, MUTED = (46, 157, 91), (208, 69, 59), (30, 27, 75), (92, 90, 120)


def download(manifest: list[dict], cache: Path) -> list[Path]:
    cache.mkdir(parents=True, exist_ok=True)
    paths = []
    with httpx.Client(headers=AGENT, timeout=90, follow_redirects=True) as client:
        for i, item in enumerate(manifest):
            path = cache / f"{i:03d}.jpg"
            for attempt in range(5):
                if path.exists():
                    break
                r = client.get(item["url"])
                if r.status_code == 200:
                    path.write_bytes(r.content)
                    time.sleep(0.3)
                else:
                    time.sleep(2 + 3 * attempt)
            if not path.exists():
                sys.exit(f"could not download {item['title']}")
            paths.append(path)
    return paths


def font(size: int, path: str | None) -> ImageFont.FreeTypeFont:
    if path and Path(path).exists():
        return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def tile(img: np.ndarray, size: int, border: tuple | None) -> Image.Image:
    pil = ImageOps.fit(Image.fromarray(img), (size, size), method=Image.LANCZOS)
    if border:
        pad = Image.new("RGB", (size + 12, size + 12), border)
        pad.paste(pil, (6, 6))
        return pad
    pad = Image.new("RGB", (size + 12, size + 12), (255, 255, 255))
    pad.paste(pil, (6, 6))
    return pad


def sift_matches(a: np.ndarray, b: np.ndarray, limit: int = 70):
    """Ratio-test, mutual-nearest-neighbour matches between two images, for drawing."""
    sift = cv2.SIFT_create(nfeatures=1200, contrastThreshold=0.02, edgeThreshold=10)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    bf = cv2.BFMatcher()
    forward = bf.knnMatch(da, db, k=2)
    back = bf.match(db, da)
    good = [
        m[0]
        for i, m in enumerate(forward)
        if len(m) == 2
        and m[0].distance < 0.75 * m[1].distance
        and back[m[0].trainIdx].trainIdx == i
    ]
    good.sort(key=lambda m: m.distance)
    return ka, kb, good, limit


def match_panel(query: np.ndarray, other: np.ndarray, height: int = 380) -> tuple[Image.Image, int]:
    def fit(img):
        scale = height / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), height), interpolation=cv2.INTER_AREA)

    q, o = fit(query), fit(other)
    ka, kb, good, limit = sift_matches(q, o)
    drawn = cv2.drawMatches(q, ka, o, kb, good[:limit], None, matchColor=(46, 157, 91),
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  # fmt: skip
    return Image.fromarray(drawn), len(good)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "poshak-demo")
    ap.add_argument("--font", help="TrueType font for the figure text")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--level", choices=list(LEVELS), default="mild")
    ap.add_argument("--list", action="store_true", help="print the candidate rows and stop")
    ap.add_argument("--pick", type=int, nargs="*", help="gallery indices to show, in order")
    args = ap.parse_args()

    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    paths = download(manifest, args.cache)
    images = [decode_rgb(p.read_bytes()) for p in paths]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        s = Settings(disable_auth=True, data_root=tmp / "data", corpus_dir=tmp / "up",
                     rmbg_dir=tmp / "rmbg", log_level="WARNING", ranker="v2")  # fmt: skip
        c = Container.build(s)
        c.startup(start_worker=False)
        for i, p in enumerate(paths):
            c.indexer.stage(f"{i:03d}.jpg", p.read_bytes())
            c.indexer.index(f"{i:03d}.jpg", i + 1, "", "Fabric", reload=False)
        c.state.reload()
        part = c.state.current.fabric
        print(
            f"indexed {len(part)} photos; ranker serving fabric: {c.search.active_ranker('fabric')}"
        )

        rows = []
        for level in LEVELS:
            for i, img in enumerate(images):
                data = degrade(img, np.random.RandomState([0, i + 1, LEVELS[level]]), level)
                query = decode_rgb(data)
                vec = c.embedder.embed(to_working_res(query))
                order, sims = part.index.search(vec, 3)
                emb_top = [int(part.entries[j].sku_id) - 1 for j in order]
                emb_rank = int(np.argsort(-sims).tolist().index(i)) + 1
                hits = c.search.search(data, "fabric", top_k=3)
                top = [int(h["sku_id"]) - 1 for h in hits]
                rows.append(
                    {
                        "level": level,
                        "i": i,
                        "query": query,
                        "emb_top": emb_top,
                        "emb_sims": [float(sims[j]) for j in order],
                        "emb_rank": emb_rank,
                        "top": top,
                        "matches": [h["total_sift_matches"] for h in hits],
                    }
                )

        n = len(rows)
        emb_ok = sum(r["emb_top"][0] == r["i"] for r in rows)
        pos_ok = sum(r["top"][0] == r["i"] for r in rows)
        print(f"{len(images)} photos, {n} re-shots: embedding-only first {emb_ok}/{n}, "
              f"poshak first {pos_ok}/{n}")  # fmt: skip
        for level in LEVELS:
            sel = [r for r in rows if r["level"] == level]
            print(f"  {level}: embedding {sum(r['emb_top'][0] == r['i'] for r in sel)}/{len(sel)}, "
                  f"poshak {sum(r['top'][0] == r['i'] for r in sel)}/{len(sel)}")  # fmt: skip

        contrast = [r for r in rows if r["level"] == args.level
                    and r["emb_top"][0] != r["i"] and r["top"][0] == r["i"]]  # fmt: skip
        contrast.sort(key=lambda r: -r["emb_rank"])
        if args.list:
            for r in contrast[:30]:
                print(
                    r["i"],
                    "embedding ranked the right photo",
                    r["emb_rank"],
                    "matches",
                    r["matches"],
                )
            return
        chosen = [next(r for r in contrast if r["i"] == i) for i in args.pick] if args.pick \
            else contrast[: args.rows]  # fmt: skip

        args.out.mkdir(parents=True, exist_ok=True)
        f_big, f_med, f_small = (font(s, args.font) for s in (34, 24, 20))
        size, gap = 210, 14
        width = 280 + 3 * (size + 12 + gap) * 2 + 80
        height = 150 + len(chosen) * (size + 90) + 120
        fig = Image.new("RGB", (width, height), (255, 255, 255))
        d = ImageDraw.Draw(fig)
        d.text((30, 22), "Search by photo: embedding only vs poshak", fill=INK, font=f_big)
        note = (
            "Query = a photo of the item, re-shot (cropped, tilted, darker, blurred, compressed)."
        )
        d.text(
            (30, 70), f"{note} Gallery: {len(images)} different photos.", fill=MUTED, font=f_small
        )
        x_emb, x_pos = 300, 300 + 3 * (size + 12 + gap) + 40
        d.text((x_emb, 112), "Embedding only (SigLIP)", fill=INK, font=f_med)
        d.text((x_pos, 112), "poshak", fill=INK, font=f_med)
        for r_i, r in enumerate(chosen):
            y = 150 + r_i * (size + 90)
            q = tile(r["query"], size, None)
            fig.paste(q, (30, y))
            d.text((30, y + size + 18), "Query", fill=MUTED, font=f_small)
            counts = [f"{m} match" + ("" if m == 1 else "es") for m in r["matches"]]
            shown = [
                (r["emb_top"], [f"cos {v:.2f}" for v in r["emb_sims"]], x_emb),
                (r["top"], counts, x_pos),
            ]
            for ids, details, x0 in shown:
                for col, (j, detail) in enumerate(zip(ids, details, strict=True)):
                    right, x = j == r["i"], x0 + col * (size + 12 + gap)
                    fig.paste(tile(images[j], size, GREEN if right else RED), (x, y))
                    colour = GREEN if right else RED
                    d.text((x, y + size + 16), f"#{col + 1}  {detail}", fill=colour, font=f_small)
                    d.text((x, y + size + 42), "same photo" if right else "different",
                           fill=colour, font=f_small)  # fmt: skip
        y = height - 90
        totals = (
            f"Over all {n} re-shots in this gallery: embedding-only ranks the right photo first"
        )
        d.text((30, y), f"{totals} {emb_ok} times, poshak {pos_ok} times.", fill=INK, font=f_small)
        credit = "Photos: Wikimedia Commons (CC0, public domain, CC BY), credits in"
        d.text(
            (30, y + 32),
            f"{credit} docs/demo/ATTRIBUTION.md. Rebuild: docs/demo/build_demo.py.",
            fill=MUTED,
            font=f_small,
        )
        fig.save(args.out / "search-demo.jpg", quality=86, optimize=True)

        # why it works: matched keypoints between the query and the winner versus a lookalike
        r = chosen[0]
        rival = next(j for j in r["emb_top"] if j != r["i"])
        win, _ = match_panel(r["query"], images[r["i"]])
        lose, _ = match_panel(r["query"], images[rival])
        # quote the pipeline's own counts (the lines are redrawn at a smaller size)
        des_q, _ = sift.compute_descriptors_by_response(to_working_res(r["query"]))
        win_n, lose_n = (
            v2.mutual_count(des_q, c.state.load_descriptors(f"{j:03d}.jpg.orig.npz")[0])
            for j in (r["i"], rival)
        )
        w = max(win.width, lose.width, 1100)
        fig2 = Image.new("RGB", (w, win.height + lose.height + 150), (255, 255, 255))
        d2 = ImageDraw.Draw(fig2)
        plural = "" if lose_n == 1 else "es"
        d2.text(
            (30, 16),
            f"Query vs the right photo: {win_n} matches (the strongest 70 are drawn)",
            fill=GREEN,
            font=f_med,
        )
        fig2.paste(win, ((w - win.width) // 2, 60))
        d2.text((30, 76 + win.height),
                f"Query vs the lookalike an embedding ranks first: {lose_n} match{plural}",
                fill=RED, font=f_med)  # fmt: skip
        fig2.paste(lose, ((w - lose.width) // 2, 120 + win.height))
        fig2.save(args.out / "match-lines.jpg", quality=86, optimize=True)
        print("wrote", args.out)
        c.shutdown()


if __name__ == "__main__":
    main()
