"""Render bench/results.json into README charts (light and dark SVG variants).

    uv run python bench/charts.py

Colors are the validated two-slot categorical palette (blue, orange), stepped
separately for the light and dark surfaces."""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).with_name("results.json")
ASSETS = ROOT / "docs" / "assets"

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "s1": "#2a78d6", "s2": "#eb6834"},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#3a3a37",
             "s1": "#3987e5", "s2": "#d95926"},
}  # fmt: skip
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
W = 760
TITLE_SCALING = "Recall grows with the catalog, verification does not"


def _fmt_ms(v: float) -> str:
    return f"{v / 1000:.2f} s" if v >= 1000 else f"{v:.0f} ms" if v >= 10 else f"{v:.1f} ms"


def _svg(h: int, t: dict, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" '
        f'viewBox="0 0 {W} {h}" role="img" font-family="{FONT}">'
        f"<title>{title}</title>"
        f'<rect width="{W}" height="{h}" rx="8" fill="{t["surface"]}"/>{body}</svg>'
    )


def _text(x, y, s, fill, size=13, anchor="start", weight=400) -> str:
    return (f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" dominant-baseline="middle">{s}</text>')  # fmt: skip


def _bar(x, y, w, h, fill, r=4) -> str:
    """Bar anchored at x with a rounded data end on the right only."""
    r = min(r, w / 2, h / 2)
    return (f'<path d="M{x},{y} H{x + w - r} Q{x + w},{y} {x + w},{y + r} V{y + h - r} '
            f'Q{x + w},{y + h} {x + w - r},{y + h} H{x} Z" fill="{fill}"/>')  # fmt: skip


def stages_chart(r: dict, t: dict) -> str:
    st, recall = r["stages"], r["recall_vs_catalog"]["100000"]["p50_ms"]
    rows = [
        ("Decode + resize to 768 px", st["decode_resize"]["p50_ms"]),
        ("Background removal (U2Net)", st["background_removal"]["p50_ms"]),
        ("SigLIP embed, 2 views", 2 * st["embed"]["p50_ms"]),
        ("SIFT keypoints, 2 views", 2 * st["sift_detect"]["p50_ms"]),
        ("Recall top 40 of 100k, 2 views", 2 * recall),
        ("SIFT verify 2 x 40, 4 threads", 2 * st["verify_40_threads"]["p50_ms"]),
    ]
    label_w, x0, top, bar_h, gap = 250, 270, 70, 20, 14
    plot_w = W - x0 - 90
    vmax = max(v for _, v in rows)
    body = [
        _text(24, 28, "Where the time goes in one cloth search", t["text"], 16, weight=600),
        _text(24, 50, f"Median per stage, 1600x1200 photo, {r['machine']['cpu']}", t["muted"], 12),
    ]
    for i, (label, v) in enumerate(rows):
        y = top + i * (bar_h + gap)
        w = max(2.0, plot_w * v / vmax)
        body += [
            _text(label_w, y + bar_h / 2, label, t["text"], 13, anchor="end"),
            _bar(x0, y, w, bar_h, t["s1"]),
            _text(x0 + w + 8, y + bar_h / 2, _fmt_ms(v), t["muted"], 12),
        ]
    h = top + len(rows) * (bar_h + gap) + 10
    return _svg(h, t, "".join(body), "Median latency of each stage of a cloth search")


def scaling_chart(r: dict, t: dict) -> str:
    sizes = [int(n) for n in r["recall_vs_catalog"]]
    recall = [r["recall_vs_catalog"][str(n)]["p50_ms"] for n in sizes]
    verify = r["stages"]["verify_40_threads"]["p50_ms"]
    x0, x1, y0, y1 = 70, W - 170, 270, 104  # plot box (y grows down)
    ymax = max(max(recall), verify) * 1.15
    lmin, lmax = math.log10(sizes[0]), math.log10(sizes[-1])

    def px(n):
        return x0 + (math.log10(n) - lmin) / (lmax - lmin) * (x1 - x0)

    def py(v):
        return y0 - v / ymax * (y0 - y1)

    body = [
        _text(24, 28, TITLE_SCALING, t["text"], 16, weight=600),
        _text(24, 50, f"One pass, median, {r['machine']['cpu']}", t["muted"], 12),
    ]
    step = 10 ** math.floor(math.log10(ymax / 4))
    tick = math.ceil(ymax / 4 / step) * step
    v = 0.0
    while v <= ymax:
        label = _fmt_ms(v) if v else "0"
        body += [f'<line x1="{x0}" x2="{x1}" y1="{py(v)}" y2="{py(v)}" stroke="{t["grid"]}"/>',
                 _text(x0 - 10, py(v), label, t["muted"], 11, "end")]  # fmt: skip
        v += tick
    for n in sizes:
        label = f"{n // 1_000_000}M" if n >= 1_000_000 else f"{n // 1000}k"
        body.append(_text(px(n), y0 + 18, label, t["muted"], 11, "middle"))
    body.append(_text((x0 + x1) / 2, y0 + 38, "photos in the index (log scale)", t["muted"], 11,
                      "middle"))  # fmt: skip

    # verification: flat reference line
    body.append(f'<line x1="{x0}" x2="{x1}" y1="{py(verify)}" y2="{py(verify)}" '
                f'stroke="{t["s2"]}" stroke-width="2" stroke-dasharray="6 4"/>')  # fmt: skip
    # end labels: keep at least 16 px apart so close values stay readable
    ly_verify, ly_recall = py(verify), py(recall[-1])
    if abs(ly_verify - ly_recall) < 16:
        mid = (ly_verify + ly_recall) / 2
        up, down = mid - 8, mid + 8
        ly_verify, ly_recall = (up, down) if ly_verify <= ly_recall else (down, up)
    body.append(_text(x1 + 10, ly_verify, f"verify 40: {_fmt_ms(verify)}", t["text"], 12))
    # recall: line + markers with a surface ring
    pts = " ".join(f"{px(n)},{py(v)}" for n, v in zip(sizes, recall, strict=True))
    body.append(f'<polyline points="{pts}" fill="none" stroke="{t["s1"]}" stroke-width="2"/>')
    for n, v in zip(sizes, recall, strict=True):
        body.append(f'<circle cx="{px(n)}" cy="{py(v)}" r="4.5" fill="{t["s1"]}" '
                    f'stroke="{t["surface"]}" stroke-width="2"/>')  # fmt: skip
    body.append(_text(x1 + 10, ly_recall, f"recall: {_fmt_ms(recall[-1])}", t["text"], 12))
    # legend
    lx = x0
    for name, color, dash in (("Recall (cosine over all photos)", t["s1"], ""),
                              ("Verify (SIFT on 40 candidates)", t["s2"], "6 4")):  # fmt: skip
        body.append(f'<line x1="{lx}" x2="{lx + 22}" y1="{y1 - 26}" y2="{y1 - 26}" '
                    f'stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>')  # fmt: skip
        body.append(_text(lx + 30, y1 - 26, name, t["text"], 12))
        lx += 250
    title = "Recall latency against catalog size, with fixed verify cost"
    return _svg(y0 + 56, t, "".join(body), title)


def main() -> None:
    r = json.loads(RESULTS.read_text(encoding="utf-8"))
    ASSETS.mkdir(parents=True, exist_ok=True)
    for mode, t in THEMES.items():
        (ASSETS / f"bench-stages-{mode}.svg").write_text(stages_chart(r, t), encoding="utf-8")
        (ASSETS / f"bench-scaling-{mode}.svg").write_text(scaling_chart(r, t), encoding="utf-8")
    print(f"wrote charts to {ASSETS}")


if __name__ == "__main__":
    main()
