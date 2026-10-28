"""Publish the read-only demo to a Hugging Face Space (a Docker Space on CPU Basic hardware).

Creating a Docker Space needs a paid Hugging Face plan (PRO, Team or Enterprise).

    hf auth login                       # once, or set HF_TOKEN
    uv run python space/deploy.py --repo your-name/poshak

Builds a clean bundle from this checkout: the code, a Dockerfile that bakes a demo gallery into
the image, the gallery's photos with their credits, and the Space's README. Then it creates the
Space if it does not exist and uploads the bundle. The Space builds the image itself, which
takes ten to twenty minutes the first time.

    --dry-run   build the bundle in space/_bundle and stop, so you can inspect it
    --cache     where the Commons photos are cached between runs"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "demo" / "manifest.json"
AGENT = {"User-Agent": "poshak-space-deploy/1.0 (https://github.com/vivekkushalch/poshak)"}
HOME = "/home/poshak"

DEMO_BLOCK = f"""\
# ---- demo: a read-only gallery baked into the image --------------------------------------
COPY demo {HOME}/demo
ENV RANKER=v2 READ_ONLY=1 DISABLE_AUTH=1 \\
    DATA_ROOT={HOME}/demo-data CORPUS_DIR={HOME}/demo-data/uploads \\
    RMBG_DIR={HOME}/demo-data/rmbg SAMPLES_DIR={HOME}/demo-samples
RUN python {HOME}/demo/seed_demo.py --demo-dir {HOME}/demo --samples {HOME}/demo-samples
"""


def download(manifest: list[dict], cache: Path) -> list[Path]:
    import httpx

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


def dockerfile() -> str:
    """The project's own Dockerfile, plus the demo block just before the port is exposed."""
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    marker = "EXPOSE 8088"
    if marker not in text:
        sys.exit("Dockerfile has no 'EXPOSE 8088' line to anchor the demo block on")
    return text.replace(marker, DEMO_BLOCK + marker, 1)


def build_bundle(out: Path, cache: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for name in ("pyproject.toml", "uv.lock", "LICENSE"):
        shutil.copy(ROOT / name, out / name)
    shutil.copytree(ROOT / "src", out / "src", ignore=ignore)
    shutil.copy(ROOT / "space" / "README.md", out / "README.md")
    (out / "Dockerfile").write_text(dockerfile(), encoding="utf-8", newline="\n")

    demo = out / "demo"
    (demo / "photos").mkdir(parents=True)
    shutil.copy(ROOT / "space" / "seed_demo.py", demo / "seed_demo.py")
    shutil.copy(ROOT / "bench" / "evaluate.py", demo / "evaluate.py")
    shutil.copy(ROOT / "docs" / "demo" / "ATTRIBUTION.md", demo / "ATTRIBUTION.md")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for i, photo in enumerate(download(manifest, cache)):
        shutil.copy(photo, demo / "photos" / f"{i:03d}.jpg")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", help="the Space, as owner/name")
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "poshak-demo")
    args = ap.parse_args()
    if not args.dry_run and not args.repo:
        ap.error("--repo is required unless --dry-run")

    out = ROOT / "space" / "_bundle"
    build_bundle(out, args.cache)
    files = [p for p in out.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files) / 1e6
    print(f"bundle: {len(files)} files, {size:.1f} MB in {out}")
    if args.dry_run:
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        user = api.whoami()["name"]
    except Exception:  # noqa: BLE001 - any failure here means "not logged in"
        sys.exit(
            "Not logged in to Hugging Face. Run `hf auth login` or set HF_TOKEN (a write token)."
        )
    print(f"signed in as {user}")
    # Skip create_repo for an existing Space: the create call is what Hugging Face refuses
    # (402) on a free account, while uploading to a Space you already own is not.
    if not api.repo_exists(args.repo, repo_type="space"):
        api.create_repo(args.repo, repo_type="space", space_sdk="docker", private=args.private)
    api.upload_folder(folder_path=str(out), repo_id=args.repo, repo_type="space",
                      commit_message="Deploy the poshak demo")  # fmt: skip
    print(
        f"uploaded. The image builds on Hugging Face now: https://huggingface.co/spaces/{args.repo}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
