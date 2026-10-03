# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repository.

## What this is

poshak is a FastAPI visual search service for Indian wear catalogs. Search is
SigLIP cosine recall (top 40) followed by SIFT geometric verification. Cloth runs
twice (background-removed cutout and original) and fuses per SKU. The algorithm
started as a research notebook that is not in this repo; `engine/` is its port.

## Commands

```bash
uv sync
uv run pytest -m "not slow"          # all non-model tests, about a minute
uv run ruff check && uv run ruff format --check
DISABLE_AUTH=1 DATA_ROOT=./.localdata uv run poshak serve
uv run poshak seed --json catalog.json
docker compose up -d --build
```

On this Windows machine the system temp dir lives on C:, which can run out of
space. If tests fail writing temp files, pass `--basetemp` on another drive.

## Layout and rules

- `api/` routes and Pydantic schemas only. Routes call one service and never
  touch infrastructure or SQL. CPU-heavy routes are plain `def` (threadpool).
- `services/` use cases. They raise `domain.errors`, never `HTTPException`.
- `engine/` pure functions, no I/O. Images, vectors and a descriptor loader come in
  as arguments.
- `domain/` dataclasses, `Protocol` ports, errors. Imports nothing from the project.
- `infrastructure/` implements ports: `storage/` (local, S3), `background/`
  (U2Net, HTTP, fallback), `embedding/` (SigLIP), `vector/` (numpy), `db/` (SQLite).
- `container.py` is the only module that names concrete classes. To add a backend,
  implement the port and add a branch there.
- Tests build the real container with fakes from `tests/fakes.py`. Don't
  monkeypatch module globals.

## Invariants

- `tests/golden/search_golden.json` pins ranked results from the original
  implementation. It runs at the engine, service and API level. Never regenerate
  it to make a refactor pass. Only regenerate when a ranking change is intended,
  and say so in the commit.
- `engine/scoring.py` constants and the SIFT parameters in `engine/sift.py` were
  tuned together. Don't change one alone.
- Working resolution is 768 px by height. Background removal runs on the full-size
  image, then the cutout is downscaled.
- The two cloth passes run sequentially to bound memory (ADR 0003).
- `Settings.model_tag` is stamped on every row and filters what loads. Changing the
  model id or working resolution changes the tag; the default tag must stay
  `siglip-base-patch16-224-v2-768` so existing indexes load.
- Memory target is a 2 GB VM. Respect the comments that explain RAM choices.

## Docs

`docs/architecture.md` (flows, concurrency, scaling path), `docs/deployment.md`,
`docs/adr/` (why the main decisions were made).
