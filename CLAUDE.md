# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A FastAPI image-search microservice for the **poshak** clothing catalog (Sherwanis,
Indowesterns, Fabrics, etc.). It is a **production port of the prototype notebook
`sharwani_large.ipynb`** — nearly every algorithm module documents the exact prototype
function it mirrors and warns against retuning constants in isolation. When changing
search/scoring/SIFT behavior, treat the notebook as the source of truth and keep parity.

The service is called by a PHP frontend (a local PHP server locally, the web host in prod) via
`EMBED_SERVICE_URL`. PHP handles the catalog DB and file uploads; this service handles
embeddings, indexing, and the search pipeline.

## Commands

Everything runs inside Docker. There is **no test suite** and no linter configured.

```bash
docker compose up -d --build              # build + start (port 8088)
docker compose logs -f                    # watch pipeline tracing (DEBUG=1 by default)
curl http://localhost:8088/health         # model + index status

# One-time / maintenance (run inside the container):
docker compose exec image-search python -m scripts.init_db            # create/migrate schema
docker compose exec image-search python -m scripts.init_db --force    # DROP+CREATE (wipes vectors)
docker compose exec image-search python -m scripts.bg_remove_batch    # write bg-removed PNGs to /corpus/data
docker compose exec image-search python -m scripts.reindex            # embed stock photos → stock_image_vectors
docker compose exec image-search python -m scripts.reindex --limit 20 # smoke test on first 20 SKUs
```

Typical first-run order: `init_db` → `bg_remove_batch` → `reindex` → restart (or `POST /reload`).
`bg_remove_batch` and `reindex` are both **idempotent and resumable** — re-running only
processes new/incomplete rows. `reindex` backfills `rembg_vec` for cloth rows once the
matching PNG exists, so the two scripts can run in either order.

The `service/` and `scripts/` dirs are bind-mounted in dev (see `docker-compose.yml`), so
code edits are picked up on container restart without a rebuild — but a Dockerfile/deps
change requires `--build`.

## Architecture

### Request → result flow
`service/main.py` is the only HTTP surface. The key endpoint is `POST /search`, which:
1. Reads the upload async, then runs the **blocking pipeline in a worker thread** so the
   event loop stays free to answer `/health` during a long search.
2. Routes by `category`: `cloth` → `unified_cloth_search`, `fabric` → `fabric_search`
   (both in `service/search.py`).
3. Always `gc.collect()`s afterward — pipelines hold transient SIFT/image buffers that
   matter under the container memory cap.

`_entry_to_result` in `main.py` deliberately promotes `json_filename` (the original JPEG)
over `filename` for the response, because cloth `filename` is the rembg PNG that only
exists on the AWS side — the browser must request the original from the web host's `uploads/`.

### The search pipeline (`service/search.py`) — the heart of the system
SIFT good-match **count** is the primary ranking signal; SigLIP cosine is only a coarse
filter and a small tiebreaker. Single pass (`search_pipeline_sift_first`):
- **Stage 1:** SigLIP embed → cosine vs the index's precomputed matrix → top-40 (one
  `embeds @ q` dot product; matrices precomputed in `IndexStore.prepare`).
- **Stage 2:** SIFT rerank the 40 candidates **in parallel** across `SIFT_POOL` (cv2
  releases the GIL, so threads give real speedup), then sort by `(sift_matches, score)`.

**Cloth uses a dual pipeline** (`unified_cloth_search`): pipeline A = rembg the query then
search the **bg-removed** index; pipeline B = original query vs the **original** index.
Results are aggregated per-SKU (`aggregate_by_sku`, best image wins) then fused
(`unified_cloth_score`). NOTE: the code comment says A and B run in parallel, but they
currently run **sequentially** to avoid doubling peak RAM (rembg + candidate decode) and
tripping OOM — stage-2 SIFT inside each is still parallel.

`highpass_trace` IoU is computed and returned but is **diagnostic only** — not used in
final ranking (kept for prototype parity).

### Index store (`service/index_store.py`, `service/loader.py`)
`STORE` is a module-level singleton holding three in-memory indexes — `cloth_bg`,
`cloth_orig`, `fabric` — each a list of per-photo dicts plus a stacked embed matrix.
`load_from_mysql` populates it at FastAPI startup (lifespan) and on `POST /reload` from
`stock_image_vectors`, joined to `stock`/`stockitem`. A missing DB at boot is swallowed so
`/embed` still works for the indexer to populate it.

Per-photo dict shape is documented at the top of `index_store.py`. Images and highpass
traces are **lazy** (`None` until `ensure_image`); the decoded image is intentionally NOT
cached (RAM), but the trace is.

### Model (`service/model.py`)
SigLIP (`google/siglip-base-patch16-224`, 768-d, L2-normalized) loaded **once at import**
— per-request loading would cost 10-30s. Torch intra-op threads are halved so parallel
embed calls don't oversubscribe CPU. The model is baked into the Docker image at build time.

### Live add/delete (`service/ingest.py`)
`POST /ingest-photo` saves the file, returns **202 immediately**, then does embed + DB
upsert + in-place `STORE` append on a background thread (`INGEST_POOL`) so the photo is
searchable on the next `/search`. `POST /forget-photo` removes file + DB row + STORE
entries synchronously. STORE mutations are guarded by `_STORE_LOCK` against concurrent
`/search` reads.

### model_tag versioning (critical invariant)
`config.MODEL_TAG` ("siglip-base-patch16-224-v1") is stamped on every
`stock_image_vectors` row. The loader filters by it, so vectors produced under a different
model are **silently ignored** — this is the guard that prevents serving mixed-model
results. **Bump `MODEL_TAG` whenever the model or preprocessing changes**, then re-`init_db`
(or re-index). The 512-d → current migration logic lives in `scripts/init_db.py`.

## Conventions specific to this codebase

- **Prototype parity over cleanliness.** `service/scoring.py`, `sift.py`, `bg_removal.py`,
  `highpass.py`, `image_io.py` are verbatim ports. Constants (SIFT `nfeatures=1200`,
  `contrastThreshold=0.02`; Lowe 0.75; the confidence tiers; the fusion weights) are tuned
  to *this* catalog — do not retune one without re-calibrating the others against the test set.
- **Memory is the dominant constraint** (`mem_limit: 6g`). Many design choices (lazy image
  loading, no decode cache, capped `SIFT_POOL` workers, sequential cloth pipelines,
  halved torch threads, aggressive gc) exist purely to stay under the cap. NOTE: rembg
  input is capped to 768px height (`REMBG_MAX_H` in `bg_removal.py`) to match the
  prototype's `load_image(max_h=768)` — bg removal never runs on full-res originals.
  Comments flag these — respect them before "optimizing."
- **Item routing** is data-driven by `config.CLOTH_ITEMS` / `FABRIC_ITEMS` matched against
  `stockitem.name`. Items in neither set are skipped at index time.
- **Tracing:** wrap timed work in `debug.timed(...)` and log stages with `debug.dbg(...)`;
  everything shows up in `docker compose logs -f`. Set `DEBUG=0` to silence in prod.
- **DB access:** always `with closing(get_connection())` (PyMySQL, `DictCursor`,
  `autocommit=False` — commit explicitly). Connection params come from `DB_*` env vars
  with a local PHP server-friendly defaults.

## Paths & volumes (from docker-compose.yml)
- `/corpus/uploads` ← the web host/a local PHP server `uploads/` (read-only mount) — original photos.
- `/corpus/data` ← `data/` — bg-removed cloth PNGs (written by `bg_remove_batch` / ingest).
- `/models` ← named volume — HF + U2Net weights, persisted across rebuilds.

## Not part of the running service
`spipeline.py`, `sharwani_large.ipynb`, `poshak.sql`, `sku-image-map.json` are
prototype/data artifacts (all in `.dockerignore`). The notebook is the algorithmic
reference; the live indexers read the `stock`/`stockitem` MySQL tables, **not**
`sku-image-map.json`.
