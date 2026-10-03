# Changelog

## Unreleased

- Admin UI rebuilt as a tabbed app: Catalog (photo and cutout side by side, filter,
  page size, replace and delete with an in-page confirmation), Upload (drag and drop,
  batch fields, live indexing progress), Jobs (progress bars, per-photo errors,
  auto-refresh while running) and Categories (view and override item routing).
  Header shows index health, light and dark themes, works at phone width, and asks for
  the API key in a dialog instead of a browser prompt.
- Images fall back to the service's own `/img/...` routes when `IMAGE_BASE_URL` is empty.
- Static assets are served with `Cache-Control: no-cache`, so an upgrade never leaves a
  stale UI in the browser.
- "Not found" errors now name the file (`ghost.jpg not found`).
- Brand kit: zari lozenge mark, outlined Sora wordmark, light and dark lockups,
  favicon for `/ui` and `/admin`, GitHub social card. See `docs/brand.md`.
- Architecture document rewritten as a full design doc: C4 context, container and
  component diagrams, sequences, data model, ranking formulas, latency and memory
  budgets, failure modes, security, scaling, alternatives.
- `bench/latency.py` and `bench/charts.py`: stage latency with the real models,
  rendered into the README.
- Layer rules from the architecture doc are enforced by a test.

## 0.3.0

Restructured into a layered, installable package. Search results are unchanged:
a golden test recorded from 0.2 passes against the new engine, services and API.

### Added

- `STORAGE_BACKEND=s3` stores photos, cutouts and descriptors in S3 or any
  S3-compatible service (`S3_BUCKET`, `S3_REGION`, `S3_ENDPOINT_URL`).
- `poshak` command with `serve` and `seed`.
- `/health/live` and `/health/ready` probes.
- Typed request validation and response models; `/docs` documents both the
  cloth and fabric result shapes.
- `MAX_CONCURRENT_SEARCHES`: extra searches wait up to 10 s, then get 503 with
  `Retry-After`.
- `SIFT_WORKERS`: candidate verification runs on a thread pool.
- `EMBED_MODEL_ID` and `LOG_LEVEL` settings; `CLOTH_ITEMS` / `FABRIC_ITEMS` as JSON lists.
- CI (ruff + pytest on every push). Docker images publish on `v*` tags.

### Changed

- `/search` no longer blocks the event loop while it runs.
- Invalid form values return 422 (was 400 for a bad `category`, `kind` or `bg_engine`).
- `/replace` on an unknown photo returns 404 (was 200 with `"status": "not_found"`).
- The index reload swaps a fully built snapshot in one step.
- Settings are validated at startup. `API_KEY` is required unless `DISABLE_AUTH=1`.
- The package is `poshak` (was `service`); compose files are `compose.yaml` and
  `compose.prod.yaml`, service name `poshak`. The image runs as uid 1000.
- Descriptor paths in SQLite become plain keys; a startup migration converts them.

### Removed

- Settings `DEBUG` (use `LOG_LEVEL`), `HOST`, `PORT`, `DATA_DIR`, `CACHE_DIR`,
  `GC_EVERY`, `LEDGER_PATH`.
- The append-only S3 sync ledger, replaced by the S3 storage backend.
- `scripts/seed.py` (use `poshak seed`), `requirements.txt`, the MySQL schema.
