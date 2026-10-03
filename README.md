<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/banner-dark.svg">
  <img alt="poshak: visual search for Indian wear. SigLIP sees the silhouette, SIFT sees the zari." src="docs/assets/brand/banner-light.svg" width="100%">
</picture>

**Finds the exact sherwani, not just one that looks like it.**

[![CI](https://github.com/vivekkushalch/poshak/actions/workflows/ci.yml/badge.svg)](https://github.com/vivekkushalch/poshak/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab.svg)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

[Quickstart](#quickstart) · [How it works](#how-it-works) · [Benchmarks](#benchmarks) · [Architecture](docs/architecture.md) · [API](#api) · [Deploy](docs/deployment.md)

</div>

<br>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/pipeline-dark.svg">
  <img alt="Search pipeline: embed, recall 40 by cosine, verify with SIFT, fuse two views per SKU" src="docs/assets/pipeline-light.svg">
</picture>

Upload a photo of a sherwani, an indowestern or a fabric swatch. poshak returns the
catalog SKUs it matches, ranked by how many local features line up.

Image embeddings like CLIP and SigLIP are good at "a long embroidered jacket".
They are bad at "*that* long embroidered jacket". In an Indian wear catalog,
hundreds of pieces share a cut and a colour and differ only in their zari,
butis and borders. poshak uses an embedding to shortlist and SIFT to decide.

## Highlights

- **Two-stage retrieval.** SigLIP cosine recalls 40 candidates. Lowe-ratio SIFT
  matching ranks them. Query cost is set by those 40, not by catalog size.
- **Dual-view matching for garments.** Each search runs on the photo and on a
  background-removed cutout, then fuses per SKU, so a busy backdrop can't hide a match.
- **Small-box friendly.** Built for a 2 GB VM: descriptors are precomputed at
  ingest, images are processed at 768 px, concurrent searches are capped.
- **Drop-in backends.** Storage, embedder, background removal and the vector index
  sit behind `Protocol` ports. Local disk and S3 (or MinIO, R2) ship today.
- **Production basics.** Async ingest with a resumable job queue, typed Pydantic
  API with OpenAPI docs, liveness and readiness probes, settings validated at boot.
- **Refactor-proof ranking.** A golden test pins ranked results from the original
  implementation at the engine, service and HTTP layers.

## Quickstart

```bash
git clone https://github.com/vivekkushalch/poshak && cd poshak
docker compose up -d --build
curl localhost:8088/health
```

Index a few photos, then search with a new shot of one of them:

```bash
curl -F image=@swatch.jpg -F photo_name=fabric_202_1.jpg -F sku_id=202 \
     -F designno=FB-202 -F item=Fabric localhost:8088/ingest
curl -F image=@query.jpg -F category=fabric -F top_k=3 localhost:8088/search
```

<details>
<summary>Example response (real output, trimmed to two results)</summary>

```json
{
  "status": "success",
  "method": "SIFT",
  "category": "fabric",
  "total_found": 3,
  "model_tag": "siglip-base-patch16-224-v2-768",
  "results": [
    {
      "sku_id": 202,
      "unified_score": 1.0495180487632751,
      "total_sift_matches": 379,
      "tier": "HIGH",
      "best_entry": {
        "sku_id": 202,
        "designno": "FB-202",
        "item": "Fabric",
        "filename": "fabric_202_1.jpg",
        "json_filename": "fabric_202_1.jpg"
      }
    },
    {
      "sku_id": 203,
      "unified_score": 1.0492178201675415,
      "total_sift_matches": 110,
      "tier": "HIGH",
      "best_entry": {
        "sku_id": 203,
        "designno": "FB-203",
        "item": "Fabric",
        "filename": "fabric_203_1.jpg",
        "json_filename": "fabric_203_1.jpg"
      }
    }
  ]
}
```

The query was a re-cropped, re-compressed photo of SKU 202. Ranking is by
`total_sift_matches` first: 379 against 110 for the runner-up, even though the
confidence scores are close.

</details>

Without Docker, using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
DISABLE_AUTH=1 DATA_ROOT=./.localdata CORPUS_DIR=./.localcorpus/uploads \
  RMBG_DIR=./.localcorpus/rmbg uv run poshak serve
```

Then open `/ui` for a phone-friendly search page, `/admin` to browse and fix
indexed photos, or `/docs` for the OpenAPI reference.

To index a catalog that is already in storage:

```bash
uv run poshak seed --json catalog.json   # resumable; --force re-indexes
```

## How it works

| Stage | What happens | Cost |
|---|---|---|
| Ingest | Photo is stored, background removed (cloth), resized to 768 px, embedded with SigLIP, SIFT descriptors computed and stored | Once per photo |
| Recall | Query embedding against every indexed view, top 40 by cosine | Grows with catalog size, one matrix product |
| Verify | Query SIFT keypoints matched to each candidate's stored descriptors with Lowe's 0.75 ratio test | Fixed: 40 candidates, run on a thread pool |
| Score | Good-match count sets a confidence tier, cosine adds a small tiebreak | |
| Fuse (cloth) | Cutout pass and original pass merged per SKU; SKUs found by both rank higher | |

Results sort by total SIFT matches, then by the fused score. The constants were
tuned together on the original catalog; [ADR 0001](docs/adr/0001-sift-verification.md)
explains why ranking works this way.

## Benchmarks

Measured with the real models on a 2017 laptop CPU (Intel i5-7200U, 2 cores and 4
threads, no GPU), 1600×1200 photos.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/bench-stages-dark.svg">
  <img alt="Median latency per stage of a cloth search" src="docs/assets/bench-stages-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/bench-scaling-dark.svg">
  <img alt="Recall latency against catalog size, with fixed verification cost" src="docs/assets/bench-scaling-light.svg">
</picture>

| | p50 | p95 |
|---|---|---|
| Cloth search, end to end | 4.89 s | 5.13 s |
| Fabric search, end to end | 1.39 s | 1.50 s |
| Index one cloth photo (cutout, 2 embeddings, 2 descriptor sets) | 5.22 s | |
| Index one fabric photo | 0.97 s | |
| Recall top 40 of 100k / 1M photos | 22 ms / 384 ms | 25 ms / 771 ms |
| Verify 40 candidates, sequential / 4 threads | 530 ms / 412 ms | 542 ms / 418 ms |

What the numbers say:

- Background removal on the query is 56% of a cloth search. It is the first thing
  to move to a GPU or a smaller segmentation model.
- Recall is negligible up to 100k photos. At a million it matches the verification
  cost on this CPU, which is the point to switch the `VectorIndex` to ANN.
- Threaded verification gains 1.3× on two physical cores, and should scale with
  cores on server hardware.

These are latency numbers on synthetic images. Retrieval accuracy needs a labelled
dataset and is the next milestone on the [roadmap](#roadmap). Reproduce with:

```bash
uv run python bench/latency.py && uv run python bench/charts.py
```

## Architecture

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/arch/components-dark.svg">
  <img alt="Component diagram: driving adapters, application core, driven adapters" src="docs/assets/arch/components-light.svg">
</picture>

Ports and adapters around a pure engine. `container.py` is the one place that picks
implementations, so adding a backend is a new class and one line of wiring. A test
enforces the layer rules on every CI run.
The [architecture document](docs/architecture.md) covers system context,
containers, request and ingest sequences, the data model, ranking formulas, the
latency and memory budgets, failure modes, security and the scaling path.
Decisions and their trade-offs are recorded as [ADRs](docs/adr).

<details>
<summary>Project layout</summary>

```
src/poshak/
├── api/              routes and Pydantic schemas
├── services/         search, indexer, jobs, catalog, live index state
├── engine/           recall, SIFT verification, scoring, fusion (no I/O)
├── domain/           models, Protocol ports, errors
├── infrastructure/
│   ├── storage/      local disk, S3
│   ├── background/   U2Net, HTTP, fallback chain
│   ├── embedding/    SigLIP
│   ├── vector/       numpy exact search
│   └── db/           SQLite repositories + migrations
├── core/             settings, logging, API-key check
├── container.py      composition root
├── main.py           create_app()
└── cli.py            poshak serve | seed
```

</details>

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/search` | Rank SKUs for a query photo (`category`: `cloth` or `fabric`) |
| `POST` | `/ingest` | Store a photo and queue it for indexing; returns `202` and a `batch_id` |
| `GET` | `/index/jobs/{batch_id}` | Batch progress and per-photo errors |
| `POST` | `/replace` | Swap a photo or its cutout and re-index |
| `POST` | `/forget` | Delete a photo, its cutout, descriptors and index row |
| `GET` | `/skus` | Page through indexed photos |
| `GET` `POST` | `/categories` | Inspect or override which pipeline an item type uses |
| `GET` | `/img/original/{name}` · `/img/rmbg/{name}` | Serve stored images |
| `GET` | `/health` · `/health/live` · `/health/ready` | Status and probes |

Data routes require an `X-API-Key` header. Full schemas at `/docs`.

<details>
<summary>Configuration</summary>

Settings come from environment variables and are validated at startup, so a bad
combination such as `STORAGE_BACKEND=s3` without `S3_BUCKET` fails before the
first request.

| Variable | Default | Notes |
|---|---|---|
| `API_KEY` | | Required unless `DISABLE_AUTH=1` |
| `STORAGE_BACKEND` | `local` | `local` or `s3` |
| `S3_BUCKET` · `S3_REGION` · `S3_ENDPOINT_URL` | | Endpoint URL for MinIO or R2 |
| `DATA_ROOT` | `/data` | SQLite index and descriptors |
| `CORPUS_DIR` · `RMBG_DIR` | `/corpus/uploads` · `/corpus/rmbg` | Local photo storage |
| `EMBED_MODEL_ID` | `google/siglip-base-patch16-224` | Changing it starts a new index tag |
| `REMBG_API_URL` | | Optional remote background removal at index time |
| `MAX_CONCURRENT_SEARCHES` | `2` | Extra searches wait 10 s, then get `503` |
| `SIFT_WORKERS` | `4` | Threads for candidate verification |
| `LOG_LEVEL` | `INFO` | `DEBUG` logs per-stage timings |

Everything else is in [.env.example](.env.example).

</details>

## Development

```bash
uv sync
uv run pytest -m "not slow"    # 141 tests, about a minute, no model downloads
uv run ruff check && uv run ruff format --check
```

Tests run the real container with in-memory fakes for the models and storage, so
they cover the same wiring as production. Storage backends share one contract
suite, and S3 runs against [moto](https://github.com/getmoto/moto). See
[CONTRIBUTING.md](CONTRIBUTING.md) before changing anything that affects ranking.

## Roadmap

- [x] Two-stage retrieval with dual-view fusion
- [x] Pluggable storage with S3 support
- [x] Async, resumable ingest
- [ ] Accuracy benchmark on a public fashion dataset (recall@k, MRR, ablations)
- [ ] ANN vector index for catalogs past a few hundred thousand photos
- [ ] Postgres repositories and a shared queue for multi-node ingest
- [ ] MCP server so agents can search a catalog
- [ ] Learned local features (LightGlue) as an alternative verifier

## Acknowledgements

poshak stands on [SigLIP](https://arxiv.org/abs/2303.15343) (Zhai et al., 2023),
[SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) (Lowe, 2004),
[U²-Net](https://arxiv.org/abs/2005.09007) through [rembg](https://github.com/danielgatis/rembg),
[OpenCV](https://opencv.org), [FastAPI](https://fastapi.tiangolo.com) and
[Hugging Face Transformers](https://github.com/huggingface/transformers).

## Citation

```bibtex
@software{poshak,
  author  = {vivekkushalch},
  title   = {poshak: fine-grained visual search for Indian wear},
  url     = {https://github.com/vivekkushalch/poshak},
  version = {0.3.0},
  year    = {2026}
}
```

## License

Code is [MIT](LICENSE). The logo and brand assets are described in
[docs/brand.md](docs/brand.md); the wordmark uses Sora under the SIL Open Font License.
