# Deployment

## Single node

One container, state on the host. This is the setup the service is tuned for: a
2 GB VM with no GPU.

```bash
mkdir -p ~/poshak/data ~/poshak/corpus/uploads ~/poshak/corpus/rmbg
cp .env.example .env        # set API_KEY at minimum
docker compose -f compose.prod.yaml pull
docker compose -f compose.prod.yaml up -d
curl localhost:8088/health
```

`compose.prod.yaml` pulls `ghcr.io/vivekkushalch/poshak`, which the release workflow publishes
when a `v*` tag is pushed (see [Releasing](#releasing)). Until a release exists, build the
image yourself with `docker compose up -d --build`.

Put photos in `~/poshak/corpus/uploads` and run `poshak seed` inside the container,
or send them through `POST /ingest`.

The image runs as uid 1000. On a host where your user is not uid 1000, `chown` the
`~/poshak` tree to 1000 first.

Back up by archiving `~/poshak`. It holds the SQLite index, the descriptors, the
photos and the cutouts.

## S3 or S3-compatible storage

```bash
STORAGE_BACKEND=s3
S3_BUCKET=my-catalog
S3_REGION=ap-south-1
# S3_ENDPOINT_URL=http://minio:9000    # MinIO, Cloudflare R2, ...
```

Photos go to `uploads/`, cutouts to `rmbg/` and descriptors to `descriptors/` in the
bucket. If the bucket is public, `IMAGE_BASE_URL` can point straight at it and the
browser pages load images without touching the service.

Credentials come from the usual AWS chain: environment variables, an instance
role, or `~/.aws`. The SQLite index stays on local disk under `DATA_ROOT`.

## Health checks

| Path | Use for | Fails when |
|---|---|---|
| `/health/live` | Liveness, container `HEALTHCHECK` | The process is stuck |
| `/health/ready` | Load balancer readiness | The index is empty |
| `/health` | Humans and dashboards | Never, while the process is up |

## Sizing

The fixed cost is the SigLIP vision tower (the text tower is dropped after load) and
the U2Net session. The index adds 3 KB per indexed view for embeddings, plus the
descriptor cache. Each running search holds the decoded query, its cutout and the
candidates' descriptors, which is why `MAX_CONCURRENT_SEARCHES` defaults to 2 on a
2 GB box. Measured numbers will come with the benchmark.

## Faster search on small catalogs (RANKER=v2)

For catalogs of up to about a thousand photos, set `RANKER=v2` and run the descriptor
refresh once. It does not re-embed anything and takes about 0.2 s per photo:

```bash
poshak refresh-descriptors
```

`/health` then reports `"ranker": "v2"`, which ranker serves each category in
`ranker_active`, and `descriptors_to_refresh` (0 when every file is in the new format).
While a category has more than `V2_MAX_VIEWS` photos (default 1000), its searches use the
legacy ranker automatically. Newly indexed photos are written in the new format. Go back
with `RANKER=legacy`; both formats stay readable. See [ADR 0006](adr/0006-progressive-ranking.md).

## Upgrading from 0.2

- The package moved from `service` to `poshak`. The start command is `poshak serve`.
- The compose service is named `poshak` and the files are `compose.yaml` and
  `compose.prod.yaml`.
- `DEBUG` is replaced by `LOG_LEVEL`. `HOST`, `PORT`, `DATA_DIR`, `CACHE_DIR`,
  `GC_EVERY` and `LEDGER_PATH` are gone.
- `API_KEY` is now required at startup unless `DISABLE_AUTH=1`.
- The existing index keeps working. On first start a migration rewrites stored
  descriptor paths to plain keys, and the model tag is unchanged.

## Releasing

1. Move the `Unreleased` entries in `CHANGELOG.md` under a new `## X.Y.Z` heading and set the
   same version in `pyproject.toml` (and `CITATION.cff`).
2. Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`.

The release workflow refuses a tag that does not match `pyproject.toml` or has no changelog
section, runs the tests, publishes the image to GitHub Container Registry (and to Docker Hub
when the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets exist), and creates the GitHub
release with that changelog section as its notes. The first time, open the package under the
repository's **Packages** and set its visibility to public, or nobody else can pull it.

The `docker` workflow builds the image on every push to `main` (and on pull requests that touch
the Dockerfile, dependencies or source) and runs `tests/smoke/docker_smoke.py` inside it:
ingest, index, search and health with the real models.
