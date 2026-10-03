"""Composition root: the only module that names concrete infrastructure classes.

Container.build() turns Settings into a fully wired object graph. Tests and
alternative deployments pass overrides (an in-memory blob store, a fake
embedder) instead of monkeypatching module globals."""

from dataclasses import dataclass

from .core.config import Settings
from .core.logging import log
from .domain.ports import BackgroundRemover, BlobStore, Embedder
from .infrastructure.background.fallback import FallbackRemover
from .infrastructure.background.http import HttpRemover
from .infrastructure.background.u2net import U2NetRemover
from .infrastructure.db.sqlite import (
    SqliteCategoryRepository,
    SqliteDatabase,
    SqliteImageRepository,
    SqliteJobRepository,
)
from .infrastructure.embedding.siglip import SiglipEmbedder
from .infrastructure.storage.local import LocalBlobStore
from .infrastructure.vector.numpy_index import NumpyIndex
from .services.catalog import CatalogService
from .services.index_state import IndexState
from .services.indexer import IndexerService
from .services.jobs import JobService
from .services.search import SearchService


def _blob_stores(s: Settings) -> tuple[BlobStore, BlobStore, BlobStore]:
    """(originals, cutouts, descriptors). S3 prefixes mirror IMAGE_BASE_URL's layout."""
    if s.storage_backend == "s3":
        import boto3

        from .infrastructure.storage.s3 import S3BlobStore

        client = boto3.client("s3", region_name=s.s3_region, endpoint_url=s.s3_endpoint_url)
        return tuple(
            S3BlobStore(s.s3_bucket, prefix, client=client)
            for prefix in ("uploads/", "rmbg/", "descriptors/")
        )
    return LocalBlobStore(s.corpus_dir), LocalBlobStore(s.rmbg_dir), LocalBlobStore(s.desc_dir)


def _index_remover(s: Settings, local: U2NetRemover) -> BackgroundRemover:
    remote = (
        HttpRemover(s.rembg_api_url, s.rembg_api_timeout, s.rembg_api_retries, s.rembg_api_max_h)
        if s.rembg_api_url
        else None
    )
    return FallbackRemover(remote, local, s.rembg_default_engine)


@dataclass
class Container:
    settings: Settings
    db: SqliteDatabase
    originals: BlobStore
    cutouts: BlobStore
    descriptors: BlobStore
    state: IndexState
    catalog: CatalogService
    indexer: IndexerService
    search: SearchService
    jobs: JobService

    @classmethod
    def build(
        cls,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        index_remover: BackgroundRemover | None = None,
        query_remover: BackgroundRemover | None = None,
        blobs: tuple[BlobStore, BlobStore, BlobStore] | None = None,
    ) -> "Container":
        s = settings
        db = SqliteDatabase(s.sqlite_path)
        images = SqliteImageRepository(db)
        originals, cutouts, descriptors = blobs or _blob_stores(s)
        embedder = embedder or SiglipEmbedder(s.embed_model_id)
        local = U2NetRemover()
        state = IndexState(images, descriptors, s.model_tag, NumpyIndex.from_rows)
        catalog = CatalogService(SqliteCategoryRepository(db), images, s.model_tag)
        indexer = IndexerService(
            images=images, catalog=catalog, originals=originals, cutouts=cutouts,
            descriptors=descriptors, embedder=embedder,
            remover=index_remover or _index_remover(s, local), state=state,
            model_tag=s.model_tag, working_max_h=s.working_max_h,
        )  # fmt: skip
        search = SearchService(
            state, embedder, query_remover or local, working_max_h=s.working_max_h,
            max_concurrent=s.max_concurrent_searches, sift_workers=s.sift_workers,
        )  # fmt: skip
        jobs = JobService(indexer, SqliteJobRepository(db), state)
        return cls(s, db, originals, cutouts, descriptors, state, catalog, indexer, search, jobs)

    def startup(self, *, start_worker: bool = True) -> None:
        self.db.init()
        self.catalog.seed_defaults(self.settings.cloth_items, self.settings.fabric_items)
        self.state.reload()
        if start_worker:
            self.jobs.start()
            resumed = self.jobs.resume()
            if resumed:
                log.info("resumed %d unfinished ingest items", resumed)

    def shutdown(self) -> None:
        self.jobs.stop()
        self.search.close()
