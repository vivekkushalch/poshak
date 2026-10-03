"""FastAPI dependency providers. Routes ask for services; the container supplies them."""

from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from ..container import Container
from ..core.security import api_key_valid
from ..services.catalog import CatalogService
from ..services.indexer import IndexerService
from ..services.jobs import JobService
from ..services.search import SearchService


def get_container(request: Request) -> Container:
    return request.app.state.container


ContainerDep = Annotated[Container, Depends(get_container)]


def require_api_key(
    container: ContainerDep, x_api_key: Annotated[str | None, Header()] = None
) -> None:
    s = container.settings
    if not s.disable_auth and not api_key_valid(x_api_key, s.api_key):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def _search(c: ContainerDep) -> SearchService:
    return c.search


def _indexer(c: ContainerDep) -> IndexerService:
    return c.indexer


def _jobs(c: ContainerDep) -> JobService:
    return c.jobs


def _catalog(c: ContainerDep) -> CatalogService:
    return c.catalog


SearchDep = Annotated[SearchService, Depends(_search)]
IndexerDep = Annotated[IndexerService, Depends(_indexer)]
JobsDep = Annotated[JobService, Depends(_jobs)]
CatalogDep = Annotated[CatalogService, Depends(_catalog)]
