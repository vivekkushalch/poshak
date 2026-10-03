"""App factory. `uvicorn poshak.main:app` serves the default app; tests and embedders
call create_app() with their own settings or a pre-built container."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api.errors import register_exception_handlers
from .api.routes import routers
from .api.routes.system import STATIC
from .container import Container
from .core.config import Settings, get_settings
from .core.logging import configure_logging

DESCRIPTION = (
    "Fine-grained visual search for Indian wear. SigLIP embeddings recall 40 "
    "candidates, SIFT geometric verification ranks them, and cloth searches fuse a "
    "background-removed view with the original."
)


class RevalidatingStatic(StaticFiles):
    """Static assets revalidate on every load (the ETag makes that a cheap 304), so an
    upgrade never leaves an admin looking at the previous version's UI."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        c = container or Container.build(settings or get_settings())
        configure_logging(c.settings.log_level)
        c.startup()
        app.state.container = c
        try:
            yield
        finally:
            c.shutdown()

    app = FastAPI(title="poshak", version=__version__, description=DESCRIPTION, lifespan=lifespan)
    register_exception_handlers(app)
    app.mount("/static", RevalidatingStatic(directory=STATIC), name="static")
    for router in routers:
        app.include_router(router)
    return app


app = create_app()
