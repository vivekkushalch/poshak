from . import catalog, images, ingest, jobs, search, system

routers = [system.router, search.router, ingest.router, catalog.router, jobs.router, images.router]
