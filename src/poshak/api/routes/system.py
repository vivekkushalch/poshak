from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

from ..deps import ContainerDep
from ..schemas.system import ClientConfig, HealthResponse

STATIC = Path(__file__).resolve().parents[2] / "static"

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(c: ContainerDep) -> dict:
    """Status summary. Always 200 while the process is up (used by Docker HEALTHCHECK)."""
    snap = c.state.current
    return {
        "model_tag": c.settings.model_tag,
        "embed_dim": c.embedder.dim,
        "index": {**snap.stats(), "model_tag": c.settings.model_tag},
        "index_ready": snap.ready,
        "ranker": c.settings.ranker,
        "ranker_active": {k: c.search.active_ranker(k) for k in ("cloth", "fabric")},
        "descriptors_to_refresh": (
            snap.descriptors_ordered[1] - snap.descriptors_ordered[0]
            if c.settings.ranker == "v2"
            else None
        ),
    }


@router.get("/health/live")
def live() -> dict:
    """Liveness: the process answers. Touches nothing else."""
    return {"status": "ok"}


@router.get("/health/ready")
def ready(c: ContainerDep) -> JSONResponse:
    """Readiness: 503 until the index has entries to search."""
    ok = c.state.current.ready
    return JSONResponse({"ready": ok}, status_code=200 if ok else 503)


@router.get("/config", response_model=ClientConfig)
def client_config(c: ContainerDep) -> ClientConfig:
    """What the browser pages need: where public images are hosted."""
    return ClientConfig(image_base_url=c.settings.image_base_url)


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def ui_page() -> HTMLResponse:
    return HTMLResponse((STATIC / "ui.html").read_text(encoding="utf-8"))


@router.get("/admin", response_class=HTMLResponse, include_in_schema=False)
def admin_page() -> HTMLResponse:
    return HTMLResponse((STATIC / "admin.html").read_text(encoding="utf-8"))
