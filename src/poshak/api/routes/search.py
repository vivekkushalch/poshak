from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ...domain.models import Category
from ..deps import ContainerDep, SearchDep, require_api_key
from ..schemas.common import ImageRef
from ..schemas.search import ClothSearchResponse, FabricSearchResponse, SearchResponse

router = APIRouter(tags=["search"], dependencies=[Depends(require_api_key)])


# Plain `def`: FastAPI runs it in a worker thread, so the CPU-heavy pipeline never
# blocks the event loop and /health keeps answering during a search.
@router.post("/search", response_model=SearchResponse)
def search(
    svc: SearchDep,
    c: ContainerDep,
    image: Annotated[UploadFile, File(description="Query photo (JPEG/PNG/WebP)")],
    category: Annotated[Category, Form(description="cloth: dual-view; fabric: single-view")],
    top_k: Annotated[int, Form(ge=1, le=50)] = 5,
) -> ClothSearchResponse | FabricSearchResponse:
    ranked = svc.search(image.file.read(), category, top_k)
    results = [
        {**{k: v for k, v in r.items() if k != "best_entry"},
         "best_entry": ImageRef.from_entry(r["best_entry"])}
        for r in ranked
    ]  # fmt: skip
    response = ClothSearchResponse if category == "cloth" else FabricSearchResponse
    return response(total_found=len(results), results=results, model_tag=c.settings.model_tag)
