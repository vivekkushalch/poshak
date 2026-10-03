import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ...domain.models import BgEngine, ImageKind, JobItem
from ..deps import IndexerDep, JobsDep, require_api_key
from ..schemas.common import PhotoName
from ..schemas.ingest import ForgetResult, IngestAccepted, ReplaceResult

router = APIRouter(tags=["catalog writes"], dependencies=[Depends(require_api_key)])


@router.post("/ingest", status_code=202, response_model=IngestAccepted)
def ingest(
    indexer: IndexerDep,
    jobs: JobsDep,
    image: Annotated[UploadFile, File()],
    sku_id: Annotated[int, Form()],
    item: Annotated[str, Form(description="Item type, e.g. Sherwani; routes the pipeline")],
    photo_name: Annotated[PhotoName, Form()],
    designno: Annotated[str, Form()] = "",
    batch_id: Annotated[str, Form(description="Group uploads; generated when empty")] = "",
    bg_engine: Annotated[BgEngine, Form()] = "auto",
    force: Annotated[bool, Form(description="Re-index even if already indexed")] = False,
) -> IngestAccepted:
    """Store the photo and queue it for indexing. Track progress at /index/jobs/{batch_id}."""
    batch_id = batch_id or uuid.uuid4().hex
    indexer.stage(photo_name, image.file.read())
    jobs.submit(JobItem(batch_id, photo_name, sku_id, designno, item, bg_engine, force))
    return IngestAccepted(batch_id=batch_id, photo_name=photo_name)


@router.post("/replace", response_model=ReplaceResult)
def replace(
    indexer: IndexerDep,
    image: Annotated[UploadFile, File()],
    photo_name: Annotated[PhotoName, Form()],
    kind: Annotated[ImageKind, Form(description="original: re-derive all; rmbg: new cutout")],
) -> ReplaceResult:
    indexer.replace(photo_name, kind, image.file.read())
    return ReplaceResult(photo_name=photo_name, kind=kind)


@router.post("/forget", response_model=ForgetResult)
def forget(indexer: IndexerDep, photo_name: Annotated[PhotoName, Form()]) -> ForgetResult:
    return ForgetResult(photo_name=photo_name, **indexer.forget(photo_name))
