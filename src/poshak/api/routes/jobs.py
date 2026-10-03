from fastapi import APIRouter, Depends

from ...domain.errors import NotFound
from ..deps import JobsDep, require_api_key
from ..schemas.jobs import JobDetail, JobList

router = APIRouter(prefix="/index/jobs", tags=["jobs"], dependencies=[Depends(require_api_key)])


@router.get("", response_model=JobList)
def list_jobs(jobs: JobsDep) -> dict:
    return {"jobs": jobs.list()}


@router.get("/{batch_id}", response_model=JobDetail)
def get_job(jobs: JobsDep, batch_id: str) -> dict:
    job = jobs.get(batch_id)
    if job is None:
        raise NotFound("job not found")
    return job
