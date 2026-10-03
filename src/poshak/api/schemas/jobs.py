from pydantic import BaseModel


class JobSummary(BaseModel):
    batch_id: str
    status: str
    total: int
    done: int
    failed: int
    skipped: int
    api_count: int
    local_count: int
    created_at: str
    updated_at: str


class JobError(BaseModel):
    photo_name: str
    error: str | None


class JobDetail(JobSummary):
    errors: list[JobError]


class JobList(BaseModel):
    jobs: list[JobSummary]
