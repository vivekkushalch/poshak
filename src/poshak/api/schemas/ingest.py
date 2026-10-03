from typing import Literal

from pydantic import BaseModel


class IngestAccepted(BaseModel):
    batch_id: str
    photo_name: str
    queued: bool = True


class ReplaceResult(BaseModel):
    status: Literal["replaced"] = "replaced"
    photo_name: str
    kind: Literal["original", "rmbg"]


class ForgetResult(BaseModel):
    status: Literal["ok"] = "ok"
    photo_name: str
    files_removed: int
    db_deleted: int
