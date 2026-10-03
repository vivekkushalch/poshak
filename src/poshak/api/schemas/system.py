from typing import Literal

from pydantic import BaseModel


class IndexStats(BaseModel):
    cloth_bg: int
    cloth_orig: int
    fabric: int
    model_tag: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_tag: str
    embed_dim: int
    index: IndexStats
    index_ready: bool


class ClientConfig(BaseModel):
    image_base_url: str
    uploads_prefix: str = "uploads"
    rmbg_prefix: str = "rmbg"
