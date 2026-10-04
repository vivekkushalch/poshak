from typing import Literal

from pydantic import BaseModel, Field


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
    ranker: Literal["legacy", "v2"] = "legacy"
    ranker_active: dict[str, str] = Field(
        default_factory=dict,
        description="Ranker serving each category now; v2 falls back to legacy on large catalogs",
    )
    descriptors_to_refresh: int | None = Field(
        None, description="v2 only: descriptor files `poshak refresh-descriptors` should rewrite"
    )


class ClientConfig(BaseModel):
    image_base_url: str
    uploads_prefix: str = "uploads"
    rmbg_prefix: str = "rmbg"
