from typing import Annotated, Literal

from pydantic import BaseModel, Field

from .common import ImageRef

Tier = Literal["HIGH", "MED", "LOW", "WEAK", "NONE"]


class _Result(BaseModel):
    sku_id: int
    unified_score: float
    total_sift_matches: int = Field(description="Primary ranking signal")
    best_entry: ImageRef


class ClothResult(_Result):
    bg_sift: int
    orig_sift: int
    bg_tier: Tier
    orig_tier: Tier
    bg_score: float
    orig_score: float
    both_found: bool = Field(description="Matched in both the cutout and original views")


class FabricResult(_Result):
    tier: Tier


class _Response(BaseModel):
    status: Literal["success"] = "success"
    method: Literal["SIFT"] = "SIFT"
    total_found: int
    model_tag: str


class ClothSearchResponse(_Response):
    category: Literal["cloth"] = "cloth"
    results: list[ClothResult]


class FabricSearchResponse(_Response):
    category: Literal["fabric"] = "fabric"
    results: list[FabricResult]


SearchResponse = Annotated[
    ClothSearchResponse | FabricSearchResponse, Field(discriminator="category")
]
