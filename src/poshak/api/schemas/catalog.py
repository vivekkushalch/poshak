from typing import Literal

from pydantic import BaseModel

from ...domain.models import Category


class ItemRow(BaseModel):
    sku_id: int
    item: str | None
    designno: str | None
    photo_name: str
    rembg_name: str | None


class ItemPage(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[ItemRow]


class CategoryMapping(BaseModel):
    item: str
    category: Category
    source: Literal["config", "inferred", "manual"]


class CategoryList(BaseModel):
    categories: list[CategoryMapping]


class CategoryUpdated(BaseModel):
    status: Literal["ok"] = "ok"
    item: str
    category: Category
