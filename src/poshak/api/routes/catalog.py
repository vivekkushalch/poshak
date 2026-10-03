from typing import Annotated

from fastapi import APIRouter, Depends, Form, Query

from ...domain.models import Category
from ..deps import CatalogDep, require_api_key
from ..schemas.catalog import CategoryList, CategoryUpdated, ItemPage

router = APIRouter(tags=["catalog"], dependencies=[Depends(require_api_key)])


@router.get("/skus", response_model=ItemPage)
def list_items(
    catalog: CatalogDep,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict:
    return catalog.items(page, page_size)


@router.get("/categories", response_model=CategoryList)
def list_categories(catalog: CatalogDep) -> dict:
    return {"categories": catalog.categories()}


@router.post("/categories", response_model=CategoryUpdated)
def set_category(
    catalog: CatalogDep,
    item: Annotated[str, Form(min_length=1)],
    category: Annotated[Category, Form()],
) -> CategoryUpdated:
    """Route an item type to a pipeline. Applies to future ingests only."""
    catalog.set_category(item, category)
    return CategoryUpdated(item=item.strip(), category=category)
