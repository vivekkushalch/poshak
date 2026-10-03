"""Catalog use cases: item listing and item-type -> pipeline routing."""

from collections.abc import Iterable
from typing import Any

from ..domain.models import Category
from ..domain.ports import CategoryRepository, ImageRepository


def infer_category(item: str) -> Category:
    """Fallback for an unseen item type: names containing "fabric" are flat textiles."""
    return "fabric" if "fabric" in item.strip().lower() else "cloth"


class CatalogService:
    def __init__(
        self, categories: CategoryRepository, images: ImageRepository, model_tag: str
    ) -> None:
        self._categories = categories
        self._images = images
        self._model_tag = model_tag

    def seed_defaults(self, cloth_items: Iterable[str], fabric_items: Iterable[str]) -> None:
        """Idempotent: never overwrites an inferred or manual mapping."""
        for item in cloth_items:
            self._categories.add(item, "cloth", "config")
        for item in fabric_items:
            self._categories.add(item, "fabric", "config")

    def resolve(self, item: str) -> Category:
        """Known mapping, or infer one and remember it so routing stays stable."""
        item = (item or "").strip()
        category = self._categories.get(item)
        if category is None:
            category = infer_category(item)
            self._categories.add(item, category, "inferred")
        return category

    def categories(self) -> list[dict[str, Any]]:
        return self._categories.list()

    def set_category(self, item: str, category: Category) -> None:
        """Affects future ingests of this item; already-indexed photos are untouched."""
        self._categories.set_manual(item, category)

    def items(self, page: int, page_size: int) -> dict[str, Any]:
        total, rows = self._images.page(self._model_tag, page, page_size)
        return {"total": total, "page": page, "page_size": page_size, "items": rows}
