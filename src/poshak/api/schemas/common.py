"""Shared API types."""

import re
from typing import Annotated

from pydantic import AfterValidator, BaseModel, StringConstraints

from ...domain.models import IndexEntry

_UNSAFE = re.compile(r"[/\\]|\.\.")


def _safe_name(v: str) -> str:
    if _UNSAFE.search(v):
        raise ValueError("must be a plain file name (no '/', '' or '..')")
    return v


PhotoName = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
    AfterValidator(_safe_name),
]  # fmt: skip


class ImageRef(BaseModel):
    """The photo a result points at. Always the original photo, which is what
    clients can display (cutouts may not be published)."""

    sku_id: int
    designno: str
    item: str
    filename: str
    json_filename: str

    @classmethod
    def from_entry(cls, e: IndexEntry) -> "ImageRef":
        return cls(
            sku_id=e.sku_id, designno=e.designno, item=e.item,
            filename=e.json_filename, json_filename=e.json_filename,
        )  # fmt: skip


class Status(BaseModel):
    status: str
