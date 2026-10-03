"""Typed, validated settings loaded from the environment.

Every knob the service reads lives here. Invalid combinations fail at startup
instead of on the first request that needs them."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MODEL_ID = "google/siglip-base-patch16-224"
# Tag stamped on every indexed row; rows with another tag are never loaded, so a
# model or preprocessing change can't silently mix embeddings. Kept at its
# historical value for the default model so existing indexes keep loading.
DEFAULT_MODEL_TAG = "siglip-base-patch16-224-v2-768"

DEFAULT_CLOTH_ITEMS = frozenset(
    {"Sherwani", "Indowestern", "Bandhgala", "Tuxedo", "Jacket", "Bomber Jacket"}
)
DEFAULT_FABRIC_ITEMS = frozenset({"Fabric"})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", protected_namespaces=(), frozen=True)

    # auth
    api_key: str = ""
    disable_auth: bool = False

    # local state
    data_root: Path = Path("/data")
    sqlite_path: Path | None = None
    desc_dir: Path | None = None
    corpus_dir: Path = Path("/corpus/uploads")
    rmbg_dir: Path = Path("/corpus/rmbg")

    # storage backend for originals, cutouts and descriptors
    storage_backend: Literal["local", "s3"] = "local"
    s3_bucket: str = ""
    s3_region: str | None = None
    s3_endpoint_url: str | None = None

    # model + preprocessing
    embed_model_id: str = DEFAULT_MODEL_ID
    working_max_h: int = Field(768, gt=0)

    # background removal at index time (search-time query removal is always local)
    rembg_api_url: str = ""
    rembg_api_timeout: int = Field(360, gt=0)
    rembg_api_retries: int = Field(1, ge=0)
    rembg_default_engine: Literal["auto", "api", "local"] = "auto"
    rembg_api_max_h: int = Field(0, ge=0)

    # item type -> search pipeline seeds (JSON list in env)
    cloth_items: frozenset[str] = DEFAULT_CLOTH_ITEMS
    fabric_items: frozenset[str] = DEFAULT_FABRIC_ITEMS

    # serving
    image_base_url: str = ""
    max_concurrent_searches: int = Field(2, ge=1)
    sift_workers: int = Field(4, ge=1)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @model_validator(mode="after")
    def _derive_and_check(self) -> "Settings":
        # frozen model: fill derived paths through object.__setattr__
        if self.sqlite_path is None:
            object.__setattr__(self, "sqlite_path", self.data_root / "index.sqlite")
        if self.desc_dir is None:
            object.__setattr__(self, "desc_dir", self.data_root / "descriptors")
        if self.storage_backend == "s3" and not self.s3_bucket:
            raise ValueError("STORAGE_BACKEND=s3 requires S3_BUCKET")
        if not self.disable_auth and not self.api_key:
            raise ValueError("API_KEY is required unless DISABLE_AUTH=1")
        return self

    @property
    def model_tag(self) -> str:
        if self.embed_model_id == DEFAULT_MODEL_ID and self.working_max_h == 768:
            return DEFAULT_MODEL_TAG
        return f"{self.embed_model_id.rsplit('/', 1)[-1]}-{self.working_max_h}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
