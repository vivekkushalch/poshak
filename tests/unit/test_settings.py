from pathlib import Path

import pytest
from pydantic import ValidationError

from poshak.core.config import DEFAULT_MODEL_TAG, Settings
from poshak.core.security import api_key_valid


def test_reads_env_and_derives_paths(monkeypatch):
    monkeypatch.setenv("API_KEY", "k")
    monkeypatch.setenv("DATA_ROOT", "/srv/poshak")
    monkeypatch.setenv("CLOTH_ITEMS", '["Kurta", "Sherwani"]')
    s = Settings()
    assert s.sqlite_path == Path("/srv/poshak/index.sqlite")
    assert s.desc_dir == Path("/srv/poshak/descriptors")
    assert s.cloth_items == frozenset({"Kurta", "Sherwani"})
    assert s.rembg_api_url == "" and s.storage_backend == "local"


def test_default_model_tag_is_stable():
    assert Settings(disable_auth=True).model_tag == DEFAULT_MODEL_TAG


def test_other_model_gets_its_own_tag():
    s = Settings(disable_auth=True, embed_model_id="google/siglip2-base-patch16-224")
    assert s.model_tag == "siglip2-base-patch16-224-768"


def test_auth_requires_key():
    with pytest.raises(ValidationError, match="API_KEY is required"):
        Settings(disable_auth=False, api_key="")


def test_s3_requires_bucket():
    with pytest.raises(ValidationError, match="S3_BUCKET"):
        Settings(disable_auth=True, storage_backend="s3")


def test_rejects_unknown_engine():
    with pytest.raises(ValidationError):
        Settings(disable_auth=True, rembg_default_engine="gpu")


@pytest.mark.parametrize(
    ("provided", "expected", "ok"),
    [("abc", "abc", True), ("abd", "abc", False), (None, "abc", False), ("", "", False)],
)
def test_api_key_valid(provided, expected, ok):
    assert api_key_valid(provided, expected) is ok
