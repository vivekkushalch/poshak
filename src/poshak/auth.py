"""API-key auth. All data routes depend on require_api_key. DISABLE_AUTH=1
bypasses (local dev). /health, /config, /ui, /admin, /img/* are NOT guarded."""

from fastapi import Header, HTTPException

from . import config


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if config.DISABLE_AUTH:
        return
    if not config.API_KEY or x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
