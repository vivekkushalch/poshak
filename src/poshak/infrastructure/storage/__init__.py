from ...domain.errors import InvalidKey


def validate_key(key: str) -> str:
    """Keys are flat names inside one namespace; anything that could escape it is rejected."""
    if not key or "/" in key or "\\" in key or ".." in key:
        raise InvalidKey(f"invalid key: {key!r}")
    return key
