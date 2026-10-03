"""API-key check, independent of the web framework."""

import secrets


def api_key_valid(provided: str | None, expected: str) -> bool:
    """Constant-time comparison so response timing doesn't leak the key."""
    if not expected or provided is None:
        return False
    return secrets.compare_digest(provided.encode(), expected.encode())
