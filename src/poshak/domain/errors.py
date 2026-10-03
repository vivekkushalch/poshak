"""Domain errors. Services raise these; the API layer maps them to HTTP status codes."""


class PoshakError(Exception):
    """Base for every error the service raises on purpose."""


class NotFound(PoshakError):
    pass


class InvalidImage(PoshakError):
    pass


class InvalidKey(PoshakError):
    """A blob key that could escape its namespace (path traversal) or is empty."""


class IndexNotReady(PoshakError):
    pass


class Overloaded(PoshakError):
    """Too many concurrent searches; the caller should retry shortly."""
