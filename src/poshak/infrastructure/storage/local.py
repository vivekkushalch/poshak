"""BlobStore on a local directory. Writes are atomic (temp file + rename)."""

import os
import uuid
from pathlib import Path

from ...domain.errors import NotFound
from . import validate_key


class LocalBlobStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / validate_key(key)

    def get(self, key: str) -> bytes:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError:
            raise NotFound(key) from None

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def delete(self, key: str) -> bool:
        path = self._path(key)
        existed = path.exists()
        path.unlink(missing_ok=True)
        return existed

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()
