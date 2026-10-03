"""In-memory adapters for tests. They satisfy the domain ports, so services and the
API are exercised exactly as in production, minus the heavy models and disks."""

from poshak.container import Container
from poshak.core.config import Settings
from poshak.domain.errors import NotFound
from poshak.engine.image_ops import encode_png
from poshak.infrastructure.storage import validate_key
from tests.golden.fixtures import fake_embed, fake_remove_bg


class MemoryBlobStore:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def get(self, key: str) -> bytes:
        try:
            return self.data[validate_key(key)]
        except KeyError:
            raise NotFound(key) from None

    def put(self, key: str, data: bytes) -> None:
        self.data[validate_key(key)] = data

    def delete(self, key: str) -> bool:
        return self.data.pop(validate_key(key), None) is not None

    def exists(self, key: str) -> bool:
        return validate_key(key) in self.data


class FakeEmbedder:
    dim = 64

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, img):
        self.calls += 1
        return fake_embed(img)


class FakeRemover:
    def __init__(self, engine: str = "local", fail: bool = False) -> None:
        self.engine, self.fail, self.calls = engine, fail, 0

    def remove(self, img, engine=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("remover down")
        return fake_remove_bg(img), self.engine


def make_container(tmp_path, **settings) -> Container:
    s = Settings(**{"disable_auth": True, "data_root": tmp_path, **settings})
    c = Container.build(
        s,
        embedder=FakeEmbedder(),
        index_remover=FakeRemover(),
        query_remover=FakeRemover(),
        blobs=(MemoryBlobStore(), MemoryBlobStore(), MemoryBlobStore()),
    )
    c.startup(start_worker=False)
    return c


def png(img) -> bytes:
    return encode_png(img)
