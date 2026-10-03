"""Every BlobStore implementation must pass the same contract."""

import boto3
import pytest
from moto import mock_aws

from poshak.domain.errors import InvalidKey, NotFound
from poshak.infrastructure.storage.local import LocalBlobStore
from poshak.infrastructure.storage.s3 import S3BlobStore


@pytest.fixture(params=["local", "s3"])
def store(request, tmp_path):
    if request.param == "local":
        yield LocalBlobStore(tmp_path / "blobs")
        return
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="poshak-test")
        yield S3BlobStore("poshak-test", prefix="uploads/", client=client)


def test_put_get_roundtrip(store):
    store.put("a.jpg", b"\x00\x01data")
    assert store.get("a.jpg") == b"\x00\x01data"
    assert store.exists("a.jpg")


def test_put_overwrites(store):
    store.put("a.jpg", b"one")
    store.put("a.jpg", b"two")
    assert store.get("a.jpg") == b"two"


def test_missing_key(store):
    assert not store.exists("nope.jpg")
    with pytest.raises(NotFound):
        store.get("nope.jpg")


def test_delete_reports_whether_removed(store):
    store.put("a.jpg", b"x")
    assert store.delete("a.jpg") is True
    assert store.delete("a.jpg") is False
    assert not store.exists("a.jpg")


@pytest.mark.parametrize("key", ["", "../etc/passwd", "a/b.jpg", "a\\b.jpg"])
def test_rejects_keys_that_escape_namespace(store, key):
    with pytest.raises(InvalidKey):
        store.put(key, b"x")


def test_s3_prefix_is_applied():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="poshak-prefix")
        S3BlobStore("poshak-prefix", prefix="rmbg/", client=client).put("x.png", b"p")
        assert client.get_object(Bucket="poshak-prefix", Key="rmbg/x.png")["Body"].read() == b"p"
