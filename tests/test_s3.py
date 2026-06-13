import boto3
import pytest
from moto import mock_aws
from service import s3


@mock_aws
def test_get_bytes(monkeypatch):
    monkeypatch.setattr(s3.config, "S3_BUCKET", "bkt")
    monkeypatch.setattr(s3.config, "S3_REGION", "ap-south-1")
    s3._client.cache_clear()
    c = boto3.client("s3", region_name="ap-south-1")
    c.create_bucket(Bucket="bkt",
                    CreateBucketConfiguration={"LocationConstraint": "ap-south-1"})
    c.put_object(Bucket="bkt", Key="uploads/x.jpg", Body=b"hello")
    assert s3.get_bytes("uploads/x.jpg") == b"hello"


@mock_aws
def test_missing_key_returns_none(monkeypatch):
    monkeypatch.setattr(s3.config, "S3_BUCKET", "bkt")
    s3._client.cache_clear()
    c = boto3.client("s3", region_name="ap-south-1")
    c.create_bucket(Bucket="bkt",
                    CreateBucketConfiguration={"LocationConstraint": "ap-south-1"})
    assert s3.get_bytes("uploads/missing.jpg") is None
