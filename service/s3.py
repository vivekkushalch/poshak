"""S3 cold-start source. Used ONLY by scripts/seed.py and ingest fallback —
never on the search path."""

from functools import lru_cache
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from . import config


@lru_cache(maxsize=1)
def _client():
    return boto3.client("s3", region_name=config.S3_REGION)


def get_bytes(key: str) -> Optional[bytes]:
    try:
        obj = _client().get_object(Bucket=config.S3_BUCKET, Key=key)
        return obj["Body"].read()
    except ClientError:
        return None
