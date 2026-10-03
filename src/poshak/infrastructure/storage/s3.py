"""BlobStore on S3 or any S3-compatible service (MinIO, R2) via S3_ENDPOINT_URL.

Needs the `s3` extra: pip install "poshak[s3]".

ponytail: descriptor reads become network GETs on the search path; the service's
in-process descriptor cache and parallel verification hide most of it. Add a local
disk cache in front if p95 latency matters."""

from typing import Any

from ...domain.errors import NotFound
from . import validate_key

_MISSING = {"404", "NoSuchKey", "NotFound"}


class S3BlobStore:
    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        client: Any = None,
        region: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        if client is None:
            import boto3

            client = boto3.client("s3", region_name=region, endpoint_url=endpoint_url)
        self.client = client
        self.bucket = bucket
        self.prefix = prefix

    def _key(self, key: str) -> str:
        return f"{self.prefix}{validate_key(key)}"

    @staticmethod
    def _is_missing(err: Exception) -> bool:
        code = getattr(err, "response", {}).get("Error", {}).get("Code")
        return code in _MISSING

    def get(self, key: str) -> bytes:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        except Exception as e:
            if self._is_missing(e):
                raise NotFound(f"{key} not found") from None
            raise
        return obj["Body"].read()

    def put(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
        except Exception as e:
            if self._is_missing(e):
                return False
            raise
        return True

    def delete(self, key: str) -> bool:
        if not self.exists(key):
            return False
        self.client.delete_object(Bucket=self.bucket, Key=self._key(key))
        return True
