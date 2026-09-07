"""One key-value backend behind every long-term store.

DynamoDB when there is AWS, a dict when there is not. Every store in
`memory/registry.py` names a partition prefix and gets its own namespace on
the same table, which is what makes a single erasure handler per store
possible.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Protocol

from app.config import settings


class Backend(Protocol):
    def get(self, pk: str) -> dict | None: ...
    def put(self, pk: str, body: dict, ttl_days: int | None) -> None: ...
    def delete_prefix(self, prefix: str) -> int: ...
    def scan_prefix(self, prefix: str) -> list[tuple[str, dict]]: ...


class InMemoryBackend:
    def __init__(self) -> None:
        self._mem: dict[str, dict] = {}

    def get(self, pk: str) -> dict | None:
        return self._mem.get(pk)

    def put(self, pk: str, body: dict, ttl_days: int | None = None) -> None:
        self._mem[pk] = body

    def delete_prefix(self, prefix: str) -> int:
        keys = [k for k in self._mem if k.startswith(prefix)]
        for k in keys:
            del self._mem[k]
        return len(keys)

    def scan_prefix(self, prefix: str) -> list[tuple[str, dict]]:
        return sorted((k, v) for k, v in self._mem.items()
                      if k.startswith(prefix))


class DynamoBackend:
    """One table, many namespaces. `delete_prefix` is the erasure handler."""

    def __init__(self, table_name: str | None = None):
        import boto3  # lazy: the module must import with no AWS installed

        cfg = settings()
        self.table_name = table_name or cfg.ddb_profile_table
        self._table = boto3.resource(
            "dynamodb", region_name=cfg.aws_region).Table(self.table_name)

    def get(self, pk: str) -> dict | None:
        try:
            item = self._table.get_item(Key={"pk": pk}).get("Item")
        except Exception:                                    # noqa: BLE001
            return None
        return json.loads(item["body"]) if item and "body" in item else None

    def put(self, pk: str, body: dict, ttl_days: int | None = None) -> None:
        item = {"pk": pk, "body": json.dumps(body, default=str)}
        if ttl_days:
            item["ttl"] = int(time.time()) + ttl_days * 86400
        self._table.put_item(Item=item)

    def delete_prefix(self, prefix: str) -> int:
        n = 0
        for pk, _ in self.scan_prefix(prefix):
            self._table.delete_item(Key={"pk": pk})
            n += 1
        return n

    def scan_prefix(self, prefix: str) -> list[tuple[str, dict]]:
        from boto3.dynamodb.conditions import Attr

        out: list[tuple[str, dict]] = []
        kwargs: dict[str, Any] = {"FilterExpression": Attr("pk").begins_with(prefix)}
        while True:
            resp = self._table.scan(**kwargs)
            for item in resp.get("Items", []):
                out.append((item["pk"], json.loads(item.get("body", "{}"))))
            if "LastEvaluatedKey" not in resp:
                return sorted(out)
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]


_BACKEND: Backend | None = None


def backend() -> Backend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = (InMemoryBackend()
                    if os.getenv("NO_AWS") == "1" or settings().no_aws
                    else DynamoBackend())
    return _BACKEND


def set_backend(b: Backend) -> None:
    """Tests inject an in-memory backend; nothing else should call this."""
    global _BACKEND
    _BACKEND = b
