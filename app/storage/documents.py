"""Document storage - the other half of keeping a thread small.

The rule: **a document never enters the message list.** Its bytes go to S3
and only its metadata comes back as a chat message. One scanned policy
schedule inline is worth a thousand turns of text, and it would push a
checkpoint past DynamoDB's 1 MB query cap on its own - so the fix is not to
page around the cap, it is to never put the document there.

What the thread holds:

    [document] policy_schedule.pdf - application/pdf, 214 KB, ref DOC-3f9a12c4

What S3 holds: the bytes, at `documents/{user}/{lob}/{sha256}`, addressed by
content so the same file uploaded twice is stored once.

The metadata is also what a tool would be given to work with - an OCR or IDP
step reads from the reference, it does not read from the transcript. That is
the same argument as the rating engine: the model names the document, the
code fetches it.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from typing import Any

from app.config import settings

MAX_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class DocumentRef:
    """Everything about a document EXCEPT its contents."""

    doc_id: str
    filename: str
    mime: str
    size: int
    sha256: str
    key: str
    user_id: str
    lob: str
    uploaded_at: float

    def as_dict(self) -> dict:
        return asdict(self)

    def as_message(self) -> str:
        """The one line that goes into the conversation."""
        kb = max(1, round(self.size / 1024))
        return (f"[document] {self.filename} - {self.mime}, {kb} KB, "
                f"ref {self.doc_id}")


class DocumentStore:
    """S3 when there is AWS, a dict when there is not."""

    def __init__(self) -> None:
        self._mem: dict[str, bytes] = {}

    # -- storage primitives, so the logic above is testable ---------------
    def _no_aws(self) -> bool:
        import os

        return os.getenv("NO_AWS") == "1" or settings().no_aws

    def _write(self, key: str, content: bytes, mime: str) -> None:
        if self._no_aws():
            self._mem[key] = content
            return
        import boto3

        cfg = settings()
        boto3.client("s3", region_name=cfg.aws_region).put_object(
            Bucket=cfg.s3_bucket, Key=key, Body=content, ContentType=mime,
            ServerSideEncryption="AES256")

    def _read(self, key: str) -> bytes | None:
        if self._no_aws():
            return self._mem.get(key)
        import boto3

        cfg = settings()
        try:
            resp = boto3.client("s3", region_name=cfg.aws_region).get_object(
                Bucket=cfg.s3_bucket, Key=key)
            return resp["Body"].read()
        except Exception:                                    # noqa: BLE001
            return None

    def _delete(self, keys: list[str]) -> int:
        if self._no_aws():
            n = sum(1 for k in keys if self._mem.pop(k, None) is not None)
            return n
        import boto3

        cfg = settings()
        s3 = boto3.client("s3", region_name=cfg.aws_region)
        n = 0
        for k in keys:
            try:
                s3.delete_object(Bucket=cfg.s3_bucket, Key=k)
                n += 1
            except Exception:                                # noqa: BLE001
                pass
        return n

    def _list_keys(self, prefix: str) -> list[str]:
        if self._no_aws():
            return [k for k in self._mem if k.startswith(prefix)]
        import boto3

        cfg = settings()
        s3 = boto3.client("s3", region_name=cfg.aws_region)
        out: list[str] = []
        token = None
        while True:
            kw: dict[str, Any] = {"Bucket": cfg.s3_bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            resp = s3.list_objects_v2(**kw)
            out.extend(o["Key"] for o in resp.get("Contents", []))
            token = resp.get("NextContinuationToken")
            if not token:
                return out

    # -- api ---------------------------------------------------------------
    def put(self, user_id: str, lob: str, filename: str, mime: str,
            content: bytes) -> DocumentRef:
        if len(content) > MAX_BYTES:
            raise ValueError(
                f"document is {len(content)} bytes; the limit is {MAX_BYTES}")
        digest = hashlib.sha256(content).hexdigest()
        key = f"documents/{user_id}/{lob}/{digest}"
        self._write(key, content, mime or "application/octet-stream")
        return DocumentRef(
            doc_id=f"DOC-{digest[:8]}", filename=filename or digest[:8],
            mime=mime or "application/octet-stream", size=len(content),
            sha256=digest, key=key, user_id=user_id, lob=lob,
            uploaded_at=time.time())

    def get(self, ref: DocumentRef | dict) -> bytes | None:
        key = ref["key"] if isinstance(ref, dict) else ref.key
        return self._read(key)

    def erase(self, user_id: str) -> int:
        """ME-8. A subject erasure has to reach the documents too."""
        return self._delete(self._list_keys(f"documents/{user_id}/"))


_STORE: DocumentStore | None = None


def documents() -> DocumentStore:
    global _STORE
    if _STORE is None:
        _STORE = DocumentStore()
    return _STORE


def reset_documents() -> None:
    global _STORE
    _STORE = None
