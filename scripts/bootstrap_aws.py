"""Create the DynamoDB tables and the S3 bucket. Idempotent."""
from __future__ import annotations

import sys

import boto3
from botocore.exceptions import ClientError

from app.config import settings


def ensure_table(ddb, name: str, sort_key: bool = False) -> None:
    """Two shapes, because two kinds of store live in DynamoDB here.

    A CHECKPOINT table needs (pk=thread, sk) so a thread's history is one
    query and a thread's erasure is one query plus deletes - both of which
    the checkpointer relies on. A key-value store needs the partition key
    alone.
    """
    try:
        ddb.describe_table(TableName=name)
        print(f"  table {name}: exists")
        return
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
    attrs = [{"AttributeName": "pk", "AttributeType": "S"}]
    schema = [{"AttributeName": "pk", "KeyType": "HASH"}]
    if sort_key:
        attrs.append({"AttributeName": "sk", "AttributeType": "S"})
        schema.append({"AttributeName": "sk", "KeyType": "RANGE"})
    ddb.create_table(
        TableName=name,
        AttributeDefinitions=attrs,
        KeySchema=schema,
        BillingMode="PAY_PER_REQUEST",
    )
    ddb.get_waiter("table_exists").wait(TableName=name)
    try:
        ddb.update_time_to_live(
            TableName=name,
            TimeToLiveSpecification={"Enabled": True, "AttributeName": "ttl"})
    except ClientError as exc:                                   # noqa: BLE001
        print(f"    (ttl not set: {exc.response['Error']['Code']})")
    print(f"  table {name}: created")


def ensure_bucket(s3, name: str, region: str) -> None:
    try:
        s3.head_bucket(Bucket=name)
        print(f"  bucket {name}: exists")
        return
    except ClientError:
        pass
    kwargs = {"Bucket": name}
    if region != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3.create_bucket(**kwargs)
    s3.put_public_access_block(
        Bucket=name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True,
            "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    s3.put_bucket_encryption(
        Bucket=name,
        ServerSideEncryptionConfiguration={"Rules": [
            {"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]})
    print(f"  bucket {name}: created (private, encrypted)")


def main() -> int:
    cfg = settings()
    if "CHANGE-ME" in cfg.s3_bucket:
        print("Set S3_BUCKET in .env to a globally unique name first.")
        return 1
    print(f"\nBootstrapping in {cfg.aws_region}\n")
    ddb = boto3.client("dynamodb", region_name=cfg.aws_region)
    # LangGraph checkpointers: partition by thread, sort by checkpoint.
    for t in (cfg.ddb_checkpoint_table, cfg.ddb_resolver_table):
        ensure_table(ddb, t, sort_key=True)
    # Key-value stores: the long-term memory stores and the audit record.
    for t in (cfg.ddb_profile_table, cfg.ddb_audit_table):
        ensure_table(ddb, t)
    ensure_bucket(boto3.client("s3", region_name=cfg.aws_region),
                  cfg.s3_bucket, cfg.aws_region)
    print("\nDone. Now: python scripts/preflight.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
