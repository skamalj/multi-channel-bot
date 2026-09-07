"""Tell me exactly what is missing before the first real run.

Credentials are never read from .env - this checks the default boto3 chain,
wherever you are running it.
"""
from __future__ import annotations

import os
import sys

from app.config import settings

OK, BAD, WARN = "  OK  ", " FAIL ", " WARN "


def line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label}" + (f"  -  {detail}" if detail else ""))


def main() -> int:
    cfg = settings()
    failures = 0
    # NO_AWS=1 keeps every store in memory. The tables are then genuinely not
    # needed, and reporting them as blocking failures teaches people to
    # ignore this script - which is the one thing it must not do.
    stores_needed = not (os.getenv("NO_AWS") == "1" or cfg.no_aws)
    print(f"\nregion={cfg.aws_region}  prompt_version={cfg.prompt_version}  "
          f"config_version={cfg.config_version}  mock_llm={cfg.mock_llm}\n"
          f"stores={'aws' if stores_needed else 'in-memory (NO_AWS=1)'}\n")

    try:
        import boto3
        ident = boto3.client("sts", region_name=cfg.aws_region).get_caller_identity()
        line(OK, "credentials", f"account {ident['Account']}")
    except Exception as exc:                                    # noqa: BLE001
        line(BAD, "credentials", f"{type(exc).__name__}: {exc}")
        print("\n  -> set AWS_PROFILE, or run `aws configure`, or use SSO.\n")
        return 1

    import boto3
    ddb = boto3.client("dynamodb", region_name=cfg.aws_region)
    for t in (cfg.ddb_checkpoint_table, cfg.ddb_resolver_table, cfg.ddb_audit_table,
              cfg.ddb_profile_table):
        try:
            ddb.describe_table(TableName=t)
            line(OK, f"table {t}")
        except Exception:                                        # noqa: BLE001
            if stores_needed:
                line(BAD, f"table {t}", "missing - run scripts/bootstrap_aws.py")
                failures += 1
            else:
                line(WARN, f"table {t}", "missing - not needed with NO_AWS=1")

    s3 = boto3.client("s3", region_name=cfg.aws_region)
    try:
        s3.head_bucket(Bucket=cfg.s3_bucket)
        line(OK, f"bucket {cfg.s3_bucket}")
    except Exception:                                            # noqa: BLE001
        if stores_needed:
            line(BAD, f"bucket {cfg.s3_bucket}", "missing or not yours")
            failures += 1
        else:
            line(WARN, f"bucket {cfg.s3_bucket}",
                 "missing - not needed with NO_AWS=1")

    try:
        br = boto3.client("bedrock", region_name=cfg.aws_region)
        ids = {m["modelId"] for m in br.list_foundation_models()["modelSummaries"]}
        for mid in (cfg.bedrock_model_id, cfg.bedrock_small_model_id):
            base = mid.split(".", 1)[-1] if mid[:3] in ("apa", "us.", "eu.") else mid
            if mid in ids or any(base in i for i in ids):
                line(OK, f"bedrock {mid}")
            else:
                line(WARN, f"bedrock {mid}",
                     "not listed here - check inference-profile id and model access")
    except Exception as exc:                                     # noqa: BLE001
        line(WARN, "bedrock", f"could not list models: {exc}")

    # The corpus is the other thing a first run needs and forgets about.
    try:
        from app.knowledge.corpus import stats

        st = stats()
        line(OK, "corpus", f"{st['documents']} documents, {st['chunks']} "
                           f"chunks ({st['source']})")
    except Exception as exc:                                     # noqa: BLE001
        line(BAD, "corpus", f"{type(exc).__name__}: {exc}"
                            f"  -  run scripts/build_corpus.py")
        failures += 1

    print()
    if failures:
        print(f"{failures} blocking issue(s). Run: "
              f"uv run python scripts/bootstrap_aws.py\n")
    else:
        print("Ready. Try:  uv run uvicorn app.main:app --port 8000\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
