"""Schema and seed for the Redshift database.

Runs inside the VPC because the workgroup is private. Invoked by CI after
`mcb-data` deploys, and deliberately NOT wired as a CloudFormation custom
resource: a migration you can only run by updating a stack is a migration you
will be afraid to run.

Every migration is a `.sql` file in `migrations/`, applied once, in filename
order, recorded in `schema_migrations`. Re-invoking is a no-op, which is what
makes it safe for CI to call on every deploy.

Redshift is not PostgreSQL where it matters here. It does not enforce unique
or primary key constraints and has no `ON CONFLICT`, so the ledger below is
guarded by an explicit existence check inside one transaction rather than by
an index. That is the same problem the write tools have with idempotency keys
(AG-6/CO-1), and this is the pattern they will use.
"""
from __future__ import annotations

import logging
import os
import pathlib

import psycopg

log = logging.getLogger()
log.setLevel(logging.INFO)

MIGRATIONS = pathlib.Path(__file__).parent / "migrations"

LEDGER = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename   VARCHAR(255) NOT NULL,
    applied_at TIMESTAMP    NOT NULL DEFAULT SYSDATE
)
"""


def _connect() -> psycopg.Connection:
    """Connect with what the environment was given at deploy time.

    The password is an environment variable rather than an SSM lookup because
    this Lambda sits in a VPC with no NAT and no interface endpoints - it
    cannot reach the SSM API at all. Parameter Store is still the store of
    record; the deploy workflow reads it and passes it in.
    """
    return psycopg.connect(
        host=os.environ["REDSHIFT_HOST"],
        port=int(os.environ.get("REDSHIFT_PORT", "5439")),
        dbname=os.environ["REDSHIFT_DB"],
        user=os.environ["REDSHIFT_USER"],
        password=os.environ["REDSHIFT_PASSWORD"],
        connect_timeout=30,
        autocommit=False,
    )


def _applied(cur) -> set[str]:
    cur.execute(LEDGER)
    cur.execute("SELECT filename FROM schema_migrations")
    return {row[0] for row in cur.fetchall()}


def _pending() -> list[pathlib.Path]:
    if not MIGRATIONS.is_dir():
        return []
    return sorted(MIGRATIONS.glob("*.sql"))


def handler(event, context):            # noqa: ANN001, ARG001
    """Apply every migration not yet recorded. Returns what it did."""
    dry_run = bool((event or {}).get("dry_run"))

    with _connect() as conn:
        with conn.cursor() as cur:
            done = _applied(cur)
            pending = [p for p in _pending() if p.name not in done]

            if dry_run:
                conn.rollback()
                return {"ok": True, "dry_run": True,
                        "applied": sorted(done),
                        "pending": [p.name for p in pending]}

            for path in pending:
                sql = path.read_text(encoding="utf-8")
                log.info("applying %s (%d bytes)", path.name, len(sql))
                cur.execute(sql)
                # The existence check and the insert are in the same
                # transaction as the migration itself, so a failure halfway
                # leaves neither the change nor the ledger row.
                cur.execute(
                    "INSERT INTO schema_migrations (filename) "
                    "SELECT %s WHERE NOT EXISTS ("
                    "  SELECT 1 FROM schema_migrations WHERE filename = %s)",
                    (path.name, path.name))
        conn.commit()

    return {"ok": True,
            "applied_now": [p.name for p in pending],
            "already_applied": sorted(done)}
