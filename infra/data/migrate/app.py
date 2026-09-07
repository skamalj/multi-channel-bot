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

**psycopg2, not psycopg3.** Redshift reports its client encoding as `UNICODE`,
which is not a codec name Python knows, and psycopg3 raises
`NotSupportedError: codec not available in Python: 'UNICODE'` on the first
query. psycopg2 ships a mapping that translates `UNICODE` to `utf_8`, which is
why it is still the right driver for Redshift specifically.
"""
from __future__ import annotations

import logging
import os
import pathlib

import psycopg2

log = logging.getLogger()
log.setLevel(logging.INFO)

MIGRATIONS = pathlib.Path(__file__).parent / "migrations"

LEDGER = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename   VARCHAR(255) NOT NULL,
    applied_at TIMESTAMP    NOT NULL DEFAULT SYSDATE
)
"""


def _connect():
    """Connect with what the environment was given at deploy time.

    The password is an environment variable rather than an SSM lookup because
    this Lambda sits in a VPC with no NAT and no interface endpoints - it
    cannot reach the SSM API at all. Parameter Store is still the store of
    record; the deploy workflow reads it and passes it in.
    """
    return psycopg2.connect(
        host=os.environ["REDSHIFT_HOST"],
        port=int(os.environ.get("REDSHIFT_PORT", "5439")),
        dbname=os.environ["REDSHIFT_DB"],
        user=os.environ["REDSHIFT_USER"],
        password=os.environ["REDSHIFT_PASSWORD"],
        connect_timeout=30,
    )


def _applied(cur) -> set[str]:
    cur.execute(LEDGER)
    cur.execute("SELECT filename FROM schema_migrations")
    return {row[0] for row in cur.fetchall()}


def _statements(sql: str) -> list[str]:
    """Split a migration into individual statements.

    Redshift will not see a schema created earlier in the same batch - send
    `CREATE SCHEMA mcb; CREATE TABLE mcb.t (...)` as one string and the second
    statement fails with `schema "mcb" does not exist`. Each statement
    therefore goes over the wire on its own, which also means a failure names
    the statement that caused it rather than the whole file.

    The split is quote- and comment-aware, because a `;` inside a string
    literal or a `--` comment is not a statement boundary.
    """
    out: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "-" and sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j == -1 else j + 1
            buf.append("\n")
            continue
        if ch == ";":
            out.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    out.append("".join(buf))
    return [t.strip() for t in out if t.strip()]


def _pending() -> list[pathlib.Path]:
    if not MIGRATIONS.is_dir():
        return []
    return sorted(MIGRATIONS.glob("*.sql"))


def handler(event, context):            # noqa: ANN001, ARG001
    """Apply every migration not yet recorded. Returns what it did."""
    dry_run = bool((event or {}).get("dry_run"))

    conn = _connect()
    try:
        with conn, conn.cursor() as cur:
            done = _applied(cur)
            pending = [p for p in _pending() if p.name not in done]

            if dry_run:
                conn.rollback()
                return {"ok": True, "dry_run": True,
                        "applied": sorted(done),
                        "pending": [p.name for p in pending]}


            for path in pending:
                stmts = _statements(path.read_text(encoding="utf-8"))
                log.info("applying %s (%d statements)", path.name, len(stmts))
                for k, stmt in enumerate(stmts, 1):
                    log.info("  [%s %d/%d] %s", path.name, k, len(stmts),
                             stmt.splitlines()[0][:80])
                    cur.execute(stmt)
                # The existence check and the insert are in the same
                # transaction as the migration itself, so a failure halfway
                # leaves neither the change nor the ledger row.
                cur.execute(
                    "INSERT INTO schema_migrations (filename) "
                    "SELECT %s WHERE NOT EXISTS ("
                    "  SELECT 1 FROM schema_migrations WHERE filename = %s)",
                    (path.name, path.name))
    finally:
        conn.close()

    return {"ok": True,
            "applied_now": [p.name for p in pending],
            "already_applied": sorted(done)}
