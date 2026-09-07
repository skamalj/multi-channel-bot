-- 001_init - the shape every later migration assumes.
--
-- Redshift specifics worth knowing before reading further:
--   * PRIMARY KEY and UNIQUE are recorded but NOT enforced. They inform the
--     query planner and nothing else. Uniqueness is our job.
--   * There is no ON CONFLICT and no partial index. Idempotency is written
--     out explicitly (see the write_ledger below).
--   * DISTSTYLE ALL is right for small dimension tables; the whole dataset
--     here is far under a gigabyte.

CREATE SCHEMA IF NOT EXISTS mcb;

-- The idempotency ledger. Twelve write tools carry an idempotency key
-- (AG-6/CO-1) and a retry must return the FIRST receipt rather than perform
-- a second write. Without a unique index to lean on, every write tool checks
-- and inserts here inside one transaction, and reads its own prior result
-- back out of `result_json` when the key is already present.
CREATE TABLE IF NOT EXISTS mcb.write_ledger (
    idempotency_key VARCHAR(128) NOT NULL,
    tool_name       VARCHAR(64)  NOT NULL,
    user_id         VARCHAR(64)  NOT NULL,
    result_json     VARCHAR(65535),
    created_at      TIMESTAMP    NOT NULL DEFAULT SYSDATE
)
DISTSTYLE ALL
SORTKEY (idempotency_key);

-- A trivial table that proves connectivity end to end without depending on
-- any of the domain schema, so the deploy pipeline can be verified before
-- the core store is ported.
CREATE TABLE IF NOT EXISTS mcb.deploy_check (
    checked_at TIMESTAMP NOT NULL DEFAULT SYSDATE,
    note       VARCHAR(256)
)
DISTSTYLE ALL;

INSERT INTO mcb.deploy_check (note) VALUES ('001_init applied');
