# Human-in-the-loop for policy issuance

Issuing a policy needs a colleague's sign-off. It runs on **agent-wait 0.4.1**
(`agent-wait`, `langgraph-wait`, `agent-wait-aws`) in **async mode** — the
customer thread never parks, so a long approval wait cannot fork it.

## How it works

**Request** — `policy_issue` is `@tool` over `@hitl(mode="async", …)`
(`app/mcpserver/tools/issuance.py`). Calling it PUBLISHES an approval question
through the announcers and returns `{"status": "pending_approval", …}` **without
issuing**. The prompt tells the model to relay that to the customer.

**Gate** — `ToolControls` (`app/agents/controls.py`) is the single point that
decides whether an issuance may be (re)published, and it runs *before* the tool:

- refuse if a human already decided this issuance (row `executed`/`rejected`/
  `blocked` → typed `already_decided`). This matters because agent-wait's
  DynamoDbAnnounce does an **unconditional `put_item`** — without this check a
  re-request would reset a decided row back to `open`;
- refuse if a blocking issuance gate (kyc/underwriting/inspection/payment) is
  unclear (`gate_not_cleared`), so a human is only ever asked what can issue.

**Receive** — the decision arrives later as a NEW message; `on_decision`
(`app/agents/approvals.py`) claims the row (`open→executing`), issues with args
**from the row**, and marks it. Then a forward turn tells the customer.

## The two rules that keep it correct

1. **One exactly-once mechanism — the core's idempotency key.** `store.policy_issue`
   replays on `sha256(policy_issue|{application_id})`, so the row DEFERS to it:
   `executing` means "call the core again, it replays", not "refuse". A crash
   between claim and issue is safe. Only `executed`/`rejected`/`blocked` refuse.
2. **Every state write is conditional** — a single `transition(frm→to)` on the
   ledger. reject is `open→rejected` (a reject after a claim is a no-op), finish
   is `executing→executed`, block is `executing→blocked`. A late message can
   never move a row backwards.

`blocked` (a gate regressed between approval and execution) is **terminal**: not
re-published. A human resets the row (below) before any re-request.

## Deploy checklist (not built — needs AWS)

Offline (no `approvals_table`) the whole loop runs against an in-memory ledger;
these are what production adds.

- **DynamoDB approvals table** + GSI **`by_status` (`status`, `expires_at`)**.
  Set `approvals_table`. Rows: `pk=THREAD#<thread>`, `sk=WAIT#<question_id>`.
- **SNS topic** for the approver surface. Set `approvals_sns_topic_arn`.
- **SQS decision consumer** → calls `on_decision`, then drives the customer
  forward turn. **FIFO with `MessageGroupId = thread_id`** so a decision and a
  customer message on one thread don't run concurrently — the conditional
  transitions make that *safe* regardless; FIFO makes it *cheap* (no wasted core
  calls). A standard queue is fine too; expect the odd `already_executing`
  no-op in logs.
- **Timeout sweep** — query `by_status` for `status = open AND expires_at <
  <now-iso>`, and for each do `transition(open→executing)` with the policy's
  `default` (`{"action":"reject","reason":"approval window elapsed"}`) as the
  answer, then mark `rejected`. Notes:
  - rows with no timeout carry the literal string **`"never"`**, which sorts
    after every ISO date, so they never match — correct;
  - treat a failed `open→executing` (someone got there first) as "a human
    decided" and move on silently.
- **Verify the AWS adapters against a real table** — agent-wait 0.4's
  DynamoDb/SNS/SQS adapters are unit-tested (moto) but not verified against real
  AWS; test `DynamoDbAnnounce` and `DynamoApprovals` against the actual table
  before relying on them.

## Runbook

- **Row is authoritative for "was the approver asked".** The announcer's row
  write and the SNS publish are independent adapters with independent failure
  (each contained by `CompositeAnnounce`) — if SNS is down the row still lands.
  The gate hook reads the row, so the row wins.
- **A `blocked` row** (gate regressed after approval) is terminal. Ops resets it
  to `open` (or deletes it) and the customer re-requests; do not auto-re-publish
  — a human should see that the world changed under an approval.
- **Re-request after rejection** is refused (`already_decided`): one approval per
  application, a human's no is final. There is no attempt nonce by design.

The package is frozen at 0.4.1; anything the library should do differently goes
to its owner, not into this code.
