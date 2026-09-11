"""Human-in-the-loop for policy issuance, on the agent-wait package (async mode).

The split, straight from the package's contract:

* **Request side** — `@hitl(mode="async", ...)` on `policy_issue` (see
  issuance.py) publishes the question through these announcers and returns
  `pending_approval` WITHOUT running the body. Nothing parks; the customer
  thread carries on. `agent-wait` writes the approval row once (`status=open`)
  and never touches it again.
* **Receive side (this module)** — the decision arrives later as a NEW message.
  `on_decision` claims the row (conditional `open -> executing`), runs the real
  issuance, and marks the row. It is ours; the library has no receive path.

Two rules the package owner was emphatic about, both honoured here:

1. **One exactly-once mechanism, the core's.** `store.policy_issue` already
   replays on its idempotency key, so the row DEFERS to it: `executing` means
   "call the core again, it replays", not "refuse". Only `executed`/`rejected`
   short-circuit. A crash between claim and issue is therefore safe.
2. **Publish only when the gate chain is clear** (enforced in controls.py, before
   the tool publishes), so the question a human approves is one that can execute.
   A gate that regresses between approval and execution is a genuine anomaly:
   the row is marked `blocked:<gate>` and NOT auto-re-published.

Args come from the ROW, never from the decision message: the approver approved
the question that was asked.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from agent_wait import InMemoryAnnounce, LogAnnounce, WaitPolicy
from agent_wait.announce import BaseAnnounce
from agent_wait.model import Transition, WaitEnvelope, iso

from app.config import settings

APPROVE = {"action": "approve"}


def issuance_policy() -> WaitPolicy:
    cfg = settings()
    return WaitPolicy(
        timeout=cfg.issuance_approval_timeout,
        default={"action": "reject", "reason": "approval window elapsed"},
        answer_ttl=cfg.issuance_answer_ttl,
        allowed_actions=("approve", "reject"),
        tags={"approver_group": cfg.issuance_approver_group},
    )


# ---------------------------------------------------------------------------
# The ledger: agent-wait writes the row (status=open); we own every transition
# on that same row. The claim is the one write that matters.
# ---------------------------------------------------------------------------
def _pk(thread_id: str) -> str:
    return f"THREAD#{thread_id}"


def _sk(question_id: str) -> str:
    return f"WAIT#{question_id}"


class MemoryApprovals:
    """In-memory ledger for offline/tests. The compare-and-set on `claim` is
    the whole point - it must be atomic, exactly as the DynamoDB conditional
    update is."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], dict] = {}
        self._lock = threading.Lock()

    def put_open(self, item: dict) -> None:
        with self._lock:
            self._rows[(item["pk"], item["sk"])] = item

    def get(self, thread_id: str, question_id: str) -> dict | None:
        with self._lock:
            row = self._rows.get((_pk(thread_id), _sk(question_id)))
            return dict(row) if row else None

    def transition(self, thread_id: str, question_id: str, frm: str, to: str,
                   **fields: Any) -> bool:
        """`frm -> to`, atomically. True if we made it. Every state change goes
        through here so a late message can never move a row backwards: whoever
        wins the write from `frm` decides, everyone else is a no-op."""
        with self._lock:
            row = self._rows.get((_pk(thread_id), _sk(question_id)))
            if row is None or row.get("status") != frm:
                return False
            row["status"] = to
            row["decided_at"] = iso(time.time())
            row.update(fields)
            return True

    def claim(self, thread_id: str, question_id: str) -> bool:
        """open -> executing, the one write that gives exactly-once."""
        return self.transition(thread_id, question_id, "open", "executing")


class DynamoApprovals:
    """The same ledger on DynamoDB. `claim` is a conditional update, which is
    what gives exactly-once across redeliveries and concurrent handlers."""

    def __init__(self, table: Any) -> None:
        self._table = table

    def get(self, thread_id: str, question_id: str) -> dict | None:
        return self._table.get_item(
            Key={"pk": _pk(thread_id), "sk": _sk(question_id)},
            ConsistentRead=True).get("Item")

    def transition(self, thread_id: str, question_id: str, frm: str, to: str,
                   **fields: Any) -> bool:
        """`frm -> to` as a conditional update - the DynamoDB form of the same
        exactly-once transition. False on ConditionalCheckFailed (someone else
        already moved it)."""
        from botocore.exceptions import ClientError

        names = {"#s": "status"}
        values = {":to": to, ":frm": frm, ":at": iso(time.time())}
        sets = ["#s = :to", "decided_at = :at"]
        for i, (k, v) in enumerate(fields.items()):
            names[f"#f{i}"] = k
            values[f":v{i}"] = v
            sets.append(f"#f{i} = :v{i}")
        try:
            self._table.update_item(
                Key={"pk": _pk(thread_id), "sk": _sk(question_id)},
                UpdateExpression="SET " + ", ".join(sets),
                ConditionExpression="attribute_exists(pk) AND #s = :frm",
                ExpressionAttributeNames=names, ExpressionAttributeValues=values)
            return True
        except ClientError as exc:                             # noqa: BLE001
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise

    def claim(self, thread_id: str, question_id: str) -> bool:
        return self.transition(thread_id, question_id, "open", "executing")


class _MemoryRowAnnounce(BaseAnnounce):
    """Offline stand-in for DynamoDbAnnounce: writes the same open row into a
    MemoryApprovals ledger, so the receive side has a row to claim without AWS."""

    name = "memory-row"

    def __init__(self, ledger: MemoryApprovals) -> None:
        super().__init__()
        self._ledger = ledger

    def deliver(self, envelope: WaitEnvelope, transition: Transition) -> None:
        self._ledger.put_open({
            "pk": _pk(envelope.thread_id), "sk": _sk(envelope.question_id),
            "status": "open", "thread_id": envelope.thread_id,
            "question_id": envelope.question_id, "question": envelope.question,
            "allowed_actions": list(envelope.allowed_actions),
            "expires_at": envelope.expires_at or "never",
            "reply_with": dict(envelope.reply_with), "event_id": envelope.event_id,
        })


# ---------------------------------------------------------------------------
# Wiring. Announcers are bound at @hitl decoration time; the ledger is read on
# the decision. Offline (no table) both point at one MemoryApprovals so the
# whole loop runs without AWS; configured, at DynamoDB + SNS.
# ---------------------------------------------------------------------------
_MEM = MemoryApprovals()
_ANNOUNCERS: list[BaseAnnounce] | None = None
_LEDGER: Any = None


def announcers() -> list[BaseAnnounce]:
    global _ANNOUNCERS
    if _ANNOUNCERS is None:
        cfg = settings()
        if cfg.approvals_table:
            from agent_wait_aws import DynamoDbAnnounce, SnsAnnounce

            out: list[BaseAnnounce] = [DynamoDbAnnounce(cfg.approvals_table,
                                                        region_name=cfg.aws_region)]
            if cfg.approvals_sns_topic_arn:
                out.append(SnsAnnounce(cfg.approvals_sns_topic_arn,
                                       region_name=cfg.aws_region))
            out.append(LogAnnounce())
            _ANNOUNCERS = out
        else:
            # Offline: also expose an InMemoryAnnounce so tests can see the
            # envelope, alongside the row-writer the receive side reads.
            _ANNOUNCERS = [_MemoryRowAnnounce(_MEM), InMemoryAnnounce(), LogAnnounce()]
    return _ANNOUNCERS


def ledger() -> Any:
    global _LEDGER
    if _LEDGER is None:
        cfg = settings()
        if cfg.approvals_table:
            import boto3
            _LEDGER = DynamoApprovals(
                boto3.resource("dynamodb", region_name=cfg.aws_region)
                .Table(cfg.approvals_table))
        else:
            _LEDGER = _MEM
    return _LEDGER


def reset() -> None:
    """Tests rebuild the wiring between cases."""
    global _ANNOUNCERS, _LEDGER
    _ANNOUNCERS = None
    _LEDGER = None
    _MEM._rows.clear()


# ---------------------------------------------------------------------------
# The receive side: a decision arrives as a new message.
# ---------------------------------------------------------------------------
def on_decision(message: dict) -> dict:
    """Handle one approval decision. Idempotent and safe to call twice.

    `message` is the reply_with stub with `answer` filled in:
    `{"thread_id": ..., "question_id": ..., "answer": {"action": "approve"|...}}`.
    Returns an outcome dict; `customer` is a line the caller can put on a
    forward turn so the model tells the customer in its own voice.
    """
    thread_id = message.get("thread_id")
    qid = message.get("question_id")
    answer = message.get("answer") or {}
    book = ledger()

    row = book.get(thread_id, qid)
    if row is None:
        return {"status": "unknown_question", "question_id": qid}

    status = row.get("status", "open")
    # executed / rejected / blocked are terminal - a human (or a prior run)
    # already decided. Never re-open, never re-execute.
    if status in ("executed", "rejected", "blocked"):
        return {"status": f"already_{status}", "question_id": qid}

    if answer.get("action") != "approve":
        # Conditional: only an OPEN question can be rejected. A reject that
        # arrives after a claim (row executing) must not overwrite it.
        if book.transition(thread_id, qid, "open", "rejected", answer=answer):
            return {"status": "rejected", "question_id": qid,
                    "customer": ("I checked, and this could not be approved "
                                 "right now, so I have not issued the policy. I "
                                 "can put you through to a colleague to sort it "
                                 "out.")}
        return {"status": f"already_{(book.get(thread_id, qid) or {}).get('status')}",
                "question_id": qid}

    # open -> executing, exactly once. If we lose the race, only executed/
    # rejected/blocked short-circuit; `executing` means "run the core again, it
    # replays on its idempotency key".
    if status == "open" and not book.claim(thread_id, qid):
        again = (book.get(thread_id, qid) or {}).get("status")
        if again in ("executed", "rejected", "blocked"):
            return {"status": f"already_{again}", "question_id": qid}

    # Args come from the ROW - the approver approved the question that was asked.
    args = (row.get("question") or {}).get("args") or {}
    application_id = args.get("application_id")

    from app.mcpserver.tools.issuance import _issue_impl

    result = _issue_impl(application_id)
    if isinstance(result, dict) and result.get("error") == "gate_not_cleared":
        # State regressed between approval and execution. Terminal, and NOT
        # re-published: a human resets the row before anyone re-asks (runbook).
        book.transition(thread_id, qid, "executing", "blocked",
                        gate=result.get("blocked_by"))
        return {"status": "blocked", "question_id": qid, "detail": result,
                "customer": ("Something changed since this was approved and it "
                             "can no longer be issued as it stands. I have not "
                             "issued it, and a colleague will follow up.")}

    ref = result.get("policy_id") if isinstance(result, dict) else None
    book.transition(thread_id, qid, "executing", "executed", policy_id=ref)
    return {"status": "executed", "question_id": qid, "result": result,
            "customer": ("Good news - your policy is issued"
                         + (f" (policy {ref})" if ref else "") + ".")}
