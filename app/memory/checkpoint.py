"""Checkpointers - the two of them, keyed apart (ME-1, ME-2).

    resolver graph   thread_id = user            ledger · profile · auth   180 d
    bot graph        thread_id = user#lob        messages · slots          30 d

LangGraph owns persistence. Nothing in this codebase serialises a message,
computes a TTL or writes a session row by hand any more; a graph is compiled
with a saver and invoked with a `thread_id`, and that is the whole contract.

Why our own DynamoDB saver rather than the community package: the checkpoint
table is where the most sensitive data in the system lives, it needs a TTL we
control per store, and `langgraph-dynamodb-checkpoint` is a third-party
package pinned to an older checkpoint API. This is ~150 lines against the
v4 `BaseCheckpointSaver` contract and we own every line of it.

The two savers differ ONLY in table and TTL. That is deliberate: the
separation is a fact about addressing and retention, not about behaviour.
"""
from __future__ import annotations

import time
from typing import Any, Iterator, Sequence

from langgraph.checkpoint.base import (BaseCheckpointSaver, ChannelVersions,
                                       Checkpoint, CheckpointMetadata,
                                       CheckpointTuple, get_checkpoint_id)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from typing_extensions import Self

from app.config import settings


def _thread(config: dict) -> str:
    return str(config.get("configurable", {}).get("thread_id", ""))


def _ns(config: dict) -> str:
    return str(config.get("configurable", {}).get("checkpoint_ns", "") or "")


class DynamoDBSaver(BaseCheckpointSaver):
    """One row per checkpoint, one row per pending write.

    Partition key is the thread, sort key orders checkpoints within it, so a
    thread's history is a single query and a thread's erasure is a single
    query plus deletes. Both of those are requirements here (ME-8), not
    conveniences.
    """

    def __init__(self, table_name: str, ttl_days: int, region: str | None = None):
        super().__init__(serde=JsonPlusSerializer())
        import boto3  # lazy: the module must import with no AWS installed

        cfg = settings()
        self.table_name = table_name
        self.ttl_seconds = ttl_days * 86400
        self._table = boto3.resource(
            "dynamodb", region_name=region or cfg.aws_region).Table(table_name)

    # -- helpers ----------------------------------------------------------
    def _sk(self, ns: str, checkpoint_id: str) -> str:
        return f"cp#{ns}#{checkpoint_id}"

    def _write_sk(self, ns: str, checkpoint_id: str, task_id: str,
                  idx: int) -> str:
        return f"wr#{ns}#{checkpoint_id}#{task_id}#{idx:04d}"

    def _ttl(self) -> int:
        return int(time.time()) + self.ttl_seconds

    def _b(self, value: bytes) -> Any:
        from boto3.dynamodb.types import Binary

        return Binary(value)

    # -- storage primitives ----------------------------------------------
    # Every boto3 call lives in these four. The saver's real logic - key
    # construction, serde, ordering, pending writes - sits above them and is
    # exercised in tests against an in-memory double, because a checkpointer
    # that has never run is a checkpointer that does not work.
    def _put(self, item: dict) -> None:
        self._table.put_item(Item=item)

    def _get(self, pk: str, sk: str) -> dict | None:
        # ConsistentRead is not optional here. DynamoDB reads are eventually
        # consistent by default, and a checkpointer that reads a stale
        # checkpoint loses the previous turn - the customer says "yes" and
        # the confirmation it was answering is not there yet.
        return self._table.get_item(Key={"pk": pk, "sk": sk},
                                    ConsistentRead=True).get("Item")

    def _query_prefix(self, pk: str, prefix: str, *, forward: bool = True,
                      limit: int | None = None) -> list[dict]:
        """Paginated, because DynamoDB caps a query response at 1 MB.

        Without the loop this truncates silently, and a checkpointer that
        silently truncates is one that reports a thread erased while rows
        remain, and shows a partial history as if it were the whole one.
        Checkpoints hold whole message lists, so a busy thread passes 1 MB
        quickly.
        """
        from boto3.dynamodb.conditions import Key

        kwargs: dict[str, Any] = {
            "KeyConditionExpression": Key("pk").eq(pk)
            & Key("sk").begins_with(prefix),
            "ScanIndexForward": forward,
            "ConsistentRead": True,
        }
        if limit:
            kwargs["Limit"] = limit
        items: list[dict] = []
        while True:
            resp = self._table.query(**kwargs)
            items.extend(resp.get("Items", []))
            nxt = resp.get("LastEvaluatedKey")
            if not nxt or (limit and len(items) >= limit):
                break
            kwargs["ExclusiveStartKey"] = nxt
        return items[:limit] if limit else items

    def _delete_many(self, keys: list[tuple[str, str]]) -> None:
        with self._table.batch_writer() as batch:
            for pk, sk in keys:
                batch.delete_item(Key={"pk": pk, "sk": sk})

    # -- checkpoint logic --------------------------------------------------
    def _pending_writes(self, thread: str, ns: str,
                        checkpoint_id: str) -> list[tuple[str, str, Any]]:
        out = []
        for item in self._query_prefix(thread, f"wr#{ns}#{checkpoint_id}#"):
            out.append((item["task_id"],
                        item["channel"],
                        self.serde.loads_typed((item["type"],
                                                bytes(item["value"])))))
        return out

    # -- read -------------------------------------------------------------
    def get_tuple(self, config: dict) -> CheckpointTuple | None:
        thread, ns = _thread(config), _ns(config)
        checkpoint_id = get_checkpoint_id(config)

        if checkpoint_id:
            item = self._get(thread, self._sk(ns, checkpoint_id))
        else:
            # Latest: the sort key is monotonic in checkpoint id.
            items = self._query_prefix(thread, f"cp#{ns}#", forward=False,
                                       limit=1)
            item = items[0] if items else None

        if not item:
            return None
        return self._to_tuple(thread, ns, item)

    def _to_tuple(self, thread: str, ns: str, item: dict) -> CheckpointTuple:
        checkpoint = self.serde.loads_typed(
            (item["type"], bytes(item["checkpoint"])))
        metadata = self.serde.loads_typed(
            (item["type"], bytes(item["metadata"])))
        parent = item.get("parent_checkpoint_id")
        return CheckpointTuple(
            config={"configurable": {"thread_id": thread, "checkpoint_ns": ns,
                                     "checkpoint_id": item["checkpoint_id"]}},
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=({"configurable": {"thread_id": thread,
                                             "checkpoint_ns": ns,
                                             "checkpoint_id": parent}}
                           if parent else None),
            pending_writes=self._pending_writes(thread, ns,
                                                item["checkpoint_id"]),
        )

    def list(self, config: dict | None, *, filter: dict | None = None,
             before: dict | None = None,
             limit: int | None = None) -> Iterator[CheckpointTuple]:
        if not config:
            return
        thread, ns = _thread(config), _ns(config)
        n = 0
        before_id = get_checkpoint_id(before) if before else None
        for item in self._query_prefix(thread, f"cp#{ns}#", forward=False):
            if before_id and item["checkpoint_id"] >= before_id:
                continue
            yield self._to_tuple(thread, ns, item)
            n += 1
            if limit and n >= limit:
                return

    # -- write ------------------------------------------------------------
    def put(self, config: dict, checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions) -> dict:
        thread, ns = _thread(config), _ns(config)
        c_type, c_bytes = self.serde.dumps_typed(checkpoint)
        _, m_bytes = self.serde.dumps_typed(metadata)
        item = {
            "pk": thread,
            "sk": self._sk(ns, checkpoint["id"]),
            "checkpoint_id": checkpoint["id"],
            "checkpoint_ns": ns,
            "type": c_type,
            "checkpoint": self._b(c_bytes),
            "metadata": self._b(m_bytes),
            "ttl": self._ttl(),
        }
        parent = get_checkpoint_id(config)
        if parent:
            item["parent_checkpoint_id"] = parent
        self._put(item)
        return {"configurable": {"thread_id": thread, "checkpoint_ns": ns,
                                 "checkpoint_id": checkpoint["id"]}}

    def put_writes(self, config: dict, writes: Sequence[tuple[str, Any]],
                   task_id: str, task_path: str = "") -> None:
        thread, ns = _thread(config), _ns(config)
        checkpoint_id = get_checkpoint_id(config)
        for idx, (channel, value) in enumerate(writes):
            v_type, v_bytes = self.serde.dumps_typed(value)
            self._put({
                "pk": thread,
                "sk": self._write_sk(ns, checkpoint_id, task_id, idx),
                "task_id": task_id, "task_path": task_path,
                "channel": channel, "type": v_type,
                "value": self._b(v_bytes), "ttl": self._ttl(),
            })

    # -- erasure (ME-8) ---------------------------------------------------
    def delete_thread(self, thread_id: str) -> None:
        """Erasure has to be complete, so it loops until the thread is empty.

        ME-8 promises erasure rather than approximating it, and a single
        pass would leave whatever a page boundary cut off.
        """
        for _ in range(50):                       # a bound, not an estimate
            items = (self._query_prefix(thread_id, "cp#")
                     + self._query_prefix(thread_id, "wr#"))
            if not items:
                return
            self._delete_many([(i["pk"], i["sk"]) for i in items])


# ---------------------------------------------------------------------------
# Factories. In-memory when there is no AWS, so the whole graph runs in tests.
# ---------------------------------------------------------------------------
# Nothing of ours goes into a checkpoint as a CLASS. LangGraph warns on
# deserialising unregistered types and will block it in a future version -
# rightly, because a checkpoint that can rehydrate arbitrary classes is a
# deserialisation gadget. The permissive default cannot be added to, so the
# durable answer is not to register our models but to persist plain data:
# `ResolverSession.model_dump()` in, `model_validate()` out. The checkpoint
# holds a dict, and a dependency bump cannot break it.
_BOT: Any = None
_RESOLVER: Any = None


def _no_aws() -> bool:
    import os

    return os.getenv("NO_AWS") == "1" or settings().no_aws


def bot_checkpointer():
    """Thread = `user#lob`. Messages, slots, documents. 30 days."""
    global _BOT
    if _BOT is None:
        cfg = settings()
        if _no_aws():
            from langgraph.checkpoint.memory import InMemorySaver

            _BOT = InMemorySaver()
        else:
            _BOT = DynamoDBSaver(cfg.ddb_checkpoint_table,
                                 cfg.session_ttl_days)
    return _BOT


def resolver_checkpointer():
    """Thread = `user`. Route ledger, shared profile, auth. 180 days."""
    global _RESOLVER
    if _RESOLVER is None:
        cfg = settings()
        if _no_aws():
            from langgraph.checkpoint.memory import InMemorySaver

            _RESOLVER = InMemorySaver()
        else:
            _RESOLVER = DynamoDBSaver(cfg.ddb_resolver_table,
                                      cfg.resolver_ttl_days)
    return _RESOLVER


def reset_checkpointers() -> None:
    """Tests start from empty threads; nothing else should call this."""
    global _BOT, _RESOLVER
    _BOT = _RESOLVER = None
