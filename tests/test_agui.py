"""The AG-UI server, driven end to end against the real orchestrator.

`MOCK_LLM=1` swaps in the scripted model, so this exercises the whole path -
resolver, binding, tools, gates, the citation guardrail, the reply - and
asserts on the event stream a browser would actually receive.

The two properties worth protecting are both about ORDER:

* `RUN_STARTED` first and `RUN_FINISHED` last, with the answer between them.
  A client that renders on RUN_FINISHED shows nothing if that never arrives.
* Exactly one `TEXT_MESSAGE_CONTENT`. The process streams and the claims do
  not - if this ever becomes several deltas, someone has started streaming
  tokens past the citation guardrail, and a fabricated premium will reach a
  customer's screen before it is retracted.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import server

    return TestClient(server.app)


def _events(resp) -> list[dict]:
    out = []
    for line in resp.text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            out.append(json.loads(line[5:].strip()))
    return out


def _turn(client, text: str, thread: str = "919820000009") -> list[dict]:
    resp = client.post("/invocations", json={
        "threadId": thread, "runId": f"run-{abs(hash(text)) % 10000}",
        "messages": [{"id": "m1", "role": "user", "content": text}],
        "state": {}, "tools": [], "context": [],
        "forwardedProps": {"channel": "webchat"},
    })
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    return _events(resp)


def test_ping_reports_healthy_without_a_moving_timestamp(client):
    """A `time_of_last_update` that advances every ping stops the idle
    timeout firing, and sessions then bill memory to MaxLifetime."""
    body = client.get("/ping").json()
    assert body == {"status": "Healthy"}


def test_a_turn_is_framed_by_run_started_and_run_finished(client):
    evs = _turn(client, "what does the health policy cover")
    assert evs[0]["type"] == "RUN_STARTED"
    assert evs[-1]["type"] == "RUN_FINISHED"
    assert evs[0]["threadId"] == evs[-1]["threadId"] == "919820000009"


def test_the_answer_is_emitted_exactly_once(client):
    """The rule the whole design rests on: the process streams, the claims
    do not. Several deltas here means tokens are being streamed past the
    citation guardrail."""
    evs = _turn(client, "what does the health policy cover")
    contents = [e for e in evs if e["type"] == "TEXT_MESSAGE_CONTENT"]
    assert len(contents) == 1
    assert contents[0]["delta"].strip()


def test_the_answer_is_wrapped_in_start_and_end_with_one_message_id(client):
    evs = _turn(client, "what does the health policy cover")
    ids = {e["messageId"] for e in evs
           if e["type"].startswith("TEXT_MESSAGE_")}
    assert len(ids) == 1
    kinds = [e["type"] for e in evs if e["type"].startswith("TEXT_MESSAGE_")]
    assert kinds == ["TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT",
                     "TEXT_MESSAGE_END"]


def test_the_answer_comes_before_run_finished(client):
    evs = _turn(client, "what does the health policy cover")
    kinds = [e["type"] for e in evs]
    assert kinds.index("TEXT_MESSAGE_END") < kinds.index("RUN_FINISHED")


def test_the_trace_reaches_the_client(client):
    """The glass box is the point. A turn that answers but explains nothing
    is a regression even though the customer sees the same words."""
    evs = _turn(client, "what does the health policy cover")
    traces = [e for e in evs if e.get("name") == "trace"]
    assert traces, "no trace events reached the client"
    kinds = {t["value"]["kind"] for t in traces}
    assert "respond" in kinds


def test_a_turn_survives_an_empty_message(client):
    """An empty body must not become an unhandled exception mid-stream."""
    evs = _turn(client, "")
    assert evs[0]["type"] == "RUN_STARTED"
    assert evs[-1]["type"] in ("RUN_FINISHED", "RUN_ERROR")


def test_two_turns_on_one_thread_both_complete(client):
    """The thread id is the person, and the checkpointer is keyed off it.
    A second turn must not trip over the first one's state."""
    first = _turn(client, "what does the health policy cover")
    second = _turn(client, "and what is the waiting period")
    for evs in (first, second):
        assert evs[-1]["type"] == "RUN_FINISHED"
