"""Test wiring.

Everything except `-m live` runs with no AWS and no Bedrock: NO_AWS=1 swaps
the stores for dicts and MOCK_LLM=1 swaps the model for the scripted stub.
That is not a convenience - a test suite that needs a cloud account is a test
suite that stops being run.

The environment is set BEFORE anything imports `app`, because `settings()` is
cached for the life of the process.
"""
from __future__ import annotations

import os

os.environ.setdefault("NO_AWS", "1")
os.environ.setdefault("MOCK_LLM", "1")
os.environ.setdefault("CORE_LATENCY_MS", "0")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def clean_state():
    """Every test starts with empty stores and an unthrottled core."""
    from app.agents.graph import reset_agents
    from app.coremock import store as core
    from app.memory.backend import InMemoryBackend, set_backend
    from app.memory.checkpoint import reset_checkpointers
    from app.resolver.graph import reset_resolver_graph

    set_backend(InMemoryBackend())
    # Fresh checkpointers, then fresh graphs compiled against them - every
    # test starts on an empty thread.
    reset_checkpointers()
    reset_resolver_graph()
    _wipe_persistent_threads()
    core.reset_chaos()
    core._QUOTES.clear()
    core._IDEMPOTENCY.clear()
    core._APPLICATIONS.clear()
    core._CLAIMS.clear()
    reset_agents()
    yield


# Test user ids the suite writes to. With in-memory savers a fresh
# checkpointer is a clean slate; against real DynamoDB the rows outlive the
# run, so a route ledger or a step-up from yesterday leaks into today's test
# and the failure looks like a product bug. The suite is only isolated if it
# says which threads are its own.
TEST_USERS = [
    "919820000001", "919820000002", "919820000009", "919877000123",
    "919899999999",
]
TEST_PREFIXES = ("thread-size-", "erase-debug-", "aws-e2e-", "reduce-live-",
                 "probe-", "live-", "smoke", "u1", "u#", "other#",
                 "health-issue-", "final-health-", "agt", "nv")


def _wipe_persistent_threads() -> None:
    """No-op on in-memory savers; a targeted delete on real ones."""
    if os.getenv("NO_AWS") == "1":
        return
    from app.memory.checkpoint import bot_checkpointer, resolver_checkpointer

    for saver in (bot_checkpointer(), resolver_checkpointer()):
        deleter = getattr(saver, "delete_thread", None)
        lister = getattr(saver, "_query_prefix", None)
        if deleter is None or lister is None:
            continue
        for user in TEST_USERS:
            for thread in (user, f"{user}#health", f"{user}#motor"):
                try:
                    deleter(thread)
                except Exception:                            # noqa: BLE001
                    pass


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


@pytest.fixture
def trace():
    from app.obs.trace import Trace

    return Trace()
