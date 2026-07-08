"""E2E: the LOW-priority thread/run CRUD gaps — driven through the langgraph_sdk's
OWN clients (the contract consumers hit), not raw httpx assertions.

Closes the remaining Agent Protocol checklist items (the SDK ``threads.update`` /
``threads.copy`` + ``runs.get`` / ``runs.delete`` / ``runs.join`` paths):

- PATCH /threads/{id}            threads.update  (metadata merge; Prefer=minimal→204)
- POST  /threads/{id}/copy       threads.copy    (fork: new id, same org+meta+state)
- GET   /threads/{id}/runs/{rid} runs.get        (one run's metadata; 404 if off-thread)
- DELETE /threads/{id}/runs/{rid} runs.delete    (cancel if in flight + drop meta; 204)
- POST   /threads/{id}/runs/{rid}/join runs.join (block until the background run done)

Route shapes + payloads match the SDK verbatim. A stub graph (ainvoke/aget_state/
aupdate_state) stands in for the compiled org so the copy's state fork + the
background-run lifecycle run for real against the shared sqlite checkpointer +
run_meta index — no Docker, no model.
"""

from __future__ import annotations

import asyncio
import types
from typing import Any

import httpx
import pytest
from httpx import ASGITransport
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore
from langgraph_sdk.client import LangGraphClient
from langgraph_sdk.errors import NotFoundError

from pux_harness.server import app
from pux_harness.threads import open_thread_store

ORG = "general"


class _StubGraph:
    """Minimal compiled-graph stand-in: a per-thread state map + the three async
    methods the CRUD paths touch (ainvoke for runs, aget_state for descriptors,
    aupdate_state for copy)."""

    def __init__(self) -> None:
        self.states: dict[str, dict[str, Any]] = {}

    async def ainvoke(self, inp: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        # A real compiled graph's checkpoint reflects the RUN RESULT, not the
        # input — so aget_state (the join/descriptor path) must see the answer.
        tid = config["configurable"]["thread_id"]
        result = {"messages": [HumanMessage("x"), AIMessage("stub-answer")]}
        self.states[tid] = result
        return result

    async def aget_state(self, config: dict[str, Any]) -> Any:
        tid = config["configurable"]["thread_id"]
        return types.SimpleNamespace(
            values=self.states.get(tid, {}),
            next=(),
            tasks=(),
            config={"configurable": {"checkpoint_id": "cp"}},
            parent_config=None,
        )

    async def aupdate_state(
        self, config: dict[str, Any], values: Any, as_node: str | None = None
    ) -> Any:
        tid = config["configurable"]["thread_id"]
        self.states[tid] = values if isinstance(values, dict) else {"values": values}
        return types.SimpleNamespace(config={"configurable": {"checkpoint_id": "cp-new"}})


@pytest.fixture
def stub_graph(monkeypatch):
    """One stub graph instance shared across the org (cached in app.state.graphs),
    + narrow discover_orgs to ORG. Returns the instance so a test can seed state."""
    g = _StubGraph()
    monkeypatch.setattr("pux_harness.server.build_graph", lambda org, **kw: g)
    monkeypatch.setattr("pux_harness.server.discover_orgs", lambda: [ORG])
    return g


async def _drive(fn):
    async with open_thread_store(db_path=":memory:") as store:
        app.state.saver = store.saver
        app.state.db = store.db
        app.state.store = store
        app.state.graphs = {}
        app.state.base_store = InMemoryStore()
        app.state.runs = {}
        app.state.run_meta = {}
        app.state.mcp = {}
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as httpx_client:
            return await fn(LangGraphClient(httpx_client))


# --- threads.update (PATCH metadata merge) -----------------------------------


def test_thread_update_merges_metadata(stub_graph):
    async def body(c):
        # pux threads are org-scoped (agent_id), which the SDK's threads.create
        # doesn't send — so create via the raw http client, then exercise the SDK
        # update + get paths.
        tid = (await c.http.post("/threads", json={"agent_id": ORG, "metadata": {"a": 1}}))[
            "thread_id"
        ]
        await c.threads.update(tid, metadata={"b": 2})
        return tid, (await c.http.get(f"/threads/{tid}"))

    tid, desc = asyncio.run(_drive(body))
    assert desc["metadata"] == {"a": 1, "b": 2}, desc


def test_thread_update_minimal_returns_no_body(stub_graph):
    async def body(c):
        tid = (await c.http.post("/threads", json={"agent_id": ORG}))["thread_id"]
        return await c.threads.update(tid, metadata={"x": 1}, return_minimal=True)

    assert asyncio.run(_drive(body)) is None  # 204 → SDK decodes empty body → None


# --- threads.copy (fork) -----------------------------------------------------


def test_thread_copy_forks_state(stub_graph):
    async def body(c):
        tid = (await c.http.post("/threads", json={"agent_id": ORG, "metadata": {"k": "v"}}))[
            "thread_id"
        ]
        # seed the source thread's graph state (a real run would); then copy.
        stub_graph.states[tid] = {"messages": [{"role": "user", "content": "hi"}]}
        new = await c.threads.copy(tid)
        return tid, new

    tid, new = asyncio.run(_drive(body))
    # The SDK copy() returns the descriptor (it discards, but we read it here).
    assert new is not None, "copy returned no descriptor"
    assert new["thread_id"] != tid, "copy did not mint a new thread id"
    assert new["agent_id"] == ORG, new
    assert new["metadata"] == {"k": "v"}, new
    assert new["values"] == {"messages": [{"role": "user", "content": "hi"}]}, new


# --- runs.get / runs.delete / runs.join (thread-scoped) ----------------------


async def _make_run(c) -> tuple[str, str]:
    tid = (await c.http.post("/threads", json={"agent_id": ORG}))["thread_id"]
    run = await c.http.post(f"/threads/{tid}/runs", json={"input": "go"})
    return tid, run["run_id"]


def test_run_get_returns_metadata(stub_graph):
    async def body(c):
        tid, rid = await _make_run(c)
        return await c.runs.get(tid, rid)

    run = asyncio.run(_drive(body))
    assert run["run_id"], run
    assert run["thread_id"], run
    assert run["status"] in {"pending", "running", "success", "interrupted", "error"}, run


def test_run_join_blocks_until_done(stub_graph):
    """join is a long-poll GET (NOT POST — the SDK's runs.join sends GET via
    request_reconnect): it blocks until the background run completes, then
    returns the thread's final STATE (the SDK docstring's 'final state of the
    thread'), so the last message content is the run's answer."""

    async def body(c):
        tid, rid = await _make_run(c)
        return await c.runs.join(tid, rid)

    state = asyncio.run(_drive(body))
    assert state["agent_id"] == ORG, state
    assert state["next"] == [], state  # no pending nodes -> run finished
    msgs = state["values"]["messages"]
    assert msgs[-1]["content"] == "stub-answer", state  # the run's answer


def test_run_delete_removes_and_then_get_404s(stub_graph):
    async def body(c):
        tid, rid = await _make_run(c)
        await c.runs.delete(tid, rid)  # 204 → None
        await c.runs.get(tid, rid)  # now 404

    with pytest.raises(NotFoundError):
        asyncio.run(_drive(body))


def test_run_get_unknown_run_404s(stub_graph):
    async def body(c):
        tid = (await c.http.post("/threads", json={"agent_id": ORG}))["thread_id"]
        await c.runs.get(tid, "nope")

    with pytest.raises(NotFoundError):
        asyncio.run(_drive(body))
