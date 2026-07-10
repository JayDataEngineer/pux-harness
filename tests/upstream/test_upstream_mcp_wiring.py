"""Unit proof: the Aegra production factory resolves + threads MCP tools.

The keystone blocker (prod built every org with ``mcp_tools=()``) is fixed in
``pux_harness.runtime.upstream.make_graph``: it is ``async``, self-resolves each
org's foreign MCP servers via ``open_org_mcp`` (the single canonical
resolve→open helper), caches per-org, and passes the tools into ``build_graph``.

This is a UNIT proof at the ``build_graph`` boundary — no Docker, no real MCP
server, no model. Heavy deps (``build_graph``'s Docker/specialist surface, the
real ``McpSessionManager.open`` probe) are stubbed, mirroring
``tests/contract/test_real_orgs_build.py``. A live end-to-end proof (real
http MCP server reached through a built graph) lives in
``test_aegra_lane_contract.py`` and is env-gated.

Governing context: [[rely-on-upstream]], [[no-fallbacks-no-aliases]].
"""

from __future__ import annotations

import asyncio
import time

import pytest
from langchain_core.tools import BaseTool


def _sentinel_tool(name: str) -> BaseTool:
    """A minimal real ``BaseTool`` we can assert by name in the captured kwarg."""
    from langchain_core.tools import tool

    @tool(name)
    def _probe() -> str:
        """Sentinel probe tool used only to assert name-based threading."""
        return "probe"

    return _probe


def _fake_spec() -> object:
    """A stand-in ``ToolServerSpec`` — only identity matters to the stubs."""
    return object()


@pytest.fixture
def clear_cache():
    """The per-org MCP cache is module-global; isolate each test from it."""
    import pux_harness.runtime.upstream as upstream

    upstream._MCP_CACHE.clear()
    yield
    upstream._MCP_CACHE.clear()


@pytest.fixture
def stub_build_graph(monkeypatch):
    """Stub the lazy ``build_graph`` import so the runtime branch doesn't build
    Docker/specialist graphs; capture the ``mcp_tools`` kwarg it receives."""
    import pux_harness.agent.graph as graph

    captured: dict = {}

    def _fake_build_graph(
        org,
        checkpointer=None,
        store=None,
        facts=None,
        mcp_tools=(),
    ):
        captured["org"] = org
        captured["mcp_tools"] = list(mcp_tools)
        # A lightweight stand-in for a CompiledStateGraph (never executed here).
        return {"org": org, "mcp_tools": captured["mcp_tools"]}

    monkeypatch.setattr(graph, "build_graph", _fake_build_graph)
    return captured


def test_make_graph_threads_mcp_into_build_graph(clear_cache, stub_build_graph, monkeypatch):
    """Runtime branch: MCP tools resolved via open_org_mcp reach build_graph."""
    import pux_harness.agent.mcp_client as mcp_client
    import pux_harness.runtime.upstream as upstream

    sentinel = _sentinel_tool("mcp__fake__probe")

    async def _fake_open(org, *, timeout=30.0):
        return [sentinel]

    monkeypatch.setattr(mcp_client, "open_org_mcp", _fake_open)
    monkeypatch.setenv("PUX_UPSTREAM_GRAPH", "runtime")

    graph = asyncio.run(upstream.make_graph("general"))
    assert graph["org"] == "general"
    names = [t.name for t in stub_build_graph["mcp_tools"]]
    assert "mcp__fake__probe" in names


def test_resolve_valueerror_degrades_to_empty(clear_cache, stub_build_graph, monkeypatch):
    """An unset ${VAR} (ValueError from resolve_tool_servers) -> mcp_tools=(),
    no exception — per-org degrade, never aborts the lane."""
    import pux_harness.agent.mcp_client as mcp_client
    import pux_harness.runtime.upstream as upstream

    def _raise(_org):
        raise ValueError("GITHUB_TOKEN is unset")

    monkeypatch.setattr(mcp_client, "resolve_tool_servers", _raise)
    monkeypatch.setenv("PUX_UPSTREAM_GRAPH", "runtime")

    # No exception should escape.
    graph = asyncio.run(upstream.make_graph("general"))
    assert graph["org"] == "general"
    assert stub_build_graph["mcp_tools"] == []


def test_open_org_mcp_times_out_to_empty(monkeypatch):
    """A hung server probe (open() slow) must not wedge startup — timeout
    degrades to [] within the budget."""

    import pux_harness.agent.mcp_client as mcp_client

    monkeypatch.setattr(mcp_client, "resolve_tool_servers", lambda _org: [_fake_spec()])

    async def _slow_open(self):  # pragma: no cover - intentionally never returns in time
        await asyncio.sleep(30)

    monkeypatch.setattr(mcp_client.McpSessionManager, "open", _slow_open)

    t0 = time.monotonic()
    out = asyncio.run(mcp_client.open_org_mcp("x", timeout=0.1))
    elapsed = time.monotonic() - t0
    assert out == []
    assert elapsed < 5.0  # returned promptly, not after the 30s sleep


def test_make_graph_caches_open_once(clear_cache, stub_build_graph, monkeypatch):
    """Two make_graph calls for the same org open MCP exactly once (cached)."""
    import pux_harness.agent.mcp_client as mcp_client
    import pux_harness.runtime.upstream as upstream

    sentinel = _sentinel_tool("mcp__fake__probe")
    calls = {"n": 0}

    async def _counting_open(org, *, timeout=30.0):
        calls["n"] += 1
        return [sentinel]

    monkeypatch.setattr(mcp_client, "open_org_mcp", _counting_open)
    monkeypatch.setenv("PUX_UPSTREAM_GRAPH", "runtime")

    asyncio.run(upstream.make_graph("general"))
    asyncio.run(upstream.make_graph("general"))
    assert calls["n"] == 1
