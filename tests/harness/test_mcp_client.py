"""McpSessionManager — the bootstrap wiring into open() (Phase C).

Phase B proved ``ensure_server`` in isolation; this file proves the WIRING: that
``open()`` calls ``_maybe_bootstrap`` BEFORE ``_to_connection``, rewrites the
stdio ``command`` to the resolved (cached/PATH) binary, and leaves specs without
a ``github:`` block untouched. The ``MultiServerMCPClient`` is faked so the test
captures the connection dict the bootstrap produced — no real subprocess, no real
MCP handshake (that's Phase E's live proof).
"""
from __future__ import annotations

import asyncio
import os

import pytest

from pux_harness.agent import mcp_bootstrap as mb
from pux_harness.agent.mcp_client import McpSessionManager
from pux_harness.agent.tool_servers import ToolServerSpec


def _spec(*, command="github-mcp-server", github=None, transport="stdio",
          name="github") -> ToolServerSpec:
    return ToolServerSpec(
        name=name, kind="mcp", transport=transport, command=command,
        args=["stdio"], env={}, github=github,
    )


def _gh():
    return {
        "repo": "github/github-mcp-server",
        "asset": "github-mcp-server_*{os}*{arch}*.tar.gz",
        "binary": "github-mcp-server",
        "version": "latest",
    }


class _FakeTool:
    """Stand-in for a BaseTool — ``_namespace_tools`` reads .name + .model_copy."""
    def __init__(self, name: str):
        self.name = name

    def model_copy(self, *, update=None):  # type: ignore[override]
        new = _FakeTool(self.name)
        if update:
            new.name = update.get("name", self.name)
        return new


def _patch_mcp_client(monkeypatch, capture: dict):
    """Replace ``MultiServerMCPClient`` with a fake that records its connections
    and returns one canned tool per ``get_tools`` call."""
    import langchain_mcp_adapters.client as _lmc

    class _FakeClient:
        def __init__(self, *, connections, **_kwargs):
            capture["connections"] = connections

        async def get_tools(self, *, server_name=None):
            return [_FakeTool("ping")]

    monkeypatch.setattr(_lmc, "MultiServerMCPClient", _FakeClient)


# --- _maybe_bootstrap (the seam) --------------------------------------------

def test_maybe_bootstrap_passthrough_non_stdio():
    spec = _spec(transport="http", github=_gh())
    out = asyncio.run(McpSessionManager._maybe_bootstrap(spec))
    assert out is spec  # untouched


def test_maybe_bootstrap_passthrough_no_github_block():
    spec = _spec(transport="stdio", github=None, command="some-bin")
    out = asyncio.run(McpSessionManager._maybe_bootstrap(spec))
    assert out is spec


def test_maybe_bootstrap_rewrites_command_to_cached_path(monkeypatch, tmp_path):
    spec = _spec(github=_gh())
    cached = (tmp_path / "github-mcp-server")
    cached.write_bytes(b"#!/bin/sh\n")
    # ensure_server returns the cached path.
    monkeypatch.setattr(mb, "ensure_server", lambda s: cached)
    out = asyncio.run(McpSessionManager._maybe_bootstrap(spec))
    assert out is not spec            # a COPY, not the input mutated
    assert out.command == str(cached)
    assert spec.command == "github-mcp-server"  # input unmutated


def test_maybe_bootstrap_failure_raises_valueerror(monkeypatch):
    spec = _spec(github=_gh())
    monkeypatch.setattr(mb, "ensure_server", lambda s: None)
    with pytest.raises(ValueError, match="could not resolve binary"):
        asyncio.run(McpSessionManager._maybe_bootstrap(spec))


# --- open() wiring -----------------------------------------------------------

def test_open_bootstraps_and_rewrites_command(monkeypatch, tmp_path):
    """A pre-cached binary → the connection built for the github spec carries
    the CACHED path as ``command`` (the bootstrap ran before _to_connection),
    and the tool flows through namespacing."""
    spec = _spec(github=_gh())
    cached = (tmp_path / ".pux" / "mcp-servers" / "github" / "latest"
              / "github-mcp-server")
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"#!/bin/sh\n")
    os.chmod(cached, 0o755)
    monkeypatch.setattr(mb, "project_root", lambda: tmp_path)
    monkeypatch.setattr(mb.shutil, "which", lambda c: None)

    capture: dict = {}
    _patch_mcp_client(monkeypatch, capture)

    mgr = McpSessionManager("general", [spec])
    asyncio.run(mgr.open())

    assert capture["connections"]["github"]["command"] == str(cached)
    assert any(t.name == "mcp__github__ping" for t in mgr.tools)


def test_open_leaves_non_github_stdio_command_alone(monkeypatch):
    """A stdio spec WITHOUT a github block is never bootstrapped — its command
    reaches _to_connection verbatim (the pre-existing PATH-or-fail behavior)."""
    spec = _spec(transport="stdio", github=None, command="some-bin", name="x")
    capture: dict = {}
    _patch_mcp_client(monkeypatch, capture)

    mgr = McpSessionManager("o", [spec])
    asyncio.run(mgr.open())

    assert capture["connections"]["x"]["command"] == "some-bin"


def test_open_bootstrap_failure_skips_server_not_bricks(monkeypatch):
    """A bootstrap that can't resolve the binary → the server is SKIPPED (logged
    ERROR, zero tools) but the OTHER server in the batch still loads (the
    per-server isolation the wiring must preserve)."""
    good = _spec(transport="stdio", github=None, command="some-bin", name="good")
    bad = _spec(github=_gh(), name="bad")
    monkeypatch.setattr(mb, "ensure_server", lambda s: None)  # bad can't resolve

    capture: dict = {}
    _patch_mcp_client(monkeypatch, capture)

    mgr = McpSessionManager("o", [bad, good])
    asyncio.run(mgr.open())

    # good made it into connections + yielded its tool; bad was dropped.
    assert "good" in capture["connections"]
    assert "bad" not in capture["connections"]
    assert any(t.name == "mcp__good__ping" for t in mgr.tools)
    assert not any(t.name.startswith("mcp__bad__") for t in mgr.tools)
