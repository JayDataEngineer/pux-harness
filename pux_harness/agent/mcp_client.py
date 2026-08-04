"""Async bridge from resolved ToolServerSpec → langchain BaseTool list.

The ``McpSessionManager`` constructs a ``MultiServerMCPClient`` from a list of
resolved ``ToolServerSpec``, calls ``get_tools()`` to probe every server and
load their tools, applies each spec's allowlist (fail-loud on missing names),
and namespaces every tool as ``mcp__<server>__<tool>``.

A server that is unreachable or fails ``tools/list`` contributes NO tools +
a loud ``ERROR`` log — the org still starts (one bad foreign server shouldn't
brick the agent).

Usage::

    mgr = McpSessionManager("my-org", specs)
    await mgr.open()
    agent_tools = mgr.tools  # list[BaseTool]
    ...
    await mgr.close()
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import replace
from typing import Sequence

from langchain_core.tools import BaseTool

from pux_harness.agent.tool_servers import ToolServerSpec, resolve_tool_servers

logger = logging.getLogger(__name__)


def _to_connection(spec: ToolServerSpec) -> dict:
    """Build a per-spec connection dict for ``MultiServerMCPClient``.

    ``MultiServerMCPClient`` connections dict expects:
    - ``url`` + ``headers`` for remote (SSE / Streamable HTTP)
    - ``command`` + ``args`` + ``env`` for stdio

    The library handles expansion of ``${VAR}`` in ``env`` values for stdio;
    we already expanded everything in ``_substitute_spec``, so values are
    ready to pass through.
    """
    if spec.transport == "stdio":
        env = dict(os.environ)
        env.update(spec.env)
        return {
            "command": spec.command,
            "args": list(spec.args),
            "env": env,
            "transport": "stdio",
        }
    elif spec.transport == "sse":
        return {
            "url": spec.url,
            "headers": dict(spec.headers),
            "transport": "sse",
        }
    elif spec.transport == "http":
        return {
            "url": spec.url,
            "headers": dict(spec.headers),
            "transport": "streamable_http",
            # The MCP streamable_http transport defaults to a 30-second general
            # timeout (streamablehttp_client signature).  Tool calls that block
            # longer than that — e.g. Ray generate() for H3 video or TRELLIS 3D,
            # which take minutes — see the httpx client closed mid-request and
            # fail with "Cannot send a request, as the client has been closed."
            # Bump to 600 s (10 min) for both the general and SSE-read timers.
            "timeout": 600,
            "sse_read_timeout": 600,
        }
    else:
        raise ValueError(f"unsupported transport {spec.transport!r}")


def _apply_allowlist(
    tools: list[BaseTool], spec: ToolServerSpec,
) -> list[BaseTool]:
    """Filter tools by the spec's allowlist. ``None`` → take everything (with
    an INFO log). Fails loud if any allowlist name is missing from the server's
    exposed tools."""
    if spec.tools is None:
        logger.info(
            "tool server %r: no allowlist set — loading all %d exposed tools",
            spec.name, len(tools),
        )
        return list(tools)
    by_name = {t.name: t for t in tools}
    missing = [n for n in spec.tools if n not in by_name]
    if missing:
        raise ValueError(
            f"tool server {spec.name!r}: allowlist names {missing} not "
            f"exposed by server (exposed: {sorted(by_name)})"
        )
    return [by_name[n] for n in spec.tools]


def _namespace_tools(
    tools: list[BaseTool], server_name: str,
) -> list[BaseTool]:
    """Rename each tool to ``mcp__<server>__<tool>``."""
    renamed: list[BaseTool] = []
    for t in tools:
        new_name = f"mcp__{server_name}__{t.name}"
        new_t = t.model_copy(update={"name": new_name})
        renamed.append(new_t)
    return renamed


class McpSessionManager:
    """Manages a set of foreign MCP server connections for one org.

    Construct with the org name + resolved specs, then ``await open()`` to
    probe servers and load tools. ``tools`` is a flat list of namespaced
    ``BaseTool`` ready for the supervisor stack.

    ``close()`` is a best-effort no-op for remote servers (each tool call opens
    a fresh per-call session). For stdio the per-call sessions self-close.
    Kept for symmetry and future stdio subprocess reaping.
    """

    def __init__(
        self, org: str, specs: Sequence[ToolServerSpec],
    ) -> None:
        self.org = org
        self.specs = list(specs)
        self._client = None
        self._tools: list[BaseTool] = []

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools)

    @staticmethod
    async def _maybe_bootstrap(spec: ToolServerSpec) -> ToolServerSpec:
        """Resolve a stdio server's binary BEFORE building its connection.

        Specs without a ``github:`` block (or non-stdio) pass through untouched
        — the operator's ``command`` must already be on PATH (or the probe fails
        loud at ``tools/list``, the pre-existing behavior). For a stdio spec WITH
        a ``github:`` block, ``ensure_server`` makes the binary present (PATH →
        cache → release-fallback download) and we return a COPY with
        ``command`` rewritten to the resolved path. The download runs in a
        thread (no event-loop blocking); a bootstrap failure raises ValueError,
        which the caller's ``except`` maps to the existing per-server skip +
        ERROR log — a broken download is indistinguishable from a broken server.
        """
        if spec.transport != "stdio" or not spec.github:
            return spec
        from pux_harness.agent.mcp_bootstrap import ensure_server  # noqa: PLC0415
        path = await asyncio.to_thread(ensure_server, spec)
        if path is None:
            raise ValueError(
                f"github bootstrap could not resolve binary "
                f"{spec.github['binary']!r} for server {spec.name!r} (not on "
                f"PATH, not in the .pux cache, and release fetch failed)"
            )
        if str(path) == spec.command:
            return spec
        return replace(spec, command=str(path))

    async def open(self) -> None:
        """Probe every spec's server and load tools.

        Each server is probed INDEPENDENTLY: a server that is unreachable or
        fails ``tools/list`` → zero tools from that server + a loud ``ERROR``
        log. Does NOT crash the org (one bad foreign server can't brick the
        batch).

        ``MultiServerMCPClient.get_tools()`` returns a FLAT list across all
        servers (no per-server bucketing), so we call it once per spec with
        ``server_name=`` to attribute tools to the right server for the
        allowlist + namespace. A bare ``get_tools()`` would ``gather`` every
        server and raise on the FIRST failure — defeating the per-server
        isolation this method promises. Proven live (2026-07-06): the old code
        read ``server_tools.get(...)`` on what is a ``list``, crashing
        ``AttributeError`` for every live server — caught only by a live
        handshake, never by the mocked unit tests."""
        if not self.specs:
            return

        connections: dict[str, dict] = {}
        for spec in self.specs:
            try:
                spec = await self._maybe_bootstrap(spec)
                connections[spec.name] = _to_connection(spec)
            except ValueError as exc:
                logger.error(
                    "org %s, tool server %r: skipped — %s",
                    self.org, spec.name, exc,
                )

        if not connections:
            return

        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(
            connections=connections,
            tool_name_prefix=False,
            handle_tool_errors=True,
        )
        self._client = client

        for spec in self.specs:
            if spec.name not in connections:
                continue  # already logged at _to_connection
            try:
                exposed = await client.get_tools(server_name=spec.name)
            except Exception as exc:
                logger.error(
                    "org %s, tool server %r: tools/list failed — %s",
                    self.org, spec.name, exc,
                )
                continue
            if not exposed:
                logger.error(
                    "org %s, tool server %r: no tools loaded — skipped",
                    self.org, spec.name,
                )
                continue
            try:
                filtered = _apply_allowlist(exposed, spec)
                renamed = _namespace_tools(filtered, spec.name)
                self._tools.extend(renamed)
                logger.info(
                    "org %s, tool server %r: loaded %d tools",
                    self.org, spec.name, len(renamed),
                )
            except ValueError as exc:
                logger.error(
                    "org %s, tool server %r: allowlist error — %s",
                    self.org, spec.name, exc,
                )

    async def close(self) -> None:
        """Best-effort close. No-op for remote today."""
        _ = self._client


async def open_org_mcp(org: str, *, timeout: float = 30.0) -> list[BaseTool]:
    """Resolve ``org``'s foreign MCP servers + open them, returning namespaced
    ``BaseTool`` ready for the supervisor stack. The single source of truth for
    the resolve→open dance shared by all three lanes (``pux direct``,
    ``pux acp``, and the Aegra production factory in ``runtime/upstream.py``).

    Failures are PER-ORG isolated and degrade to zero tools — the agent still
    starts. ``McpSessionManager.open()`` already isolates per-**server** (one
    bad server → zero tools from it); this adds a per-org wrapper so one bad
    org's resolution/timeout can't abort a lane that builds many orgs.

    * ``resolve_tool_servers`` raising ``ValueError`` (an unset ``${VAR}``) →
      ``[]`` (the operator's env is missing a required var; loud ``ERROR`` log).
    * ``open()`` taking longer than ``timeout`` → ``[]`` (a hung server probe
      can't wedge the whole startup).
    * any other ``Exception`` from open → ``[]``.

    No ``close()`` is taken on the manager — verified no-op, and each loaded
    tool opens a fresh per-call session on whatever loop later invokes it (see
    ``MultiServerMCPClient.get_tools`` docstring: "A new session will be
    created for each tool call"), so the loop the probe ran on is irrelevant.
    """
    try:
        specs = resolve_tool_servers(org)
    except ValueError as exc:
        logger.error("org %s: tool_servers resolution failed — %s", org, exc)
        return []
    if not specs:
        return []
    mgr = McpSessionManager(org, specs)
    try:
        await asyncio.wait_for(mgr.open(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(
            "org %s: MCP open() timed out after %.1fs — starting with no MCP tools",
            org, timeout,
        )
        return []
    except Exception as exc:  # noqa: BLE001 — per-org degrade, don't brick the lane
        logger.error("org %s: MCP open() failed — %s", org, exc)
        return []
    return list(mgr.tools)
