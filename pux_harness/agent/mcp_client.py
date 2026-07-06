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

import logging
import os
from typing import Sequence

from langchain_core.tools import BaseTool

from pux_harness.agent.tool_servers import ToolServerSpec

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

    async def open(self) -> None:
        """Probe every spec's server and load tools.

        A server that is unreachable / fails ``tools/list`` → zero tools from
        that server + a loud ``ERROR`` log. Does NOT crash the org."""
        if not self.specs:
            return

        connections: dict[str, dict] = {}
        for spec in self.specs:
            try:
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

        try:
            server_tools: dict[str, list[BaseTool]] = await client.get_tools()
        except Exception as exc:
            logger.error(
                "org %s: get_tools() failed for all servers — %s",
                self.org, exc,
            )
            return

        for spec in self.specs:
            exposed = server_tools.get(spec.name, [])
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
