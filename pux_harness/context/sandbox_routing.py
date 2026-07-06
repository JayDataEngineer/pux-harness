"""Tool-output sandboxing + routing enforcement middleware (Phase 10).

Intercepts tool calls and enforces routing rules:
- Deny raw network tools (curl, wget, httpie) — suggest sandbox alternatives
- Log all routing decisions to the event store for observability
- Configurable per-org via profile.yaml ``routing:`` block

Our sandbox IS Docker, so the "sandboxed subprocess" aspect is already
handled by PuxSandboxBackend.  This middleware adds the *routing
enforcement* layer — denying commands that bypass the sandbox or produce
excessive output.
"""
from __future__ import annotations

import re
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

from pux_harness.context.events import shared_event_store

# Default deny patterns — commands that should never raw-enter context.
_DEFAULT_DENY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bcurl\b(?!.*-o\b)(?!.*--output\b)(?!.*-s\b)"),
    re.compile(r"\bwget\b(?!.*-q\b)(?!.*--quiet\b)(?!.*-O\b)"),
    re.compile(r"\bhttpie\b|\bhttp\s+GET\b|\bhttp\s+POST\b"),
]

# Tools whose args contain shell commands worth inspecting.
_INTERCEPT_TOOLS = frozenset({"execute", "bash", "pux_sandbox_execute"})

# Guidance message when a command is denied.
_DENY_MSG = (
    "[routing] Command blocked by routing policy. "
    "Use ctx_execute to run code in a sandboxed subprocess, "
    "or use a dedicated tool (e.g. ctx_fetch_and_index for URLs). "
    "Raw network commands are denied to keep context lean."
)


class RoutingMiddleware(AgentMiddleware):
    """Enforce routing rules on tool calls.

    Set ``enabled=False`` to disable (tests, specific orgs).

    Config:
        deny_patterns: compiled regex patterns — if any match the command
            string in an intercepted tool's args, the call is denied.
        intercept_tools: tool names to inspect (default: execute, bash).
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        deny_patterns: list[re.Pattern[str]] | None = None,
        intercept_tools: frozenset[str] | None = None,
    ) -> None:
        self.enabled = enabled
        self.deny_patterns = deny_patterns if deny_patterns is not None else _DEFAULT_DENY_PATTERNS
        self.intercept_tools = intercept_tools or _INTERCEPT_TOOLS

    def _tool_name(self, request: Any) -> str:
        tc = getattr(request, "tool_call", None) or {}
        if isinstance(tc, dict):
            name = tc.get("name")
            return str(name) if name is not None else "tool"
        return "tool"

    def _extract_command(self, request: Any) -> str:
        """Pull the command string from tool args (execute/bash take a 'command' arg)."""
        tc = getattr(request, "tool_call", None) or {}
        if not isinstance(tc, dict):
            return ""
        args = tc.get("args", {})
        if isinstance(args, dict):
            return str(args.get("command", args.get("cmd", "")))
        return ""

    def _is_denied(self, command: str) -> bool:
        """True if any deny pattern matches the command."""
        if not command:
            return False
        return any(p.search(command) for p in self.deny_patterns)

    def _thread_id(self, request: Any) -> str:
        state = getattr(request, "state", None) or {}
        if isinstance(state, dict):
            return state.get("configurable", {}).get("thread_id", "")
        return ""

    # -- sync ------------------------------------------------------------------

    def wrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
        if not self.enabled:
            return handler(request)

        tool = self._tool_name(request)
        if tool not in self.intercept_tools:
            return handler(request)

        command = self._extract_command(request)
        if not self._is_denied(command):
            return handler(request)

        # Denied — log to event store and return error ToolMessage.
        thread_id = self._thread_id(request)
        try:
            shared_event_store().capture(
                "routing_denied",
                {"tool": tool, "command": command[:200]},
                thread_id=thread_id,
            )
            shared_event_store().flush()
        except Exception:
            pass  # never block the agent on observability

        tc = getattr(request, "tool_call", None) or {}
        tc_id = tc.get("id", "") if isinstance(tc, dict) else ""
        return ToolMessage(content=_DENY_MSG, tool_call_id=str(tc_id), name=tool)

    # -- async -----------------------------------------------------------------

    async def awrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
        if not self.enabled:
            return await handler(request)

        tool = self._tool_name(request)
        if tool not in self.intercept_tools:
            return await handler(request)

        command = self._extract_command(request)
        if not self._is_denied(command):
            return await handler(request)

        thread_id = self._thread_id(request)
        try:
            shared_event_store().capture(
                "routing_denied",
                {"tool": tool, "command": command[:200]},
                thread_id=thread_id,
            )
            shared_event_store().flush()
        except Exception:
            pass

        tc = getattr(request, "tool_call", None) or {}
        tc_id = tc.get("id", "") if isinstance(tc, dict) else ""
        return ToolMessage(content=_DENY_MSG, tool_call_id=str(tc_id), name=tool)
