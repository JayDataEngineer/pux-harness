"""Tool-output sandboxing + routing enforcement middleware.

Intercepts tool calls and enforces routing rules:
- Deny raw network tools (curl, wget, httpie) — suggest sandbox alternatives
- Redirect declared-script exec — a script exposed as a typed
  ``pux_sandbox_*`` tool must be called via that tool, not raw ``execute``
  (the exec-guard: declaring a tool TAKES the script out of the agent's exec
  surface, so context carries ONE representation of the capability)
- Log all routing decisions to the event store for observability
- Configurable per-org via profile.yaml ``routing:`` block

Our sandbox IS Docker, so the "sandboxed subprocess" aspect is already
handled by PuxSandboxBackend.  This middleware adds the *routing
enforcement* layer — denying commands that bypass the sandbox or produce
excessive output.

The exec-guard seam: agent-via-``execute`` is a TOOL CALL (intercepted here);
a declared tool's own ``func`` calls ``exec_client.exec(cmd)`` DIRECTLY (not a
tool call, never seen here). So redirecting declared-script exec does NOT
break the declared tool's own in-container execution.
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

# Guidance message when a command targets a declared script. ``{tool}`` is the
# typed ``pux_sandbox_*`` name the agent should call instead.
_REDIRECT_MSG = (
    "[routing] This command targets a script exposed as the typed tool `{tool}`. "
    "Call `{tool}(...)` directly instead — it is typed, schema-validated, runs "
    "in-container, and is audited. Raw exec of declared scripts is blocked to "
    "keep context lean."
)


class RoutingMiddleware(AgentMiddleware):
    """Enforce routing rules on tool calls.

    Set ``enabled=False`` to disable (tests, specific orgs).

    Config:
        deny_patterns: compiled regex patterns — if any match the command
            string in an intercepted tool's args, the call is denied.
        intercept_tools: tool names to inspect (default: execute, bash).
        declared_redirects: ``(pattern, target_tool)`` pairs compiled from the
            org's declared tools (``declared.build_script_redirects``). If a
            pattern matches the command, the call is REDIRECTED — returns a
            ToolMessage naming ``target_tool`` instead of running the command.
            Default ``[]`` → byte-identical behavior for orgs that declare
            nothing. The deny check runs first (network egress wins over a
            declared-script redirect when a compound command matches both).
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        deny_patterns: list[re.Pattern[str]] | None = None,
        intercept_tools: frozenset[str] | None = None,
        declared_redirects: list[tuple[re.Pattern[str], str]] | None = None,
    ) -> None:
        self.enabled = enabled
        self.deny_patterns = deny_patterns if deny_patterns is not None else _DEFAULT_DENY_PATTERNS
        self.intercept_tools = intercept_tools or _INTERCEPT_TOOLS
        self.declared_redirects: list[tuple[re.Pattern[str], str]] = (
            declared_redirects if declared_redirects is not None else []
        )

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

    def _declared_redirect(self, command: str) -> str | None:
        """The typed tool name to redirect to if ``command`` targets a declared
        script, else ``None``. First match wins; declared redirects are disjoint
        per-``(script, subcommand)`` in practice (two tools exposing the same
        script+subcommand would be a config smell caught upstream)."""
        if not command:
            return None
        for pattern, target in self.declared_redirects:
            if pattern.search(command):
                return target
        return None

    def _thread_id(self, request: Any) -> str:
        state = getattr(request, "state", None) or {}
        if isinstance(state, dict):
            return state.get("configurable", {}).get("thread_id", "")
        return ""

    def _respond(
        self,
        request: Any,
        tool: str,
        command: str,
        *,
        event_type: str,
        content: str,
        extra: dict[str, Any] | None = None,
    ) -> ToolMessage:
        """Log a routing decision to the event store + return a synthetic
        ``ToolMessage`` that REPLACES the tool's real output (the command never
        runs). Shared by the deny path (network egress) and the redirect path
        (declared script). Observability is best-effort — never block the agent
        on it (a capture/flush failure is swallowed)."""
        thread_id = self._thread_id(request)
        payload: dict[str, Any] = {"tool": tool, "command": command[:200]}
        if extra:
            payload.update(extra)
        try:
            shared_event_store().capture(event_type, payload, thread_id=thread_id)
            shared_event_store().flush()
        except Exception:
            pass  # never block the agent on observability
        tc = getattr(request, "tool_call", None) or {}
        tc_id = tc.get("id", "") if isinstance(tc, dict) else ""
        return ToolMessage(content=content, tool_call_id=str(tc_id), name=tool)

    # -- sync ------------------------------------------------------------------

    def wrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
        if not self.enabled:
            return handler(request)

        tool = self._tool_name(request)
        if tool not in self.intercept_tools:
            return handler(request)

        command = self._extract_command(request)
        # Deny first (network egress wins over a declared-script redirect when
        # a compound command matches both), then redirect, else allow.
        if self._is_denied(command):
            return self._respond(request, tool, command,
                                 event_type="routing_denied", content=_DENY_MSG)
        redirect = self._declared_redirect(command)
        if redirect is not None:
            return self._respond(request, tool, command,
                                 event_type="routing_redirected",
                                 content=_REDIRECT_MSG.format(tool=redirect),
                                 extra={"target": redirect})
        return handler(request)

    # -- async -----------------------------------------------------------------

    async def awrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
        if not self.enabled:
            return await handler(request)

        tool = self._tool_name(request)
        if tool not in self.intercept_tools:
            return await handler(request)

        command = self._extract_command(request)
        if self._is_denied(command):
            return self._respond(request, tool, command,
                                 event_type="routing_denied", content=_DENY_MSG)
        redirect = self._declared_redirect(command)
        if redirect is not None:
            return self._respond(request, tool, command,
                                 event_type="routing_redirected",
                                 content=_REDIRECT_MSG.format(tool=redirect),
                                 extra={"target": redirect})
        return await handler(request)
