"""Tool-output sandboxing + routing enforcement middleware.

Intercepts tool calls and enforces routing rules:
- Deny raw network tools (curl, wget, httpie, requests.get, urllib, httpx) —
  UNIVERSAL: fires on EVERY tool call, not just execute/bash. This is the
  parity surface with context-mode's PreToolUse hook. A declared dynamic tool,
  an MCP tool, or any future tool whose string args carry a network-fetch
  pattern gets denied the same as ``execute("curl ...")`` would.
- Redirect declared-script exec — a script exposed as a typed
  ``pux_sandbox_*`` tool must be called via that tool, not raw ``execute``
  (the exec-guard: declaring a tool TAKES the script out of the agent's exec
  surface, so context carries ONE representation of the capability). Scoped
  to exec-shaped tools only (``intercept_tools``).
- Log all routing decisions to the event store for observability
- Configurable per-org via profile.yaml ``routing:`` block

Our sandbox IS Docker, so the "sandboxed subprocess" aspect is already
handled by BaseSandbox.  This middleware adds the *routing
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

# Default SHELL deny patterns — network egress that dumps a response body into
# context. Matched universally against ALL string args of ANY tool call. The
# positive lookahead ``(?=\s+\S)`` requires curl/wget/httpie to be followed by
# space + content (a URL or flag), so the BARE word ``curl`` in a non-exec arg
# (e.g. ``grep("curl")`` or documentation text) does NOT false-positive.
# The negative lookahead preserves the ``-s``/``-o`` exemption: a curl that
# writes to a file (silent + output flags) is allowed through.
_DEFAULT_DENY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bcurl(?=\s+\S)(?!.*(?:-o\b|--output\b|-s\b))"),
    re.compile(r"\bwget(?=\s+\S)(?!.*(?:-q\b|--quiet\b|-O\b))"),
    re.compile(r"\bhttpie\s+\S|\bhttp\s+(?:GET|POST)\s+\S"),
]

# Default PYTHON deny patterns — ``requests.get(url)`` / ``urllib.request.
# urlopen(url)`` / ``httpx.get(url)`` / subprocess-wrapping-curl. Matched
# universally against ALL string args of ANY tool. These patterns are specific
# enough (``requests.get(`` is unambiguous) that false positives are rare; the
# one known tradeoff is grepping for a code pattern that literally contains
# ``requests.get(`` — that IS denied (parity with context-mode).
_DEFAULT_PY_DENY_PATTERNS: list[re.Pattern[str]] = [
    # requests: requests.get(url), requests.post(url), requests.request("GET", url)
    re.compile(r"\brequests\.(?:get|post|put|patch|delete|head|request)\s*\("),
    # urllib.request.urlopen(url) / urlretrieve(url, path)
    re.compile(r"\burllib\.request\.urlopen\s*\("),
    re.compile(r"\burllib\.request\.urlretrieve\s*\("),
    # bare urlopen() after ``from urllib.request import urlopen``
    re.compile(r"\burlopen\s*\("),
    # httpx: httpx.get(url), httpx.Client(), httpx.AsyncClient()
    re.compile(r"\bhttpx\.(?:get|post|put|patch|delete|head|request|Client|AsyncClient)\s*\("),
    # subprocess / os wrapping curl or wget (the shell deny bypass via Python)
    re.compile(
        r"\b(?:subprocess\.(?:run|call|check_output|Popen)|os\.system|os\.popen)\s*"
        r"\([^)]*\b(?:curl|wget)\b"
    ),
]

# Tools eligible for the DECLARED-REDIRECT path. The deny path is universal
# (every tool scanned); redirect only makes sense for exec-shaped tools whose
# ``command``/``cmd`` arg targets a declared script. A pux_sandbox_python or
# MCP tool targeting a declared script is nonsensical and not redirected.
_INTERCEPT_TOOLS = frozenset({"execute", "bash", "pux_sandbox_execute"})

# Guidance message when a SHELL command is denied (curl/wget/httpie).
_DENY_MSG = (
    "[routing] Command blocked by routing policy. "
    "Use ctx_execute to run code in a sandboxed subprocess, "
    "or use a dedicated tool (e.g. ctx_fetch_and_index for URLs). "
    "Raw network commands are denied to keep context lean."
)

# Guidance message when PYTHON code is denied (requests/urllib/httpx).
_PY_DENY_MSG = (
    "[routing] Python network call blocked by routing policy. "
    "Fetching a URL dumps the response body into context. "
    "Use ctx_execute to fetch inside the sandbox (only stdout enters context), "
    "or ctx_fetch_and_index to fetch + index in one step. "
    "Raw requests.get / urllib.request.urlopen / httpx.get are denied."
)

# Tools EXEMPT from the deny — their design ALREADY keeps output out of context
# (they stash/index internally via the EventStore). The deny would be a false
# positive on these tools: e.g. ctx_batch_execute running ``curl -s ...`` is the
# SANCTIONED fetch-into-store path (output auto-indexed), not raw context-dumping.
# ctx_execute running ``requests.get(...).text`` returns only stdout. These ARE
# the context-saving surface — denying network inside them defeats their purpose.
#
# This is NOT the allowlist hack (where uninspected tools bypassed deny). The deny
# is still universal — it fires on every tool NOT in this set. The set is narrow
# (4 named tools) and semantically justified: each one guarantees context-safe
# output handling by design.
_DENY_EXEMPT_TOOLS = frozenset({
    "ctx_execute",
    "ctx_execute_file",
    "ctx_batch_execute",
    "ctx_fetch_and_index",
})

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

    The DENY path is UNIVERSAL: every tool call's string args are concatenated
    and pattern-matched against ``deny_patterns`` (shell: curl/wget/httpie) and
    ``py_deny_patterns`` (Python: requests/urllib/httpx). There is no allowlist
    gate — a declared dynamic tool, an MCP tool, or any future tool is scanned
    the same as ``execute``. This is the parity surface with context-mode's
    PreToolUse hook.

    The DECLARED-REDIRECT path is SCOPED to ``intercept_tools`` (exec-shaped
    tools only): a script exposed as ``pux_sandbox_<name>`` must be called via
    that typed tool, not raw ``execute``. Redirecting a ``pux_sandbox_python``
    or MCP tool to a declared script is nonsensical and never fires.

    Config:
        deny_patterns: compiled regex patterns — if any match ANY string arg of
            ANY tool, the call is denied. Tightened so the bare word ``curl``
            (e.g. in a grep arg) does not false-positive — the pattern requires
            curl/wget to be followed by space + content (a URL or flag).
        py_deny_patterns: compiled regex patterns for Python network egress
            (requests/urllib/httpx/subprocess-wraps-curl). Checked against all
            string args of all tools.
        intercept_tools: tool names eligible for the declared-redirect path
            ONLY (default: execute, bash, pux_sandbox_execute). The deny path
            ignores this — it is universal.
        declared_redirects: ``(pattern, target_tool)`` pairs compiled from the
            org's declared tools. If a pattern matches the ``command`` arg of
            an intercept-eligible tool, the call is REDIRECTED. Default ``[]``.
            Deny wins over redirect when a compound command matches both.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        deny_patterns: list[re.Pattern[str]] | None = None,
        py_deny_patterns: list[re.Pattern[str]] | None = None,
        intercept_tools: frozenset[str] | None = None,
        declared_redirects: list[tuple[re.Pattern[str], str]] | None = None,
    ) -> None:
        self.enabled = enabled
        self.deny_patterns = deny_patterns if deny_patterns is not None else _DEFAULT_DENY_PATTERNS
        self.py_deny_patterns = (
            py_deny_patterns if py_deny_patterns is not None else _DEFAULT_PY_DENY_PATTERNS
        )
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

    def _extract_scan(self, request: Any) -> str:
        """Concatenate ALL string-valued tool args into one inspection string.

        UNIVERSAL — every tool call is scanned, not just execute/bash. This is
        the parity surface: a declared tool (``pux_sandbox_*``), an MCP tool,
        or any future tool whose string args carry a curl/requests.get pattern
        gets denied the same as ``execute("curl ...")`` would. Non-string args
        (ints, bools, lists, nested objects) are skipped — only string-valued
        args can carry code/command patterns.
        """
        tc = getattr(request, "tool_call", None) or {}
        if not isinstance(tc, dict):
            return ""
        args = tc.get("args", {})
        if not isinstance(args, dict):
            return ""
        return "\n".join(str(v) for v in args.values() if isinstance(v, str))

    def _extract_command(self, request: Any) -> str:
        """Pull the command string from a SHELL tool's args (command/cmd).

        Used ONLY by the declared-redirect path, which fires for exec-shaped
        tools in ``intercept_tools``. The deny path uses ``_extract_scan``
        (universal — all string args of all tools).
        """
        tc = getattr(request, "tool_call", None) or {}
        if not isinstance(tc, dict):
            return ""
        args = tc.get("args", {})
        if isinstance(args, dict):
            return str(args.get("command", args.get("cmd", "")))
        return ""

    def _deny_tag(self, scan: str, *, tool: str = "") -> str:
        """Reason tag if ``scan`` matches a deny list, else ``""``.

        UNIVERSAL — both pattern lists are checked against the full scan string
        (all string args of any tool). Python patterns checked first (they're
        the most specific — ``requests.get(`` is unambiguous network egress
        regardless of which tool carries it). Then shell patterns. If a shell
        pattern matches inside ``pux_sandbox_python`` (e.g. ``pty.spawn("curl
        ...")`` — a Python pattern we don't cover), tag ``"python"`` anyway so
        the message is actionable for someone writing Python, not shell.

        Returns ``"python"``, ``"shell"``, or ``""``.
        """
        if not scan:
            return ""
        if any(p.search(scan) for p in self.py_deny_patterns):
            return "python"
        if any(p.search(scan) for p in self.deny_patterns):
            return "python" if tool == "pux_sandbox_python" else "shell"
        return ""

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
        scan = self._extract_scan(request)

        # UNIVERSAL deny — fires on EVERY tool call EXCEPT the 4 context-saving
        # tools in _DENY_EXEMPT_TOOLS (ctx_execute / ctx_execute_file /
        # ctx_batch_execute / ctx_fetch_and_index). Those tools already keep
        # output out of context by design (stash/index); denying network inside
        # them would defeat their purpose. Every OTHER tool is scanned. This is
        # the parity surface: context-mode's PreToolUse fires on all tools.
        if tool not in _DENY_EXEMPT_TOOLS:
            tag = self._deny_tag(scan, tool=tool)
            if tag:
                msg = _PY_DENY_MSG if tag == "python" else _DENY_MSG
                return self._respond(request, tool, scan,
                                     event_type="routing_denied", content=msg,
                                     extra={"deny_family": tag})

        # Declared redirect — scoped to exec-shaped tools only. Only these
        # tools carry a ``command``/``cmd`` arg that targets a declared script.
        if tool in self.intercept_tools:
            command = self._extract_command(request)
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
        scan = self._extract_scan(request)

        if tool not in _DENY_EXEMPT_TOOLS:
            tag = self._deny_tag(scan, tool=tool)
            if tag:
                msg = _PY_DENY_MSG if tag == "python" else _DENY_MSG
                return self._respond(request, tool, scan,
                                     event_type="routing_denied", content=msg,
                                     extra={"deny_family": tag})

        if tool in self.intercept_tools:
            command = self._extract_command(request)
            redirect = self._declared_redirect(command)
            if redirect is not None:
                return self._respond(request, tool, command,
                                     event_type="routing_redirected",
                                     content=_REDIRECT_MSG.format(tool=redirect),
                                     extra={"target": redirect})
        return await handler(request)
