"""ACP (Agent Client Protocol) stdio server — exposes ``build_graph(org)`` to
editors (Zed / VS Code via vscode-acp / Neovim). The editor IS the TUI; this
module owns no UI code.

ACP (``agentclientprotocol.com``) is a stdio JSON-RPC protocol between coding
agents and editors. ``deepagents-acp``'s ``AgentServerACP`` wraps a deepagents
graph as such a server. We hand it a **factory** that returns the compiled
per-org graph; the server caches the first build (one graph instance serves all
sessions, keyed by ``thread_id=session_id`` in the checkpointer) — so the org
is fixed at startup, not per-session.

Org resolution (first wins): ``--org`` flag → ``$PUX_ORG`` → ``general``. An
unknown org fails loud. The sandbox self-boots lazily on first tool use
(``build_graph`` → ``shared_backend()`` → ``shared_exec()`` → ``ensure()``,
), so — like ``pux direct`` — ``pux acp`` needs no prior
``pux sandbox start``.

Run: ``pux acp --org invest``. Stdin/stdout are the protocol —
this process must not print to stdout. Errors go to stderr.

The stdout contract is ENFORCED, not just stated: ``run_acp`` calls
``bootstrap_env_and_logging(pin_stderr=True)`` FIRST — the SHARED kit helper
(``pux_harness.kit``) that loads ``./.env`` (the editor's shell lacks the
user's key export) and pins the root logger to ``stderr`` (``force=True``) so
no library can auto-configure a stdout handler that would corrupt the
JSON-RPC stream. The same helper (``pin_stderr=False``) is the seam that makes
the Aegra runtime / ``pux direct`` / exported runners load their consumer ``.env``
seamlessly. See ``tests/harness/test_bootstrap.py``.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Callable
from typing import Any

from acp import run_agent as run_acp_agent
from acp import update_agent_thought_text
from acp.exceptions import RequestError
from acp.schema import (
    CloseSessionResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    PromptResponse,
    SessionCapabilities,
    SessionCloseCapabilities,
    SessionInfo,
    SessionListCapabilities,
    SessionResumeCapabilities,
    UserMessageChunk,
)
from acp.schema import (
    CloseSessionRequest,
    ResumeSessionRequest,
    ResumeSessionResponse,
)
from acp.utils import param_model

# ``session/delete`` types were added in a newer acp version. The root
# orchestrator venv (used by ``bin/pux``) may lag behind pux-harness/.venv
# (used by ``uv run`` direct). Guard the import so the server boots
# on both — the delete capability degrades gracefully when unavailable.
try:
    from acp.schema import (
        DeleteSessionRequest,
        DeleteSessionResponse,
        SessionDeleteCapabilities,
    )
    _HAS_SESSION_DELETE = True
except ImportError:
    _HAS_SESSION_DELETE = False

# ``build_agent_router`` in the upstream ``acp`` package routes every session
# lifecycle method EXCEPT ``session/delete`` — the method name exists in
# ``AGENT_METHODS`` and the request/response schema types ship, but the router
# never wires them. Patch it post-hoc so ``session/delete`` actually reaches
# our handler instead of 404ing. Applied to BOTH the router module and the
# connection module (which holds its own imported reference).
if _HAS_SESSION_DELETE:
    from acp.meta import AGENT_METHODS as _AGENT_METHODS
    from acp.utils import normalize_result as _normalize_result
    import acp.agent.router as _router_mod
    import acp.agent.connection as _conn_mod

    _orig_build_router = _router_mod.build_agent_router

    def _build_agent_router_with_delete(agent, use_unstable_protocol=False):
        router = _orig_build_router(agent, use_unstable_protocol=use_unstable_protocol)
        router.route_request(
            _AGENT_METHODS["session_delete"],
            DeleteSessionRequest,
            agent,
            "delete_session",
            adapt_result=_normalize_result,
            unstable=True,
        )
        return router

    _router_mod.build_agent_router = _build_agent_router_with_delete
    _conn_mod.build_agent_router = _build_agent_router_with_delete
from deepagents_acp.server import AgentServerACP, AgentSessionContext, text_block, update_agent_message
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from langchain_core.tools import BaseTool

from pux_harness.agent.graph import build_graph
from pux_harness.agent.model import (
    available_model_ids,
    driver_multimodal,
    is_multimodal,
    resolve_model_id,
)
from pux_harness.agent.orgs import discover_orgs
from pux_harness.agent.stack import RuntimeFacts, autonomous_from_env
from pux_harness.kit import bootstrap_env_and_logging
from pux_harness.kit._paths import project_root
from pux_harness.threads import open_thread_store

DEFAULT_ORG = "general"

# Module logger — stderr-bound by ``bootstrap_env_and_logging(pin_stderr=True)``
# in ``run_acp`` so library tracebacks never corrupt the stdout JSON-RPC stream.
_log = logging.getLogger("pux.acp")

# Stream-stall classifier + policy live in ``pux_harness.agent.retry`` (shared
# with ``agent.graph``, which wires the policy onto the deepagents ``model``
# node — the LangGraph-native layer that retried a stall in-place without
# losing in-flight work). The prompt-boundary retry loop below is the
# DEFENSE-IN-DEPTH outer layer: it catches stalls that escape the node-level
# retry (e.g. a stall inside a middleware that runs OUTSIDE the model node, or
# a stall in a node we didn't attach the policy to).
from pux_harness.agent.retry import retry_on_stream_stall as _is_stream_stall_recoverable

# How many times ``PuxAgentServer.prompt()`` re-enters ``super().prompt()`` on a
# transient model-stream stall before giving up and surfacing the end_turn +
# resume notice. This is the FALLBACK path — the primary retry happens at the
# model-node level via RetryPolicy (see ``agent.graph.build_graph`` and
# ``agent.retry.attach_stream_stall_retry``). The wrapper-level retry catches
# stalls that escape the node-level policy.
#
# NOTE on resume semantics: re-entering ``super().prompt()`` re-passes the
# user prompt as NEW input, which causes LangGraph to APPEND the message and
# re-run the turn from the start. Prior TURNS are preserved (conversation
# history is checkpointed); the in-flight TURN's partial work is re-done.
# This is acceptable as a fallback because the primary node-level retry
# handles the common case (stall inside the model node) WITHOUT losing
# in-flight work. Tuned so 4 × 120s (``stream_chunk_timeout``) + 14s of
# backoff ≈ 8 min of patience before handing control back with the resume
# notice.
_PROMPT_MAX_ATTEMPTS = 4
_PROMPT_BACKOFF_BASE = 2.0  # seconds; doubled each retry → 2s, 4s, 8s

# Traffic log — writes every JSON-RPC method call to a file so we can see
# EXACTLY what the editor (Zed) calls and what it ignores. Written to
# ``.pux/acp-traffic.log`` (one line per call, JSON format).
import json as _json_mod
_TRAFFIC_LOG = project_root() / ".pux" / "acp-traffic.log"


def _log_traffic(method: str, **detail: Any) -> None:
    """Append a method call to the traffic log."""
    try:
        _TRAFFIC_LOG.parent.mkdir(parents=True, exist_ok=True)
        entry = {"method": method, **detail}
        with open(_TRAFFIC_LOG, "a") as f:
            f.write(_json_mod.dumps(entry) + "\n")
    except Exception:
        pass


# --- ask_user: end-turn-and-resume (cross-prompt HITL) ---------------------------
#
# The ONE ask_user mechanic over ACP — mechanically identical to ending a turn:
#   1. the ``ask_user`` tool (``hitl.make_ask_user_tool("acp")``) raises a
#      langgraph ``interrupt({"ask_user": {question, options, default}})``;
#   2. ``_handle_interrupts`` (below) detects it, PRESENTS the question (+opts)
#      to the client as a chat message, then raises ``_AskUserPause`` → the turn
#      ENDS (``PromptResponse(stop_reason="end_turn")``). The interrupt persists
#      in the thread checkpoint — exactly like a normal turn end.
#   3. the user's NEXT ``session/prompt`` — a freeform message ("A", "B", any
#      text) — is the resume signal: ``prompt`` sees the pending interrupt at
#      entry and hands the message to ``_resume_ask_user``, which feeds
#      ``Command(resume={"decisions":[{"type":"ask_user","answer": msg}]})``
#      FIRST. That answer becomes the tool's return value and the agent continues.
#
# Works on Zed/Toad/Hermes TODAY — no ``elicitation`` capability needed (options
# are merely *presented*; the reply is freeform). Proven end-to-end on the real
# deepagents graph + real glm-5.2 (``reference_acp_ask_user_resume_mechanic``):
# feeding ``Command(resume=...)`` first delivers the answer; the stock loop's
# ``{"messages":[reply]}``-first shape POISONS it (cancels the tool call), so
# ``_resume_ask_user`` owns the loop instead of delegating to the base.


class _AskUserPause(Exception):
    """Control-flow signal: an ask_user interrupt is pending, so end the turn.

    Raised out of ``_handle_interrupts`` (during inline streaming) to unwind to
    ``prompt``, which catches it and returns ``end_turn``. The interrupt persists
    in the checkpoint; the user's next message resumes it (``_resume_ask_user``).
    A plain ``Exception`` subclass so it interops with the base loop's own
    try/except and is never confused with a real error."""


def _content_to_text(blocks: Any) -> str:
    """Flatten an ACP ``session/prompt`` content-block list to plain text.

    The user's reply to an ask_user is the resume answer; ACP delivers it as a
    list of ``TextContentBlock`` (``.text``) / image / resource blocks. We take
    the text (the answer is text — "A", "B", or any phrase) and drop non-text
    blocks. Empty when the client sent no text (→ the tool's ``default``)."""
    parts = []
    for block in blocks or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
    return "\n".join(parts).strip()


def _extract_text(content: Any) -> str:
    """Pull a flat string out of a streamed message chunk's ``content``
    (str | list[text/str blocks] | other). Mirrors the base ``prompt`` loop."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                out.append(block.get("text", ""))
            elif isinstance(block, str):
                out.append(block)
        return "".join(out)
    return str(content)


def _make_factory(
    org: str, saver, mcp_tools: list[BaseTool] | None = None,
    facts: RuntimeFacts | None = None,
) -> Callable[[AgentSessionContext], CompiledStateGraph]:
    """Build a graph factory bound to ``org`` + the SHARED persistent saver.

    ``saver`` is the ``AsyncSqliteSaver`` from ``open_thread_store``
    — the SAME ``.pux/agent-protocol.sqlite`` ``serve``/``direct`` use. ACP
    session checkpoints (keyed by ``thread_id=session_id``) now persist across
    ``pux acp`` process restarts instead of dying with the ephemeral
    ``MemorySaver``.

    ``context.cwd`` (the editor's project dir) IS honored as the sandbox
    workspace: ``_RegisteringAgentServerACP.new_session`` exports it as
    ``PUX_PROJECT_PATH`` so the lazily-booted container mounts the editor's
    folder at ``/sandbox/workspace`` (via ``container.resolve_project_path``)
    — the agent reads / edits the project the editor opened, like Claude Code.
    An absent cwd falls back to the harness ``project_root()``. The org is fixed
    at server startup. ``context.model`` IS honored: the editor-selected id
    (set by ``session/set_config_option``; ``None`` until the client switches)
    is threaded as ``build_graph(base_model_override=…)``. ``AgentServerACP``
    rebuilds the graph from this factory on every model switch (``_reset_agent``
    → ``factory(context)``), so the picker is live — what the editor shows ==
    what runs.

    ``facts`` carries the ACP runtime: ``transport="acp"`` makes an opted-in
    ``ask_user`` raise a langgraph ``interrupt`` that this server turns into an
    end-turn-and-resume (the user's next freeform message resumes the thread —
    see ``_AskUserPause`` / ``_resume_ask_user``) + ``autonomous``
    (``PUX_AUTONOMOUS``) drops ask_user entirely.
    """

    _tools = list(mcp_tools) if mcp_tools else []
    # prepare_warmup=True: ACP is a serve-class transport (editors connect to
    # a long-running server). The browser cold-start MUST happen in the
    # before_agent hook, NOT on the first LLM-driven browser tool call —
    # otherwise every editor session (Zed/Toad/Hermes) trips the 60s browser
    # timeout on the first navigate. Warn-and-continue: warmup failure never
    # blocks the run (agent cold-starts on first use, exactly as before).
    # Matches the Aegra runtime's RuntimeFacts(prepare_warmup=True).
    _facts = facts or RuntimeFacts(transport="acp", prepare_warmup=True)

    def factory(context: AgentSessionContext) -> CompiledStateGraph:
        # context.model = editor-selected id (None → org's tier base). Thread it
        # as the base-role override so the advertised picker is authoritative.
        override = getattr(context, "model", None)
        graph = build_graph(
            org, checkpointer=saver, facts=_facts, mcp_tools=_tools,
            base_model_override=override,
        )
        # Langfuse tracing is applied centrally in build_graph (via
        # _with_langfuse_tracing) so ALL entry points produce traces.
        # The transport tag is derived from _facts.transport ("acp" here).
        return graph

    return factory


def _advertised_models(org: str) -> list[dict[str, str]]:
    """The model selector the editor (Zed) sees for this org's agent.

    ``AgentServerACP`` only populates ``new_session.config_options`` — the model
    dropdown the editor renders — when ``models=[...]`` is passed at construction.
    Pass nothing and the editor falls back to its own built-in model list (Zed
    shows ChatGPT/OpenAI models even though this agent runs its own models).

    Advertises EVERY id in ``models.yaml`` (``available_model_ids``), with the
    org's base-role model FIRST so it is the ``select``'s default
    ``currentValue``. Picking another id flows ``context.model`` → the factory →
    ``build_graph(base_model_override=…)`` and re-resolves the supervisor model
    on the next ``_reset_agent`` (``session/set_config_option`` triggers it). An
    explicit pick is a hard pin: it disables the tier's ``base_fallbacks``
    (mirrors the ``resolve_model_id`` override stack) — the picker says exactly
    what runs.
    """
    base = resolve_model_id(role="base", org=org)
    ordered = [base] + [mid for mid in available_model_ids() if mid != base]
    options: list[dict[str, str]] = []
    for mid in ordered:
        kind = "multimodal" if is_multimodal(mid) else "text"
        options.append({"value": mid, "name": mid, "description": f"{mid} · {kind}"})
    return options


def _capture_editor_cwd(cwd: str | None) -> None:
    """Export the editor's project dir as ``PUX_PROJECT_PATH``.

    The ACP ``session/new`` ``cwd`` is the folder the editor opened — WHEN the
    client sends one. TOAD's CLI sends it (its launch dir); **Zed does NOT**
    (its ACP client leaves ``cwd`` unset, so ``cwd`` arrives as ``None`` and
    this function no-ops). Relying on this alone was the root cause of the
    cross-project leak: a Zed-opened ``ray`` project got ``PUX_PROJECT_PATH``
    unset → ``resolve_project_path()`` fell back to the harness repo → the
    agent edited the orchestrator's own files.

    The Zed path is now covered ONE layer up: ``bin/pux`` (the package entry
    Zed launches by name via PATH) captures the editor's CWD (``$PWD`` at
    launch, before ``cd "$REPO"``) and exports ``PUX_PROJECT_PATH`` itself, so
    ``setdefault`` here finds it already set. ``run_acp`` is the last-resort
    net (derives from process CWD + logs loud). This function remains the
    wire-cwd path for clients (TOAD) that send it.

    An explicit ``PUX_PROJECT_PATH=…`` export from the shell wins
    (``setdefault``); an absent / non-dir cwd falls through unchanged.

    Why this is safe: ``PUX_PROJECT_PATH`` ONLY selects the container
    bind-mount source + the discovery label + the cache-volume name. The
    harness still resolves ``orgs/`` on the HOST from ``PUX_PROJECT_ROOT`` (a
    DIFFERENT env var — see ``kit._paths.project_root``), so the org spec,
    system prompt, and middleware stack are compiled from the orchestrator
    repo unchanged. Each unique project path gets its own container (keyed by
    the ``openshell.project-path`` label), so multi-project isolation holds.
    """
    if not cwd:
        return
    try:
        abs_cwd = os.path.abspath(cwd)
    except (ValueError, OSError):
        return
    if os.path.isdir(abs_cwd):
        prev = os.environ.get("PUX_PROJECT_PATH")
        os.environ.setdefault("PUX_PROJECT_PATH", abs_cwd)
        if prev is None:
            sys.stderr.write(
                f"[pux acp] editor cwd → PUX_PROJECT_PATH={abs_cwd}\n"
            )
        else:
            sys.stderr.write(
                f"[pux acp] editor cwd ignored (PUX_PROJECT_PATH already "
                f"pinned to {prev})\n"
            )


def _normalize_todos(todos: Any) -> list[dict[str, Any]]:
    """Coerce a ``write_todos`` payload into the list-of-dicts shape the base
    ``_handle_todo_update`` requires (each item a dict with ``content``).

    GLM-5.2 and other OpenAI-compat models routinely emit todos as a list of
    PLAIN STRINGS (``["write design doc", "review code"]``) instead of the
    ``[{"content": ..., "status": ...}]`` the schema declares. The base handler
    calls ``todo.get("content")`` unconditionally → ``AttributeError: 'str'
    object has no attribute 'get'`` mid-stream → the turn dies the instant the
    model starts planning (the deterministic "stall"; proven in
    ``.pux/stall.log``). This normalizer makes that crash impossible regardless
    of the shape the model improvises. Pure + unit-tested so the fix is
    provable without instantiating the ACP server."""
    out: list[dict[str, Any]] = []
    if isinstance(todos, list):
        for todo in todos:
            if isinstance(todo, dict):
                if "content" not in todo:
                    todo = {**todo, "content": ""}
                out.append(todo)
            elif isinstance(todo, str):
                out.append({"content": todo, "status": "pending"})
            else:
                out.append({"content": str(todo), "status": "pending"})
    elif todos is not None:
        # Top-level schema violation (not even a list) — salvage rather than
        # crash the whole turn.
        out.append({"content": str(todos), "status": "pending"})
    return out


class _RegisteringAgentServerACP(AgentServerACP):
    """AgentServerACP that indexes ACP sessions + backs ``load``/``list``.

    ``AgentServerACP`` keys checkpoints by ``thread_id=session_id`` but never
    tells ``pux_threads`` about them (so ACP sessions were invisible to
    ``pux resume`` / ``pux show``), and advertises no session-persistence
    capability at all. The graph factory cannot fix either: the base caches ONE
    compiled graph and serves opaque ``session_id``s whose value never reaches
    the ``AgentSessionContext`` the factory sees.

    This override closes both gaps against the shared ``pux_threads`` index +
    persistent checkpointer: ``new_session`` registers each freshly minted
    ``session_id`` (idempotent ``INSERT OR IGNORE``); ``initialize`` advertises
    ``load_session`` + ``session_capabilities.list``; ``load_session`` verifies
    + re-hydrates a prior session so a client resumes across a pux restart;
    ``list_sessions`` enumerates the org's threads. ``fork``/``resume``/``close``
    are deliberately left unbacked (UNSTABLE in the spec) — see
    [[protocol-surface-map]].
    """

    def __init__(self, agent, store, org, models=None):
        super().__init__(agent=agent, models=models)
        self._store = store
        self._org = org
        # ``_cancelled`` is owned by the base ``AgentServerACP`` (set by the
        # cancellation handler, reset at the top of each ``prompt``). The base
        # module ships no py.typed, so annotate it here for mypy — ``_resume_ask_user``
        # checks it the same way the base loop does.
        self._cancelled: bool = False
        # Per-session prompt serialization. The ACP dispatcher spawns each
        # ``session/prompt`` as its own asyncio task with NO per-session
        # serialization, and ``_cancelled`` above is a SINGLE shared flag on
        # ``self``. Without this lock, two prompts on the same session run
        # ``agent.astream`` CONCURRENTLY on the same ``thread_id`` (corrupting
        # the checkpointer) and race on ``_cancelled`` (one prompt's cancel
        # aborts the OTHER's in-flight tool calls — the "async seems broken"
        # symptom where all tool calls come back cancelled). Keyed by
        # ``session_id`` so different sessions still run in parallel.
        self._prompt_locks: dict[str, asyncio.Lock] = {}

    def _prompt_lock(self, session_id: str) -> asyncio.Lock:
        """One ``asyncio.Lock`` per session, lazily created.

        ``dict.setdefault`` is atomic under the GIL, so two prompts racing the
        first-time creation still resolve to the SAME lock object (both lookups
        return the one ``setdefault`` installed) — serializing correctly."""
        return self._prompt_locks.setdefault(session_id, asyncio.Lock())

    async def new_session(self, cwd, mcp_servers=None, **kwargs):
        # Honor the editor's cwd BEFORE the lazy container boot: export it as
        # PUX_PROJECT_PATH so the sandbox mounts the editor's project at
        # /sandbox/workspace (Claude-Code-style "spawn in folder"). See
        # ``_capture_editor_cwd`` for the safety argument.
        _capture_editor_cwd(cwd)
        response = await super().new_session(cwd=cwd, mcp_servers=mcp_servers, **kwargs)
        await self._store.register_thread(
            response.session_id, self._org, metadata={"source": "acp"}
        )
        _log_traffic("new_session", session_id=response.session_id)
        return response

    async def _process_tool_call_chunks(
        self, session_id: str, message_chunk: Any,
        active_tool_calls: dict, tool_call_accumulator: dict,
    ) -> None:
        """Thought hook, then delegate to the base tool-call chunk handler.

        The base ``prompt`` loop calls this for EVERY streamed message chunk
        (root + subgraph) BEFORE it decides whether the chunk carries text or a
        tool call — so it is the one small seam that sees the raw chunk's
        ``additional_kwargs``. We surface the model's reasoning here, then hand
        off to ``super`` unchanged. deepagents-acp otherwise emits only message
        text (``_log_text``) and silently discards the reasoning our
        :class:`ReasoningChatOpenAI` adapter captures — so without this hook NO
        thinking ever reaches the ACP wire (the ``agent_thought_chunk`` gap)."""
        await self._maybe_emit_thought(session_id, message_chunk)
        await super()._process_tool_call_chunks(
            session_id, message_chunk, active_tool_calls, tool_call_accumulator,
        )

    async def _maybe_emit_thought(self, session_id: str, message_chunk: Any) -> None:
        """Emit one ``AgentThoughtChunk`` per reasoning delta, if any.

        ``ReasoningChatOpenAI`` accumulates provider reasoning
        (``delta.reasoning_content`` — DeepSeek / MiMo / OpenRouter, proven
        live vs ``mimo-v2.5``) onto ``message_chunk.additional_kwargs
        ["reasoning_content"]`` as a per-chunk DELTA. Each non-empty delta → one
        thought chunk, so the editor streams the reasoning live exactly like the
        message text. Empty/absent reasoning → no-op (non-reasoning providers and
        the text-only ``glm-5.2`` are unaffected)."""
        if isinstance(message_chunk, str):
            return
        ak = getattr(message_chunk, "additional_kwargs", None) or {}
        reasoning = ak.get("reasoning_content") or ak.get("reasoning") or ""
        if not reasoning:
            return
        await self._conn.session_update(
            session_id=session_id,
            update=update_agent_thought_text(reasoning),
            source="DeepAgent",
        )

    async def _handle_todo_update(
        self, session_id: str, todos: list[Any], *, log_plan: bool = True,
    ) -> None:
        """Normalize ``write_todos`` payloads before the base handler sees them.

        deepagents-acp's base ``_handle_todo_update`` assumes every todo is a
        DICT with ``content``/``status`` keys. GLM-5.2 (and other OpenAI-compat
        models that improvise the ``write_todos`` schema) routinely emit todos
        as a list of PLAIN STRINGS — ``["write design doc", "review code"]`` —
        so the base ``todo.get("content")`` raises
        ``AttributeError: 'str' object has no attribute 'get'`` mid-stream,
        killing the turn the instant the model starts planning. That was the
        deterministic "stall" at the design-doc boundary (proven in
        ``.pux/stall.log`` — session a0af08…, org coder). Coerce every item to
        the dict shape via :func:`_normalize_todos` so the base handler never
        sees a non-dict: the turn survives and the plan renders correctly."""
        await super()._handle_todo_update(
            session_id, _normalize_todos(todos), log_plan=log_plan,
        )

    async def initialize(self, protocol_version, client_capabilities=None,
                         client_info=None, **kwargs):
        """Advertise the full session-persistence surface we back.

        The base class advertises ``prompt_capabilities.image`` alone. We add
        ``load_session=True`` + ``session_capabilities`` for ``list``,
        ``resume``, ``close``, and ``delete`` — ALL backed by overrides below
        against the shared ``pux_threads`` index + the persistent checkpointer.

        Without these capabilities, editors like Zed disable their history UI
        entirely (never call ``session/list`` or ``session/load``), which
        orphaned every prior conversation in SQLite. Advertising them wakes
        up the editor's native multi-slot sidebar so the user can browse,
        load, resume, close, and delete sessions."""
        resp = await super().initialize(
            protocol_version=protocol_version,
            client_capabilities=client_capabilities,
            client_info=client_info,
            **kwargs,
        )
        _log_traffic("initialize", client_info=str(client_info),
                     client_capabilities=str(client_capabilities),
                     protocol_version=protocol_version)
        caps = resp.agent_capabilities
        # Truthful image cap (#69). The base class hardcodes
        # ``prompt_capabilities.image=True`` for every org; but image backing
        # depends on the org's BASE (supervisor) model — the role that first
        # receives the prompt's ImageContentBlocks. A text-only base
        # (e.g. glm-5.2) would make the editor offer image-attach on a model
        # that can't ingest image blocks. Gate on the SAME seam
        # ``BrowserVisionMiddleware`` uses; backing for multimodal bases
        # (mimo-v2.5) is LIVE-PROVEN by the browser-vision work.
        if caps.prompt_capabilities is not None:
            caps.prompt_capabilities.image = driver_multimodal(
                role="base", org=self._org
            )
        caps.load_session = True
        sess = caps.session_capabilities or SessionCapabilities()
        if sess.list is None:
            sess.list = SessionListCapabilities()
        if sess.resume is None:
            sess.resume = SessionResumeCapabilities()
        if sess.close is None:
            sess.close = SessionCloseCapabilities()
        if _HAS_SESSION_DELETE and hasattr(sess, "delete") and sess.delete is None:
            sess.delete = SessionDeleteCapabilities()
        caps.session_capabilities = sess
        # ``agentInfo`` — the base class never sets it. Zed uses this to
        # identify the agent and key session persistence. Without it, Zed
        # can't associate prior sessions with this agent and falls back to
        # ``session/new`` every restart.
        from acp.schema import Implementation as _Impl
        resp.agent_info = _Impl(
            name=f"pux-{self._org}",
            title=f"PUX · {self._org}",
            version="1.0.0",
        )
        resp.auth_methods = resp.auth_methods or []
        # ``mcp_capabilities`` is INTENTIONALLY left at the schema default
        # (http=False, sse=False): we do NOT back client-passed ``mcp_servers``
        # — deepagents-acp 0.0.8 drops them. The truthful False is locked by
        # contract in ``tests/server/test_acp.py`` (#71).
        return resp

    async def prompt(self, prompt, session_id, message_id=None, **kwargs):
        """Process a user prompt — with cross-prompt ``ask_user`` resume.

        Two cases (the ONE ask_user mechanic = end-turn-and-resume):

        * **Resume** — a PRIOR turn ended on an ``ask_user`` interrupt (still
          pending in this thread's checkpoint). The user's freeform message here
          IS the answer: hand it to ``_resume_ask_user``, which feeds
          ``Command(resume={"decisions":[{"type":"ask_user","answer": msg}]})``
          FIRST (the stock loop's ``{"messages":[msg]}``-first shape would CANCEL
          the paused tool call — proven poison).
        * **Normal** — no pending ask_user: delegate to the base ``prompt`` loop.
          If the agent calls ``ask_user`` mid-turn, ``_handle_interrupts`` raises
          ``_AskUserPause`` (caught here) and we end the turn normally; the
          interrupt persists for the user's next message to resume.

        The whole body is guarded by a per-session ``asyncio.Lock``: the ACP
        dispatcher runs each ``session/prompt`` as its own task, and
        ``_cancelled`` is shared instance state, so without serialization two
        prompts on one session race on the LangGraph checkpointer AND on
        ``_cancelled`` (spurious tool-call cancellations — "the async seems
        broken"). The lock makes prompts run one-at-a-time per session;
        different sessions still run in parallel.
        """
        # Capture the first user prompt as a title for the session sidebar.
        # ``merge_metadata`` is idempotent: we only set ``title`` if not already
        # set, so the FIRST prompt becomes the title (Zed's history sidebar
        # shows it instead of a bare session_id).
        try:
            first_text = _content_to_text(getattr(prompt, "content", None))
            if first_text:
                existing = await self._store.get_thread(session_id)
                if existing is not None:
                    import json as _json
                    meta = _json.loads(existing.get("metadata") or "{}")
                    if not meta.get("title"):
                        await self._store.merge_metadata(
                            session_id, {"title": first_text[:80]},
                        )
        except Exception:  # noqa: BLE001 — title capture must never break prompt
            pass
        async with self._prompt_lock(session_id):
            # A cancel may have arrived WHILE we were queued behind another
            # prompt. The base ``prompt`` would reset ``_cancelled=False`` at
            # its top and run anyway; honor the cancel here instead.
            if self._cancelled:
                self._cancelled = False
                return PromptResponse(stop_reason="cancelled")
            # RETRY LOOP around the turn body. A transient model-stream stall
            # (TCP alive, provider silent → ``StreamChunkTimeoutError`` after
            # ``stream_chunk_timeout`` seconds) MUST NOT end the turn — re-enter
            # ``super().prompt()``; LangGraph's checkpointer resumes from the
            # last completed node, so the retry picks up exactly where the stall
            # killed the prior attempt. Only deterministic errors or
            # exhausted retries fall through to the end_turn + notice path.
            #
            # Why we still guarantee a ``PromptResponse`` even on final failure:
            # the ACP dispatcher runs each ``session/prompt`` on a detached
            # supervisor task, and the IncomingMessage store only records
            # status (``fail_incoming``) — it holds no Future to reject and
            # writes nothing to the wire. An unhandled exception is swallowed
            # by the supervisor's error log and the JSON-RPC ``session/prompt``
            # RESPONSE is never sent. The editor (Zed) spins forever — the
            # "response just freezes" symptom. So we always end the turn
            # cleanly, but only AFTER retrying, not instead of retrying.
            #
            # ``asyncio.CancelledError`` / ``KeyboardInterrupt`` /
            # ``SystemExit`` are ``BaseException`` and are intentionally NOT
            # caught — the cancel path returns ``stop_reason="cancelled"``
            # inside the base loop, and a genuine cancellation must keep
            # propagating.
            last_exc: Exception | None = None
            last_recoverable: bool = False
            attempts_made = 0
            for attempt in range(_PROMPT_MAX_ATTEMPTS):
                attempts_made = attempt + 1
                try:
                    if self._agent is None:
                        self._reset_agent(session_id)
                        # Kick off background browser warmup NOW (not on the
                        # first browser tool call inside a subagent). Chrome
                        # cold-starts in ~15-20s; starting it at first-prompt
                        # time means the CTO can think + dispatch to web-agent
                        # while Chrome warms in parallel. By the time web-agent
                        # calls browser_navigate, it's ready. Fire-and-forget.
                        self._maybe_warmup_browser()
                    agent = self._agent
                    if agent is not None:
                        config: dict[str, Any] = {"configurable": {"thread_id": session_id}}
                        try:
                            state = await agent.aget_state(config)
                        except Exception:  # noqa: BLE001 — fresh/empty thread has no snapshot yet
                            state = None
                        if state is not None and self._has_ask_user_interrupt(state):
                            return await self._resume_ask_user(session_id, prompt, config)
                    return await super().prompt(
                        prompt=prompt, session_id=session_id, message_id=message_id, **kwargs
                    )
                except _AskUserPause:
                    # The turn ended on an ask_user: question was presented in
                    # ``_handle_interrupts``; the interrupt persists. ``end_turn``
                    # so the editor hands control back to the user.
                    return PromptResponse(stop_reason="end_turn")
                except Exception as exc:
                    last_exc = exc
                    recoverable = _is_stream_stall_recoverable(exc)
                    last_recoverable = recoverable
                    more_attempts = attempt < _PROMPT_MAX_ATTEMPTS - 1
                    if recoverable and more_attempts:
                        backoff = _PROMPT_BACKOFF_BASE * (2 ** attempt)
                        _log.warning(
                            "prompt() stream stall for session %s (attempt %d/%d): "
                            "%s: %s — re-entering super().prompt() in %.1fs "
                            "(LangGraph resumes from last checkpoint, no work lost)",
                            session_id, attempts_made, _PROMPT_MAX_ATTEMPTS,
                            type(exc).__name__, exc, backoff,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    # Unrecoverable OR exhausted retries → fall through to the
                    # end_turn + notice path below.
                    break

            # FINAL-ATTEMPT FAILURE (or deterministic error). Surface end_turn
            # with a resume-aware notice so the user knows their work IS
            # checkpointed — re-sending the same message resumes from the last
            # completed step, it does NOT restart the turn.
            assert last_exc is not None, "retry loop exited without an exception"
            exc = last_exc
            _log.exception(
                "prompt() raised for session %s after %d/%d attempts; surfacing "
                "end_turn with resume notice: %s: %s",
                session_id, attempts_made, _PROMPT_MAX_ATTEMPTS,
                type(exc).__name__, exc,
            )
            # Durable evidence dump. stderr is pinned to Zed's stdio pipe
            # (pin_stderr=True) so the traceback above is LOST to the editor
            # and never reaches a file — leaving us to guess. Write the full
            # traceback to .pux/stall.log so the operator can see EXACTLY
            # which layer raised (supervisor astream vs a middleware vs a
            # subagent vs a grader). Best-effort: never break the turn end.
            try:
                import traceback as _tb
                from datetime import datetime as _dt
                _stall = project_root() / ".pux" / "stall.log"
                _stall.parent.mkdir(parents=True, exist_ok=True)
                with _stall.open("a") as _f:
                    _f.write(f"\n{'=' * 72}\n")
                    _f.write(f"[{_dt.now().isoformat()}] session={session_id} "
                             f"org={getattr(self, '_org', '?')} "
                             f"attempts={attempts_made}/{_PROMPT_MAX_ATTEMPTS}\n")
                    _f.write(f"exception={type(exc).__name__}: {exc}\n\n")
                    _f.write("".join(_tb.format_exception(exc)))
            except Exception:  # noqa: BLE001
                pass
            try:
                # Pick a notice that honestly describes WHICH layer raised.
                # The pre-fix text unconditionally called every failure a
                # "model stream stalled", which sent the operator looking at
                # provider health when the actual cause was a deterministic
                # tool-side timeout (e.g. ExecTimeout from a recursive
                # ``uv run pux`` the agent shelled out to) or an
                # unrecoverable code error (AttributeError from a malformed
                # tool schema, KeyError on missing config). ``last_recoverable``
                # from the classifier above tells us which branch we're in:
                # ``False`` → unrecoverable exception that bailed after 1
                # attempt; ``True`` → stall that exhausted
                # ``_PROMPT_MAX_ATTEMPTS`` retries.
                from pux_harness.sandbox.docker_exec import (
                    ExecTimeout as _SandboxExecTimeout,
                )
                if isinstance(exc, _SandboxExecTimeout):
                    notice = (
                        f"⚠️ A tool call hit the sandbox wall-clock timeout "
                        f"({exc}) and didn't finish. This is NOT a model "
                        f"stream stall — the agent's command exceeded its "
                        f"time budget (often a sign it shelled out to a "
                        f"long-running subprocess like ``uv run pux`` and "
                        f"waited inline). Your work up to this point is "
                        f"checkpointed: re-send the same message and the "
                        f"subagent will resume from the last completed step."
                    )
                elif not last_recoverable:
                    # Unrecoverable exception that bailed after 1 attempt —
                    # NOT a stream stall. Surface the actual exception name
                    # and message so the operator knows where to look.
                    notice = (
                        f"⚠️ This turn ended early — the agent raised "
                        f"{type(exc).__name__}: {exc}. This is a "
                        f"deterministic error (not a transient stream "
                        f"stall) and will not change shape on retry. Your "
                        f"work up to this point is checkpointed: re-send "
                        f"the same message and the subagent will resume "
                        f"from the last completed step, not start over."
                    )
                else:
                    notice = (
                        f"⚠️ This turn ended early — the model stream stalled "
                        f"({type(exc).__name__}) and didn't recover after "
                        f"{attempts_made} attempt(s). Your work up to this point "
                        f"is checkpointed: re-send the same message and the "
                        f"subagent will resume from the last completed step, "
                        f"not start over."
                    )
                await self._log_text(
                    session_id=session_id,
                    text=notice,
                )
            except Exception:  # noqa: BLE001 — surfacing is best-effort
                _log.debug(
                    "could not surface stall notice via _log_text", exc_info=True,
                )
            return PromptResponse(stop_reason="end_turn")

    @staticmethod
    def _has_ask_user_interrupt(state: Any) -> bool:
        """``True`` iff ``state`` has a pending ``{"ask_user": ...}`` interrupt
        (i.e. a prior turn ended paused on ask_user, awaiting the user's reply)."""
        return any(
            isinstance(i.value, dict) and "ask_user" in i.value
            for i in (getattr(state, "interrupts", None) or [])
        )

    def _maybe_warmup_browser(self) -> None:
        """Kick off background browser warmup if this org has browser tools.

        Called once at first-prompt time (when the graph is first built).
        Scans the org's specialist tools for any ``pux_sandbox_browser_*`` —
        if found, fires ``warmup_ephemeral_browser`` in a daemon thread so
        Chrome cold-starts in parallel with the agent's first turn. No-op
        for orgs without browser tools. Never raises."""
        try:
            from pux_harness.sandbox.tools.browser import warmup_ephemeral_browser
            from pux_harness.sandbox.docker_exec import shared_exec
            warmup_ephemeral_browser(shared_exec())
        except Exception:  # noqa: BLE001 — warmup must never break the agent
            pass

    async def _handle_interrupts(
        self, *, current_state: Any, session_id: str, pending_interrupts: Any = None,
    ) -> list[dict[str, Any]]:
        """Route ``ask_user`` interrupts to end-turn-and-resume; delegate the rest.

        The base class only understands ``action_requests``-shaped interrupts and
        hard-errors (``RequestError`` -32600) on free-form ones. We intercept
        ``{"ask_user": ...}`` interrupts BEFORE the base sees them: present the
        question (+options) as a chat message, then raise ``_AskUserPause`` so the
        turn ends — mechanically identical to a normal turn end. The interrupt
        persists in the checkpoint; the user's next freeform message resumes it
        (``prompt`` → ``_resume_ask_user``). Everything else (tool-gates) is
        delegated to the parent's inline ``request_permission`` path.
        """
        interrupts = (
            list(pending_interrupts)
            if pending_interrupts is not None
            else list(current_state.interrupts or [])
        )
        ask_ix = [
            i for i in interrupts
            if isinstance(i.value, dict) and "ask_user" in i.value
        ]
        tool_ix = [
            i for i in interrupts
            if not (isinstance(i.value, dict) and "ask_user" in i.value)
        ]
        if ask_ix:
            for it in ask_ix:
                await self._present_ask_user(it.value["ask_user"], session_id)
            raise _AskUserPause()
        if tool_ix:
            return await super()._handle_interrupts(
                current_state=current_state, session_id=session_id, pending_interrupts=tool_ix,
            )
        return []

    async def _present_ask_user(self, payload: dict[str, Any], session_id: str) -> None:
        """Emit the question (+options) as a chat message so the user can answer.

        The options are merely *presented* — the user's next freeform message (a
        letter, the word, or any text) resumes the thread as the answer. This is
        the whole UX: present, end turn, let the reply continue."""
        question = str(payload.get("question", ""))
        options = [str(o) for o in (payload.get("options") or [])]
        default = payload.get("default")
        if options:
            line = f"❓ {question}\nOptions: {' / '.join(options)}"
            if default:
                line += f" [default: {default}]"
            line += "\n(reply with your choice to continue)"
        else:
            line = f"❓ {question}\n(reply to answer; your next message continues)"
        await self._log_text(session_id=session_id, text=line)

    async def _resume_ask_user(
        self, session_id: str, prompt: Any, config: dict[str, Any],
    ) -> PromptResponse:
        """Own the cross-prompt ask_user resume.

        The inherited ``prompt`` loop feeds ``{"messages":[reply]}`` as its FIRST
        ``astream`` whenever ``user_decisions`` is empty — which, on a thread
        paused on an ask_user interrupt, CANCELS the paused tool call (proven
        poison; ``reference_acp_ask_user_resume_mechanic``). So we cannot delegate
        the resume to the base. This minimal loop feeds
        ``Command(resume={"decisions":[{"type":"ask_user","answer": reply}]})``
        FIRST, then mirrors the base's streaming body — assistant text + todo
        updates + nested-interrupt handling + cancellation.

        Tool-call *card* rendering (the base's ``_process_tool_call_chunks`` +
        result-card block) is intentionally NOT mirrored: it would fork ~60 lines
        of drift-prone card logic for a cosmetic gain. Tools still EXECUTE
        (deepagents runs tool nodes server-side during astream); only their visual
        cards are suppressed on a resumed turn. The assistant's text reply — what
        the user needs — streams normally. Nested ask_user re-raises
        ``_AskUserPause`` (propagates to ``prompt``); a nested tool-gate is
        resolved inline by ``_handle_interrupts`` → base, then the loop resumes.
        """
        agent = self._agent
        answer = _content_to_text(prompt)
        decisions: list[dict[str, Any]] = [{"type": "ask_user", "answer": answer}]
        current_state = None
        while current_state is None or current_state.interrupts:
            if self._cancelled:
                self._cancelled = False
                return PromptResponse(stop_reason="cancelled")
            pending_interrupts: tuple = ()
            async for stream_chunk in agent.astream(
                Command(resume={"decisions": decisions}),
                config=config,
                stream_mode=["messages", "updates"],
                subgraphs=True,
            ):
                if not (isinstance(stream_chunk, tuple) and len(stream_chunk) == 3):
                    continue
                namespace, stream_mode, data = stream_chunk
                if self._cancelled:
                    self._cancelled = False
                    return PromptResponse(stop_reason="cancelled")
                if stream_mode == "updates":
                    if isinstance(data, dict) and "__interrupt__" in data:
                        objs = data.get("__interrupt__")
                        if objs:
                            pending_interrupts = objs
                        continue
                    if isinstance(data, dict):
                        for node_name, update in data.items():
                            if (
                                node_name == "tools"
                                and isinstance(update, dict)
                                and "todos" in update
                            ):
                                todos = update.get("todos", [])
                                if todos:
                                    await self._handle_todo_update(
                                        session_id, todos, log_plan=False
                                    )
                    continue
                # messages mode — stream the assistant's text reply.
                message_chunk, _metadata = data
                if isinstance(message_chunk, str):
                    if not namespace:
                        await self._log_text(text=message_chunk, session_id=session_id)
                elif getattr(message_chunk, "content", None):
                    text = _extract_text(message_chunk.content)
                    if text and not namespace:
                        await self._log_text(text=text, session_id=session_id)
            current_state = await agent.aget_state(config)
            if pending_interrupts:
                decisions = await self._handle_interrupts(
                    current_state=current_state,
                    session_id=session_id,
                    pending_interrupts=pending_interrupts,
                )
                if decisions:
                    current_state = None
        return PromptResponse(stop_reason="end_turn")

    # --- session history replay ------------------------------------------------
    #
    # When Zed loads a session (``session/load`` or ``session/resume``) it
    # expects the server to REPLAY the conversation history as
    # ``session_update`` notifications. Without this, the editor's conversation
    # panel is empty — the session is "loaded" but the user sees no prior
    # messages.
    #
    # We read messages DIRECTLY from the langgraph ``writes`` table — NOT via
    # ``agent.aget_state``. This is critical: ``aget_state`` requires building
    # the full graph (model init, sandbox boot, specialist tools), which is
    # expensive and can FAIL during a lightweight ``load_session`` call. The
    # writes table stores every message write to the ``messages`` channel as a
    # serialized blob; we deserialize each with langgraph's ``JsonPlusSerializer``
    # and emit it. No graph build, no sandbox, no model init — pure SQLite read.

    async def _replay_history(self, session_id: str) -> None:
        """Replay conversation messages from the checkpointer to the editor.

        Called from ``load_session`` / ``resume_session`` so the conversation
        panel shows prior messages instead of appearing empty. Reads messages
        directly from the ``writes`` table (channel='messages', root namespace),
        deserializes each blob with ``JsonPlusSerializer``, and emits each
        message as a ``session_update`` notification.

        Failures are caught + logged — replay must NEVER break session loading.
        A session that loads without history is still usable; the next ``prompt``
        call resumes off the checkpoint normally.
        """
        try:
            messages = await self._load_messages_from_writes(session_id)
            if not messages:
                return
            for msg in messages:
                await self._emit_replay_message(session_id, msg)
        except Exception as exc:  # noqa: BLE001 — replay must never break load
            sys.stderr.write(f"[pux acp] history replay failed: {exc}\n")

    async def _load_messages_from_writes(self, session_id: str) -> list[Any]:
        """Read + deserialize all messages from the ``writes`` table.

        langgraph stores each message write as a row in ``writes`` with
        ``channel='messages'``. The root graph namespace (``checkpoint_ns=''``)
        holds the main conversation; subgraph namespaces (tools, middleware) hold
        sub-agent traffic we don't want in the top-level replay. Each row's
        ``value`` blob is serialized via langgraph's ``JsonPlusSerializer`` —
        ``loads_typed((type, value))`` reconstructs the original langchain
        message objects.

        Messages are deduplicated by ID, matching langgraph's ``add_messages``
        reducer semantics: when a message with the same ID is written again, it
        REPLACES the prior occurrence (not appends). Without dedup, the same
        message would be replayed multiple times (once per superstep that wrote
        it), swamping the conversation panel.
        """
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        cur = await self._store.db.execute(
            "SELECT type, value FROM writes "
            "WHERE thread_id = ? AND channel = 'messages' AND checkpoint_ns = '' "
            "ORDER BY rowid ASC",
            (session_id,),
        )
        rows = await cur.fetchall()
        await cur.close()
        if not rows:
            return []
        serde = JsonPlusSerializer()
        # Build the accumulated message list with add_messages semantics:
        # same-ID writes replace, different-ID writes append.
        result: list[Any] = []
        id_to_index: dict[str, int] = {}
        for row in rows:
            dtype, dvalue = row[0], row[1]
            try:
                decoded = serde.loads_typed((dtype, dvalue))
            except Exception:  # noqa: BLE001 — one corrupt write must not kill replay
                continue
            batch = decoded if isinstance(decoded, list) else [decoded]
            for msg in batch:
                mid = getattr(msg, "id", None)
                if mid and mid in id_to_index:
                    result[id_to_index[mid]] = msg
                elif mid:
                    id_to_index[mid] = len(result)
                    result.append(msg)
                else:
                    result.append(msg)
        return result

    async def _emit_replay_message(self, session_id: str, msg: Any) -> None:
        """Emit one langchain message as a ``session_update`` notification.

        ``HumanMessage`` → ``UserMessageChunk``, ``AIMessage`` →
        ``AgentMessageChunk``. ``SystemMessage`` / ``ToolMessage`` are skipped:
        system prompts are internal, and tool-call cards would need card-rendering
        machinery we deliberately don't replay (the tool EXECUTED server-side
        during the original turn; replaying its card is cosmetic).
        """
        from langchain_core.messages import AIMessage, HumanMessage

        content = getattr(msg, "content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    parts.append(b.get("text", ""))
                elif isinstance(b, str):
                    parts.append(b)
            text = "\n".join(p for p in parts if p)
        else:
            text = str(content) if content else ""

        if not text.strip():
            return

        msg_id = str(getattr(msg, "id", None) or "")

        if isinstance(msg, HumanMessage):
            update = UserMessageChunk(
                session_update="user_message_chunk",
                content=text_block(text),
            )
            if msg_id:
                update.message_id = msg_id
            await self._conn.session_update(
                session_id=session_id, update=update, source="DeepAgent",
            )
        elif isinstance(msg, AIMessage):
            update = update_agent_message(text_block(text))
            if msg_id:
                update.message_id = msg_id
            await self._conn.session_update(
                session_id=session_id, update=update, source="DeepAgent",
            )
        # SystemMessage, ToolMessage, etc. — skip

    async def load_session(self, cwd, session_id, mcp_servers=None,
                           additional_directories=None, **kwargs):
        """Resume a previously-created session across a ``pux acp`` restart.

        Editors/daemons (Hermes, acpx) call ``session/load`` to pick a thread
        back up after pux exits. The conversation state lives in the langgraph
        checkpointer keyed by ``thread_id=session_id``; this method (a) verifies
        the session is ours + belongs to this org (raise otherwise — no handle
        to a foreign session), (b) re-populates the per-session in-memory state
        the way ``new_session`` does so the editor's config-options render and a
        subsequent ``prompt`` resumes cleanly off the persisted checkpoint.

        ``mcp_servers`` from the client is accepted but NOT honored yet — the
        graph is built from OUR resolved ``tool_servers`` (client-MCP passthrough
        is the deferred #68b sub-feature; we deliberately do not advertise
        ``mcp_capabilities`` until it lands)."""
        _capture_editor_cwd(cwd)
        _log_traffic("load_session", session_id=session_id)
        row = await self._store.get_thread(session_id)
        if row is None or row["org"] != self._org:
            raise RequestError(
                code=-32001,
                message=(
                    f"session {session_id!r} not found for org {self._org!r}"
                ),
            )
        self._session_cwds[session_id] = cwd
        if self._modes is not None:
            self._session_modes[session_id] = self._modes.current_mode_id
            self._session_mode_states[session_id] = self._modes
        if self._models is not None and len(self._models) > 0:
            self._session_models[session_id] = self._models[0]["value"]
        config_options = None
        if self._modes is not None or self._models is not None:
            config_options = self._build_config_options(session_id)
        # Replay conversation history so the editor's conversation panel shows
        # prior messages instead of appearing empty after a restart.
        await self._replay_history(session_id)
        return LoadSessionResponse(
            config_options=config_options,
            modes=self._modes if self._modes is not None else None,
        )

    async def list_sessions(self, cwd=None, cursor=None, **kwargs):
        """Enumerate this org's sessions — backs ``session/list``.

        acpx's parallel workstreams and Hermes' session picker call
        ``session/list`` to discover prior threads. pux mounts the editor's cwd
        (via ``PUX_PROJECT_PATH``, captured at ``session/new``) as the sandbox
        workspace, so every listed session under one ``pux acp`` process shares
        that cwd. ``updated_at`` reflects the session's CREATION time (the
        pux_threads row timestamp) — we don't yet track last-prompt activity;
        honest about that limit rather than fabricating a fresher timestamp."""
        _log_traffic("list_sessions")
        rows = await self._store.list_threads(org=self._org)
        # Reflect the ACTUAL workspace the container mounts (the editor's cwd
        # via PUX_PROJECT_PATH, else the harness project_root).
        shared_cwd = os.environ.get("PUX_PROJECT_PATH") or str(project_root())
        sessions = []
        for r in rows:
            import json as _json
            meta = _json.loads(r.get("metadata") or "{}")
            sessions.append(SessionInfo(
                session_id=r["thread_id"],
                cwd=shared_cwd,
                updated_at=r["created_at"],
                title=meta.get("title"),
            ))
        return ListSessionsResponse(sessions=sessions, next_cursor=None)

    @param_model(ResumeSessionRequest)
    async def resume_session(self, session_id, cwd,
                             additional_directories=None, mcp_servers=None,
                             **kwargs):
        """Resume a previously-created session — backs ``session/resume``.

        Zed's history sidebar calls this to switch to a prior conversation.
        Mechanically identical to ``load_session``: verify ownership +
        re-hydrate per-session state so config-options render and a
        subsequent ``prompt`` resumes off the persisted checkpoint."""
        _capture_editor_cwd(cwd)
        _log_traffic("resume_session", session_id=session_id)
        row = await self._store.get_thread(session_id)
        if row is None or row["org"] != self._org:
            raise RequestError(
                code=-32001,
                message=(
                    f"session {session_id!r} not found for org {self._org!r}"
                ),
            )
        self._session_cwds[session_id] = cwd
        if self._modes is not None:
            self._session_modes[session_id] = self._modes.current_mode_id
            self._session_mode_states[session_id] = self._modes
        if self._models is not None and len(self._models) > 0:
            self._session_models[session_id] = self._models[0]["value"]
        config_options = None
        if self._modes is not None or self._models is not None:
            config_options = self._build_config_options(session_id)
        # Replay conversation history (same as load_session — Zed's history
        # sidebar needs the prior messages to display the conversation).
        await self._replay_history(session_id)
        return ResumeSessionResponse(
            config_options=config_options,
            modes=self._modes if self._modes is not None else None,
        )

    @param_model(CloseSessionRequest)
    async def close_session(self, session_id, **kwargs):
        """Close a session — backs ``session/close``.

        The editor removes the session from its active list. Thread data
        persists in the checkpointer — it can still be loaded/resumed
        later. A no-op acknowledgment is sufficient."""
        _log_traffic("close_session", session_id=session_id)
        return CloseSessionResponse()


# --- session/delete (conditional on acp version) ------------------------------
#
# ``session/delete`` types were added in a newer acp than the root orchestrator
# venv ships. Define the handler ONLY when the types are importable, and attach
# it to the class with the ``@param_model`` decorator so the JSON-RPC router
# discovers it. The router in ``build_agent_router`` does NOT route
# ``session/delete`` yet (only ``new/load/list/resume/close/fork``), so this
# method serves as a forward-compat stub: it works once the upstream adds the
# route, and is harmless dead code until then. The capability advertisement
# (``sess.delete``) is similarly guarded.

if _HAS_SESSION_DELETE:
    @param_model(DeleteSessionRequest)
    async def _delete_session(self, session_id, **kwargs):
        """Delete a session permanently — backs ``session/delete``.

        Removes the session from ``pux_threads`` AND the checkpointer
        (checkpoints + writes tables). The conversation is gone after this.
        Zed's history sidebar offers this as 'Delete'."""
        row = await self._store.get_thread(session_id)
        if row is None or row["org"] != self._org:
            raise RequestError(
                code=-32001,
                message=(
                    f"session {session_id!r} not found for org {self._org!r}"
                ),
            )
        await self._store.delete_thread(session_id)
        self._session_cwds.pop(session_id, None)
        self._session_models.pop(session_id, None)
        self._session_modes.pop(session_id, None)
        self._session_mode_states.pop(session_id, None)
        self._session_mcp_servers.pop(session_id, None)
        return DeleteSessionResponse()

    _RegisteringAgentServerACP.delete_session = _delete_session  # type: ignore[attr-defined]


# --- Public API (called from the unified CLI) ---------------------------------


async def _acp_main(org: str) -> None:
    """Async wrapper that opens MCP sessions, builds the ACP server, then runs.

    The shared thread store is held open for the process lifetime:
    ``run_acp_agent`` blocks serving stdio until the editor disconnects, and the
    ACP server must keep writing checkpoints into ``.pux/agent-protocol.sqlite``
    for that whole span (sessions now persist across restarts)."""
    from pux_harness.agent.mcp_client import open_org_mcp  # noqa: PLC0415
    mcp_tools: list[BaseTool] = []
    try:
        mcp_tools = await open_org_mcp(org)
    except Exception as exc:
        sys.stderr.write(f"pux acp: tool_servers resolution failed: {exc}\n")
    async with open_thread_store() as store:
        acp_agent = _RegisteringAgentServerACP(
            agent=_make_factory(
                org, saver=store.saver, mcp_tools=mcp_tools,
                facts=RuntimeFacts(transport="acp",
                                   autonomous=autonomous_from_env()),
            ),
            store=store,
            org=org,
            models=_advertised_models(org),
        )
        # ``use_unstable_protocol=True``: the ACP SDK marks ``session/resume``,
        # ``session/close``, and ``session/fork`` as UNSTABLE. Without this flag
        # the router REJECTS those method calls (``RequestError.method_not_found``)
        # even though we advertise the capabilities in ``initialize``. Zed's
        # history sidebar needs resume + close to actually work — the capability
        # advertisement alone is useless if the handler 404s.
        await run_acp_agent(acp_agent, use_unstable_protocol=True)


def run_acp(org: str = DEFAULT_ORG) -> None:
    """Run the deepagents org graph as an ACP stdio server (editor = TUI)."""
    # Stdio bootstrap FIRST: load ./.env (editor shell lacks the key export) +
    # pin logging to stderr (stdout IS the JSON-RPC wire). The shared kit helper
    # is the same seam serve/direct/exported runners use (pin_stderr=False there).
    bootstrap_env_and_logging(pin_stderr=True)
    known = discover_orgs()
    if org not in known:
        sys.stderr.write(f"pux acp: unknown org {org!r}; discovered: {known}\n")
        raise SystemExit(2)

    # ---- HARD PROJECT-ISOLATION GUARD -------------------------------------
    # The sandbox bind-mounts ``$PUX_PROJECT_PATH`` (or, as a FALLBACK,
    # ``$PUX_PROJECT_ROOT``) at ``/sandbox/workspace`` — that is what the agent
    # reads and writes. If a launcher (Zed / aegra / TOAD / a bare ``pux acp``)
    # forgets to pin the edit target, the fallback silently binds THIS harness
    # repo, so a coder org spawned against any OTHER project edits the
    # orchestrator's own files. That cross-project leak is a hard isolation
    # failure; it must NEVER happen silently.
    #
    # Defense in depth: ``bin/pux`` (the package entry launched by name over
    # PATH) captures the caller's CWD and exports ``PUX_PROJECT_PATH`` from it
    # BEFORE cd-ing into the harness repo. This block is the LAST-RESORT net:
    # if the var is still unset when we get here, derive it from the process
    # CWD and LOG LOUDLY so a wrong bind is visible in the editor's stderr, not
    # silent. We never fall back to ``project_root()`` here — that is the
    # exact foot-gun.
    _harness_root = str(project_root())
    _pp = os.environ.get("PUX_PROJECT_PATH")
    if not _pp:
        _pp = os.getcwd()
        os.environ["PUX_PROJECT_PATH"] = _pp
        sys.stderr.write(
            f"pux acp: WARNING — PUX_PROJECT_PATH was unset; pinning sandbox "
            f"workspace to the process CWD ({_pp}). If this is not the project "
            f"you opened in the editor, your launcher is not exporting "
            f"PUX_PROJECT_PATH and the agent will edit {_pp}. "
            f"(harness root / fallback-to-avoid: {_harness_root})\n"
        )
    # Loud one-line bind confirmation either way — the editor's stderr shows
    # EXACTLY which host dir the agent can touch. Cheap, permanent audit trail.
    sys.stderr.write(
        f"pux acp: sandbox workspace bound to PUX_PROJECT_PATH={_pp} "
        f"(harness root={_harness_root}, edit target is "
        f"{'THE HARNESS REPO ITSELF' if os.path.realpath(_pp) == os.path.realpath(_harness_root) else 'a separate project'})\n"
    )

    asyncio.run(_acp_main(org))


def main() -> None:
    """Legacy CLI entry point (argparse). Replaced by ``pux_harness.cli.main``."""
    ap = argparse.ArgumentParser(
        prog="pux acp",
        description="Run the deepagents org graph as an ACP stdio server (editor = TUI).",
    )
    ap.add_argument(
        "--org",
        default=os.environ.get("PUX_ORG", DEFAULT_ORG),
        help=f"org to serve (default: $PUX_ORG or {DEFAULT_ORG!r})",
    )
    ap.add_argument(
        "--sandbox-id",
        default=None,
        help="override the per-project sandbox id (container name + persist "
        "volume key). DEFAULT: derived from the project path, so two "
        "DIFFERENT projects never collide and you do not need to set this. "
        "Set it ONLY when running two concurrent `pux acp` sessions against "
        "the SAME project (a deliberate collision).",
    )
    args = ap.parse_args()
    # Export so the lazily-booted SandboxContainer (deep in the graph) picks it
    # up via the env path in __init__ — acp does not construct the container
    # directly, so the env var is the one seam that reaches it.
    if args.sandbox_id:
        os.environ["PUX_SANDBOX_ID"] = args.sandbox_id
    run_acp(args.org)


if __name__ == "__main__":
    main()
