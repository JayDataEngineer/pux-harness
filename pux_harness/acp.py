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
import os
import sys
from collections.abc import Callable
from typing import Any

from acp import run_agent as run_acp_agent
from acp import update_agent_thought_text
from acp.exceptions import RequestError
from acp.schema import (
    ListSessionsResponse,
    LoadSessionResponse,
    PromptResponse,
    SessionCapabilities,
    SessionInfo,
    SessionListCapabilities,
)
from deepagents_acp.server import AgentServerACP, AgentSessionContext
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

    ``context.cwd`` (the editor's project dir) is intentionally ignored: the
    Pux sandbox workspace is the bind-mounted project at ``/sandbox/
    workspace/``, fixed by the container, not the editor's cwd. The org is fixed
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
    _facts = facts or RuntimeFacts(transport="acp")

    def factory(context: AgentSessionContext) -> CompiledStateGraph:
        # context.model = editor-selected id (None → org's tier base). Thread it
        # as the base-role override so the advertised picker is authoritative.
        override = getattr(context, "model", None)
        return build_graph(
            org, checkpointer=saver, facts=_facts, mcp_tools=_tools,
            base_model_override=override,
        )

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

    async def new_session(self, cwd, mcp_servers=None, **kwargs):
        response = await super().new_session(cwd=cwd, mcp_servers=mcp_servers, **kwargs)
        await self._store.register_thread(
            response.session_id, self._org, metadata={"source": "acp"}
        )
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
        (``delta.reasoning_content`` — DeepSeek / MiMo / OpenCode Zen Go, proven
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

    async def initialize(self, protocol_version, client_capabilities=None,
                         client_info=None, **kwargs):
        """Advertise ONLY the session surfaces we actually back.

        The base class advertises ``prompt_capabilities.image`` alone. We add
        ``load_session=True`` + ``session_capabilities.list`` — both backed by
        the overrides below against the shared ``pux_threads`` index + the
        persistent checkpointer. ``fork``/``resume``/``close`` stay unset
        (UNSTABLE in the spec and unbacked) so a client never offers a resume
        path that 404s — the truthful-capability half of the ACP mastery work
        ([[protocol-surface-map]] audit rows 3 + 7)."""
        resp = await super().initialize(
            protocol_version=protocol_version,
            client_capabilities=client_capabilities,
            client_info=client_info,
            **kwargs,
        )
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
        caps.session_capabilities = sess
        # ``mcp_capabilities`` is INTENTIONALLY left at the schema default
        # (http=False, sse=False): we do NOT back client-passed ``mcp_servers``
        # — deepagents-acp 0.0.8 drops them (``new_session``/``load_session``
        # accept the param but never store it; ``AgentSessionContext`` is a
        # frozen cwd/mode/model dataclass, so the factory can't receive them).
        # Per-session honoring needs a graph rebuild + ``McpSessionManager``
        # lifecycle per session; deferred until a dispatcher needs it. Do NOT
        # set True here without that backing — a client would send MCP servers
        # we silently ignore. The truthful False is locked by contract in
        # ``tests/server/test_acp.py`` (#71).
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
        """
        try:
            if self._agent is None:
                self._reset_agent(session_id)
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
            # ``_handle_interrupts``; the interrupt persists. ``end_turn`` so the
            # editor hands control back to the user (mechanically a normal turn end).
            return PromptResponse(stop_reason="end_turn")

    @staticmethod
    def _has_ask_user_interrupt(state: Any) -> bool:
        """``True`` iff ``state`` has a pending ``{"ask_user": ...}`` interrupt
        (i.e. a prior turn ended paused on ask_user, awaiting the user's reply)."""
        return any(
            isinstance(i.value, dict) and "ask_user" in i.value
            for i in (getattr(state, "interrupts", None) or [])
        )

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
        return LoadSessionResponse(
            config_options=config_options,
            modes=self._modes if self._modes is not None else None,
        )

    async def list_sessions(self, cwd=None, cursor=None, **kwargs):
        """Enumerate this org's sessions — backs ``session/list``.

        acpx's parallel workstreams and Hermes' session picker call
        ``session/list`` to discover prior threads. pux runs ONE project-root
        cwd per invocation (the sandbox workspace is bind-mounted, not the
        editor's cwd), so every listed session shares that cwd. ``updated_at``
        reflects the session's CREATION time (the pux_threads row timestamp) —
        we don't yet track last-prompt activity; honest about that limit rather
        than fabricating a fresher timestamp."""
        rows = await self._store.list_threads(org=self._org)
        shared_cwd = str(project_root())
        sessions = [
            SessionInfo(
                session_id=r["thread_id"],
                cwd=shared_cwd,
                updated_at=r["created_at"],
            )
            for r in rows
        ]
        return ListSessionsResponse(sessions=sessions, next_cursor=None)


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
        await run_acp_agent(acp_agent)


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
    args = ap.parse_args()
    run_acp(args.org)


if __name__ == "__main__":
    main()
