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
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Callable

from acp import run_agent as run_acp_agent
from acp.exceptions import RequestError
from acp.schema import (
    ListSessionsResponse,
    LoadSessionResponse,
    SessionCapabilities,
    SessionInfo,
    SessionListCapabilities,
)
from deepagents_acp.server import AgentServerACP, AgentSessionContext
from langgraph.graph.state import CompiledStateGraph

from langchain_core.tools import BaseTool

from pux_harness.agent.graph import build_graph
from pux_harness.agent.model import driver_multimodal, resolve_model_id
from pux_harness.agent.orgs import discover_orgs
from pux_harness.agent.tool_servers import resolve_tool_servers
from pux_harness.kit._paths import project_root
from pux_harness.threads import open_thread_store

DEFAULT_ORG = "general"


def _make_factory(
    org: str, saver, mcp_tools: list[BaseTool] | None = None,
) -> Callable[[AgentSessionContext], CompiledStateGraph]:
    """Build a graph factory bound to ``org`` + the SHARED persistent saver.

    ``saver`` is the ``AsyncSqliteSaver`` from ``open_thread_store``
    — the SAME ``.pux/agent-protocol.sqlite`` ``serve``/``direct`` use. ACP
    session checkpoints (keyed by ``thread_id=session_id``) now persist across
    ``pux acp`` process restarts instead of dying with the ephemeral
    ``MemorySaver``.

    ``context.cwd`` (the editor's project dir) is intentionally ignored: the
    Pux sandbox workspace is the bind-mounted project at ``/sandbox/
    workspace/``, fixed by the container, not the editor's cwd. The factory is
    called once and cached by ``AgentServerACP`` (it keys sessions by
    ``thread_id`` in the checkpointer, not by rebuilding the graph), so the org
    cannot vary per session — it is fixed at server startup.
    """

    _tools = list(mcp_tools) if mcp_tools else []

    def factory(_context: AgentSessionContext) -> CompiledStateGraph:
        return build_graph(org, checkpointer=saver, mcp_tools=_tools)

    return factory


def _advertised_models(org: str) -> list[dict[str, str]]:
    """The model selector the editor (Zed) sees for this org's agent.

    ``AgentServerACP`` only populates ``new_session.config_options`` — the model
    dropdown the editor renders — when ``models=[...]`` is passed at construction.
    Pass nothing and the editor falls back to its own built-in model list: Zed
    shows ChatGPT/OpenAI models even though this agent runs MiMo via OpenCode Go
    (the "asks for OpenAI models" bug).

    The advertised id is the main agent's base-role model — the SAME id
    ``build_graph`` compiles — resolved via ``resolve_model_id`` (id only: no
    ``ChatOpenAI``, no key, no network). So what the editor shows == what runs.

    The factory ignores ``context.model``, so the single advertised option is
    authoritative. Honoring Zed-side model switching (threading ``context.model``
    into ``build_graph``) is a deliberate future option, not v1.
    """
    mid = resolve_model_id(role="base", org=org)
    return [{"value": mid, "name": mid, "description": "OpenCode Go (MiMo)"}]


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

    async def new_session(self, cwd, mcp_servers=None, **kwargs):
        response = await super().new_session(cwd=cwd, mcp_servers=mcp_servers, **kwargs)
        await self._store.register_thread(
            response.session_id, self._org, metadata={"source": "acp"}
        )
        return response

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
    from pux_harness.agent.mcp_client import McpSessionManager  # noqa: PLC0415
    mcp_tools: list[BaseTool] = []
    _mcp_mgr = None
    try:
        specs = resolve_tool_servers(org)
        if specs:
            _mcp_mgr = McpSessionManager(org, specs)
            await _mcp_mgr.open()
            mcp_tools = _mcp_mgr.tools
    except Exception as exc:
        sys.stderr.write(f"pux acp: tool_servers resolution failed: {exc}\n")
    async with open_thread_store() as store:
        acp_agent = _RegisteringAgentServerACP(
            agent=_make_factory(org, saver=store.saver, mcp_tools=mcp_tools),
            store=store,
            org=org,
            models=_advertised_models(org),
        )
        await run_acp_agent(acp_agent)
    if _mcp_mgr is not None:
        await _mcp_mgr.close()


def run_acp(org: str = DEFAULT_ORG) -> None:
    """Run the deepagents org graph as an ACP stdio server (editor = TUI)."""
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
