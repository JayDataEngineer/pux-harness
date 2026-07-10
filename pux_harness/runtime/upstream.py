"""Upstream-runtime graph declaration for ``langgraph-api`` — the OFFICIAL
Agent Protocol server. ``langgraph.json`` points ``langgraph build`` (the k3s
Docker image) — and ``langgraph dev`` (local smoke of THIS lane) — here.

TWO PROTOCOLS, TWO SURFACES (per the pi-pivot scoping):
  * LOCAL  -> Agent CLIENT Protocol (ACP): ``deepagents-acp`` stdio JSON-RPC
              (editors). Served by ``pux acp``; NOT here.
  * k3s    -> Agent Protocol (AP): ``langgraph-api`` HTTP REST (threads/runs/
              store/assistants/SSE). THIS module + ``langgraph.json``.
This is the k3s/AP lane. langgraph-api OWNS the REST surface — pux's hand-rolled
``server.py`` REST lane was RETIRED (Aegra phase D; see [[protocol-surface-map]],
[[rely-on-upstream]], [[no-legacy-left-behind]]). Each org = one ``graph_id`` =
one assistant; the wire format is ``langgraph_sdk``'s, so consumers keep working.

Three graph regimes, ONE ``graph_id``, EXPLICIT (no silent fallback — see
[[no-fallbacks-no-aliases]]); select via ``PUX_UPSTREAM_GRAPH``:

* ``keystone`` (legacy ``PUX_UPSTREAM_DEV``) — minimal scripted ``create_agent``.
  Portable contract keystone: proves langgraph-api serves a pux-style graph with
  the full SDK surface, keyless + Dockerless. Driven by ``scripts/upstream_keystone.py``.
* ``org`` — the REAL org graph via ``compile_org`` (genuine roster + prompt +
  deepagents middleware), scripted supervisor, no specialist tools. Keyless proof
  that the real org graph SHAPE serves upstream.
* unset (production/k3s) — ``build_graph(org)``: Docker specialists + real model
  + the declarative stack. Needs the provider key + Docker. langgraph-api OWNS the
  checkpointer (``build_graph`` called with ``checkpointer=None``) per
  [[unified-thread-store]] — the owner shifts from pux's ``open_thread_store`` to
  langgraph-api's runtime checkpointer.

MULTI-ORG: every discovered org is its own ``graph_id`` = its own assistant
(one org per graph_id — the SDK dispatch key). ``langgraph.json`` lists each;
this module registers one lazy 0-param FACTORY closure per org
(``graph__<slug>``). langgraph-api classifies any callable attr as a factory
(``_graph_from_spec`` → ``classify_factory``) and invokes a 0-param factory with
no args, so the org graph is built lazily on first request — not all up-front at
import (some orgs need Docker/a key). The manifest is regenerated from the
orgs/ tree by ``scripts/gen_langgraph_json.py``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph


class _ScriptedModel(BaseChatModel):
    """Deterministic supervisor for the keyless contract graphs.

    A real ``BaseChatModel`` so the compiled graph + its JSON Schemas are genuine,
    with the LLM itself scripted (no API key, no Docker). ``_llm_type`` is a
    PROPERTY returning a string — matching ``ChatOpenAI``'s contract — because
    deepagents' ``SummarizationMiddleware`` reads ``model._llm_type`` as a string
    ATTRIBUTE (``model._llm_type.startswith``), not a callable.
    """

    @property
    def _llm_type(self) -> str:
        return "upstream-keystone"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedModel":  # noqa: ANN401
        return self

    def _generate(  # noqa: ANN001
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="upstream ok"))])


def _project_root() -> Path | None:
    """Where the ``orgs/`` tree lives. The parent repo owns it (pux-harness is its
    submodule); ``PUX_PROJECT_ROOT`` overrides for standalone kit use."""
    env = os.environ.get("PUX_PROJECT_ROOT")
    if env:
        return Path(env)
    candidate = Path(__file__).resolve().parents[3]  # .../auto-developer-orchestrator
    return candidate if (candidate / "orgs").is_dir() else None


# Per-org MCP-tool cache for the production runtime lane. ``make_graph`` resolves
# each org's foreign MCP servers ONCE on its first invocation (load time under
# Aegra, when the 0-arg factory is awaited) and the result is cached so aegra's
# graph cache + this never re-probe. A failure caches as ``[]`` — the org starts
# MCP-less with a loud ERROR log; restart after fixing the env. See delta #5 in
# ``docs/AEGRA_PROD.md`` + ``open_org_mcp`` (``agent/mcp_client.py``).
_MCP_CACHE: dict[str, Sequence[BaseTool]] = {}


async def _mcp_for(org: str) -> Sequence[BaseTool]:
    """Resolve + cache ``org``'s MCP tools for the production runtime lane.

    Lazy import mirrors the lazy ``build_graph`` import below — keeps the
    keyless ``keystone`` / ``org`` contract graphs (and ``import upstream`` in
    tests) free of the MCP/Docker dependency surface. Cache-on-failure is
    acceptable: ``make_graph`` runs once per process (aegra caches the built
    graph), and prod env is set at process start, so an empty cache is a
    misconfig SIGNAL (loud ERROR in ``open_org_mcp``), not a steady-state bug.
    """
    if org in _MCP_CACHE:
        return _MCP_CACHE[org]
    from pux_harness.agent.mcp_client import open_org_mcp  # noqa: PLC0415

    _MCP_CACHE[org] = await open_org_mcp(org)
    return _MCP_CACHE[org]


async def make_graph(org: str) -> CompiledStateGraph:
    """Build the ``graph_id`` graph for ``org`` (Agent Protocol / k3s lane).

    ``async`` because the production runtime branch resolves foreign MCP servers
    (``await _mcp_for(org)``) — the thing ``pux direct`` / ``pux acp`` do inline
    but the Aegra lane historically omitted (every org started MCP-less). Aegra
    awaits a 0-arg async factory (``langgraph_service`` invokes
    ``await graph()`` for ``asyncio.iscoroutinefunction(graph)``), so making it
    async is free — the keystone/org contract branches don't resolve MCP.

    Graph selection is EXPLICIT — no silent fallback (see [[no-fallbacks-no-aliases]]):

    * ``PUX_UPSTREAM_GRAPH=keystone`` (or legacy ``PUX_UPSTREAM_DEV``) — a minimal
      scripted ``create_agent`` graph. Portable contract keystone: proves
      langgraph-api serves a pux-style compiled graph with the full SDK surface,
      keyless + Dockerless. Driven by ``scripts/upstream_keystone.py``.
    * ``PUX_UPSTREAM_GRAPH=org`` — the REAL org graph via ``compile_org`` (genuine
      roster + prompt + deepagents middleware) with a scripted supervisor + no
      specialist tools. Keyless/Dockerless contract proof that the real org graph
      SHAPE serves upstream (stronger than the keystone).
    * unset (production / k3s) — ``build_graph(org)``: Docker specialists + real
      model + the declarative stack + the org's MCP tools. Needs the provider key
      + Docker (+ ``GITHUB_TOKEN`` / ``GODOT_PATH`` / ``PUX_MCP_*_URL`` for MCP).
      langgraph-api OWNS the checkpointer (``build_graph`` called with
      ``checkpointer=None``) per [[unified-thread-store]] — the owner shifts from
      pux's ``open_thread_store`` to langgraph-api's runtime checkpointer.

    This is the k3s lane. LOCAL is ACP (``deepagents-acp``, stdio JSON-RPC) — a
    separate surface, not served here (see [[protocol-surface-map]]).
    """
    mode = os.environ.get("PUX_UPSTREAM_GRAPH")
    if mode is None:
        mode = "keystone" if os.environ.get("PUX_UPSTREAM_DEV") else "runtime"

    if mode == "keystone":
        return create_agent(_ScriptedModel(), [])

    if mode == "org":
        from pux_harness.kit.compile import compile_org

        root = _project_root()
        if root is None:
            raise RuntimeError(
                "PUX_UPSTREAM_GRAPH=org needs the orgs/ tree — set PUX_PROJECT_ROOT"
            )
        return compile_org(org, model=_ScriptedModel(), tools=[], project_root=root)

    # runtime (production / k3s): pux's runtime factory. langgraph-api owns the
    # checkpointer + store; passing None lets the runtime inject its own.
    from pux_harness.agent.graph import build_graph
    from pux_harness.agent.stack import RuntimeFacts

    # Resolve this org's foreign MCP tool-servers (github-mcp-server,
    # godot-mcp-runtime, web_research, media, equibles, godot-ai — whichever the
    # org arms). ``open_org_mcp`` is per-ORG isolated: an unset env var or a hung
    # server degrades to zero tools (loud ERROR), never aborts the lane. Cached so
    # the once-per-process resolution isn't repeated.
    mcp_tools = await _mcp_for(org)

    # ``prepare_warmup=True`` arms the PrepareWarmupMiddleware (the
    # ``before_agent`` hook), the serve-lane owner of the ``prepare()`` seam.
    # Aegra owns the run loop itself — unlike ``pux direct`` it has no pux
    # entry point to call ``prepare()`` from, so the warmup (declared
    # ``jobs:`` like ``warmup_browser`` + universal ``warmup_webhook``) would
    # silently NOT fire without this. ``transport`` defaults to ``serve`` ⇒
    # ``universal_warmup=True`` (matches the historical serve lane — formerly
    # the retired ``server.py``). See ``context/prepare_warmup.py``.
    return build_graph(
        org,
        checkpointer=None,
        store=None,
        facts=RuntimeFacts(prepare_warmup=True),
        mcp_tools=mcp_tools,
    )


# --- multi-org factory registration --------------------------------------
# langgraph.json key (graph_id) -> module attr ``graph__<slug>``. Each attr is a
# lazy 0-param factory closing over its org; langgraph-api calls it on first
# request. See module docstring + ``scripts/gen_langgraph_json.py``.

def graph_attr_name(org: str) -> str:
    """Module attr name for ``org``'s graph factory (``graph__<slug>``).

    ``slug`` keeps it a valid Python identifier (hyphens -> underscores) so
    langgraph.json's ``module:variable`` resolves. The graph_id (langgraph.json
    KEY) keeps the org's real name (e.g. ``deep-research-engine``)."""
    slug = re.sub(r"[^0-9a-zA-Z_]", "_", org)
    if not slug or slug[0].isdigit():
        slug = f"_{slug}"
    return f"graph__{slug}"


def discover_upstream_orgs() -> list[str]:
    """Orgs to serve as graph_ids. The full discovered set when the orgs/ tree is
    reachable; otherwise just ``general`` (standalone-kit fallback so the
    keystone smoke still has a graph to serve)."""
    root = _project_root()
    if root is None:
        return ["general"]
    from pux_harness.kit.loaders import discover_orgs

    orgs = discover_orgs(root)
    return orgs if orgs else ["general"]


def _factory_for(org: str) -> Callable[[], Awaitable[CompiledStateGraph]]:
    """A 0-param async factory closure building ``org``'s graph lazily.

    ``async`` so the production runtime branch can ``await`` MCP resolution
    inside ``make_graph``. Aegra's ``langgraph_service`` awaits a 0-arg factory
    when it is an ``asyncio.iscoroutinefunction`` (``graph = await graph()``),
    so the closure is async at every regime — the keyless ``keystone``/``org``
    branches return without awaiting anything."""

    async def factory() -> CompiledStateGraph:
        return await make_graph(org)

    factory.__name__ = graph_attr_name(org)
    factory.__qualname__ = graph_attr_name(org)
    return factory


# Register one factory attr per org at import. langgraph-api resolves
# ``pux_harness.runtime.upstream:graph__<slug>`` from langgraph.json.
for _org in discover_upstream_orgs():
    globals()[graph_attr_name(_org)] = _factory_for(_org)
del _org
