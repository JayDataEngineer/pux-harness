"""Upstream-runtime graph declaration for ``langgraph-api`` — the OFFICIAL
Agent Protocol server. ``langgraph.json`` points ``langgraph build`` (the k3s
Docker image) — and ``langgraph dev`` (local smoke of THIS lane) — here.

TWO PROTOCOLS, TWO SURFACES (per the pi-pivot scoping):
  * LOCAL  -> Agent CLIENT Protocol (ACP): ``deepagents-acp`` stdio JSON-RPC
              (editors). Served by ``pux acp``; NOT here.
  * k3s    -> Agent Protocol (AP): ``langgraph-api`` HTTP REST (threads/runs/
              store/assistants/SSE). THIS module + ``langgraph.json``.
This is the k3s/AP lane. langgraph-api OWNS the REST surface — pux's hand-rolled
``server.py`` REST lane is retire-eligible (see [[protocol-surface-map]],
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

This ships the ``general`` graph_id; multi-org discovery is a follow-up.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
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


def make_graph(org: str) -> CompiledStateGraph:
    """Build the ``graph_id`` graph for ``org`` (Agent Protocol / k3s lane).

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
      model + the declarative stack. Needs the provider key + Docker. langgraph-api
      OWNS the checkpointer (``build_graph`` called with ``checkpointer=None``) per
      [[unified-thread-store]] — the owner shifts from pux's ``open_thread_store``
      to langgraph-api's runtime checkpointer.

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

    return build_graph(org, checkpointer=None, store=None)


# langgraph.json -> {"graphs": {"general": "pux_harness.runtime.upstream:general"}}
# Each graph is a CompiledStateGraph resolved at import; langgraph-api runs it
# with the runtime's checkpointer + store injected.
general: CompiledStateGraph = make_graph("general")
