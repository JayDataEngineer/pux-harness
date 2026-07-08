"""Upstream-runtime graph declaration for ``langgraph-api`` — the OFFICIAL
Agent Protocol server. ``langgraph.json`` points ``langgraph dev`` (localhost)
and ``langgraph build`` (Docker image) here.

UPSTREAM CONTRACT BINDING (the pi-pivot): the threads / runs / store /
assistants / SSE REST surface is owned by ``langgraph-api`` (LangChain's
reference Agent Protocol server), NOT by pux's hand-rolled ``server.py``. Each
org = one ``graph_id`` = one assistant; the wire format is ``langgraph_sdk``'s,
so every consumer that already talks to ``server.py`` keeps working unchanged
against the upstream runtime. See [[protocol-surface-map]],
[[rely-on-upstream]], [[no-legacy-left-behind]].

Two regimes, ONE ``graph_id``, explicit (no silent fallback — see
[[no-fallbacks-no-aliases]]):

* ``PUX_UPSTREAM_DEV=1`` -> a deterministic scripted-model graph (the contract
  keystone: proves langgraph-api serves a pux-style compiled graph with the full
  SDK surface, keyless + Dockerless — the live-server analog of the test
  harness). Driven by ``scripts/upstream_keystone.py``.
* absent               -> the real ``build_graph(org)`` runtime factory (Docker
  specialists + real model + the declarative stack). langgraph-api OWNS the
  checkpointer (``build_graph`` is called with ``checkpointer=None`` here) per
  [[unified-thread-store]] — the owner shifts from pux's ``open_thread_store`` to
  langgraph-api's runtime checkpointer.

P2 declares every discovered org as its own ``graph_id``; this module ships the
keystone (``general``) now.
"""

from __future__ import annotations

import os
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph.state import CompiledStateGraph


class _ScriptedModel(BaseChatModel):
    """Deterministic supervisor for the ``PUX_UPSTREAM_DEV`` keystone graph.

    Mirrors the test-harness scripted models: a real ``BaseChatModel`` so the
    compiled graph + its JSON Schemas are genuine, with the LLM itself scripted
    (no API key, no Docker).
    """

    def _llm_type(self) -> str:
        return "upstream-keystone"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedModel":  # noqa: ANN401
        return self

    def _generate(  # noqa: ANN001
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="upstream ok"))])


def make_graph(org: str) -> CompiledStateGraph:
    """Build the ``graph_id`` graph for ``org``.

    ``PUX_UPSTREAM_DEV`` selects the deterministic keystone graph; otherwise the
    real runtime factory. langgraph-api injects its OWN checkpointer at runtime,
    so the production path passes ``checkpointer=None`` (the graph must not carry
    a competing checkpointer — see [[unified-thread-store]]).
    """
    if os.environ.get("PUX_UPSTREAM_DEV"):
        return create_agent(_ScriptedModel(), [])

    # Production: pux's runtime factory. langgraph-api owns the checkpointer +
    # store; passing None lets the runtime inject its own (the whole point of
    # binding the checkpoint+memory surface upstream too).
    from pux_harness.agent.graph import build_graph

    return build_graph(org, checkpointer=None, store=None)


# langgraph.json -> {"graphs": {"general": "pux_harness.runtime.upstream:general"}}
# Each graph is a CompiledStateGraph resolved at import; langgraph-api runs it
# with the runtime's checkpointer + store injected.
general: CompiledStateGraph = make_graph("general")
