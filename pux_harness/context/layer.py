"""The single context-layer wiring seam (Phase 19 unification).

``build_context_layer()`` returns ``(middleware, retrieval_tools)`` — the one
tuple both the MAIN agent (``agent.graph.build_graph``) and EVERY subagent
(``agent.orgs._build_sub``) import, so capture + offload + retrieval reach the
whole agent tree with zero duplication. Both bind to the same process-wide
``EventStore`` (``shared_event_store``), so a blob offloaded by any agent is
recallable by any other via ``ctx_recall`` / ``ctx_search``.

Why a seam object (not two free functions called independently): it guarantees
the middleware and the tools share ONE store instance — building them apart
would let a test pass different stores to each and silently split the layer.
"""
from __future__ import annotations

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import BaseTool

from pux_harness.context.events import EventStore, shared_event_store
from pux_harness.context.middleware import ContextMiddleware
from pux_harness.context.tools import build_context_tools


def build_context_layer(
    store: EventStore | None = None,
    *,
    threshold: int | None = None,
    preview: int | None = None,
    enabled: bool = True,
) -> tuple[list[AgentMiddleware], list[BaseTool]]:
    """Build the context layer: one ``ContextMiddleware`` + the
    ``ctx_recall``/``ctx_search`` retrieval tools, both bound to ``store``
    (default: the shared event store).

    ``threshold``/``preview``/``enabled`` forward to the middleware; ``None``
    means "the middleware default" (8000 / 1500 / True). Returns a fresh tuple
    each call so callers can mutate the lists (e.g. append more middleware)
    without aliasing across the main agent and every subagent."""
    s = store or shared_event_store()
    mw_kwargs: dict[str, object] = {"enabled": enabled}
    if threshold is not None:
        mw_kwargs["threshold"] = threshold
    if preview is not None:
        mw_kwargs["preview"] = preview
    middleware: list[AgentMiddleware] = [ContextMiddleware(s, **mw_kwargs)]
    tools = build_context_tools(s)
    return middleware, tools
