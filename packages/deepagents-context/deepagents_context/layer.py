"""The single context-layer wiring seam.

``build_context_layer()`` returns ``(middleware, retrieval_tools)`` — the one
tuple both the MAIN agent and EVERY subagent import, so capture + offload +
retrieval reach the whole agent tree with zero duplication. Both bind to the
same ``EventStore``, so a blob offloaded by any agent is recallable by any
other via ``ctx_recall`` / ``ctx_search``.

Why a seam object (not two free functions called independently): it guarantees
the middleware and the tools share ONE store instance — building them apart
would let a test pass different stores to each and silently split the layer.

Pux-specific exec tools (Docker) are injected via ``extra_tools`` by the pux
adapter — this package has no Docker dependency.
"""
from __future__ import annotations

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import BaseTool

from deepagents_context.store import EventStore, shared_event_store
from deepagents_context.middleware import ContextMiddleware
from deepagents_context.tools import build_context_tools


def build_context_layer(
    store: EventStore | None = None,
    *,
    threshold: int | None = None,
    preview: int | None = None,
    enabled: bool = True,
    extra_tools: list[BaseTool] | None = None,
) -> tuple[list[AgentMiddleware], list[BaseTool]]:
    """Build the context layer: one ``ContextMiddleware`` + the
    ``ctx_recall``/``ctx_search`` retrieval tools, both bound to ``store``
    (default: the shared event store).

    ``threshold``/``preview``/``enabled`` forward to the middleware; ``None``
    means "the middleware default" (8000 / 1500 / True). ``extra_tools`` is an
    optional list of pre-built tools to append (e.g. Docker exec tools injected
    by the pux adapter). Returns a fresh tuple each call so callers can mutate
    the lists without aliasing across the main agent and every subagent."""
    s = store or shared_event_store()
    mw_kwargs: dict[str, object] = {"enabled": enabled}
    if threshold is not None:
        mw_kwargs["threshold"] = threshold
    if preview is not None:
        mw_kwargs["preview"] = preview
    middleware: list[AgentMiddleware] = [ContextMiddleware(s, **mw_kwargs)]
    tools = build_context_tools(s, extra_tools=extra_tools)
    return middleware, tools
