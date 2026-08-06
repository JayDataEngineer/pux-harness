"""deepagents-context — proactive context management for deepagents.

A standalone package extracted from the Pux harness. Provides capture, offload,
and retrieval middleware that keeps large tool results out of the context window
BEFORE they accumulate (proactive), complementing deepagents' reactive
SummarizationMiddleware.

Zero ``pux_harness.*`` imports — pure stdlib + langchain/langgraph/deepagents.

Quick start (native deepagents, no pux required):

    from deepagents_context import (
        EventStore, ContextMiddleware, build_context_layer,
    )

    store = EventStore(".myapp/events.sqlite")
    middleware, tools = build_context_layer(store=store, threshold=8000)
    agent = create_deep_agent(tools=[*my_tools, *tools], middleware=middleware)
"""
from deepagents_context.store import EventStore, shared_event_store
from deepagents_context.snapshot import build_snapshot
from deepagents_context.middleware import ContextMiddleware
from deepagents_context.prompt_capture import PromptCaptureMiddleware
from deepagents_context.session_guide import SessionGuideMiddleware
from deepagents_context.tools import build_context_tools
from deepagents_context.layer import build_context_layer
from deepagents_context.prefix_caching import FullPrefixCachingMiddleware
from deepagents_context.audit import AuditMiddleware
from deepagents_context.web_tools import build_web_tools

__all__ = [
    "EventStore",
    "shared_event_store",
    "build_snapshot",
    "ContextMiddleware",
    "PromptCaptureMiddleware",
    "SessionGuideMiddleware",
    "build_context_tools",
    "build_context_layer",
    "FullPrefixCachingMiddleware",
    "AuditMiddleware",
    "build_web_tools",
]
