"""Agent-callable retrieval surface over the unified context store.

Two tools, both bound to the same ``EventStore`` the middleware writes to:

* ``ctx_search`` — BM25 search across the UNION of offloaded blobs (full tool
  results parked behind ``ctx:<id>``) + structured events (tool-call previews,
  errors, decisions). Returns matching handles + a snippet each, tagged
  ``[blob]`` (recoverable in full via ``ctx_recall``) or ``[event]``.
* ``ctx_recall`` — pull the FULL content of a stashed blob back by its handle.

Only the slice the agent asks for re-enters its context — fewer tokens
recalled = lower cost per call. Both are EXEMPT from offload (the middleware's
``_RETRIEVAL_TOOLS``): their job is to inject content, so re-stashing their
output would trap the agent the instant it retrieves a large stash.

This replaces the old ``event_recent`` / ``event_query`` pair (the
resume snapshot already gives chronological orientation; ``ctx_search`` covers
query-based recall over both blobs and events in one tool).
"""
from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from pux_harness.context.events import EventStore, SearchHit, shared_event_store


class _RecallArgs(BaseModel):
    handle: str = Field(
        ...,
        description='A ctx handle like "ctx:1a2b3c4d5e6f", shown at the top of a truncated tool result.',
    )


class _SearchArgs(BaseModel):
    query: str = Field(
        ...,
        description=(
            "A distinctive phrase to find across prior tool outputs and events. "
            "Ranked by relevance; returns handles + a snippet each."
        ),
    )
    limit: int = Field(8, description="Max hits to return (default 8).")


def _format_hits(hits: list[SearchHit]) -> str:
    if not hits:
        return "no prior tool output or event matched that query."
    lines = [f"{len(hits)} hit(s):"]
    for h in hits:
        tag = f"[{h.kind}]"  # [blob] (recallable in full) or [event]
        tool = f" {h.tool}" if h.tool else ""
        handle = f" {h.handle}" if h.handle else ""
        evtype = f" ({h.type})" if h.type else ""
        lines.append(f"- {tag}{tool}{evtype}{handle}: {h.snippet}")
    return "\n".join(lines)


def build_context_tools(store: EventStore | None = None) -> list[StructuredTool]:
    """The ``ctx_recall`` + ``ctx_search`` tools, bound to ``store`` (default:
    the process-wide shared event store). Built fresh per call so a test can
    pass its own ``EventStore(tmp_path)`` and have offload + recall share it."""
    s = store or shared_event_store()

    def _recall(handle: str) -> str:
        out = s.recall_blob(handle)
        return out if out is not None else f"no truncated result found for handle {handle!r}"

    def _search(query: str, limit: int = 8) -> str:
        return _format_hits(s.search_context(query, limit=limit))

    recall = StructuredTool.from_function(
        _recall,
        name="ctx_recall",
        description=(
            "Return the complete output of a tool call whose result was truncated. "
            "Pass the ctx: handle shown at the top of the truncated result, "
            'e.g. "ctx:1a2b3c4d5e6f".'
        ),
        args_schema=_RecallArgs,
    )
    search = StructuredTool.from_function(
        _search,
        name="ctx_search",
        description=(
            "Search prior tool outputs and events for a phrase; returns matching "
            "handles + a snippet each. Use when you remember a detail but not which "
            "call produced it, then ctx_recall the handle for the full text."
        ),
        args_schema=_SearchArgs,
    )
    return [recall, search]
