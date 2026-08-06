"""deepagents-context — proactive context management for deepagents.

A standalone package extracted from the Pux harness. Provides:

- ``EventStore`` — unified SQLite store for structured events + offloaded blobs,
  behind a single FTS5 (BM25) search surface.
- ``shared_event_store()`` — process-wide singleton accessor.
- ``ContextMiddleware`` — capture + offload in one ``wrap_tool_call`` pass
  (coming as files migrate in).
- ``build_context_layer()`` — the one-stop wiring seam (coming).

Zero ``pux_harness.*`` imports — pure stdlib + langchain/langgraph/deepagents.
"""
from deepagents_context.store import EventStore, shared_event_store

__all__ = ["EventStore", "shared_event_store"]
