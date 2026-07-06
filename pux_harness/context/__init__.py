"""Unified context-saving layer (Phase 19 unification).

One store, one middleware, one search surface, every agent.

- ``events`` — the ``EventStore`` (``.pux/events.sqlite``): structured events
  (tool calls, errors, decisions, …) AND offloaded blobs (full oversized tool
  results behind ``ctx:<id>`` handles), both FTS5/BM25 searchable.
- ``middleware`` — ``ContextMiddleware``: in ONE ``wrap_tool_call`` pass it
  captures each tool call as an event AND offloads oversized results to a blob.
- ``tools`` — the ``ctx_recall`` (full blob by handle) + ``ctx_search`` (BM25
  over events+blobs) retrieval surface.
- ``layer`` — ``build_context_layer()``: the single seam that returns the
  middleware + retrieval tools; imported by BOTH ``agent.graph`` (main agent)
  and ``agent.orgs._build_sub`` (every subagent), so capture + offload +
  retrieval reach the whole agent tree.

Downstream (snapshot.py, session_guide.py) read the same store for the
structured compaction snapshot + cross-session rehydration.
"""
