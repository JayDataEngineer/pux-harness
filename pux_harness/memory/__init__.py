"""Agent-managed persistent memory (Phase 18).

Memory files live in ``.pux/memory/`` (gitignored) and are loaded into the
system prompt at conversation start via deepagents' ``MemoryMiddleware``. The
agent reads memory on startup and can write to it during conversations via
``edit_file`` — the model does the work, not the harness.

Architecture:
  - ``CompositeBackend`` routes ``/memories/`` to a ``StoreBackend`` (persistent,
    cross-conversation) and everything else to the existing ``PuxSandboxBackend``.
  - ``StoreBackend`` uses LangGraph's ``BaseStore`` with a project-scoped namespace
    (based on the working directory path) — matching Claude's per-project memory
    model. Each working directory gets its own isolated memory store.
  - ``MemoryMiddleware`` loads ``/memories/AGENTS.md`` at startup and injects it
    into the system prompt inside ``<agent_memory>`` tags.
  - The agent updates memory via ``edit_file`` on the backend, which persists
    through the store.

The runner (``main.py``) passes ``store=None`` and gets an ephemeral
``InMemoryStore`` (created inside ``build_memory_backend``); the server
(``server.py``) passes its own store for cross-restart survival. A ``None``
store can NOT reach ``StoreBackend`` — it has no in-graph fallback and crashes
on the first ``download_files``.
"""
from __future__ import annotations

from pux_harness.memory.backend import build_memory_backend
from pux_harness.memory.config import MEMORY_SOURCES, memory_namespace

__all__ = ["MEMORY_SOURCES", "build_memory_backend", "memory_namespace"]
