"""Memory paths and namespace configuration.

Memory lives in ``.pux/memory/`` (gitignored) — agent-managed, not tracked.
The namespace is based on the project root path, matching Claude's per-project
memory model: each working directory gets its own isolated memory store.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

MEMORY_SOURCES: list[str] = ["/memories/AGENTS.md"]
"""Memory file paths loaded at conversation start. The agent can create
additional files under ``/memories/`` during conversations, but only these
are injected into the system prompt at startup."""


def _project_key() -> str:
    """Derive a stable project key from the working directory.

    Flattens the absolute path like Claude does:
    ``/home/ubuntu/.../auto-developer-orchestrator``
    becomes ``-home-ubuntu-...-auto-developer-orchestrator``.
    """
    return str(Path.cwd()).replace("/", "-").lstrip("-")


def memory_namespace(org: str) -> callable:
    """Return a namespace factory that scopes memory to the project.

    The factory signature matches ``StoreBackend``'s ``NamespaceFactory``
    protocol: ``(Runtime) -> tuple[str, ...]``.

    Using the project root path (not the org name) means memory is shared
    across all orgs in the same working directory — matching Claude's
    per-project memory model.
    """

    def _namespace(rt: Runtime) -> tuple[str, ...]:
        return (_project_key(),)

    return _namespace
