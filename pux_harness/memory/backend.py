"""CompositeBackend factory for memory routing.

Routes ``/memories/*`` to a ``StoreBackend`` (persistent, agent-managed) and
everything else to the existing ``PuxSandboxBackend`` (sandbox fs/shell).

The ``CompositeBackend`` is constructed as a *factory* (``lambda rt: ...``) so
that ``StoreBackend`` receives the runtime context needed to resolve the
namespace. This matches the deepagents pattern where ``backend=`` can be a
callable that takes ``Runtime`` and returns a ``BackendProtocol``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.state import StateBackend

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore

    from pux_harness.sandbox.backend import PuxSandboxBackend

from pux_harness.memory.config import memory_namespace


def build_memory_backend(
    org: str,
    default_backend: PuxSandboxBackend,
    store: BaseStore | None = None,
) -> tuple[callable, BaseStore]:
    """Build the composite backend and store for memory.

    Returns:
        ``(backend_factory, store)`` — the factory is passed as ``backend=``
        to ``create_deep_agent()``; the store is passed as ``store=``. Both
        share the SAME store object so the graph's store and the
        ``StoreBackend``'s store are one and the same.

        When ``store`` is ``None`` an :class:`InMemoryStore` is created here.
        ``StoreBackend(store=None)`` has NO in-graph fallback — it holds
        ``None`` and crashes on ``store.get`` the first time
        ``MemoryMiddleware.before_agent`` downloads memory files
        (``download_files`` → ``store.get`` → ``AttributeError: 'NoneType'
        object has no attribute 'get'``). So a real store MUST be supplied;
        the ephemeral default gives callers (the ``pux direct`` runner) a
        working one-shot memory without having to know this. Callers wanting
        cross-restart survival (the server) pass their own persistent store.
    """
    from langgraph.store.memory import InMemoryStore

    if store is None:
        # Do NOT pass None through to StoreBackend — see docstring.
        store = InMemoryStore()
    namespace_fn = memory_namespace(org)

    def _backend_factory(rt):
        """Resolve composite backend at graph execution time.

        ``StoreBackend`` needs the runtime to resolve the namespace factory.
        The store is the one resolved above (caller-supplied or the
        ephemeral InMemoryStore default) — never None.
        """
        from deepagents.backends.store import StoreBackend

        memory_store = StoreBackend(
            store=store,
            namespace=namespace_fn,
        )
        return CompositeBackend(
            default=default_backend,
            routes={"/memories/": memory_store},
        )

    return _backend_factory, store
