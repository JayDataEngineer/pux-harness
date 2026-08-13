"""CompositeBackend instance for memory routing.

Routes ``/memories/*`` to a ``StoreBackend`` (persistent, agent-managed) and
everything else to the existing ``BaseSandbox`` (sandbox fs/shell).

``CompositeBackend`` is built ONCE here (a ``BackendProtocol`` instance) and
passed as ``backend=`` to ``create_deep_agent()``. ``StoreBackend`` takes the
namespace as a *factory* (``namespace=``) it resolves per-access, so no
``Runtime`` is needed at construction — the backend is stateless path-prefix
routing over the shared sandbox backend. (The old callable-factory form
re-invoked construction on every fs op and tripped deepagents' 0.7.0
deprecation: ``backend=`` must be an instance, not a callable.)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.backends.composite import CompositeBackend

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore

    from deepagents.backends.sandbox import BaseSandbox

from pux_harness.memory.config import memory_namespace


def build_memory_backend(
    org: str,
    default_backend: BaseSandbox,
    store: BaseStore | None = None,
) -> tuple[CompositeBackend, BaseStore]:
    """Build the composite backend and store for memory.

    Returns:
        ``(backend, store)`` — the ``CompositeBackend`` instance is passed as
        ``backend=`` to ``create_deep_agent()``; the store is passed as
        ``store=``. Both
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

    # Build the composite backend INSTANCE once, at graph-build time.
    # ``StoreBackend`` takes the namespace as a *factory* (``namespace=``) that
    # it calls per-access with the runtime context — it does NOT need the
    # ``Runtime`` at construction. So unlike the old callable-factory form
    # (which deepagents re-invoked on every fs op), this instance is built
    # once and reused. ``CompositeBackend`` is stateless path-prefix routing
    # over the shared ``default_backend`` + the per-access namespace factory,
    # so a single instance is correct. This also clears deepagents' 0.7.0
    # deprecation: ``backend=`` must be a ``BackendProtocol`` instance, not a
    # callable (the old ``_backend_factory(rt)`` ignored ``rt`` anyway).
    from deepagents.backends.store import StoreBackend  # noqa: PLC0415

    memory_store_be = StoreBackend(
        store=store,
        namespace=namespace_fn,
    )
    backend = CompositeBackend(
        default=default_backend,
        routes={"/memories/": memory_store_be},
    )

    return backend, store
