"""Per-org deepagents graph builder, shared by the in-process runner
(``main.py``) and the Agent Protocol server (``server.py``).

One ``DockerExecClient`` + one ``PuxSandboxBackend`` serve the whole process
(the client is a thin Docker SDK wrapper; the backend is stateless apart from
an observation log). Per-org compiled graphs are built lazily and cached by
the caller — building is expensive (model init + subagent assembly) and the
only per-org variation is system_prompt + subagents + the specialist-tool
whitelist. All 13 specialists are native Python tools (Phase 8i deleted the Go
bridge that used to supply them over MCP).

**Phase 21 — this module is THIN.** It owns the runtime DEPS (the model, the
specialist tools, the loaded profile + rubric gate, the memory backend, the
checkpointer) and the final BINDING (``create_deep_agent``), but NO stack
assembly — every middleware, the prompt, the tool list, and the subagents are
resolved by the factory ``stack.build_stack``. There is no second
hand-maintained middleware list here; the ``no-legacy-middleware-in-graph``
contract tripwire enforces that (this module imports NEITHER
``RoutingMiddleware`` / ``SessionGuideMiddleware`` / ``RubricMiddleware``).
"""

from __future__ import annotations

from typing import Any, Sequence

from deepagents import create_deep_agent
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from pux_harness.agent.model import get_model
from pux_harness.agent.profile import load_profile, load_rubric_gate
from pux_harness.agent.stack import build_stack
from pux_harness.memory import MEMORY_SOURCES, build_memory_backend
from pux_harness.sandbox.backend import PuxSandboxBackend
from pux_harness.sandbox.docker_exec import DockerExecClient, get_exec_client
from pux_harness.sandbox.tools import build_native_specialists

_exec: DockerExecClient | None = None  # direct docker exec — fs/shell + specialists
_backend: PuxSandboxBackend | None = None


def shared_exec() -> DockerExecClient:
    """One docker-exec client for the process (lazy — discovery hits Docker)."""
    global _exec
    if _exec is None:
        _exec = get_exec_client()
    return _exec


def shared_backend() -> PuxSandboxBackend:
    """One sandbox backend over the shared docker-exec client."""
    global _backend
    if _backend is None:
        _backend = PuxSandboxBackend(shared_exec())
    return _backend


def build_graph(
    org: str,
    *,
    checkpointer: Any,
    store: Any | None = None,
    mcp_tools: Sequence[BaseTool] = (),
) -> CompiledStateGraph:
    """Compile the deepagents graph for ``org`` against ``checkpointer``.

    Specialist ``pux_sandbox_*`` tools come from ``tools=`` (all native);
    native fs/shell tools come from ``FilesystemMiddleware`` via the shared
    backend (auto-injected into the main agent + every subagent by
    ``create_deep_agent``). The checkpointer is caller-supplied so the runner
    can use an ephemeral ``MemorySaver`` while the server uses a persistent
    ``AsyncSqliteSaver``.

    The whole stack — supervisor tools, supervisor middleware (in order),
    supervisor prompt, and the compiled subagents — is resolved by
    ``stack.build_stack`` from the deps below + the org's ``profile.yaml``.
    This function only supplies those deps + the memory backend + checkpointer
    and binds the result via ``create_deep_agent``.

    Phase 18: ``store`` is an optional ``BaseStore`` for persistent memory.
    When provided, memory survives server restarts. When ``None`` (the runner
    default), ``build_memory_backend`` supplies an ephemeral ``InMemoryStore``
    — a real store is required because ``StoreBackend(store=None)`` crashes on
    the first ``download_files`` (no in-graph fallback exists).
    """
    # Roles (Phase 17.B.0): the CTO runs on `base`; describe_image runs on
    # `multimodal` (decoupled so an org can pin a vision model != the driver).
    # Both resolve through models.yaml + org profile + env, never a hardcoded id.
    base_model = get_model(role="base", org=org)
    specialists = build_native_specialists(
        shared_exec(), vision_model=get_model(role="multimodal", org=org), org=org,
        backend=shared_backend(),
    )
    cfg = load_profile(org)
    gate = load_rubric_gate(org)

    # The factory owns everything that varies per-org across tools, middleware,
    # subagents, and the prompt. Byte-identical to the pre-factory build when
    # the org ships no profile (same middleware order, same tools, same prompt).
    plan = build_stack(
        org,
        specialists=specialists,
        profile=cfg,
        rubric_gate=gate,
        exec_client=shared_exec(),
        mcp_tools=mcp_tools,
    )

    # Phase 18: agent-managed persistent memory. The composite backend routes
    # /memories/ to a StoreBackend (project-scoped namespace) and everything
    # else to the existing PuxSandboxBackend. MemoryMiddleware loads
    # /memories/AGENTS.md at startup and injects it into the system prompt.
    # The agent updates memory via edit_file — the model does the work.
    memory_backend, memory_store = build_memory_backend(
        org=org,
        default_backend=shared_backend(),
        store=store,
    )

    return create_deep_agent(
        model=base_model,
        system_prompt=plan.supervisor_prompt,
        tools=plan.supervisor_tools,
        memory=MEMORY_SOURCES,
        subagents=plan.subagents,
        middleware=plan.supervisor_middleware,
        backend=memory_backend,
        store=memory_store,
        checkpointer=checkpointer,
    )
