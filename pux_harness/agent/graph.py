"""Per-org deepagents graph builder, shared by the in-process runner
(``main.py``) and the Agent Protocol server (Aegra / ``langgraph-api`` in prod;
``langgraph dev`` / ``aegra dev`` in dev).

One ``BaseSandbox`` + one ``ExecClient`` adapter serve the whole process
(constructed in ``sandbox.exec`` — OpenShell by default, local-filesystem
fallback). Per-org compiled graphs are built lazily and cached by the caller —
building is expensive (model init + subagent assembly) and the only per-org
variation is system_prompt + subagents + the specialist-tool whitelist. All 13
specialists are native Python tools (the Go bridge that used to supply them
over MCP was deleted; the in-process Docker reimplementation was deleted in
favor of the upstream OpenShell ``BaseSandbox``).

**This module is THIN.** It owns the runtime DEPS (the model, the
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
import os

from deepagents import create_deep_agent
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from deepagents.backends.sandbox import BaseSandbox

from pux_harness.agent.model import get_model
from pux_harness.agent.profile import load_profile, load_rubric_gate
from pux_harness.agent.stack import RuntimeFacts, build_stack
from pux_harness.memory import MEMORY_SOURCES, build_memory_backend
from pux_harness.sandbox.exec import (
    ExecClient,
    shared_backend as _shared_backend,
    shared_exec as _shared_exec,
)
from pux_harness.sandbox.tools import make_specialist_tools


def shared_exec() -> ExecClient:
    """One exec client for the process (lazy — backed by the shared BaseSandbox)."""
    return _shared_exec()


def shared_backend() -> BaseSandbox:
    """One sandbox backend for the process (lazy — OpenShell or local)."""
    return _shared_backend()


def build_graph(
    org: str,
    *,
    checkpointer: Any,
    store: Any | None = None,
    facts: RuntimeFacts | None = None,
    mcp_tools: Sequence[BaseTool] = (),
    base_model_override: str | None = None,
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

    ``facts`` carries RUNTIME-level decisions (transport, autonomous) that
    ``build_stack`` can't derive from the org alone — today the ``ask_user``
    HITL tool's construction gate (web → interrupt; editor → turn-based; mcp /
    autonomous → dropped). ``None`` → the default ``RuntimeFacts()`` (transport
    ``serve``, not autonomous), which is correct for the runner + the AG-UI web
    path; the ACP / direct / mcp entrypoints pass a real ``RuntimeFacts``.

    ``store`` is an optional ``BaseStore`` for persistent memory.
    When provided, memory survives server restarts. When ``None`` (the runner
    default), ``build_memory_backend`` supplies an ephemeral ``InMemoryStore``
    — a real store is required because ``StoreBackend(store=None)`` crashes on
    the first ``download_files`` (no in-graph fallback exists).
    """
    # Roles: the CTO runs on `base`; describe_image runs on
    # `multimodal` (decoupled so an org can pin a vision model != the driver).
    # Both resolve through models.yaml + org profile + env, never a hardcoded id.
    # ``base_model_override`` (from the ACP model picker, via ``context.model``)
    # re-pins the supervisor/base role to a literal id; None → the org's tier
    # base. An explicit pin disables the tier's ``base_fallbacks`` (see
    # ``resolve_model_id``), matching the frontmatter/org/env override stack.
    base_model = get_model(role="base", org=org, model=base_model_override)
    specialists = make_specialist_tools(
        shared_backend(), vision_model=get_model(role="multimodal", org=org), org=org,
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
        facts=facts,
        mcp_tools=mcp_tools,
    )

    # Agent-managed persistent memory. The composite backend routes
    # /memories/ to a StoreBackend (project-scoped namespace) and everything
    # else to the existing PuxSandboxBackend. MemoryMiddleware loads
    # /memories/AGENTS.md at startup and injects it into the system prompt.
    # The agent updates memory via edit_file — the model does the work.
    memory_backend, memory_store = build_memory_backend(
        org=org,
        default_backend=shared_backend(),
        store=store,
    )

    transport = (facts.transport if facts else "serve")
    graph = create_deep_agent(
        model=base_model,
        system_prompt=plan.supervisor_prompt,
        tools=plan.supervisor_tools,
        memory=MEMORY_SOURCES,
        subagents=plan.subagents,
        middleware=plan.supervisor_middleware,
        # Native SkillsMiddleware on the supervisor (progressive
        # disclosure: skill metadata in the prompt, body via read_file).
        # ``None`` for a no-skills org (supervisor_skills == []) -> deepagents
        # mounts no SkillsMiddleware (byte-identical to the legacy build).
        skills=plan.supervisor_skills or None,
        backend=memory_backend,
        store=memory_store,
        checkpointer=checkpointer,
    )
    # LangGraph-native stream-stall retry. Attaches a RetryPolicy to the
    # ``model`` node so a stalled upstream model stream (TCP alive, provider
    # silent → StreamChunkTimeoutError, a subclass of asyncio.TimeoutError,
    # after stream_chunk_timeout seconds) is retried IN-PLACE — the
    # checkpointer preserves every node that completed before the stall,
    # the stalled model node picks up from its own beginning, and the rest
    # of the turn continues normally. No work re-done. No exception
    # propagated to the caller. No '⚠️ This turn ended early' notice.
    #
    # This is the PRIMARY retry layer. ``pux_harness.acp.PuxAgentServer.prompt``
    # has a SECONDARY wrapper-level retry as defense-in-depth for stalls that
    # escape the model node (e.g. a stall inside a middleware running in a
    # different node). See ``agent/retry.py`` for the full rationale and the
    # shared classifier.
    from pux_harness.agent.retry import attach_stream_stall_retry
    attach_stream_stall_retry(graph)
    return _with_langfuse_tracing(graph, org, transport=transport)


def _with_langfuse_tracing(
    graph: CompiledStateGraph, org: str, *, transport: str = "serve"
) -> CompiledStateGraph:
    """Wrap a compiled graph's ``astream`` to inject the Langfuse CallbackHandler.

    Uses LangChain's standard callback system — the handler automatically traces
    every LLM call, tool use, subagent spawn, and reasoning step in the
    deepagents graph. Each ``astream`` call = one Langfuse trace; the
    ``thread_id`` from the config becomes the ``langfuse_session_id`` (groups
    all prompts in a thread into one session in the UI); org/transport become
    tags (UI filter axes).

    Applied centrally in ``build_graph`` so EVERY entry point (direct, acp,
    serve, mcp, tui) produces traces — not just ACP. The ``org`` parameter
    tags the trace with the originating org for UI filtering.

    No-op when langfuse is not installed or credentials are absent — the graph
    runs identically with or without. A down/unreachable Langfuse host never
    blocks a run: the handler queues locally and flushes best-effort.
    """
    try:
        from langfuse.langchain import CallbackHandler as _LangfuseHandler
    except ImportError:
        return graph

    if not (
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
    ):
        return graph

    _orig_astream = graph.astream

    def _traced_astream(input, config=None, **kwargs):
        """Synchronous wrapper that injects Langfuse callbacks into config,
        then delegates to the original ``astream``.

        MUST be a regular function (NOT ``async def``) because ``astream``
        returns an async ITERATOR (you ``async for chunk in graph.astream(...)``),
        not a coroutine. An ``async def`` wrapper returns a coroutine, which
        breaks ``async for`` with ``TypeError: __aiter__``.

        ``config`` is explicitly the 2nd positional arg (matching
        ``Pregel.astream(input, config=None, *, ...)``).  Using ``*args``
        + ``config=config`` caused "got multiple values for argument 'config'"
        because ``ainvoke`` passes config positionally: ``astream(input, config)``
        — our ``*args`` captured it AND we re-passed it as keyword.
        """
        if config is None:
            config = {}
        # Only inject if not already present (avoid double-tracing on re-entry)
        if not config.get("callbacks"):
            thread_id = (
                config.get("configurable", {}).get("thread_id", "unknown")
                if isinstance(config.get("configurable"), dict)
                else "unknown"
            )
            handler = _LangfuseHandler()
            config["callbacks"] = [handler]
            config["metadata"] = config.get("metadata") or {}
            config["metadata"].setdefault("langfuse_session_id", thread_id)
            config["metadata"].setdefault(
                "langfuse_tags", [f"org:{org}", f"transport:{transport}"]
            )
        return _orig_astream(input, config, **kwargs)

    graph.astream = _traced_astream
    return graph
