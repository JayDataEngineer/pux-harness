"""Pux Agent Protocol server — serves the deepagents org graphs over the
LangChain Agent Protocol REST subset AND the AG-UI protocol for CopilotKit.

**Org → agent_id.** One ``AsyncSqliteSaver`` (persistent threads/history) is
shared across all org graphs; per-org compiled graphs are cached lazily. Runs
are ephemeral executions tracked in-memory; the durable thread state (the
messages + checkpoints) lives in SQLite. A small ``pux_threads`` index table
in the same DB maps thread_id → org (so a thread remembers which org's graph
owns it across restarts).

Endpoints (subset of the published Agent Protocol spec —
https://langchain-ai.github.io/agent-protocol/):

  GET    /ok                          health
  POST   /agents/search               list orgs as agents
  GET    /agents/{agent_id}           org info
  POST   /threads                     create a thread for an agent
  POST   /threads/search              list/search threads
  GET    /threads/{thread_id}         thread state
  DELETE /threads/{thread_id}         delete a thread
  GET    /threads/{thread_id}/history revision history (langgraph checkpoints)
  POST   /threads/{thread_id}/runs    background run -> run_id
  GET    /threads/{thread_id}/runs    list a thread's runs
  POST   /runs/wait                   ephemeral blocking run (create+run+return)
  GET    /runs/{run_id}/wait          block for a background run's final output
  POST   /runs/{run_id}/cancel        cancel a background run
  POST   /runs/stream                 ephemeral run -> SSE event stream
  POST   /threads/{thread_id}/runs/stream  thread run -> SSE event stream

**SSE wire format:** standard SSE frames ``event: <mode>\\ndata: <json>\\n\\n``
— the exact bytes the ``langgraph_sdk`` ``SSEDecoder`` parses, so a ``RunClient``
consumes our stream with no adapter. Events: a leading ``metadata`` (run_id),
then ``messages`` (token + tool-call chunks), ``updates`` (per-node diffs), and
``values`` (full state after each step); ``error`` on exception. An
``interrupt()`` (e.g. ``ask_user``) surfaces as a ``__interrupt__`` key in the
final ``values``/``updates`` events — the client reads it + resumes by POSTing a
new run with ``command={"resume": ...}`` on the same thread.

**Interrupt correctness (the #3 fix):** ``ainvoke``/``astream`` RETURN on
``interrupt()`` (they do not hang) — ``state.next`` is non-empty. So the polled
endpoints report ``status="interrupted"`` + the interrupt payload, never a
silent ``status="success"`` with a stale tool-call message; the stream emits the
interrupt as an event. Resume drives ``ainvoke(Command(resume=...))``.

**Implementation choice:** thin FastAPI implementing the published spec, NOT
``langgraph-api`` (the Platform runtime). Rationale: minimalist (we own the LOC
vs adopting an opinionated runtime), and the REST contract is identical either
way — swapping the server impl behind these endpoints is invisible to clients,
so the choice is reversible.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import StreamingResponse
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel

from langchain_core.tools import BaseTool

from pux_harness.agent.graph import build_graph
from pux_harness.agent.observability import build_invoke_config
from pux_harness.agent.stack import RuntimeFacts, autonomous_from_env
from pux_harness.threads import open_thread_store
from pux_harness.agent.orgs import discover_orgs, org_agent_slugs
from pux_harness.agent.profile import default_rubric
from pux_harness.agent.tool_servers import resolve_tool_servers
from pux_harness.run_events import EventBus

PUX_API_HOST = os.environ.get("PUX_API_HOST", "127.0.0.1")
PUX_API_PORT = int(os.environ.get("PUX_API_PORT", "9988"))
DEFAULT_RECURSION_LIMIT = 60


# --- request / response models (kept loose: the spec's input/output is free-form) -


class ThreadCreate(BaseModel):
    agent_id: str
    metadata: dict[str, Any] = {}


class ThreadSearch(BaseModel):
    metadata: dict[str, Any] | None = None
    agent_id: str | None = None


class AssistantSearch(BaseModel):
    """``POST /assistants/search`` body (the SDK ``assistants.search``).
    ``graph_id`` narrows to one org (= one assistant); ``metadata`` is accepted
    for wire compatibility but ignored (org metadata is declarative)."""

    metadata: dict[str, Any] | None = None
    graph_id: str | None = None
    limit: int = 10
    offset: int = 0


class RunCreate(BaseModel):
    input: Any = None  # str | {"messages": [...]} | dict
    metadata: dict[str, Any] = {}
    recursion_limit: int = DEFAULT_RECURSION_LIMIT
    # Resume an interrupted run: ``{"resume": <value>}`` drives
    # ``Command(resume=<value>)`` on the same thread (the value becomes the
    # ``interrupt()`` return value). Absent => a fresh-input run.
    command: dict[str, Any] | None = None
    # Push-notification callback: when this background run reaches a terminal
    # state, its metadata is POSTed here so a fire-and-forget caller (MCP
    # ``start_run``) learns the run finished WITHOUT polling ``list_runs``.
    # Falls back to ``PUX_RUN_WEBHOOK_URL`` when unset. Best-effort — see
    # ``_dispatch_run_webhook``.
    webhook_url: str | None = None


class EphemeralRun(BaseModel):
    agent_id: str
    input: Any = None
    metadata: dict[str, Any] = {}
    recursion_limit: int = DEFAULT_RECURSION_LIMIT
    command: dict[str, Any] | None = None  # resume — see RunCreate.command


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_input(raw: Any) -> dict[str, Any]:
    """Accept a plain task string OR a full ``{"messages": [...]}`` dict and
    return the deepagents graph input shape."""
    if isinstance(raw, str):
        return {"messages": [{"role": "user", "content": raw}]}
    if isinstance(raw, dict) and "messages" in raw:
        return raw
    if raw is None:
        raise HTTPException(status_code=422, detail="run `input` is required")
    # any other dict: treat as a single user message (json-serialized)
    return {"messages": [{"role": "user", "content": json.dumps(raw)}]}


def _final_answer(result: dict[str, Any]) -> str:
    msgs = result.get("messages") or []
    if not msgs:
        return ""
    last = msgs[-1]
    content = getattr(last, "content", last)
    if isinstance(content, list):  # multimodal blocks → extract text per block
        parts: list[str] = []
        for c in content:
            parts.append(str(c.get("text", c)) if isinstance(c, dict) else str(c))
        content = "\n".join(parts)
    return str(content) if content else ""


def _agent_descriptor(org: str) -> dict[str, Any]:
    slugs = org_agent_slugs(org)
    return {
        "agent_id": org,
        "name": org,
        "description": f"Pux org '{org}'; specialists: {', '.join(slugs) or '(none)'}",
        "metadata": {"specialists": slugs},
    }


def _assistant_descriptor(org: str) -> dict[str, Any]:
    """The langgraph-api ``Assistant`` shape — what the SDK ``assistants`` client
    + Studio's agent picker consume. An org IS an assistant: ``assistant_id`` and
    ``graph_id`` both carry the org name (one compiled graph per org). Read-only
    (orgs are declarative — created on disk, not via POST /assistants)."""
    slugs = org_agent_slugs(org)
    ts = getattr(app.state, "started_at", None) or _now()
    return {
        "assistant_id": org,
        "graph_id": org,
        "name": org,
        "description": f"Pux org '{org}'; specialists: {', '.join(slugs) or '(none)'}",
        "metadata": {"created_by": "system", "specialists": slugs},
        "config": {},
        "context": {},
        "version": 1,
        "created_at": ts,
        "updated_at": ts,
    }


def _json_schema(model: Any) -> dict[str, Any]:
    """Coerce a langgraph/pydantic schema object to a JSON Schema dict — the
    shape ``/assistants/{id}/schemas`` returns for each of state/input/output/
    config. Falls back to ``{"type": "object"}`` when the compiled graph doesn't
    expose a given schema (e.g. a stub graph in tests, or an older langgraph).

    Subtlety: ``config_schema`` is a bound METHOD (calling it yields the schema
    model), while ``input_schema``/``output_schema`` ARE the schema model classes
    themselves — so only invoke non-class callables. Instantiating a class with
    required fields would ValidationError and silently drop a real schema."""
    if model is None:
        return {"type": "object"}
    if not isinstance(model, type) and callable(model):
        try:
            model = model()
        except Exception:  # noqa: BLE001 - schema extraction must never 500
            return {"type": "object"}
    dump = getattr(model, "model_json_schema", None)
    if callable(dump):
        try:
            return dump()
        except Exception:  # noqa: BLE001
            return {"type": "object"}
    if isinstance(model, dict):
        return model
    return {"type": "object"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The shared checkpointer + pux_threads index live in ONE sqlite file owned
    # by open_thread_store(). server / acp / direct all share it, so a
    # thread created by `pux direct` is visible to `pux show` / `pux resume`.
    async with open_thread_store() as store:
        app.state.saver = store.saver
        app.state.db = store.db
        app.state.store = store
        app.state.graphs: dict[str, CompiledStateGraph] = {}
        app.state.runs: dict[str, asyncio.Task] = {}
        app.state.run_meta: dict[str, dict[str, Any]] = {}
        app.state.mcp: dict[str, list[BaseTool]] = {}
        # Stable per-process timestamp for the assistant descriptors' created_at/
        # updated_at — orgs are declarative (discovered from disk), so this server
        # start is the natural "when this assistant began being served".
        app.state.started_at = _now()
        # ONE shared long-term-memory store (BaseStore) for every org + the
        # /store/* REST surface. Namespaces isolate tenants/threads, so a single
        # store is correct (and the only way a REST ``put_item`` is visible to a
        # graph's memory tools — previously each org built a private InMemoryStore
        # in ``_get_graph``, invisible across the seam). Ephemeral by default
        # (matches the localhost single-user model); swap to AsyncSqliteStore for
        # persistence in a follow-up.
        from langgraph.store.memory import InMemoryStore  # noqa: PLC0415

        app.state.base_store = InMemoryStore()

        # Run-completion event bus — the receiver-of-last-resort that lives ON
        # the pux side (see run_events.py). An MCP client with no webhook
        # receiver (Hermes) subscribes to GET /events/stream once and gets every
        # background-run completion across all orgs, instead of polling
        # list_runs. Persisted to .pux/run_events.jsonl for cross-restart
        # catch-up (.pux is HARD_EXCLUDE from packs).
        from pux_harness.kit._paths import project_root  # noqa: PLC0415

        app.state.events = EventBus(log_path=project_root() / ".pux" / "run_events.jsonl")

        # Load foreign MCP tool servers for every org that declares them.
        from pux_harness.agent.mcp_client import McpSessionManager  # noqa: PLC0415

        _mcp_managers: dict[str, McpSessionManager] = {}
        for org_name in discover_orgs():
            try:
                specs = resolve_tool_servers(org_name)
            except ValueError:
                specs = []
            if specs:
                mgr = McpSessionManager(org_name, specs)
                await mgr.open()
                _mcp_managers[org_name] = mgr
                app.state.mcp[org_name] = mgr.tools

        # Register AG-UI endpoints now that the checkpointer is ready — but only
        # for orgs that declare the `agui` surface in policy.yaml (absent ->
        # DEFAULT both, so this only ever NARROWS: an org opting `protocols:
        # [acp]` is excluded from the web/AG-UI mount). See
        # policy.protocols_for_org.
        if _HAS_AG_UI:
            from pux_harness.sandbox import policy as policy_mod  # noqa: PLC0415
            from pux_harness.kit._paths import project_root  # noqa: PLC0415

            # The module-level ``app`` is reused across lifespans (in tests every
            # TestClient context re-enters lifespan; production runs one). Mounting
            # APPENDS a route per call, so without dropping the prior mount a later
            # request would still route to the FIRST registration — a graph bound
            # to the previous lifespan's sqlite checkpointer, now closed
            # (``ValueError: no active connection``). Rebuild the /agui/* surface
            # fresh each lifespan so the active graph's connection is live.
            # No-op on a fresh app; correct for any in-process re-mount.
            app.router.routes[:] = [
                r for r in app.router.routes if not getattr(r, "path", "").startswith("/agui/")
            ]
            for org_name in discover_orgs():
                if "agui" not in policy_mod.protocols_for_org(org_name, project_root()):
                    continue
                add_langgraph_fastapi_endpoint(
                    app=app,
                    agent=LangGraphAgent(
                        name=org_name,
                        graph=_get_graph(org_name),
                        description=f"Pux org '{org_name}'",
                    ),
                    path=f"/agui/{org_name}",
                )

        try:
            yield
        finally:
            for mgr in _mcp_managers.values():
                await mgr.close()
            # open_thread_store() owns the connection close on context exit.


app = FastAPI(title="Pux Agent Protocol", version="0.1.0", lifespan=lifespan)

# ── AG-UI support (lazy import, registered after lifespan starts) ─────────────
_HAS_AG_UI = False
try:
    # Both pieces come from ONE upstream (ag-ui-langgraph): the endpoint AND the
    # agent class. ``add_langgraph_fastapi_endpoint`` is type-annotated to accept
    # ``ag_ui_langgraph.agent.LangGraphAgent``, so that is what we pass — not the
    # deprecated ``copilotkit.LangGraphAGUIAgent`` (the copilotkit dep was dropped
    # in 76659a0; the orphaned import survived only while the package lingered in
    # the venv). The interrupt handling the ask_user web proof depends on lives in
    # ag_ui_langgraph/agent.py, so LangGraphAgent covers it natively.
    from ag_ui_langgraph import LangGraphAgent, add_langgraph_fastapi_endpoint  # noqa: F401

    _HAS_AG_UI = True
except ImportError:
    pass


def _get_graph(org: str) -> CompiledStateGraph:
    if org not in app.state.graphs:
        # ``serve`` hosts BOTH the AG-UI SSE surface (the live web path —
        # CopilotKit ``useInterrupt`` resolves an ask_user) AND the REST lane
        # (SSE stream + polled endpoints). One graph serves both, so ask_user
        # uses the web/interrupt branch on both — which is correct now that the
        # REST lane is interrupt-aware: ``ainvoke`` RETURNS on interrupt (it
        # does not hang), the run reports ``status="interrupted"``, and the
        # client resumes via ``command={"resume": ...}`` (over SSE the interrupt
        # is a stream event; over the polled lane it's the run's status).
        # ``PUX_AUTONOMOUS`` drops ask_user entirely (headless serve).
        # The store is the SHARED ``app.state.base_store`` (created in the
        # lifespan) — so a ``/store/items`` put is visible to the graph's memory
        # tools, and the REST + graph memory surfaces are one backend.
        app.state.graphs[org] = build_graph(
            org,
            checkpointer=app.state.saver,
            store=app.state.base_store,
            facts=RuntimeFacts(transport="serve", autonomous=autonomous_from_env()),
            mcp_tools=app.state.mcp.get(org, ()),
        )
    return app.state.graphs[org]


async def _require_thread(thread_id: str) -> str:
    """Return the org that owns ``thread_id``, or 404."""
    cur = await app.state.db.execute(
        "SELECT org FROM pux_threads WHERE thread_id = ?", (thread_id,)
    )
    row = await cur.fetchone()
    await cur.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"unknown thread {thread_id!r}")
    return row[0]


@dataclass
class RunOutcome:
    """Result of one graph invocation on a thread.

    ``interrupted`` is True when the graph paused on an ``interrupt()`` (e.g.
    ``ask_user``): ``ainvoke`` RETURNS in that state (it does not hang) with
    ``state.next`` non-empty. The polled run endpoints report this as
    ``status="interrupted"`` (never a silent ``success``); the client resumes by
    POSTing a new run carrying ``command={"resume": ...}`` on the same thread.
    """

    output: str
    interrupted: bool
    interrupts: list[dict[str, Any]]


def _graph_input(org: str, body: Any) -> Any:
    """Resolve the ainvoke/astream input for a run body.

    A resume ``command`` -> ``Command(**command)`` (drives the interrupted
    node's ``interrupt()`` return value). Otherwise the fresh ``input``,
    normalized + injected with the org's default rubric (the gate that arms an
    opted-in org's ``RubricMiddleware``; a caller-supplied ``rubric`` key wins).
    """
    if getattr(body, "command", None):
        return Command(**body.command)
    state = _normalize_input(body.input)
    if "rubric" not in state:
        dr = default_rubric(org)
        if dr:
            state["rubric"] = dr
    return state


async def _invoke_once(org: str, thread_id: str, body: Any, recursion_limit: int) -> RunOutcome:
    """Run one org graph invocation on a thread; return the outcome with
    interrupt status. Handles both fresh input and resume ``Command`` payloads."""
    graph = _get_graph(org)
    config = build_invoke_config(thread_id, recursion_limit, org, transport="serve")
    result = await graph.ainvoke(_graph_input(org, body), config=config)
    snap = await graph.aget_state(config)
    # ainvoke returns on interrupt with state.next non-empty (the pending node).
    interrupted = bool(snap.next)
    interrupts: list[dict[str, Any]] = []
    for task in snap.tasks:
        for intr in getattr(task, "interrupts", None) or []:
            interrupts.append(_jsonable(getattr(intr, "value", intr)))
    output = "" if interrupted else _final_answer(result)
    return RunOutcome(output=output, interrupted=interrupted, interrupts=interrupts)


# --- health -------------------------------------------------------------------


@app.get("/ok")
async def health() -> dict[str, Any]:
    return {"ok": True, "orgs": discover_orgs()}


# --- run-completion event stream (receiver-of-last-resort for webhook-less
# MCP clients like Hermes; see run_events.py) ------------------------------


@app.get("/events/health")
async def events_health() -> dict[str, Any]:
    return app.state.events.health()


@app.get("/events")
async def events_list(since: str | None = None, limit: int = 100) -> dict[str, Any]:
    """Recent run-completion events (catch-up / poll). ``since`` is an ISO-8601
    ts; events with ``ts > since`` are returned. A webhook-less client polls
    this instead of per-thread ``list_runs``."""
    return {"events": app.state.events.recent(since=since, limit=limit)}


@app.get("/events/stream")
async def events_stream() -> StreamingResponse:
    """Live SSE feed of EVERY background-run completion across all orgs. A
    webhook-less MCP client (Hermes) subscribes here once and receives
    ``event: run.completed`` frames as runs finish — no per-run ``webhook_url``
    and no receiver to host. Reconnect + ``GET /events?since=<last ts>`` resyncs
    anything missed (the bus drops to a slow subscriber)."""

    async def gen() -> Any:
        q = app.state.events.subscribe()
        try:
            yield _sse("metadata", {"stream": "run_events"})
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15.0)
                except TimeoutError:  # asyncio.wait_for raises TimeoutError (3.11+)
                    yield ": keep-alive\n\n"  # SSE comment — holds the conn open
                    continue
                yield _sse(str(ev.get("event", "run.completed")), ev)
        finally:
            app.state.events.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers=_SSE_HEADERS)


# --- agents (introspection) ---------------------------------------------------


@app.post("/agents/search")
async def agents_search() -> list[dict[str, Any]]:
    return [_agent_descriptor(o) for o in discover_orgs()]


@app.get("/agents/{agent_id}")
async def agent_get(agent_id: str) -> dict[str, Any]:
    if agent_id not in discover_orgs():
        raise HTTPException(status_code=404, detail=f"unknown agent {agent_id!r}")
    return _agent_descriptor(agent_id)


# --- assistants (the langgraph-api surface Studio + the SDK ``assistants`` client hit) ---
# An org maps 1:1 to an assistant: read-only search/get/get_schemas. The mutating
# paths (create/update/delete/versions) don't apply — orgs are declarative on
# disk — so they're deliberately absent (FastAPI's default 405/404 is the honest
# "unsupported" signal, not a silent no-op).


@app.post("/assistants/search")
async def assistants_search(body: AssistantSearch = AssistantSearch()) -> list[dict[str, Any]]:
    orgs = discover_orgs()
    if body.graph_id:
        orgs = [o for o in orgs if o == body.graph_id]
    return [_assistant_descriptor(o) for o in orgs][body.offset : body.offset + body.limit]


@app.get("/assistants/{assistant_id}")
async def assistant_get(assistant_id: str) -> dict[str, Any]:
    if assistant_id not in discover_orgs():
        raise HTTPException(status_code=404, detail=f"unknown assistant {assistant_id!r}")
    return _assistant_descriptor(assistant_id)


@app.get("/assistants/{assistant_id}/schemas")
async def assistant_schemas(assistant_id: str) -> dict[str, Any]:
    """The assistant's input/output/config/state JSON Schemas (the SDK
    ``assistants.get_schemas`` path — what Studio reads to render the run form).
    Derived LIVE from the org's compiled graph; ``{"type": "object"}`` placeholders
    where the graph doesn't expose a given schema."""
    if assistant_id not in discover_orgs():
        raise HTTPException(status_code=404, detail=f"unknown assistant {assistant_id!r}")
    graph = _get_graph(assistant_id)
    return {
        "graph_id": assistant_id,
        "state_schema": _json_schema(
            getattr(graph, "state_schema", None) or getattr(graph, "schema", None)
        ),
        "input": _json_schema(getattr(graph, "input_schema", None)),
        "output": _json_schema(getattr(graph, "output_schema", None)),
        "config_schema": _json_schema(getattr(graph, "config_schema", None)),
        "context_schema": {"type": "object"},
    }


# --- threads ------------------------------------------------------------------


@app.post("/threads")
async def thread_create(body: ThreadCreate) -> dict[str, Any]:
    if body.agent_id not in discover_orgs():
        raise HTTPException(status_code=404, detail=f"unknown agent {body.agent_id!r}")
    thread_id = str(uuid.uuid4())
    await app.state.store.register_thread(thread_id, body.agent_id, body.metadata)
    return {
        "thread_id": thread_id,
        "agent_id": body.agent_id,
        "status": "idle",
        "metadata": body.metadata,
        "values": {},
    }


@app.post("/threads/search")
async def threads_search(body: ThreadSearch = ThreadSearch()) -> list[dict[str, Any]]:
    query = "SELECT thread_id, org, metadata, created_at FROM pux_threads"
    params: list[Any] = []
    if body.agent_id is not None:
        query += " WHERE org = ?"
        params.append(body.agent_id)
    query += " ORDER BY created_at DESC"
    cur = await app.state.db.execute(query, params)
    rows = await cur.fetchall()
    await cur.close()
    return [
        {
            "thread_id": r[0],
            "agent_id": r[1],
            "metadata": json.loads(r[2] or "{}"),
            "created_at": r[3],
        }
        for r in rows
    ]


@app.get("/threads/{thread_id}")
async def thread_get(thread_id: str) -> dict[str, Any]:
    org = await _require_thread(thread_id)
    return await _thread_descriptor(thread_id, org)


async def _thread_descriptor(thread_id: str, org: str) -> dict[str, Any]:
    """The canonical thread shape GET/PATCH/copy return: id + org + status +
    merged metadata + current graph values. Shared so the descriptor can't drift
    between the read + write endpoints."""
    graph = _get_graph(org)
    snap = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    row = await app.state.store.get_thread(thread_id) or {}
    return {
        "thread_id": thread_id,
        "agent_id": org,
        "status": _status_from_snapshot(snap),
        "metadata": json.loads(row.get("metadata") or "{}"),
        "values": _jsonable(snap.values or {}),
        "next": snap.next,
    }


@app.delete("/threads/{thread_id}")
async def thread_delete(thread_id: str) -> dict[str, Any]:
    org = await _require_thread(thread_id)
    # deletion is a checkpointer operation, not a graph one
    await app.state.saver.adelete_thread(thread_id)
    await app.state.db.execute("DELETE FROM pux_threads WHERE thread_id = ?", (thread_id,))
    await app.state.db.commit()
    return {"thread_id": thread_id, "agent_id": org, "deleted": True}


@app.get("/threads/{thread_id}/history")
async def thread_history(thread_id: str) -> list[dict[str, Any]]:
    org = await _require_thread(thread_id)
    graph = _get_graph(org)
    out: list[dict[str, Any]] = []
    async for snap in graph.aget_state_history({"configurable": {"thread_id": thread_id}}):
        out.append(
            {
                "checkpoint_id": snap.config["configurable"].get("checkpoint_id"),
                "parent_checkpoint_id": snap.parent_config["configurable"].get("checkpoint_id")
                if snap.parent_config
                else None,
                "next": list(snap.next) if snap.next else [],
                "values": _jsonable(snap.values or {}),
            }
        )
    return out


class ThreadUpdate(BaseModel):
    """``PATCH /threads/{id}`` body (the SDK ``threads.update``). ``metadata`` is
    shallow-merged into the stored metadata; ``ttl`` is accepted for wire
    compatibility but ignored (the sqlite index has no TTL — a follow-up)."""

    metadata: dict[str, Any] = {}
    ttl: int | dict[str, Any] | None = None


@app.patch("/threads/{thread_id}")
async def thread_update(
    thread_id: str,
    body: ThreadUpdate,
    prefer: str | None = Header(default=None, alias="Prefer"),
) -> Any:
    """Merge metadata into a thread (the SDK ``threads.update`` path). Honors
    ``Prefer: return=minimal`` → 204 no body; otherwise returns the updated
    thread descriptor."""
    org = await _require_thread(thread_id)
    await app.state.store.merge_metadata(thread_id, body.metadata)
    if prefer and "return=minimal" in prefer:
        return Response(status_code=204)
    return await _thread_descriptor(thread_id, org)


@app.post("/threads/{thread_id}/copy")
async def thread_copy(thread_id: str) -> dict[str, Any]:
    """Fork a thread: register a new thread_id under the same org + copied
    metadata, then copy the current checkpoint state across (the SDK
    ``threads.copy`` path; returns the new descriptor — the SDK itself discards
    it, but raw clients use it).

    Current-state copy (not full history): ``AsyncSqliteSaver`` exposes no
    history-duplicate op, so the new thread carries the source's latest
    checkpoint — enough to continue the conversation from the fork point.
    """
    org = await _require_thread(thread_id)
    src = await app.state.store.get_thread(thread_id) or {}
    new_id = str(uuid.uuid4())
    await app.state.store.register_thread(
        new_id,
        org,
        json.loads(src.get("metadata") or "{}"),
    )
    graph = _get_graph(org)
    snap = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    if snap.values:
        await graph.aupdate_state(
            {"configurable": {"thread_id": new_id}},
            snap.values,
        )
    return await _thread_descriptor(new_id, org)


# --- runs ---------------------------------------------------------------------


@app.post("/runs/wait")
async def run_ephemeral(body: EphemeralRun) -> dict[str, Any]:
    """Create a persistent thread, run on it synchronously, return the final
    output (or an ``interrupted`` status when the graph paused on ``ask_user`` —
    resume by POSTing again with ``command={"resume": ...}``). The thread is
    kept (resumable) — a superset of the spec's 'ephemeral' semantics, more
    useful for a single-user local agent."""
    if body.agent_id not in discover_orgs():
        raise HTTPException(status_code=404, detail=f"unknown agent {body.agent_id!r}")
    thread_id = str(uuid.uuid4())
    await app.state.store.register_thread(thread_id, body.agent_id, body.metadata)
    try:
        outcome = await _invoke_once(body.agent_id, thread_id, body, body.recursion_limit)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - surface as a failed run, not a 500 stack
        return {
            "thread_id": thread_id,
            "agent_id": body.agent_id,
            "status": "error",
            "output": "",
            "error": f"{type(exc).__name__}: {exc}",
        }
    base = {"thread_id": thread_id, "agent_id": body.agent_id}
    if outcome.interrupted:
        return {**base, "status": "interrupted", "output": "", "interrupts": outcome.interrupts}
    return {**base, "status": "success", "output": outcome.output}


_log = logging.getLogger(__name__)

#: Best-effort upper bound on a webhook POST — a slow/dead target must not stall
#: the background run that produced the result beyond this.
WEBHOOK_TIMEOUT: float = 5.0


async def _dispatch_run_webhook(meta: dict[str, Any]) -> None:
    """Push-notification for a BACKGROUND run's completion.

    Closes the gap an MCP client that fired ``start_run`` and ended its turn
    otherwise cannot: with nothing pushing completion it must poll
    ``list_runs``. Here the run's terminal metadata is POSTed to the
    ``webhook_url`` recorded on the run (per-run ``RunCreate.webhook_url`` or
    the ``PUX_RUN_WEBHOOK_URL`` default), so the caller learns the run finished
    with NO polling.

    Fire-and-forget semantics: a down/unreachable target is logged + swallowed —
    it must NEVER fail the run that produced the result beyond
    ``WEBHOOK_TIMEOUT`` ([[no-fallbacks-no-aliases]]: this is a notification
    channel, not a reliability gate; non-delivery degrades to the existing
    poll-with-``list_runs`` path). The payload drops ``webhook_url`` itself (the
    caller's callback location is not echoed back) and tags an ``event`` so one
    receiver can demux multiple run kinds later.
    """
    url = meta.get("webhook_url")
    if not url:
        return
    payload = {k: v for k, v in meta.items() if k != "webhook_url"}
    payload["event"] = "run.completed"
    try:
        async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code >= 400:
                _log.warning("run webhook %s returned HTTP %s", url, resp.status_code)
    except Exception as exc:  # noqa: BLE001 - best-effort; never fail the run
        _log.warning("run webhook dispatch to %s failed: %r", url, exc)


async def _run_task(run_id: str, org: str, thread_id: str, body: Any, rl: int) -> None:
    app.state.run_meta[run_id]["status"] = "running"
    # Run prep jobs after container is up, before the agent loop.
    try:
        from pux_harness.sandbox.container import prepare  # noqa: PLC0415

        # universal_warmup=True => every serve sandbox also probes the
        # run-completion event endpoint (so a webhook-less client like Hermes
        # can observe this run's finish). direct-mode (main.py) leaves it off.
        #
        # Offloaded to a worker thread: prepare() does blocking Docker I/O
        # (ensure container + exec warmup_webhook) and would otherwise stall the
        # SINGLE event loop — during which a webhook-less client's GET /events
        # poll would time out and mistake serve for dead. to_thread keeps the
        # loop serving /events, /ok, and new runs while prep runs in parallel.
        job_results = await asyncio.to_thread(prepare, org, universal_warmup=True)
        from pux_harness.sandbox.container import SandboxContainer  # noqa: PLC0415

        _watch_url = SandboxContainer(org=org).watch_url
        if _watch_url:
            app.state.run_meta[run_id]["watch_url"] = _watch_url
        if job_results:
            failed = [r for r in job_results if r["status"] != "ok"]
            if failed:
                app.state.run_meta[run_id].setdefault("warnings", [])
                app.state.run_meta[run_id]["warnings"] = [
                    f"job {r['name']}: {r['status']}" for r in failed
                ]
    except Exception:  # noqa: BLE001
        # Jobs failing shouldn't block the agent run — warn only.
        pass
    try:
        outcome = await _invoke_once(org, thread_id, body, rl)
        if outcome.interrupted:
            app.state.run_meta[run_id].update(
                status="interrupted",
                output="",
                error=None,
                interrupts=outcome.interrupts,
            )
        else:
            app.state.run_meta[run_id].update(status="success", output=outcome.output, error=None)
    except Exception as exc:  # noqa: BLE001
        app.state.run_meta[run_id].update(
            status="error", output="", error=f"{type(exc).__name__}: {exc}"
        )
    # Push-notification: tell the out-of-process caller the background run
    # reached a terminal state (success/interrupted/error) WITHOUT it polling
    # list_runs. Best-effort — _dispatch_run_webhook swallows delivery failures.
    await _dispatch_run_webhook(app.state.run_meta[run_id])
    # Same completion, second channel: fan to the in-process event bus so an MCP
    # client with NO webhook receiver (Hermes — "can't make webhooks on the
    # sandbox") can subscribe to GET /events/stream once and get every background
    # run's completion across all orgs. Identical payload to the outbound POST.
    await app.state.events.publish(
        {k: v for k, v in app.state.run_meta[run_id].items() if k != "webhook_url"}
        | {"event": "run.completed"}
    )


@app.post("/threads/{thread_id}/runs")
async def thread_run_create(thread_id: str, body: RunCreate) -> dict[str, Any]:
    org = await _require_thread(thread_id)
    run_id = str(uuid.uuid4())
    app.state.run_meta[run_id] = {
        "run_id": run_id,
        "thread_id": thread_id,
        "agent_id": org,
        "status": "pending",
        "started_at": _now(),
        "output": "",
        "error": None,
        "webhook_url": getattr(body, "webhook_url", None) or os.environ.get("PUX_RUN_WEBHOOK_URL"),
    }
    app.state.runs[run_id] = asyncio.create_task(
        _run_task(run_id, org, thread_id, body, body.recursion_limit)
    )
    return app.state.run_meta[run_id]


@app.get("/threads/{thread_id}/runs")
async def thread_runs_list(thread_id: str) -> list[dict[str, Any]]:
    await _require_thread(thread_id)
    return [
        {**meta, "alive": (rid in app.state.runs and not app.state.runs[rid].done())}
        for rid, meta in app.state.run_meta.items()
        if meta["thread_id"] == thread_id
    ]


@app.get("/threads/{thread_id}/runs/{run_id}")
async def thread_run_get(thread_id: str, run_id: str) -> dict[str, Any]:
    """Get one run's metadata (the SDK ``runs.get`` path). 404 if the run isn't
    on this thread."""
    await _require_thread(thread_id)
    meta = app.state.run_meta.get(run_id)
    if meta is None or meta.get("thread_id") != thread_id:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    return {**meta, "alive": (run_id in app.state.runs and not app.state.runs[run_id].done())}


@app.delete("/threads/{thread_id}/runs/{run_id}")
async def thread_run_delete(thread_id: str, run_id: str) -> Response:
    """Delete a run: cancel the task if still in flight, then drop its metadata
    (the SDK ``runs.delete`` path). 204 no body — the SDK discards."""
    await _require_thread(thread_id)
    meta = app.state.run_meta.get(run_id)
    if meta is None or meta.get("thread_id") != thread_id:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    task = app.state.runs.pop(run_id, None)
    if task is not None and not task.done():
        task.cancel()
    app.state.run_meta.pop(run_id, None)
    return Response(status_code=204)


@app.get("/threads/{thread_id}/runs/{run_id}/join")
async def thread_run_join(thread_id: str, run_id: str) -> dict[str, Any]:
    """Block until the run completes, then return the thread's final state (the
    SDK ``runs.join`` path — a long-poll GET for a background run's result). 404
    if the run isn't on this thread."""
    org = await _require_thread(thread_id)
    meta = app.state.run_meta.get(run_id)
    if meta is None or meta.get("thread_id") != thread_id:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    task = app.state.runs.get(run_id)
    if task is not None:
        await task  # the _run_task coroutine; resolves to None on completion
    return await _thread_descriptor(thread_id, org)


@app.get("/runs/{run_id}/wait")
async def run_wait(run_id: str) -> dict[str, Any]:
    meta = app.state.run_meta.get(run_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    task = app.state.runs.get(run_id)
    if task is not None and not task.done():
        await task
    return app.state.run_meta[run_id]


@app.post("/runs/{run_id}/cancel")
async def run_cancel(run_id: str) -> dict[str, Any]:
    meta = app.state.run_meta.get(run_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    task = app.state.runs.get(run_id)
    if task is not None and not task.done():
        task.cancel()
    meta["status"] = "cancelled"
    return meta


# --- store / long-term memory -------------------------------------------------
# The langgraph Store/Memory REST surface (the MEDIUM-priority Agent Protocol
# gap). Backed by the SHARED ``app.state.base_store`` the graph also reads/writes
# (``_get_graph`` passes it to ``build_graph``), so a memory written over REST is
# visible to a graph's memory tools on the next run, and vice versa — one backend
# across the seam. Routes + payloads match what ``langgraph_sdk``'s
# ``StoreClient`` sends verbatim (the contract Studio + the SDK + any
# langgraph-api client hits): PUT/GET/DELETE ``/store/items``,
# POST ``/store/items/search``, POST ``/store/namespaces``.


class StorePutItem(BaseModel):
    namespace: list[str]
    key: str
    value: dict[str, Any]
    index: bool | list[str] | None = None
    ttl: float | None = None


class StoreDeleteItem(BaseModel):
    namespace: list[str]
    key: str


class StoreSearchItems(BaseModel):
    namespace_prefix: list[str] = []
    filter: dict[str, Any] | None = None
    limit: int = 10
    offset: int = 0
    query: str | None = None
    refresh_ttl: bool | None = None


class StoreListNamespaces(BaseModel):
    prefix: list[str] | None = None
    suffix: list[str] | None = None
    max_depth: int | None = None
    limit: int = 100
    offset: int = 0


def _store_item(item: Any) -> dict[str, Any]:
    """Serialize a langgraph ``Item`` to the SDK's expected shape: namespace
    tuple → list, datetimes → ISO-8601 strings (``_jsonable`` mishandles both)."""
    return {
        "namespace": list(item.namespace),
        "key": item.key,
        "value": item.value,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
    }


@app.put("/store/items")
async def store_put_item(body: StorePutItem) -> dict[str, Any]:
    """Upsert one item (the SDK ``StoreClient.put_item`` path). Namespace labels
    may not contain ``.`` — the SDK joins the namespace on it for GET."""
    for label in body.namespace:
        if "." in label:
            raise HTTPException(
                status_code=422,
                detail=f"namespace label {label!r} may not contain '.'",
            )
    await app.state.base_store.aput(
        tuple(body.namespace),
        body.key,
        body.value,
        body.index,
        ttl=body.ttl,
    )
    return {}  # the SDK discards the PUT response


@app.get("/store/items")
async def store_get_item(
    namespace: str,
    key: str,
    refresh_ttl: bool | None = None,
) -> dict[str, Any]:
    """Read one item by dotted namespace + key (the SDK ``get_item`` path). 404
    when absent — the SDK raises for it."""
    ns = tuple(namespace.split(".")) if namespace else ()
    item = await app.state.base_store.aget(ns, key, refresh_ttl=refresh_ttl)
    if item is None:
        raise HTTPException(status_code=404, detail="item not found")
    return _store_item(item)


@app.delete("/store/items")
async def store_delete_item(body: StoreDeleteItem) -> dict[str, Any]:
    """Remove one item (the SDK ``delete_item`` path)."""
    await app.state.base_store.adelete(tuple(body.namespace), body.key)
    return {}


@app.post("/store/items/search")
async def store_search_items(body: StoreSearchItems) -> dict[str, Any]:
    """Search within a namespace prefix (the SDK ``search_items`` path). Returns
    ``{"items": [...]}`` — the SDK unwraps the list itself."""
    items = await app.state.base_store.asearch(
        tuple(body.namespace_prefix),
        query=body.query,
        filter=body.filter,
        limit=body.limit,
        offset=body.offset,
        refresh_ttl=body.refresh_ttl,
    )
    return {"items": [_store_item(i) for i in items]}


@app.post("/store/namespaces")
async def store_list_namespaces(body: StoreListNamespaces) -> list[list[str]]:
    """List namespaces under a prefix/suffix/depth (the SDK ``list_namespaces``
    path). Returns the raw list of namespace paths."""
    ns = await app.state.base_store.alist_namespaces(
        prefix=body.prefix,
        suffix=body.suffix,
        max_depth=body.max_depth,
        limit=body.limit,
        offset=body.offset,
    )
    return [list(n) for n in ns]


# --- streaming (SSE) ---------------------------------------------------------


def _sse(event: str, data: Any) -> str:
    """One SSE v1 frame — ``event: <e>\\ndata: <json>\\n\\n`` — the exact wire
    format the ``langgraph_sdk`` ``SSEDecoder`` parses, so a ``RunClient``
    consumes our stream with no adapter."""
    return f"event: {event}\ndata: {json.dumps(_jsonable(data), default=str)}\n\n"


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


async def _stream_run(
    org: str,
    thread_id: str,
    body: Any,
    recursion_limit: int,
    run_id: str,
) -> Any:
    """Yield v1 SSE frames for one run: a leading ``metadata`` event, then the
    langgraph stream modes (``messages``/``updates``/``values``), then an
    ``error`` frame on exception. The interrupt surfaces as a ``__interrupt__``
    key in the final ``values``/``updates`` events — the client reads it +
    resumes with ``command={"resume": ...}`` on the same thread."""
    graph = _get_graph(org)
    config = build_invoke_config(thread_id, recursion_limit, org, transport="serve")
    yield _sse("metadata", {"run_id": run_id})
    try:
        async for mode, chunk in graph.astream(
            _graph_input(org, body),
            config=config,
            stream_mode=["messages", "updates", "values"],
        ):
            if mode == "messages":
                msg, meta = chunk
                yield _sse("messages", [_jsonable(msg), _jsonable(meta)])
            elif mode == "updates":
                yield _sse("updates", chunk)
            else:  # values — carries __interrupt__ on the interrupted step
                yield _sse("values", chunk)
    except Exception as exc:  # noqa: BLE001 - emit an error frame, don't crash mid-stream
        yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})


@app.post("/runs/stream")
async def run_stream(body: EphemeralRun) -> StreamingResponse:
    """Ephemeral run streamed as SSE (create thread + stream + the thread is
    kept for resume). The wire format is what ``langgraph_sdk`` consumes."""
    if body.agent_id not in discover_orgs():
        raise HTTPException(status_code=404, detail=f"unknown agent {body.agent_id!r}")
    thread_id = str(uuid.uuid4())
    await app.state.store.register_thread(thread_id, body.agent_id, body.metadata)
    run_id = str(uuid.uuid4())
    return StreamingResponse(
        _stream_run(body.agent_id, thread_id, body, body.recursion_limit, run_id),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@app.post("/threads/{thread_id}/runs/stream")
async def thread_run_stream(thread_id: str, body: RunCreate) -> StreamingResponse:
    """Stream a run on an existing thread (SSE). POST ``command={"resume": ...}``
    to resume an interrupted run on the thread."""
    org = await _require_thread(thread_id)
    run_id = str(uuid.uuid4())
    return StreamingResponse(
        _stream_run(org, thread_id, body, body.recursion_limit, run_id),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# --- jobs (post-create prep steps) -------------------------------------------


class JobsRunRequest(BaseModel):
    job: str | None = None  # run a specific job by name, or all if None


@app.post("/jobs/{org}/run")
async def jobs_run(org: str, body: JobsRunRequest = JobsRunRequest()) -> dict[str, Any]:
    """Run prep jobs inside the org's sandbox container. Returns per-job results."""
    if org not in discover_orgs():
        raise HTTPException(status_code=404, detail=f"unknown org {org!r}")

    from pux_harness.sandbox.container import SandboxContainer  # noqa: PLC0415
    from pux_harness.sandbox.docker_exec import DockerExecClient  # noqa: PLC0415
    from pux_harness.sandbox.jobs import run_jobs  # noqa: PLC0415
    from pux_harness.sandbox import policy as policy_mod  # noqa: PLC0415
    from pux_harness.kit._paths import project_root  # noqa: PLC0415

    try:
        pol = policy_mod.load(org, project_root())
    except policy_mod.NoPolicy:
        return {"org": org, "jobs": [], "message": "no policy.yaml — no jobs declared"}

    specs = policy_mod.job_specs(pol)
    if not specs:
        return {"org": org, "jobs": [], "message": "no jobs declared"}

    # Filter to specific job if requested
    if body.job:
        specs = [s for s in specs if s.name == body.job]
        if not specs:
            raise HTTPException(status_code=404, detail=f"no job named {body.job!r}")

    # Ensure sandbox is running
    try:
        sb = SandboxContainer(org=org)
        container_name = sb.ensure()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"sandbox error: {exc}") from exc

    ec = DockerExecClient(container=container_name)
    results = run_jobs(pol, ec)
    # Filter results if specific job requested
    if body.job:
        results = [r for r in results if r.name == body.job]

    return {
        "org": org,
        "jobs": [
            {"name": r.name, "status": r.status, "error": r.error, "duration": round(r.duration, 1)}
            for r in results
        ],
    }


@app.get("/jobs/{org}/status")
async def jobs_status(org: str) -> dict[str, Any]:
    """Show declared jobs for an org and their specs. Status is derived from
    the latest run if available."""
    if org not in discover_orgs():
        raise HTTPException(status_code=404, detail=f"unknown org {org!r}")

    from pux_harness.sandbox import policy as policy_mod  # noqa: PLC0415
    from pux_harness.kit._paths import project_root  # noqa: PLC0415

    try:
        pol = policy_mod.load(org, project_root())
    except policy_mod.NoPolicy:
        return {"org": org, "jobs": [], "message": "no policy.yaml"}

    specs = policy_mod.job_specs(pol)
    return {
        "org": org,
        "jobs": [
            {
                "name": s.name,
                "script": s.script,
                "args": s.args,
                "timeout": s.timeout,
                "description": s.description,
            }
            for s in specs
        ],
    }


# --- helpers ------------------------------------------------------------------


def _status_from_snapshot(snap: Any) -> str:
    if snap.next:  # more nodes to run -> interrupted mid-graph
        return "interrupted"
    values = snap.values or {}
    msgs = values.get("messages") or []
    if msgs and getattr(msgs[-1], "tool_calls", None):
        return "interrupted"
    return "idle" if not msgs else "finished"


def _jsonable(obj: Any) -> Any:
    """Best-effort coercion of langgraph message objects to JSON-serializable
    dicts (messages → role/content; tool_calls preserved)."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    # langchain message object
    role = getattr(obj, "type", None) or getattr(obj, "role", None) or type(obj).__name__
    return {
        "role": role,
        "content": _jsonable(getattr(obj, "content", "")),
        "tool_calls": getattr(obj, "tool_calls", None) or None,
    }


def main() -> None:
    import uvicorn

    uvicorn.run(
        "pux_harness.server:app",
        host=PUX_API_HOST,
        port=PUX_API_PORT,
        reload=False,
        log_level=os.environ.get("PUX_API_LOG", "info"),
    )


if __name__ == "__main__":
    main()
