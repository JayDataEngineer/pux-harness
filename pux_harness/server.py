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

**Implementation choice:** thin FastAPI implementing the published spec, NOT
``langgraph-api`` (the Platform runtime). Rationale: minimalist (we own ~250
LOC vs adopting an opinionated runtime), and the REST contract is identical
either way — swapping the server impl behind these endpoints is invisible to
clients, so the choice is reversible. SSE streaming (run lifecycle + tool +
nested-subagent events) is deferred.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from langchain_core.tools import BaseTool

from pux_harness.agent.graph import build_graph
from pux_harness.threads import open_thread_store
from pux_harness.agent.orgs import discover_orgs, org_agent_slugs
from pux_harness.agent.profile import default_rubric
from pux_harness.agent.tool_servers import resolve_tool_servers

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


class RunCreate(BaseModel):
    input: Any = None            # str | {"messages": [...]} | dict
    metadata: dict[str, Any] = {}
    recursion_limit: int = DEFAULT_RECURSION_LIMIT


class EphemeralRun(BaseModel):
    agent_id: str
    input: Any = None
    metadata: dict[str, Any] = {}
    recursion_limit: int = DEFAULT_RECURSION_LIMIT


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

        # Register AG-UI endpoints now that the checkpointer is ready.
        if _HAS_AG_UI:
            for org_name in discover_orgs():
                add_langgraph_fastapi_endpoint(
                    app=app,
                    agent=LangGraphAGUIAgent(
                        name=org_name,
                        description=f"Pux org '{org_name}'",
                        graph=_get_graph(org_name),
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
    from ag_ui_langgraph import add_langgraph_fastapi_endpoint  # noqa: F401
    from copilotkit import LangGraphAGUIAgent  # noqa: F401
    _HAS_AG_UI = True
except ImportError:
    pass


def _get_graph(org: str) -> CompiledStateGraph:
    if org not in app.state.graphs:
        from langgraph.store.memory import InMemoryStore  # noqa: PLC0415
        store = InMemoryStore()
        app.state.graphs[org] = build_graph(
            org, checkpointer=app.state.saver, store=store,
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


async def _execute(org: str, thread_id: str, raw_input: Any, recursion_limit: int) -> str:
    """Run one org graph invocation on a thread; return the final answer text.

    Injects the org's default rubric when the caller supplied
    none — this is what arms an opted-in org's ``RubricMiddleware`` gate. A
    caller-supplied ``rubric`` key (e.g. ``--rubric`` from the CLI, flowing
    through ``_normalize_input``) wins and is left untouched."""
    graph = _get_graph(org)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    state = _normalize_input(raw_input)
    if "rubric" not in state:
        dr = default_rubric(org)
        if dr:
            state["rubric"] = dr
    result = await graph.ainvoke(state, config=config)
    return _final_answer(result)


# --- health -------------------------------------------------------------------

@app.get("/ok")
async def health() -> dict[str, Any]:
    return {"ok": True, "orgs": discover_orgs()}


# --- agents (introspection) ---------------------------------------------------

@app.post("/agents/search")
async def agents_search() -> list[dict[str, Any]]:
    return [_agent_descriptor(o) for o in discover_orgs()]


@app.get("/agents/{agent_id}")
async def agent_get(agent_id: str) -> dict[str, Any]:
    if agent_id not in discover_orgs():
        raise HTTPException(status_code=404, detail=f"unknown agent {agent_id!r}")
    return _agent_descriptor(agent_id)


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
    graph = _get_graph(org)
    snap = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    values = snap.values or {}
    return {
        "thread_id": thread_id,
        "agent_id": org,
        "status": _status_from_snapshot(snap),
        "values": _jsonable(values),
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


# --- runs ---------------------------------------------------------------------

@app.post("/runs/wait")
async def run_ephemeral(body: EphemeralRun) -> dict[str, Any]:
    """Create a persistent thread, run on it synchronously, return the final
    output. The thread is kept (resumable) — a superset of the spec's
    'ephemeral' semantics, more useful for a single-user local agent."""
    if body.agent_id not in discover_orgs():
        raise HTTPException(status_code=404, detail=f"unknown agent {body.agent_id!r}")
    thread_id = str(uuid.uuid4())
    await app.state.store.register_thread(thread_id, body.agent_id, body.metadata)
    try:
        answer = await _execute(body.agent_id, thread_id, body.input, body.recursion_limit)
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
    return {
        "thread_id": thread_id,
        "agent_id": body.agent_id,
        "status": "success",
        "output": answer,
    }


async def _run_task(run_id: str, org: str, thread_id: str, raw_input: Any, rl: int) -> None:
    app.state.run_meta[run_id]["status"] = "running"
    # Run prep jobs after container is up, before the agent loop.
    try:
        from pux_harness.sandbox.container import prepare  # noqa: PLC0415
        job_results = prepare(org)
        if job_results:
            failed = [r for r in job_results if r["status"] != "ok"]
            if failed:
                app.state.run_meta[run_id].setdefault("warnings", [])
                app.state.run_meta[run_id]["warnings"] = [
                    f"job {r['name']}: {r['status']}" for r in failed
                ]
    except Exception as exc:  # noqa: BLE001
        # Jobs failing shouldn't block the agent run — warn only.
        pass
    try:
        answer = await _execute(org, thread_id, raw_input, rl)
        app.state.run_meta[run_id].update(status="success", output=answer, error=None)
    except Exception as exc:  # noqa: BLE001
        app.state.run_meta[run_id].update(
            status="error", output="", error=f"{type(exc).__name__}: {exc}"
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
    }
    app.state.runs[run_id] = asyncio.create_task(
        _run_task(run_id, org, thread_id, body.input, body.recursion_limit)
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
            {"name": r.name, "status": r.status, "error": r.error,
             "duration": round(r.duration, 1)}
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
            {"name": s.name, "script": s.script, "args": s.args,
             "timeout": s.timeout, "description": s.description}
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
