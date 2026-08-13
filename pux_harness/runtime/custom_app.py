"""Pux's custom FastAPI app for the langgraph-api ``user_router`` seam.

When ``langgraph serve`` / ``langgraph dev`` runs, langgraph-api builds its full
Agent Protocol surface ON TOP of this app (``langgraph_api/server.py``:
``app = user_router`` then mounts all CRUD routes on it, and *combines* its base
lifespan with ours via ``combine_lifespans`` — so the EventBus lifespan below
runs alongside langgraph-api's own). See ``plan-p3-server-rest-retirement``.

This app owns the **pux-unique** surfaces only — the things upstream langgraph-api
has NO equivalent for:

* ``/events*``  — the run-completion EventBus (push-notification
  receiver-of-last-resort for webhook-less clients like Hermes; ``run_events.py``).
* ``/jobs/{org}/*`` — prep/warmup jobs (warmup_browser etc.; ``sandbox/jobs.py``).

Everything else (threads/runs/store/assistants/agents CRUD) is served by
upstream langgraph-api and is intentionally NOT re-rolled here — this is the
rely-on-upstream cutover vehicle ([[rely-on-upstream]]). (Health is the custom
``/events/health`` below — Aegra does not expose a ``/ok``.)

Wired in ``langgraph.json`` via::

    "http": {"app": "pux_harness.runtime.custom_app:app"}

Handlers originated in the now-retired ``server.py`` (lifted verbatim; only the
state access changed — ``app.state.events`` -> ``request.app.state.events`` — so
they are robust to being mounted as a sub-app).
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pux_harness.agent.orgs import discover_orgs
from pux_harness.run_events import EventBus

# --- SSE helpers (the exact wire format langgraph_sdk's SSEDecoder parses) ------

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _sse(event: str, data: Any) -> str:
    """One SSE v1 frame — ``event: <e>\\ndata: <json>\\n\\n``."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


# --- lifespan: mount the EventBus (langgraph-api combines this with its own) ----
#
# MUST use the ``lifespan=`` context-manager form, NOT ``@app.on_event(...)``:
# langgraph_api's ``validate_router_lifespan_hooks`` rejects apps that declare
# ``on_startup``/``on_shutdown`` hooks (it merges via ``combine_lifespans``).


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    app.state.events = EventBus()
    yield


app = FastAPI(lifespan=lifespan, title="pux custom surfaces")


# --- run-completion event stream -----------------------------------------------


@app.get("/events/health")
async def events_health(request: Request) -> dict[str, Any]:
    return request.app.state.events.health()


@app.get("/events")
async def events_list(request: Request, since: str | None = None, limit: int = 100) -> dict[str, Any]:
    """Recent run-completion events (catch-up / poll). ``since`` is an ISO-8601
    ts; events with ``ts > since`` are returned. A webhook-less client polls
    this instead of per-thread ``list_runs``."""
    return {"events": request.app.state.events.recent(since=since, limit=limit)}


@app.get("/events/stream")
async def events_stream(request: Request) -> StreamingResponse:
    """Live SSE feed of EVERY background-run completion across all orgs. A
    webhook-less MCP client (Hermes) subscribes here once and receives
    ``event: run.completed`` frames as runs finish — no per-run ``webhook_url``
    and no receiver to host. Reconnect + ``GET /events?since=<last ts>`` resyncs
    anything missed (the bus drops to a slow subscriber)."""

    async def gen() -> Any:
        q = request.app.state.events.subscribe()
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
            request.app.state.events.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers=_SSE_HEADERS)


# --- prep / warmup jobs ---------------------------------------------------------
#
# These handlers do REAL blocking I/O (filesystem scan via ``discover_orgs``,
# YAML policy load, Docker container ensure/exec). Under langgraph-api that work
# runs in the ASGI event loop, so it is offloaded to a worker thread
# (``asyncio.to_thread``) — both to satisfy langgraph-api's dev-mode blocking-call
# guard (``blockbuster``) AND because tying up the loop with sync I/O degrades
# every co-served AP request. Same lesson as the EventBus ``prepare`` offload
# ([[context-offload-prevents-ai-calls]]). ``HTTPException`` raised in the thread
# propagates back through the ``await`` into FastAPI's exception handling.


class JobsRunRequest(BaseModel):
    job: str | None = None  # run a specific job by name, or all if None


def _jobs_status_sync(org: str) -> dict[str, Any]:
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


@app.get("/jobs/{org}/status")
async def jobs_status(org: str) -> dict[str, Any]:
    """Show declared jobs for an org and their specs. Status is derived from
    the latest run if available."""
    return await asyncio.to_thread(_jobs_status_sync, org)


def _jobs_run_sync(org: str, job: str | None) -> dict[str, Any]:
    if org not in discover_orgs():
        raise HTTPException(status_code=404, detail=f"unknown org {org!r}")

    from pux_harness.sandbox.exec import shared_backend  # noqa: PLC0415
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
    if job:
        specs = [s for s in specs if s.name == job]
        if not specs:
            raise HTTPException(status_code=404, detail=f"no job named {job!r}")

    # The shared backend auto-starts the sandbox on first use (OpenShell).
    backend = shared_backend()
    results = run_jobs(pol, backend)
    # Filter results if specific job requested
    if job:
        results = [r for r in results if r.name == job]

    return {
        "org": org,
        "jobs": [
            {"name": r.name, "status": r.status, "error": r.error, "duration": round(r.duration, 1)}
            for r in results
        ],
    }


@app.post("/jobs/{org}/run")
async def jobs_run(org: str, body: JobsRunRequest = JobsRunRequest()) -> dict[str, Any]:
    """Run prep jobs inside the org's sandbox. Returns per-job results."""
    return await asyncio.to_thread(_jobs_run_sync, org, body.job)
