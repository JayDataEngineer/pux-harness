"""MCP server that wraps the Pux Agent Protocol REST API.

Exposes the org graph as MCP tools so any MCP client (Hermes, OpenClaw, Claude
Desktop, Zed) can drive Pux without knowing the Agent Protocol wire format.
Transport: SSE on :9987 (``pux mcp``).

Requires the Agent Protocol server to be running: ``pux serve`` (FastAPI on
:9988). Call ``health`` first to confirm the backend is up.

Two execution models:
  - ``run_agent`` — blocking, one-shot (quick tasks).
  - ``start_run`` … — async lifecycle for long tasks:
        create_thread -> start_run -> list_runs (poll) | wait_for_run (block)
                       -> cancel_run (stop)

Errors are returned as text prefixed ``ERROR:`` so any client — even ones that
treat a raised tool error as fatal — can read and recover.
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx
from fastmcp import FastMCP

PUX_API = os.environ.get("PUX_API_URL", "http://127.0.0.1:9988")
TIMEOUT = float(os.environ.get("PUX_MCP_TIMEOUT", "300"))
DEFAULT_RECURSION_LIMIT = 60  # matches server.py:60 + cli.py default

mcp = FastMCP(
    "pux",
    instructions=(
        "Pux multi-org agent orchestrator. Workflow:\n"
        "1. `health` — confirm the backend is up (call first if anything errors).\n"
        "2. `list_orgs` — pick an org; each is a self-contained multi-agent team.\n"
        "3. Quick task: `run_agent(task, org)` blocks until the answer is ready.\n"
        "4. Long task: `create_thread(org)` -> `start_run(thread_id, task)` -> "
        "poll `list_runs(thread_id)` or block `wait_for_run(run_id)` -> "
        "`cancel_run(run_id)` to stop early.\n"
        "5. Corpus prep: `get_jobs(org)` then `run_jobs(org)`.\n"
        "Threads persist across calls (see get_thread, get_thread_history). "
        "Tool failures come back as text prefixed `ERROR:`."
    ),
)


# --- transport ---------------------------------------------------------------
# Lazy singleton so calls reuse one connection pool. Tests monkeypatch `_client`
# with an httpx.AsyncClient over httpx.MockTransport (see tests/test_mcp_server.py).
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(base_url=PUX_API, timeout=TIMEOUT)
    return _client


async def _request(method: str, path: str, **kw: Any) -> Any:
    """HTTP wrapper. Returns parsed JSON (or {} for empty bodies). Raises
    ValueError with a clean, actionable message on failure."""
    client = _get_client()
    try:
        r = await client.request(method, path, **kw)
    except httpx.ConnectError as exc:
        raise ValueError(
            f"cannot reach Pux API at {PUX_API} ({exc}); is `pux serve` running?"
        ) from exc
    except httpx.HTTPError as exc:  # timeouts, etc.
        raise ValueError(f"Pux API transport error: {exc}") from exc
    if r.status_code >= 400:
        detail = f"HTTP {r.status_code}"
        try:
            body = r.json()
            detail = str(body.get("detail", body))
        except Exception:  # noqa: BLE001
            detail = r.text or detail
        raise ValueError(detail)
    if not r.content:
        return {}
    return r.json()


async def _call(method: str, path: str, **kw: Any) -> tuple[Any, str | None]:
    """Never-raise variant for tools: returns (data, None) on success or
    (None, 'ERROR: …') on failure."""
    try:
        return await _request(method, path, **kw), None
    except ValueError as exc:
        return None, f"ERROR: {exc}"


# --- discovery ---------------------------------------------------------------

@mcp.tool()
async def health() -> str:
    """Check the Pux backend (Agent Protocol server) is reachable and list loaded
    orgs. Call this first if any other tool returns an ERROR."""
    data, err = await _call("GET", "/ok")
    if err:
        return err
    orgs = data.get("orgs", []) if isinstance(data, dict) else []
    return f"ok — backend up; {len(orgs)} org(s): {', '.join(orgs)}"


@mcp.tool()
async def list_orgs() -> str:
    """List available Pux orgs with their specialist subagents and description.
    Each org is a self-contained multi-agent team; pass its id as `org` to
    run_agent / create_thread."""
    data, err = await _call("POST", "/agents/search", json={"metadata": {}, "page": 1})
    if err:
        return err
    # Real server returns a bare list; accept {"agents":[...]} too for safety.
    agents = data if isinstance(data, list) else (data.get("agents", []) if isinstance(data, dict) else [])
    if not agents:
        return "No orgs found."
    lines = []
    for a in agents:
        specialists = a.get("metadata", {}).get("specialists") or []
        sp = ", ".join(specialists) if specialists else "(no subagents)"
        desc = a.get("description", "")
        lines.append(f"- **{a.get('agent_id', '?')}** — {sp}\n    {desc}")
    return "\n".join(lines)


@mcp.tool()
async def get_org(org: str) -> str:
    """Get details for one org (specialist subagents + description).

    Args:
        org: org id (see list_orgs).
    """
    data, err = await _call("GET", f"/agents/{org}")
    if err:
        return err
    specialists = data.get("metadata", {}).get("specialists") or []
    return (
        f"**{data.get('agent_id', org)}**\n"
        f"description: {data.get('description', '(none)')}\n"
        f"specialists: {', '.join(specialists) or '(none)'}"
    )


# --- execution ---------------------------------------------------------------

@mcp.tool()
async def run_agent(
    task: str,
    org: str = "general",
    thread_id: str | None = None,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> str:
    """Run a task on a Pux org and block until the final answer is ready. Use for
    quick tasks. For long tasks use create_thread + start_run so you can poll
    (list_runs) and cancel (cancel_run) instead of blocking.

    Args:
        task: the prompt to send.
        org: org id (default 'general'; see list_orgs).
        thread_id: optional existing thread to continue a conversation.
        recursion_limit: LangGraph node-step cap (default 60).
    """
    body: dict[str, Any] = {"agent_id": org, "input": task, "metadata": {}}
    if thread_id:
        body["thread_id"] = thread_id
    body["recursion_limit"] = recursion_limit
    data, err = await _call("POST", "/runs/wait", json=body)
    if err:
        return err
    return _extract_run_output(data)


@mcp.tool()
async def start_run(
    thread_id: str,
    task: str,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> str:
    """Start a background run on an existing thread — returns immediately with a
    run_id (status 'pending'). Poll with list_runs(thread_id), block for the
    result with wait_for_run(run_id), or stop early with cancel_run(run_id). Use
    this (not run_agent) for long tasks. Create a thread first with create_thread.

    Args:
        thread_id: an existing thread (from create_thread).
        task: the prompt to send.
        recursion_limit: LangGraph node-step cap (default 60).
    """
    data, err = await _call(
        "POST",
        f"/threads/{thread_id}/runs",
        json={"input": task, "recursion_limit": recursion_limit},
    )
    if err:
        return err
    return _format_run_meta(data)


@mcp.tool()
async def wait_for_run(run_id: str) -> str:
    """Block until a background run finishes, then return its final result
    (status, output, error, warnings). Worst-case latency equals run_agent; for
    non-blocking progress use list_runs(thread_id).

    Args:
        run_id: run id from start_run.
    """
    data, err = await _call("GET", f"/runs/{run_id}/wait")
    if err:
        return err
    return _format_run_meta(data)


@mcp.tool()
async def list_runs(thread_id: str) -> str:
    """List runs on a thread with their current status (non-blocking poll). Use
    to watch a background run without blocking.

    Args:
        thread_id: the thread whose runs to list.
    """
    data, err = await _call("GET", f"/threads/{thread_id}/runs")
    if err:
        return err
    if not data:
        return "No runs."
    return "\n".join(_format_run_meta(r) for r in data)


@mcp.tool()
async def cancel_run(run_id: str) -> str:
    """Cancel a background run.

    Args:
        run_id: run id from start_run.
    """
    data, err = await _call("POST", f"/runs/{run_id}/cancel")
    if err:
        return err
    return _format_run_meta(data)


# --- threads -----------------------------------------------------------------

@mcp.tool()
async def create_thread(org: str = "general") -> str:
    """Create a new conversation thread for an org. Returns the thread_id to pass
    to start_run / run_agent for multi-turn conversations."""
    data, err = await _call("POST", "/threads", json={"agent_id": org, "metadata": {}})
    if err:
        return err
    return data.get("thread_id", "")


@mcp.tool()
async def get_thread(thread_id: str) -> str:
    """Get a thread's current state (full message history).

    Args:
        thread_id: the thread to read.
    """
    data, err = await _call("GET", f"/threads/{thread_id}")
    if err:
        return err
    messages = (data.get("values") or {}).get("messages", []) if isinstance(data, dict) else []
    if not messages:
        return "Thread is empty."
    return "\n\n".join(_format_message(m) for m in messages)


@mcp.tool()
async def list_threads(org: str | None = None) -> str:
    """List recent threads, optionally filtered by org.

    Args:
        org: if given, only list threads for this org.
    """
    body: dict[str, Any] = {"metadata": {}, "page": 1}
    if org:
        body["agent_id"] = org
    data, err = await _call("POST", "/threads/search", json=body)
    if err:
        return err
    # Real server returns a bare list; accept {"threads":[...]} too for safety.
    threads = data if isinstance(data, list) else (data.get("threads", []) if isinstance(data, dict) else [])
    if not threads:
        return "No threads found."
    lines = []
    for t in threads:
        tid = t.get("thread_id", "?")
        created = (t.get("created_at") or "?")[:19]
        agent = t.get("agent_id", "?")
        lines.append(f"- `{tid}` — {agent} — {created}")
    return "\n".join(lines)


@mcp.tool()
async def get_thread_history(thread_id: str) -> str:
    """Get the full revision history (checkpoints) of a thread.

    Args:
        thread_id: the thread whose history to read.
    """
    data, err = await _call("GET", f"/threads/{thread_id}/history")
    if err:
        return err
    history = data if isinstance(data, list) else (data.get("history", []) if isinstance(data, dict) else [])
    if not history:
        return "No history."
    lines = []
    for i, h in enumerate(history):
        ts = (h.get("created_at") or "?")[:19]
        msgs = (h.get("values") or {}).get("messages", [])
        last_content = ""
        if msgs:
            last = msgs[-1]
            content = last.get("content", "") if isinstance(last, dict) else last
            last_content = _content_text(content)[:120]
        lines.append(f"{i+1}. [{ts}] {last_content}")
    return "\n".join(lines)


@mcp.tool()
async def delete_thread(thread_id: str) -> str:
    """Delete a thread.

    Args:
        thread_id: the thread to delete.
    """
    _, err = await _call("DELETE", f"/threads/{thread_id}")
    if err:
        return err
    return f"Thread {thread_id} deleted."


# --- jobs (corpus prep) ------------------------------------------------------

@mcp.tool()
async def run_jobs(org: str, job: str | None = None) -> str:
    """Run prep jobs inside an org's sandbox (e.g. corpus ingestion for
    deep-research). Blocks until the jobs finish; returns per-job status.

    Args:
        org: org id.
        job: run only this job by name (default: all declared jobs).
    """
    body = {"job": job} if job else {}
    data, err = await _call("POST", f"/jobs/{org}/run", json=body)
    if err:
        return err
    return _format_jobs(data)


@mcp.tool()
async def get_jobs(org: str) -> str:
    """Show the jobs declared for an org (name, script, description). Does not run them.

    Args:
        org: org id.
    """
    data, err = await _call("GET", f"/jobs/{org}/status")
    if err:
        return err
    return _format_jobs(data)


# --- formatting --------------------------------------------------------------

def _content_text(content: Any) -> str:
    """Flatten a message content (str or list of content blocks) to text."""
    if isinstance(content, list):
        parts = []
        for b in content:
            parts.append(str(b.get("text", b)) if isinstance(b, dict) else str(b))
        return "\n".join(parts)
    return str(content) if content else ""


def _format_message(m: Any) -> str:
    if isinstance(m, dict):
        return f"**{m.get('role', '?')}**: {_content_text(m.get('content', ''))}"
    return str(m)


def _extract_run_output(data: Any) -> str:
    """Render a ``/runs/wait`` response as the bare answer text an MCP client
    wants (not a status blob). The REAL shape is a run-meta dict
    (server.py:356-361): ``status`` success/error with ``output``/``error``.
    On error it returns ``ERROR: <error>`` (consistent with the error-text
    convention); on success it returns ``output``. The legacy ``{messages:[…]}``
    LangGraph-agent shape is handled as a defensive fallback."""
    if isinstance(data, dict) and "status" in data:
        if data.get("status") == "error":
            return f"ERROR: {data.get('error') or 'run failed'}"
        out = data.get("output")
        if out is not None:
            return _content_text(out) or "(empty output)"
    return _extract_answer(data)


def _extract_answer(data: Any) -> str:
    """Pull the final assistant message from a legacy {messages:[...]} response.
    Kept as the fallback path of _extract_run_output; the live /runs/wait route
    returns a run-meta dict, not this shape."""
    messages = data.get("messages", []) if isinstance(data, dict) else []
    if not messages:
        return json.dumps(data, indent=2)
    last = messages[-1]
    content = last.get("content", "") if isinstance(last, dict) else last
    return _content_text(content) or json.dumps(data, indent=2)


def _format_run_meta(meta: Any) -> str:
    """Render a run-meta dict (status/output/error/warnings) as compact text.
    Surfaces warnings so the _run_task job-failure notices aren't swallowed."""
    if not isinstance(meta, dict):
        return str(meta)
    parts = [f"run `{meta.get('run_id', '?')}` — {meta.get('status', '?')}"]
    if meta.get("output"):
        parts.append(f"output: {meta['output']}")
    if meta.get("error"):
        parts.append(f"error: {meta['error']}")
    if meta.get("warnings"):
        parts.append("warnings: " + "; ".join(str(w) for w in meta["warnings"]))
    return "\n".join(parts)


def _format_jobs(data: Any) -> str:
    if not isinstance(data, dict):
        return str(data)
    header = f"org `{data.get('org', '?')}`"
    msg = data.get("message")
    if msg:
        return f"{header}: {msg}"
    jobs = data.get("jobs", [])
    if not jobs:
        return f"{header}: no jobs."
    lines = [header]
    for j in jobs:
        name = j.get("name", "?")
        if "status" in j:  # run_jobs result
            extra = ""
            if j.get("error"):
                extra += f" ({j['error']})"
            if "duration" in j:
                extra += f" [{j['duration']}s]"
            lines.append(f"- {name}: {j['status']}{extra}")
        else:  # get_jobs spec
            lines.append(f"- {name}: {j.get('description') or j.get('script', '')}")
    return "\n".join(lines)


def main() -> None:
    port = int(os.environ.get("PUX_MCP_PORT", "9987"))
    mcp.run(transport="sse", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
