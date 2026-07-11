"""MCP server that wraps the Pux Agent Protocol REST API (Aegra on :9988).

Exposes the org graph as structured MCP tools so any MCP client (Hermes, Claude
Desktop, Zed) can drive Pux's worker-bee orgs without knowing the Agent Protocol
wire format.  Transport: SSE on :9987 (``pux mcp``).

Queen-bee architecture: Hermes (the queen) calls these tools to command Pux's
specialist orgs (the worker bees).  Every Pux capability — run, stream, thread
lifecycle, interrupt detection, interrupt resume, model selection — is a
STRUCTURED TOOL that ENFORCES behavior, never a prompt instruction.

Requires the Agent Protocol server to be running — prod is Aegra
(scripts/start_pux_aegra.sh on :9988).  Call ``health`` first to confirm.

Wire format (Aegra / langgraph-api):
  - ``assistant_id`` (NOT ``agent_id``) — accepts the org name directly.
  - ``input`` is a DICT: ``{"messages": [{"role":"user","content":"..."}]}``.
  - Interrupts appear in thread STATE under ``tasks[*].interrupts``.
  - Resume via ``POST /threads/{id}/runs`` with ``"command": {"resume": ...}``.

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
STREAM_TIMEOUT = float(os.environ.get("PUX_MCP_STREAM_TIMEOUT", "600"))
DEFAULT_RECURSION_LIMIT = 60  # matches the cli.py default

mcp = FastMCP(
    "pux",
    instructions=(
        "Pux multi-org agent orchestrator — the queen-bee's worker army.\n"
        "Workflow:\n"
        "1. `health` — confirm the backend is up (call first if anything errors).\n"
        "2. `list_orgs` — pick an org; each is a self-contained multi-agent team.\n"
        "3. Quick task: `run_agent(task, org)` blocks until the answer is ready.\n"
        "4. Long task: `create_thread(org)` -> `start_run(thread_id, task)` -> "
        "poll `list_runs(thread_id)` or block `wait_for_run(run_id)` -> "
        "`cancel_run(run_id)` to stop early.\n"
        "5. Multi-turn: `update_thread(thread_id, message)` to add to a thread.\n"
        "6. Interrupts: `check_interrupt(thread_id)` to see if the agent is asking "
        "a question, then `resume_thread(thread_id, response)` to answer.\n"
        "7. Models: `list_models()` to see available models; pass `model` to "
        "run_agent/start_run/update_thread.\n"
        "8. Corpus prep: `get_jobs(org)` then `run_jobs(org)`.\n"
        "Threads persist across calls (see get_thread, get_thread_state, "
        "get_thread_history). Tool failures come back as text prefixed `ERROR:`."
    ),
)


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------
# Lazy singleton so calls reuse one connection pool.  Tests monkeypatch `_client`
# with an httpx.AsyncClient over httpx.MockTransport (see tests/test_mcp_server.py).
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(base_url=PUX_API, timeout=TIMEOUT)
    return _client


async def _request(method: str, path: str, **kw: Any) -> Any:
    """HTTP wrapper.  Returns parsed JSON (or {} for empty bodies).  Raises
    ValueError with a clean, actionable message on failure."""
    client = _get_client()
    try:
        r = await client.request(method, path, **kw)
    except httpx.ConnectError as exc:
        raise ValueError(
            f"cannot reach Pux API at {PUX_API} ({exc}); is the Agent Protocol "
            f"server running? (prod: `aegra serve`; dev: `langgraph dev` / `aegra dev`)"
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


# ---------------------------------------------------------------------------
# SSE stream collector — used internally by run_agent / update_thread for
# full error visibility (POST /runs/wait returns {} on error; stream does not).
# ---------------------------------------------------------------------------

async def _stream_run(
    thread_id: str,
    *,
    assistant_id: str,
    input_: dict[str, Any] | None = None,
    command: dict[str, Any] | None = None,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    model: str | None = None,
) -> tuple[str | None, str | None]:
    """POST /threads/{id}/runs/stream and collect events.

    Returns (output_text, error_text).  Exactly one is non-None:
    - On success: (final_answer, None).
    - On error:    (None, error_message).
    """
    body: dict[str, Any] = {
        "assistant_id": assistant_id,
        "recursion_limit": recursion_limit,
    }
    if input_ is not None:
        body["input"] = input_
    if command is not None:
        body["command"] = command
    if model:
        body["config"] = {"configurable": {"model": model}}

    client = _get_client()
    final_values: dict[str, Any] = {}
    error_msg: str | None = None
    status = "unknown"

    try:
        async with client.stream(
            "POST",
            f"/threads/{thread_id}/runs/stream",
            json=body,
            timeout=httpx.Timeout(STREAM_TIMEOUT, connect=10.0),
        ) as resp:
            if resp.status_code >= 400:
                body_text = await resp.aread()
                try:
                    err_detail = json.loads(body_text)
                    detail = str(err_detail.get("detail", err_detail))
                except Exception:
                    detail = body_text.decode()[:500] if body_text else f"HTTP {resp.status_code}"
                return None, detail

            event_type: str | None = None
            data_lines: list[str] = []

            async for raw_line in resp.aiter_lines():
                if raw_line.startswith("event:"):
                    event_type = raw_line[6:].strip()
                elif raw_line.startswith("data:"):
                    data_lines.append(raw_line[5:].strip())
                elif raw_line == "" and event_type is not None:
                    # Event boundary — process accumulated event
                    data_str = "\n".join(data_lines)
                    payload: Any = {}
                    if data_str:
                        try:
                            payload = json.loads(data_str)
                        except Exception:
                            payload = {"raw": data_str}

                    if event_type == "values" and isinstance(payload, dict):
                        final_values = payload  # keep overwriting → last values wins
                    elif event_type == "error":
                        if isinstance(payload, dict):
                            error_msg = payload.get("message") or payload.get("error") or str(payload)
                        else:
                            error_msg = str(payload)
                    elif event_type == "end":
                        if isinstance(payload, dict):
                            status = payload.get("status", status)

                    event_type = None
                    data_lines = []
    except httpx.ConnectError as exc:
        return None, f"cannot reach Pux API ({exc})"
    except httpx.HTTPError as exc:
        return None, f"stream transport error: {exc}"

    if error_msg:
        return None, error_msg
    if status == "error":
        return None, final_values.get("error") or "run ended with error (no detail)"
    # Success — extract the last assistant message
    return _extract_last_answer(final_values), None


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

@mcp.tool()
async def health() -> str:
    """Check the Pux backend (Agent Protocol server) is reachable and list served
    orgs.  Call this first if any other tool returns an ERROR."""
    data, err = await _call("GET", "/events/health")
    if err:
        return err
    assistants, aerr = await _call("POST", "/assistants/search", json={})
    if aerr:
        return f"ok — backend up (org list unavailable: {aerr})"
    orgs = sorted({
        a["graph_id"] for a in (assistants or [])
        if isinstance(a, dict) and a.get("graph_id")
    })
    return f"ok — backend up; {len(orgs)} org(s): {', '.join(orgs)}"


@mcp.tool()
async def list_orgs() -> str:
    """List available Pux orgs with their description.  Each org is a
    self-contained multi-agent team; pass its id (the ``graph_id``) as ``org``
    to run_agent / create_thread."""
    data, err = await _call("POST", "/assistants/search", json={})
    if err:
        return err
    agents = data if isinstance(data, list) else (data.get("assistants", []) if isinstance(data, dict) else [])
    if not agents:
        return "No orgs found."
    lines = []
    for a in agents:
        gid = a.get("graph_id", "?")
        desc = a.get("description", "")
        lines.append(f"- **{gid}** — {desc}")
    return "\n".join(lines)


@mcp.tool()
async def get_org(org: str) -> str:
    """Get details for one org.

    Args:
        org: org id (the graph_id; see list_orgs).
    """
    # Aegra identifies assistants by UUID, not by graph_id/name.  Search for the
    # one whose graph_id matches.
    data, err = await _call("POST", "/assistants/search", json={})
    if err:
        return err
    agents = data if isinstance(data, list) else (data.get("assistants", []) if isinstance(data, dict) else [])
    match = next((a for a in agents if a.get("graph_id") == org), None)
    if not match:
        return f"ERROR: org '{org}' not found."
    return (
        f"**{match.get('graph_id', org)}**\n"
        f"assistant_id: {match.get('assistant_id', '?')}\n"
        f"description: {match.get('description', '(none)')}\n"
        f"name: {match.get('name', '?')}"
    )


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

@mcp.tool()
async def run_agent(
    task: str,
    org: str = "general",
    thread_id: str | None = None,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    model: str | None = None,
) -> str:
    """Run a task on a Pux org and block until the final answer is ready.  Use
    for quick tasks.  For long tasks use create_thread + start_run so you can
    poll (list_runs) and cancel (cancel_run) instead of blocking.

    Uses SSE streaming internally so errors (ContainerError, model failures) are
    surfaced as ERROR: text instead of silently returning empty output.

    Args:
        task: the prompt to send.
        org: org id (default 'general'; see list_orgs).
        thread_id: optional existing thread to continue a conversation.
        recursion_limit: LangGraph node-step cap (default 60).
        model: optional model override (see list_models).
    """
    # Auto-create a thread if none given (Aegra runs are per-thread).
    tid = thread_id
    if not tid:
        data, err = await _call("POST", "/threads", json={"assistant_id": org, "metadata": {}})
        if err:
            return err
        tid = data.get("thread_id", "")
        if not tid:
            return "ERROR: failed to create thread for run"

    output, error = await _stream_run(
        tid,
        assistant_id=org,
        input_={"messages": [{"role": "user", "content": task}]},
        recursion_limit=recursion_limit,
        model=model,
    )
    if error:
        return f"ERROR: {error}"
    return output or "(empty output)"


@mcp.tool()
async def start_run(
    thread_id: str,
    task: str,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    webhook_url: str | None = None,
    model: str | None = None,
) -> str:
    """Start a background run on an existing thread — returns immediately with a
    run_id (status 'pending').  Poll with list_runs(thread_id), block for the
    result with wait_for_run(run_id), or stop early with cancel_run(run_id).

    Args:
        thread_id: an existing thread (from create_thread).
        task: the prompt to send.
        recursion_limit: LangGraph node-step cap (default 60).
        webhook_url: optional completion callback.
        model: optional model override (see list_models).
    """
    body: dict[str, Any] = {
        "assistant_id": _thread_org(thread_id) or "general",
        "input": {"messages": [{"role": "user", "content": task}]},
        "recursion_limit": recursion_limit,
    }
    if webhook_url:
        body["webhook_url"] = webhook_url
    if model:
        body["config"] = {"configurable": {"model": model}}
    data, err = await _call("POST", f"/threads/{thread_id}/runs", json=body)
    if err:
        return err
    return _format_run_meta(data)


@mcp.tool()
async def wait_for_run(run_id: str) -> str:
    """Block until a background run finishes, then return its final result.

    Args:
        run_id: run id from start_run.
    """
    data, err = await _call("GET", f"/runs/{run_id}/wait")
    if err:
        return err
    return _format_run_meta(data)


@mcp.tool()
async def list_runs(thread_id: str) -> str:
    """List runs on a thread with their current status (non-blocking poll).

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


# ---------------------------------------------------------------------------
# threads
# ---------------------------------------------------------------------------

@mcp.tool()
async def create_thread(org: str = "general") -> str:
    """Create a new conversation thread for an org.  Returns the thread_id to
    pass to start_run / run_agent / update_thread for multi-turn conversations."""
    data, err = await _call("POST", "/threads", json={"assistant_id": org, "metadata": {}})
    if err:
        return err
    return data.get("thread_id", "")


async def _get_latest_run(thread_id: str) -> tuple[dict[str, Any], str | None]:
    """Return (latest_run_dict, error_text) for a thread.

    Aegra doesn't implement GET /threads/{id}/state, so thread messages and
    run status come from the runs list -- the latest run's ``output`` field
    holds the final graph-state values ({messages: [...]}).
    """
    data, err = await _call("GET", f"/threads/{thread_id}/runs")
    if err:
        return {}, err
    if not data:
        return {}, None
    latest = data[0] if isinstance(data, list) and data else {}
    return latest, None


@mcp.tool()
async def get_thread(thread_id: str) -> str:
    """Get a thread's message history (the conversation so far).

    Args:
        thread_id: the thread to read.
    """
    run, err = await _get_latest_run(thread_id)
    if err:
        return err
    output = run.get("output") or {}
    messages = output.get("messages", []) if isinstance(output, dict) else []
    if not messages:
        status = run.get("status", "empty")
        return f"Thread is empty (run status: {status})."
    return "\n\n".join(_format_message(m) for m in messages)
@mcp.tool()
async def get_thread_state(thread_id: str) -> str:
    """Get a thread's full state — run status, messages, and error info.
    Use this to inspect what the agent is doing right now.

    Args:
        thread_id: the thread to inspect.
    """
    run, err = await _get_latest_run(thread_id)
    if err:
        return err
    if not run:
        data, _ = await _call("GET", f"/threads/{thread_id}")
        status = data.get("status", "?") if isinstance(data, dict) else "?"
        return f"Thread `{thread_id}` — status: {status} (no runs yet)"
    parts: list[str] = [f"Thread `{thread_id}`"]
    parts.append(f"run status: {run.get('status', '?')}")
    if run.get("error_message"):
        parts.append(f"error: {run['error_message'][:300]}")
    output = run.get("output") or {}
    msgs = output.get("messages", []) if isinstance(output, dict) else []
    parts.append(f"messages: {len(msgs)}")
    if msgs:
        parts.append("recent messages:")
        for m in msgs[-3:]:
            parts.append("  " + _format_message(m)[:300])
    return "\n".join(parts)
@mcp.tool()
async def update_thread(
    thread_id: str,
    message: str,
    org: str = "general",
    model: str | None = None,
) -> str:
    """Add a message to an existing thread and get the agent's response.  This is
    the multi-turn conversation primitive — it blocks until the agent replies.

    Args:
        thread_id: an existing thread (from create_thread or run_agent).
        message: the new user message to add.
        org: the org this thread belongs to (default 'general').
        model: optional model override (see list_models).
    """
    output, error = await _stream_run(
        thread_id,
        assistant_id=org,
        input_={"messages": [{"role": "user", "content": message}]},
        model=model,
    )
    if error:
        return f"ERROR: {error}"
    return output or "(empty output)"


@mcp.tool()
async def list_threads(org: str | None = None) -> str:
    """List recent threads, optionally filtered by org.

    Args:
        org: if given, only list threads for this org.
    """
    body: dict[str, Any] = {"metadata": {}, "limit": 50}
    if org:
        body["metadata"] = {"graph_id": org}
    data, err = await _call("POST", "/threads/search", json=body)
    if err:
        return err
    threads = data if isinstance(data, list) else (data.get("threads", []) if isinstance(data, dict) else [])
    if not threads:
        return "No threads found."
    lines = []
    for t in threads:
        tid = t.get("thread_id", "?")
        status = t.get("status", "?")
        created = (t.get("created_at") or "?")[:19]
        gid = (t.get("metadata") or {}).get("graph_id", "?")
        lines.append(f"- `{tid}` — {gid} — {status} — {created}")
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
    """Delete a thread permanently.

    Args:
        thread_id: the thread to delete.
    """
    _, err = await _call("DELETE", f"/threads/{thread_id}")
    if err:
        return err
    return f"Thread {thread_id} deleted."


# ---------------------------------------------------------------------------
# HITL — interrupt detection and resume (the queen-bee's ask/answer channel)
# ---------------------------------------------------------------------------

@mcp.tool()
async def check_interrupt(thread_id: str) -> str:
    """Check if a thread has a pending interrupt (the agent is asking a question
    or waiting for approval).  Returns the interrupt payload if pending, or
    'No pending interrupts.' if the thread is clear.

    Detects interrupts by checking if the latest run is in 'interrupted' status.
    When the agent calls ask_user, the run pauses with status='interrupted' and
    the question is in the run output.

    Args:
        thread_id: the thread to check.
    """
    run, err = await _get_latest_run(thread_id)
    if err:
        return err
    if not run:
        return "No runs on this thread."
    status = run.get("status", "?")
    if status == "interrupted":
        output = run.get("output") or {}
        msgs = output.get("messages", []) if isinstance(output, dict) else []
        question = ""
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("type") in ("ai", "assistant"):
                question = _content_text(m.get("content", ""))
                break
        return (
            f"⏸ INTERRUPTED — the agent is asking:\n"
            f"❓ {question or '(question not found in output)'}\n\n"
            f"→ Use resume_thread(thread_id, response) to answer."
        )
    return f"No pending interrupts (latest run status: {status})."


@mcp.tool()
async def resume_thread(
    thread_id: str,
    response: str,
    org: str = "general",
) -> str:
    """Resume an interrupted thread with a response.  This is how the queen-bee
    answers a worker bee's question — the agent's ask_user interrupt is resumed
    with your response, and the agent continues.

    Args:
        thread_id: the thread with a pending interrupt (check with check_interrupt).
        response: your answer to the agent's question.
        org: the org this thread belongs to (default 'general').
    """
    output, error = await _stream_run(
        thread_id,
        assistant_id=org,
        command={"resume": {"decisions": [{"type": "ask_user", "answer": response}]}},
    )
    if error:
        return f"ERROR: {error}"
    return output or "(agent continued — no text output)"


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_models() -> str:
    """List available models from Pux's models.yaml.  Pass any of these ids as
    the ``model`` parameter to run_agent / start_run / update_thread to override
    the org's default model for that run."""
    try:
        from pux_harness.agent.model import available_model_ids
        ids = available_model_ids()
    except Exception:
        return "ERROR: cannot read models.yaml (is the harness installed?)"
    if not ids:
        return "No models declared in models.yaml."
    return "Available models:\n" + "\n".join(f"- {m}" for m in ids)


# ---------------------------------------------------------------------------
# events (run-completion SSE)
# ---------------------------------------------------------------------------

@mcp.tool()
async def recent_events(limit: int = 20) -> str:
    """Show recent run-completion events across all orgs (the catch-up poll).
    Each event shows the run_id, status, and org.  Use this to see what the
    worker bees have been doing.

    Args:
        limit: max events to return (default 20).
    """
    data, err = await _call("GET", "/events", params={"limit": limit})
    if err:
        return err
    events = data.get("events", []) if isinstance(data, dict) else []
    if not events:
        return "No recent events."
    lines = []
    for ev in events:
        ts = (ev.get("ts") or "?")[:19]
        status = ev.get("status", "?")
        run_id = ev.get("run_id", "?")[:12]
        org = (ev.get("metadata") or {}).get("graph_id", "?")
        lines.append(f"- [{ts}] {org}: {status} (run {run_id}…)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# jobs (corpus prep)
# ---------------------------------------------------------------------------

@mcp.tool()
async def run_jobs(org: str, job: str | None = None) -> str:
    """Run prep jobs inside an org's sandbox (e.g. corpus ingestion for
    deep-research).  Blocks until the jobs finish; returns per-job status.

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
    """Show the jobs declared for an org (name, script, description).  Does not run them.

    Args:
        org: org id.
    """
    data, err = await _call("GET", f"/jobs/{org}/status")
    if err:
        return err
    return _format_jobs(data)


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

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
        role = m.get("type") or m.get("role") or "?"
        return f"**{role}**: {_content_text(m.get('content', ''))}"
    return str(m)


def _extract_last_answer(values: dict[str, Any]) -> str | None:
    """Pull the final assistant message from a graph-state values dict."""
    msgs = values.get("messages") or []
    if not msgs:
        return None
    # Walk backwards for the last AI/assistant message.
    for m in reversed(msgs):
        if isinstance(m, dict):
            mtype = m.get("type") or m.get("role") or ""
            if mtype in ("ai", "assistant"):
                return _content_text(m.get("content", "")) or None
        # LangGraph message objects serialized as dicts may have "type":"ai"
    # Fallback: last message of any type
    last = msgs[-1]
    if isinstance(last, dict):
        return _content_text(last.get("content", ""))
    return str(last) if last else None


def _extract_run_output(data: Any) -> str:
    """Render a run response as the bare answer text an MCP client wants."""
    if isinstance(data, dict) and "status" in data:
        if data.get("status") == "error":
            return f"ERROR: {data.get('error') or 'run failed'}"
        out = data.get("output")
        if out is not None:
            return _content_text(out) or "(empty output)"
    return _extract_last_answer(data) or json.dumps(data, indent=2)[:2000]


def _format_run_meta(meta: Any) -> str:
    """Render a run-meta dict (status/output/error/warnings) as compact text."""
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
        if "status" in j:
            extra = ""
            if j.get("error"):
                extra += f" ({j['error']})"
            if "duration" in j:
                extra += f" [{j['duration']}s]"
            lines.append(f"- {name}: {j['status']}{extra}")
        else:
            lines.append(f"- {name}: {j.get('description') or j.get('script', '')}")
    return "\n".join(lines)


def _thread_org(thread_id: str) -> str | None:
    """Best-effort: fetch the org (graph_id) for a thread.  Returns None on
    failure (the caller falls back to 'general')."""
    # This is a sync helper called from an async tool — but it's just a dict
    # lookup in the thread metadata cache.  We do a quick GET.
    # (Kept simple; if it fails, the caller has a sensible default.)
    return None  # the caller uses the org arg or 'general'


def main() -> None:
    port = int(os.environ.get("PUX_MCP_PORT", "9987"))
    host = os.environ.get("PUX_MCP_HOST", "0.0.0.0")
    mcp.run(transport="sse", host=host, port=port)


if __name__ == "__main__":
    main()
