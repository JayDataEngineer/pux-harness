"""Queen-bee MCP server — pure ACP delegation to Pux worker orgs.

Hermes (the queen) connects via MCP-SSE. This server spawns ``pux acp --org X``
subprocesses and speaks ACP (Agent Client Protocol) over stdio. The queen
manages worker bees — she delegates tasks to orgs, answers their questions,
and manages sessions. She does NOT hold granular tools; she delegates.

Architecture:
    Hermes (queen) → MCP-SSE :9987 → [this server] → ACP stdio → pux acp --org X

Each org gets one cached ACP subprocess (the hive). Sessions within that
subprocess are individual worker conversations. The ACP ask_user interrupt
mechanic means a worker asking a question simply ends its turn; the queen's
next prompt_worker() call IS the resume answer.
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from acp import spawn_agent_process
from acp.interfaces import Agent
from acp.schema import TextContentBlock

from pux_harness.agent.orgs import discover_orgs
from pux_harness.agent.model import available_model_ids

# PUX_PROJECT_ROOT = where orgs/, .env, .pux/ live (the Pux project root —
# the parent of pux-harness). pux acp discovers orgs from here.
PUX_PROJECT_ROOT = os.environ.get(
    "PUX_PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
# PUX_HARNESS_DIR = where the Python package + venv live.
PUX_HARNESS_DIR = os.path.join(PUX_PROJECT_ROOT, "pux-harness")
# PUX_BIN = the venv pux executable (avoids `uv run` overhead + env issues).
PUX_BIN = os.path.join(PUX_HARNESS_DIR, ".venv", "bin", "pux")

MCP = FastMCP(
    "pux-queen",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,  # internal Tailscale-only
    ),
)


# ═══════════════════════════════════════════════════════════════════════════
# QueenClient — minimal ACP client that collects agent messages
# ═══════════════════════════════════════════════════════════════════════════

class QueenClient:
    """Collects agent messages + thoughts per session during prompt() calls."""

    def __init__(self) -> None:
        self._buffers: dict[str, dict[str, list[str]]] = {}

    def reset(self, session_id: str) -> None:
        self._buffers[session_id] = {"messages": [], "thoughts": []}

    def messages(self, session_id: str) -> list[str]:
        return self._buffers.get(session_id, {}).get("messages", [])

    @staticmethod
    def _text(content: Any) -> str:
        return getattr(content, "text", "") or ""

    # ── Client protocol (Protocol class — only session_update is load-bearing)

    def on_connect(self, conn: Agent) -> None:
        pass

    async def session_update(self, session_id: str, update: Any, **kw: Any) -> None:
        buf = self._buffers.setdefault(
            session_id, {"messages": [], "thoughts": []}
        )
        kind = getattr(update, "session_update", "")
        if kind in ("agent_message_chunk", "agent_thought_chunk"):
            text = self._text(update.content)
            if text:
                key = "messages" if kind == "agent_message_chunk" else "thoughts"
                # Streaming chunks arrive fragmented; concatenate into one string.
                if not buf[key]:
                    buf[key].append("")
                buf[key][-1] += text

    # Stubs — Pux workers use their own Docker sandbox; editor fs/terminal
    # callbacks are never invoked in transport="acp" mode.
    async def request_permission(self, *a: Any, **kw: Any) -> None: ...
    async def read_text_file(self, *a: Any, **kw: Any) -> None: ...
    async def write_text_file(self, *a: Any, **kw: Any) -> None: ...
    async def create_terminal(self, *a: Any, **kw: Any) -> None: ...
    async def terminal_output(self, *a: Any, **kw: Any) -> None: ...
    async def release_terminal(self, *a: Any, **kw: Any) -> None: ...
    async def kill_terminal(self, *a: Any, **kw: Any) -> None: ...
    async def wait_for_terminal_exit(self, *a: Any, **kw: Any) -> None: ...
    async def create_elicitation(self, *a: Any, **kw: Any) -> None: ...
    async def complete_elicitation(self, *a: Any, **kw: Any) -> None: ...
    async def ext_method(self, *a: Any, **kw: Any) -> dict[str, Any]:
        return {}

    async def ext_notification(self, *a: Any, **kw: Any) -> None: ...


# ═══════════════════════════════════════════════════════════════════════════
# OrgConnection — one ACP subprocess per org (the hive)
# ═══════════════════════════════════════════════════════════════════════════

class OrgConnection:
    """One cached ACP subprocess for an org. Lives for the server lifetime."""

    def __init__(self, org: str) -> None:
        self.org = org
        self.conn: Any = None          # ClientSideConnection (Agent interface)
        self.process: Any = None
        self.client = QueenClient()
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._start_error: Exception | None = None

    @property
    def alive(self) -> bool:
        return (
            self.conn is not None
            and self.process is not None
            and self.process.returncode is None
        )

    async def start(self) -> None:
        """Spawn ``pux acp --org X`` and hold the ACP connection open."""
        self._ready.clear()
        self._stop.clear()
        self._start_error = None
        self._task = asyncio.create_task(self._hold())
        await self._ready.wait()
        if self._start_error:
            raise self._start_error

    async def _hold(self) -> None:
        """Background task that keeps the ``async with`` alive."""
        # Redirect stderr to a per-org log file to prevent pipe-buffer deadlock.
        log_path = f"/tmp/pux-acp-{self.org}.stderr"
        stderr_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        try:
            async with spawn_agent_process(
                self.client,
                PUX_BIN, "acp", "--org", self.org,
                cwd=PUX_PROJECT_ROOT,
                transport_kwargs={"stderr": stderr_fd},
            ) as (conn, process):
                os.close(stderr_fd)  # parent copy; subprocess has its own
                self.conn = conn
                self.process = process
                await conn.initialize(
                    protocol_version=1,
                    client_info={"name": "hermes-queen", "version": "1.0"},
                )
                self._ready.set()
                await self._stop.wait()  # hold the async with open
        except Exception as exc:
            self._start_error = exc
            self._ready.set()

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
        self._task = None
        self.conn = None
        self.process = None
        self._ready.clear()
        self._stop.clear()

    async def ensure(self) -> "OrgConnection":
        if not self.alive:
            await self.stop()
            await self.start()
        return self

    # ── ACP operations ──────────────────────────────────────────────────

    async def new_session(self, model: str | None = None) -> str:
        async with self._lock:
            await self.ensure()
            resp = await self.conn.new_session(cwd=PUX_PROJECT_ROOT)
            sid = resp.session_id
            if model:
                await self._set_model(sid, model)
            return sid

    async def prompt(self, session_id: str, message: str) -> tuple[str, str]:
        async with self._lock:
            await self.ensure()
            self.client.reset(session_id)
            resp = await self.conn.prompt(
                session_id,
                [TextContentBlock(type="text", text=message)],
            )
            msgs = self.client.messages(session_id)
            text = "\n".join(msgs) if msgs else ""
            return text, resp.stop_reason

    async def list_sessions_raw(self) -> list[dict]:
        async with self._lock:
            await self.ensure()
            resp = await self.conn.list_sessions(cwd=PUX_PROJECT_ROOT)
            sessions = getattr(resp, "sessions", []) or []
            return [
                {
                    "session_id": s.session_id,
                    "org": self.org,
                    "title": getattr(s, "title", None),
                    "updated_at": getattr(s, "updated_at", None),
                }
                for s in sessions
            ]

    async def cancel(self, session_id: str) -> None:
        async with self._lock:
            await self.ensure()
            await self.conn.cancel(session_id)

    async def load(self, session_id: str) -> bool:
        async with self._lock:
            await self.ensure()
            resp = await self.conn.load_session(
                cwd=PUX_PROJECT_ROOT, session_id=session_id
            )
            return resp is not None

    async def set_model_raw(self, session_id: str, model: str) -> None:
        async with self._lock:
            await self.ensure()
            await self._set_model(session_id, model)

    async def _set_model(self, session_id: str, model: str) -> None:
        """Set the model via ACP config option. The config_id is 'model'."""
        await self.conn.set_config_option("model", session_id, model)


# ═══════════════════════════════════════════════════════════════════════════
# Pool — one OrgConnection per org, lazily spawned
# ═══════════════════════════════════════════════════════════════════════════

_pool: dict[str, OrgConnection] = {}
_session_org: dict[str, str] = {}  # session_id → org


async def _get_org(org: str) -> OrgConnection:
    """Get (or create + start) the OrgConnection for an org."""
    if org not in _pool:
        _pool[org] = OrgConnection(org)
    return await _pool[org].ensure()


def _find_org_for_session(session_id: str) -> OrgConnection | None:
    """Find the OrgConnection that owns a session."""
    org = _session_org.get(session_id)
    if org and org in _pool and _pool[org].alive:
        return _pool[org]
    # Fallback: search all alive connections
    for oc in _pool.values():
        if oc.alive:
            return oc
    return None


# ═══════════════════════════════════════════════════════════════════════════
# MCP Tools — queen-bee worker management (pure ACP)
# ═══════════════════════════════════════════════════════════════════════════

@MCP.tool()
async def list_orgs() -> str:
    """List available worker orgs (specialist agents) the queen can delegate to.

    Each org is a specialist worker bee: coder, deep-research-engine,
    twitter-agent, etc. Delegate with new_session(org).
    """
    orgs = sorted(discover_orgs())
    lines = [f"**{len(orgs)} worker orgs available:**"]
    for o in orgs:
        lines.append(f"- `{o}`")
    return "\n".join(lines)


@MCP.tool()
async def list_models() -> str:
    """List available models the worker bees can use.

    Pass a model name to new_session(org, model) or set_model(session_id, model).
    """
    models = available_model_ids()
    lines = [f"**{len(models)} models available:**"]
    for m in models:
        lines.append(f"- `{m}`")
    return "\n".join(lines)


@MCP.tool()
async def new_session(org: str = "general", model: str | None = None) -> str:
    """Recruit a new worker bee — creates an ACP session on an org.

    Args:
        org: Worker org to delegate to (e.g. 'coder', 'general',
             'deep-research-engine'). Use list_orgs() for all options.
        model: Optional model override (e.g. 'glm-5.2'). See list_models().

    Returns:
        The session_id for the new worker. Use prompt_worker() to delegate.
    """
    known = discover_orgs()
    if org not in known:
        return f"Error: unknown org '{org}'. Available: {', '.join(sorted(known))}"
    try:
        conn = await _get_org(org)
        sid = await conn.new_session(model=model)
        _session_org[sid] = org
        tag = f" (model: {model})" if model else ""
        return (
            f"Worker recruited.\n"
            f"  session: `{sid}`\n"
            f"  org: `{org}`{tag}\n"
            f"Use prompt_worker() to delegate tasks."
        )
    except Exception as exc:
        return f"Error creating session on '{org}': {exc}"


@MCP.tool()
async def prompt_worker(session_id: str, message: str) -> str:
    """Delegate a task to a worker, or answer a worker's question.

    This is the universal delegation primitive:
    - Give a task:    prompt_worker(session_id, "build a script that does X")
    - Answer a question: prompt_worker(session_id, "use Python")
      (After an end_turn, the next message IS the resume answer — ACP-native.
       The ask_user interrupt persists in the checkpoint; your answer unblocks it.)

    Args:
        session_id: Worker session from new_session() or load_session().
        message: Task, follow-up, or answer to send.

    Returns:
        Worker's response text + stop reason. stop_reason='end_turn' means the
        worker finished OR is asking a question (the question is in the text).
    """
    oc = _find_org_for_session(session_id)
    if oc is None:
        return (
            f"Error: no active connection for session '{session_id}'.\n"
            f"Create one with new_session(org)."
        )
    try:
        text, stop = await oc.prompt(session_id, message)
        parts = []
        if text:
            parts.append(text)
        else:
            parts.append("(worker produced no text)")
        parts.append(f"\n*[{stop}]*")
        if stop == "end_turn":
            parts.append(
                "\n(Worker done or asking a question. "
                "If it asked something, answer with another prompt_worker() call.)"
            )
        elif stop == "cancelled":
            parts.append("\n(Task cancelled.)")
        return "\n".join(parts)
    except Exception as exc:
        return f"Error: {exc}"


@MCP.tool()
async def list_sessions(org: str | None = None) -> str:
    """List active worker sessions.

    Args:
        org: Optional org filter. If omitted, lists across all orgs.

    Returns:
        Active sessions with org, title, and last update time.
    """
    orgs_to_check = [org] if org else list(_pool.keys())
    if not orgs_to_check:
        return "No worker sessions. Use new_session(org) to recruit a worker."
    all_sessions: list[dict] = []
    for o in orgs_to_check:
        if o in _pool and _pool[o].alive:
            try:
                all_sessions.extend(await _pool[o].list_sessions_raw())
            except Exception:
                pass
    if not all_sessions:
        return "No active worker sessions."
    lines = [f"**{len(all_sessions)} session(s):**"]
    for s in all_sessions:
        title = f" — {s['title']}" if s.get("title") else ""
        ts = f" ({s['updated_at']})" if s.get("updated_at") else ""
        lines.append(f"- `{s['session_id']}` org=`{s['org']}`{title}{ts}")
    return "\n".join(lines)


@MCP.tool()
async def load_session(session_id: str, org: str) -> str:
    """Resume a past worker session.

    Args:
        session_id: The session to resume.
        org: The org the session belongs to.

    Returns:
        Confirmation. Use prompt_worker() to continue the conversation.
    """
    try:
        conn = await _get_org(org)
        ok = await conn.load(session_id)
        if ok:
            _session_org[session_id] = org
            return f"Session `{session_id}` resumed on `{org}`."
        return f"Session `{session_id}` not found on `{org}`."
    except Exception as exc:
        return f"Error: {exc}"


@MCP.tool()
async def cancel_session(session_id: str) -> str:
    """Cancel an in-progress task on a worker.

    Args:
        session_id: The worker session to cancel.

    Returns:
        Confirmation of cancellation.
    """
    oc = _find_org_for_session(session_id)
    if oc is None:
        return f"Error: no active connection for session '{session_id}'."
    try:
        await oc.cancel(session_id)
        return f"Cancelled task on `{session_id}`."
    except Exception as exc:
        return f"Error: {exc}"


@MCP.tool()
async def set_model(session_id: str, model: str) -> str:
    """Change the model on a worker session.

    Args:
        session_id: The worker session.
        model: Model to use (e.g. 'glm-5.2'). See list_models().

    Returns:
        Confirmation of model change.
    """
    oc = _find_org_for_session(session_id)
    if oc is None:
        return f"Error: no active connection for session '{session_id}'."
    try:
        await oc.set_model_raw(session_id, model)
        return f"Model set to `{model}` on `{session_id}`."
    except Exception as exc:
        return f"Error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    import uvicorn  # noqa: PLC0415

    host = os.environ.get("PUX_MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("PUX_MCP_PORT", "9987"))
    sys.stderr.write(f"[pux-queen] MCP-SSE on {host}:{port}\n")
    sys.stderr.write(f"[pux-queen] ACP subprocess root: {PUX_PROJECT_ROOT}\n")
    app = MCP.sse_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
