"""MCP server — pure ACP delegation to Pux subagents.

Hermes (the orchestrator) connects via MCP-SSE. This server spawns
``pux acp --org X`` subprocesses and speaks ACP (Agent Client Protocol) over
stdio. Hermes delegates tasks to orgs (subagents), answers their questions,
and manages sessions. It does NOT hold granular tools; it delegates.

Architecture:
    Hermes (orchestrator) -> MCP-SSE :9987 -> [this server] -> ACP stdio -> pux acp --org X

Each org gets one cached ACP subprocess. Sessions within that subprocess are
individual subagent conversations. The ACP ask_user interrupt mechanic means
a subagent asking a question simply ends its turn; the orchestrator's next
prompt() call IS the resume answer.
"""
from __future__ import annotations

import asyncio
import base64
import os
import re
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ImageContent as MCPImageContent
from mcp.types import TextContent as MCPTextContent

from acp import spawn_agent_process
from acp.interfaces import Agent
from acp.schema import ImageContentBlock, TextContentBlock

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
    "pux",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,  # internal Tailscale-only
    ),
)


# ═══════════════════════════════════════════════════════════════════════════
# SubagentClient — minimal ACP client that collects agent messages
# ═══════════════════════════════════════════════════════════════════════════

class SubagentClient:
    """Collects agent messages + thoughts per session during prompt() calls."""

    def __init__(self) -> None:
        self._buffers: dict[str, dict[str, list]] = {}

    def reset(self, session_id: str) -> None:
        self._buffers[session_id] = {"messages": [], "thoughts": [], "images": []}

    def messages(self, session_id: str) -> list[str]:
        return self._buffers.get(session_id, {}).get("messages", [])

    def thoughts(self, session_id: str) -> list[str]:
        return self._buffers.get(session_id, {}).get("thoughts", [])

    def images(self, session_id: str) -> list[dict]:
        """Return captured image content blocks as ``{"data": b64, "mime_type": str}``."""
        return self._buffers.get(session_id, {}).get("images", [])

    @staticmethod
    def _text(content: Any) -> str:
        return getattr(content, "text", "") or ""

    # ── Client protocol (Protocol class — only session_update is load-bearing)

    def on_connect(self, conn: Agent) -> None:
        pass

    async def session_update(self, session_id: str, update: Any, **kw: Any) -> None:
        buf = self._buffers.setdefault(
            session_id, {"messages": [], "thoughts": [], "images": []}
        )
        kind = getattr(update, "session_update", "")
        if kind in ("agent_message_chunk", "agent_thought_chunk"):
            content = update.content
            text = self._text(content)
            if text:
                key = "messages" if kind == "agent_message_chunk" else "thoughts"
                # Streaming chunks arrive fragmented; concatenate into one string.
                if not buf[key]:
                    buf[key].append("")
                buf[key][-1] += text
            # Capture image content blocks emitted by the agent (screenshots,
            # generated charts, downloaded images returned inline). The ACP
            # spec allows ImageContentBlock in agent_message_chunk — see the
            # ContentBlock symmetry doc. Without this capture the images are
            # silently dropped and never reach Hermes.
            ctype = getattr(content, "type", None) or ""
            if ctype == "image" or (
                hasattr(content, "data")
                and hasattr(content, "mime_type")
                and not text
            ):
                data = getattr(content, "data", None)
                mime = getattr(content, "mime_type", None) or getattr(
                    content, "mimeType", None
                )
                if data and mime:
                    buf["images"].append({"data": data, "mime_type": mime})

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
# OrgConnection — one ACP subprocess per org
# ═══════════════════════════════════════════════════════════════════════════

class OrgConnection:
    """One cached ACP subprocess for an org. Lives for the server lifetime."""

    def __init__(self, org: str) -> None:
        self.org = org
        self.conn: Any = None          # ClientSideConnection (Agent interface)
        self.process: Any = None
        self.client = SubagentClient()
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
                    client_info={"name": "hermes", "version": "1.0"},
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

    async def new_session(self, model: str | None = None,
                           cwd: str | None = None) -> str:
        async with self._lock:
            await self.ensure()
            workdir = cwd or PUX_PROJECT_ROOT
            resp = await self.conn.new_session(cwd=workdir)
            sid = resp.session_id
            if model:
                await self._set_model(sid, model)
            return sid

    async def prompt(self, session_id: str, message: str,
                     images: list[ImageContentBlock] | None = None,
                     ) -> tuple[str, str, str, list[dict]]:
        """Send a prompt and collect the agent's response.

        Returns ``(text, thoughts, stop_reason, agent_images)`` where
        ``agent_images`` is a list of ``{"data": b64, "mime_type": str}`` dicts
        for any ImageContentBlock the agent emitted (screenshots, generated
        charts, downloaded images). Previously these were silently dropped —
        only text was captured. Now they flow back to the MCP caller so Hermes
        sees org-produced assets natively.
        """
        async with self._lock:
            await self.ensure()
            self.client.reset(session_id)
            blocks: list[Any] = [TextContentBlock(type="text", text=message)]
            if images:
                blocks.extend(images)
            resp = await self.conn.prompt(session_id, blocks)
            msgs = self.client.messages(session_id)
            text = "\n".join(msgs) if msgs else ""
            thoughts = "\n".join(self.client.thoughts(session_id))
            agent_images = list(self.client.images(session_id))
            return text, thoughts, resp.stop_reason, agent_images

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
# MCP Tools — subagent management (pure ACP)
# ═══════════════════════════════════════════════════════════════════════════

# Build org + model lists at load time so they're baked into tool descriptions.
# The LLM reads the description on connect (ListToolsRequest) and immediately
# knows what orgs/models exist — no separate discovery call needed. Same pattern
# as Pux's `task` tool (subagents.py: TASK_TOOL_DESCRIPTION.format(available_agents=...)).
_ORGS = sorted(discover_orgs())
_MODELS = available_model_ids()

_ORG_LINES = "\n".join(f"  - `{o}`" for o in _ORGS)
_MODEL_LINES = "\n".join(f"  - `{m}`" for m in _MODELS)

_NEW_SESSION_DESC = (
    "Start a new subagent session on an org.\n\n"
    "Available orgs:\n"
    f"{_ORG_LINES}\n\n"
    "Args:\n"
    "  org: One of the orgs listed above. Default: 'general'.\n"
    f"  model: Optional override. Available models:\n{_MODEL_LINES}\n"
    "  cwd: Optional working directory (absolute path). When set, the agent's\n"
    "    filesystem operations are relative to this directory. Default: the\n"
    "    sandbox project root.\n\n"
    "Returns the session_id. Use prompt() to delegate tasks."
)


@MCP.tool(description=_NEW_SESSION_DESC)
async def new_session(org: str = "general", model: str | None = None,
                      cwd: str | None = None) -> str:
    known = discover_orgs()
    if org not in known:
        return f"Error: unknown org '{org}'. Available: {', '.join(sorted(known))}"
    try:
        conn = await _get_org(org)
        sid = await conn.new_session(model=model, cwd=cwd)
        _session_org[sid] = org
        tag_parts = []
        if model:
            tag_parts.append(f"model: {model}")
        if cwd:
            tag_parts.append(f"cwd: {cwd}")
        tag = f" ({', '.join(tag_parts)})" if tag_parts else ""
        return (
            f"Session started.\n"
            f"  session: `{sid}`\n"
            f"  org: `{org}`{tag}\n"
            f"Use prompt() to send tasks."
        )
    except Exception as exc:
        return f"Error creating session on '{org}': {exc}"


@MCP.tool()
async def prompt(session_id: str, message: str,
                 images: list[dict] | None = None):
    """Send a message to a subagent session — delegate a task, ask a follow-up,
    or answer a question. Optionally attach images.

    After an end_turn, the next prompt() IS the resume answer (ACP-native:
    the ask_user interrupt persists in the checkpoint; your answer unblocks it).

    Args:
        session_id: From new_session() or load_session().
        message: Task, follow-up, or answer.
        images: Optional image attachments (ACP ImageContentBlock). Each item:
          ``{"data": "<base64-encoded>", "mime_type": "image/png"}``.
          The image is forwarded natively to the agent — the deepagents-acp
          adapter converts it to the model's multimodal format so the agent
          SEES the image. Requires an org whose base model is multimodal
          (the agent declares ``promptCapabilities.image`` at initialize).
          Use this for VISION tasks ("describe this", "what's on this screen").
          For FILE-UPLOAD tasks ("post this image to Twitter" via
          browser_upload), use stage_file() instead — a vision model that
          sees an image content block cannot extract its bytes to disk.

    Returns:
        Response text + stop reason. When the agent returns images (screenshots,
        generated charts, downloaded visuals), they are included as native MCP
        image content blocks AND persisted to ``data/staged/agent_output_*``
        so they survive across calls. end_turn = done or asking a question
        (the question is in the text).
    """
    oc = _find_org_for_session(session_id)
    if oc is None:
        return (
            f"Error: no active connection for session '{session_id}'.\n"
            f"Create one with new_session(org)."
        )
    image_blocks: list[ImageContentBlock] | None = None
    if images:
        image_blocks = []
        for img in images:
            data = img.get("data")
            mime = img.get("mime_type") or img.get("mimeType")
            if not data or not mime:
                return (
                    "Error: each image needs 'data' (base64) and "
                    "'mime_type' (e.g. 'image/png')."
                )
            image_blocks.append(
                ImageContentBlock(type="image", data=data, mime_type=mime)
            )
    try:
        text, thoughts, stop, agent_images = await oc.prompt(
            session_id, message, images=image_blocks,
        )
        parts = []
        if text:
            parts.append(text)
        else:
            parts.append("(no response)")
        parts.append(f"\n*[{stop}]*")
        if stop == "end_turn":
            parts.append(
                "\n(Done or asking a question. "
                "If it asked something, answer with another prompt() call.)"
            )
        elif stop == "cancelled":
            parts.append("\n(Task cancelled.)")

        # ── Asset passthrough ───────────────────────────────────────────
        # If the agent produced images (screenshots, generated charts, etc.),
        # persist them to staged/ AND return them as native MCP ImageContent
        # blocks so the orchestrator (Hermes) sees them inline — no manual
        # read_file() needed. The files are the durable copy; the MCP image
        # blocks are the zero-friction display path.
        if agent_images:
            _STAGED_HOST_DIR.mkdir(parents=True, exist_ok=True)
            import time
            ts = int(time.time())
            saved_paths = []
            for i, img in enumerate(agent_images):
                ext = "png" if "png" in img["mime_type"] else "jpg"
                fname = f"agent_output_{ts}_{i}.{ext}"
                try:
                    raw = base64.b64decode(img["data"])
                    (_STAGED_HOST_DIR / fname).write_bytes(raw)
                    saved_paths.append(f"{_STAGED_CONTAINER_DIR}/{fname}")
                except Exception:
                    pass  # persist is best-effort; inline display still works
            if saved_paths:
                parts.append("\n📸 Agent images saved:")
                for p in saved_paths:
                    parts.append(f"  `{p}`")
            # Return native MCP content blocks: text + each image inline.
            content: list = [MCPTextContent(type="text", text="\n".join(parts))]
            for img in agent_images:
                content.append(MCPImageContent(
                    type="image",
                    data=img["data"],
                    mimeType=img["mime_type"],
                ))
            return content
        return "\n".join(parts)
    except Exception as exc:
        return f"Error: {exc}"


@MCP.tool()
async def list_sessions(org: str | None = None) -> str:
    """List active subagent sessions.

    Args:
        org: Optional org filter. If omitted, lists across all orgs.

    Returns:
        Active sessions with org, title, and last update time.
    """
    orgs_to_check = [org] if org else list(_pool.keys())
    if not orgs_to_check:
        return "No subagent sessions. Use new_session(org) to start one."
    all_sessions: list[dict] = []
    for o in orgs_to_check:
        if o in _pool and _pool[o].alive:
            try:
                all_sessions.extend(await _pool[o].list_sessions_raw())
            except Exception:
                pass
    if not all_sessions:
        return "No active subagent sessions."
    lines = [f"**{len(all_sessions)} session(s):**"]
    for s in all_sessions:
        title = f" — {s['title']}" if s.get("title") else ""
        ts = f" ({s['updated_at']})" if s.get("updated_at") else ""
        lines.append(f"- `{s['session_id']}` org=`{s['org']}`{title}{ts}")
    return "\n".join(lines)


@MCP.tool()
async def load_session(session_id: str, org: str) -> str:
    """Resume a past subagent session.

    Args:
        session_id: The session to resume.
        org: The org the session belongs to.

    Returns:
        Confirmation. Use prompt() to continue the conversation.
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
    """Cancel an in-progress task on a subagent.

    Args:
        session_id: The subagent session to cancel.

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
async def reset_session(session_id: str) -> str:
    """Reset the sandbox backing a session — force-remove its container so the
    next prompt() recreates a fresh one. Use when the sandbox is stuck or broken
    (a tool is hanging, the container is wedged, files got corrupt).

    This is a RECOVERY primitive: it does NOT save the sandbox's current state
    (installed packages, Chrome profile) — the point is a clean slate. The
    session's conversation history is preserved (durable checkpoint); only the
    container is replaced. Any in-flight task on the session is cancelled first.
    Scoped to the session's org — it resets the one sandbox container that org
    runs in.

    Args:
        session_id: The session whose sandbox should be reset.

    Returns:
        Confirmation that the sandbox was reset (or was already absent).
    """
    oc = _find_org_for_session(session_id)
    if oc is None:
        return f"Error: no active connection for session '{session_id}'."
    org = oc.org
    # Cancel any in-flight task first so we don't force-remove the container out
    # from under a running tool call (best-effort; harmless if the session is
    # idle). The exec layer re-ensures on NotFound, so a tool mid-flight when
    # the reset lands auto-recovers onto the fresh container.
    try:
        await oc.cancel(session_id)
    except Exception:  # noqa: BLE001 — cancel must never block the reset
        pass
    try:
        from pux_harness.sandbox.container import SandboxContainer  # noqa: PLC0415

        SandboxContainer(org=org).reset()
        return (
            f"Sandbox reset for org `{org}` (session `{session_id}`).\n"
            f"A fresh container is recreated automatically on the next prompt()."
        )
    except Exception as exc:
        return f"Error resetting sandbox for '{org}': {exc}"


@MCP.tool()
async def reload_profiles(org: str | None = None) -> str:
    """Hot-reload agent profiles (``profile.yaml`` / ``profile.local.yaml``)
    by bouncing the cached ACP subprocess for an org — or every active org
    when ``org`` is omitted.

    Use this AFTER editing an org's profile so the NEXT ``new_session()``
    picks up the change. Without it, the harness reuses the already-running
    ACP subprocess (which holds the OLD profile in memory for the whole
    server lifetime), forcing a full ``pux`` server restart from a host
    terminal — the round-trip this tool eliminates.

    What it does NOT do:
      • It does NOT touch the sandbox container (installed packages, browser
        profile, running browser) — only the ACP agent process. Use
        ``reset_session`` for a stuck sandbox.
      • Existing live sessions on the bounced org are interrupted (their ACP
        process is gone); their conversation history is durable in the
        checkpointer, so ``load_session`` resurrects them on the fresh
        process with the NEW profile.

    Args:
        org: The org whose profiles should be reloaded. ``None`` (default)
        reloads ALL currently-active orgs. Unknown / inactive orgs are
        reported, not treated as errors.

    Returns:
        A per-org report of what was bounced + the instruction to call
        ``new_session`` (or ``load_session``) to use the refreshed profile.
    """
    if org is not None:
        targets = [org]
    else:
        targets = list(_pool.keys())

    bounced: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []
    for o in targets:
        oc = _pool.get(o)
        if oc is None:
            skipped.append(f"{o} (not active)")
            continue
        try:
            await oc.stop()  # kills the cached ACP subprocess
            bounced.append(o)
        except Exception as exc:  # noqa: BLE001 — one org's failure must not abort the rest
            errors.append(f"{o}: {exc}")

    parts = []
    if bounced:
        parts.append(
            "Reloaded profiles for: " + ", ".join(bounced) + ". "
            "The next new_session() (or load_session()) on these orgs spawns a "
            "fresh ACP process that re-reads profile.yaml + profile.local.yaml."
        )
    if skipped:
        parts.append("Skipped (no active subprocess): " + "; ".join(skipped))
    if errors:
        parts.append("Errors: " + "; ".join(errors))
    if not parts:
        parts.append("No active orgs to reload.")
    return " ".join(parts)


@MCP.tool()
async def set_model(session_id: str, model: str) -> str:
    """Change the model on a session. See new_session for available models."""
    oc = _find_org_for_session(session_id)
    if oc is None:
        return f"Error: no active connection for session '{session_id}'."
    try:
        await oc.set_model_raw(session_id, model)
        return f"Model set to `{model}` on `{session_id}`."
    except Exception as exc:
        return f"Error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# File staging — for UPLOAD tasks (complementary to ACP ImageContentBlock)
# ═══════════════════════════════════════════════════════════════════════════
#
# prompt(images=...) uses ACP ImageContentBlock — the agent SEES the image
# (vision). But for UPLOAD tasks ("post this image to Twitter" via
# browser_upload), the agent needs the image as a FILE it can pass to
# <input type="file">. A vision model that sees an image block cannot extract
# its bytes to disk. stage_file() bridges that gap: write the bytes here
# (host-side), the agent reads them at the container path (workspace bind mount).

_STAGED_HOST_DIR = Path(PUX_PROJECT_ROOT) / "data" / "staged"
_STAGED_CONTAINER_DIR = "/sandbox/workspace/data/staged"
_SAFE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
_MAX_STAGE_BYTES = 50 * 1024 * 1024  # 50 MB


@MCP.tool()
async def stage_file(filename: str, content_b64: str) -> str:
    """Stage a file (base64-encoded) for an agent to upload or process.

    Use this when the agent needs the file BYTES on disk — e.g. posting an
    image to Twitter (browser_upload needs a file path, not a vision block).
    For VISION tasks (the agent just needs to SEE an image), use
    prompt(images=...) instead — that's the native ACP path with no disk I/O.

    The project root is bind-mounted into the sandbox container at
    /sandbox/workspace, so a file written here is instantly visible at the
    returned container path. data/staged/ is gitignored (runtime, not source).

    Args:
        filename: Filename only (no path separators). Alphanumeric + ._-
        content_b64: Base64-encoded file content.

    Returns:
        Container-absolute path the agent uses in browser_upload, etc.
    """
    if not _SAFE_NAME.match(filename):
        return (
            f"Error: unsafe filename {filename!r}. "
            "Use alphanumeric + dot/underscore/hyphen only."
        )
    try:
        raw = base64.b64decode(content_b64, validate=True)
    except Exception as exc:
        return f"Error: invalid base64 ({exc})"
    if not raw:
        return "Error: decoded content is empty"
    if len(raw) > _MAX_STAGE_BYTES:
        return (
            f"Error: file too large ({len(raw):,} bytes; "
            f"cap {_MAX_STAGE_BYTES // (1024 * 1024)} MB)"
        )
    _STAGED_HOST_DIR.mkdir(parents=True, exist_ok=True)
    (_STAGED_HOST_DIR / filename).write_bytes(raw)
    container_path = f"{_STAGED_CONTAINER_DIR}/{filename}"
    return (
        f"Staged {len(raw):,} bytes.\n"
        f"  container_path: `{container_path}`\n"
        f"  Pass this path in prompt() so the agent knows where to find the file."
    )


# ═══════════════════════════════════════════════════════════════════════════
# File retrieval — read back files agents saved to the workspace
# ═══════════════════════════════════════════════════════════════════════════
#
# stage_file() is write-only (host → container). But agents also WRITE files:
# deep-research-engine downloads images to data/images/, browser-agent saves
# screenshots, etc. The workspace bind mount means those files exist on the
# host too — read_file() lets Hermes pull them back to show the user, forward
# to another org, or re-stage for a different agent.
#
# Scoped to data/ (the runtime area, gitignored). No source code, no configs.

_DATA_HOST_DIR = Path(PUX_PROJECT_ROOT) / "data"


@MCP.tool()
async def read_file(path: str) -> str:
    """Read a file from the workspace data/ directory, returning base64 content.

    Use this to retrieve files that an agent saved during a task — e.g.
    images downloaded by deep-research-engine, screenshots captured by
    browser-agent, transcripts generated by media analysis. The content
    comes back as base64 so you can display it, forward it, or re-stage it
    for another agent via stage_file().

    Only files under data/ are readable (runtime area — images, downloads,
    session state). Source code, configs, and everything outside data/ is
    off-limits.

    Args:
        path: Relative path within data/ (e.g. "images/news1.jpg",
              "staged/tweet.jpg"). No .. or absolute paths.

    Returns:
        Base64-encoded content + byte size + detected mime type, or an error.
    """
    # Reject path traversal and absolute paths.
    if path.startswith("/") or ".." in Path(path).parts:
        return (
            f"Error: unsafe path {path!r}. "
            "Use a relative path within data/ (no .. or leading /)."
        )
    target = (_DATA_HOST_DIR / path).resolve()
    try:
        target.relative_to(_DATA_HOST_DIR.resolve())
    except ValueError:
        return f"Error: path {path!r} escapes data/"
    if not target.is_file():
        return f"Error: not found — {path!r} (resolved: {target})"
    if target.is_symlink():
        return f"Error: symlinks are not readable (security)"
    raw = target.read_bytes()
    if len(raw) > _MAX_STAGE_BYTES:
        return (
            f"Error: file too large ({len(raw):,} bytes; "
            f"cap {_MAX_STAGE_BYTES // (1024 * 1024)} MB)"
        )
    b64 = base64.b64encode(raw).decode("ascii")
    # Lightweight mime sniff from extension.
    ext = target.suffix.lower()
    _MIME = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml",
        ".pdf": "application/pdf", ".mp4": "video/mp4", ".mp3": "audio/mpeg",
        ".wav": "audio/wav", ".json": "application/json",
        ".txt": "text/plain", ".csv": "text/csv",
    }
    mime = _MIME.get(ext, "application/octet-stream")
    return (
        f"Read {len(raw):,} bytes ({mime}).\n"
        f"  path: data/{path}\n"
        f"  base64:\n{b64}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    import uvicorn  # noqa: PLC0415

    host = os.environ.get("PUX_MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("PUX_MCP_PORT", "9987"))
    sys.stderr.write(f"[pux] MCP-SSE on {host}:{port}\n")
    sys.stderr.write(f"[pux] ACP subprocess root: {PUX_PROJECT_ROOT}\n")
    app = MCP.sse_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
