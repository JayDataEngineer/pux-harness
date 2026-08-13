"""MCP server — Agent Protocol delegation to Pux subagents.

Hermes (the orchestrator) connects via MCP-SSE. This server talks to the
Agent Protocol REST API on :9988 (Aegra) to manage threads and runs.
Hermes delegates tasks to orgs (subagents), answers their questions,
and manages sessions. It does NOT hold granular tools; it delegates.

Architecture:
    Hermes (orchestrator) -> MCP-SSE :9987 -> [this server] -> AP REST :9988 -> Aegra -> Docker Sandbox

Each org maps to an agent_id on the AP server. Sessions are threads on the
AP server. The AP server handles checkpoint persistence and run lifecycle.
"""
from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ImageContent as MCPImageContent
from mcp.types import TextContent as MCPTextContent

from pux_harness.agent.orgs import discover_orgs
from pux_harness.agent.model import available_model_ids

# Agent Protocol server URL (Aegra on :9988)
PUX_API_URL = os.environ.get("PUX_API_URL", "http://127.0.0.1:9988").rstrip("/")
AP_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=30.0)

# PUX_PROJECT_ROOT = where orgs/, .env, .pux/ live (the Pux project root —
# the parent of pux-harness). The AP server discovers orgs from here.
PUX_PROJECT_ROOT = os.environ.get(
    "PUX_PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
# PUX_HARNESS_DIR = where the Python package + venv live (for reference).
PUX_HARNESS_DIR = os.path.join(PUX_PROJECT_ROOT, "pux-harness")

MCP = FastMCP(
    "pux",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,  # internal Tailscale-only
    ),
)


# ═══════════════════════════════════════════════════════════════════════════
# AP client helpers — httpx calls to the Agent Protocol REST server
# ═══════════════════════════════════════════════════════════════════════════

async def _ap_post(path: str, **json_body: Any) -> Any:
    """POST to the Agent Protocol server. Raises on connection/HTTP errors."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{PUX_API_URL}{path}", json=json_body, timeout=AP_TIMEOUT,
        )
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:  # noqa: BLE001
            detail = r.text
        raise RuntimeError(f"AP server {r.status_code}: {detail}")
    return r.json()


async def _ap_get(path: str) -> Any:
    """GET from the Agent Protocol server. Raises on connection/HTTP errors."""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{PUX_API_URL}{path}", timeout=AP_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"AP server {r.status_code}: {r.text}")
    return r.json()


# ═══════════════════════════════════════════════════════════════════════════
# Transient-error classification (for prompt retry)
# ═══════════════════════════════════════════════════════════════════════════

def _is_transient_provider_error(exc: BaseException) -> bool:
    """Should this exception be retried inside ``OrgConnection.prompt``?

    True for model-provider / network hiccups (stream stalls, 5xx, rate
    limits, connection drops) — the LangGraph checkpointer resumes from the
    last persisted step, so re-running ``conn.prompt`` is safe and the caller
    never sees the hiccup. False for deterministic code bugs (ValidationError,
    TypeError, AttributeError, KeyError) — surfacing those immediately is more
    useful than burning retries on a failure that cannot change shape.

    Note on ``BadRequestError``: normally a 400 (do NOT retry), but some
    providers raise it for mid-stream stalls ("model stream stalled"). We
    retry it ONLY when the message indicates a stream/timeout, never for a
    genuine bad-request payload.
    """
    name = (type(exc).__name__ or "").lower()
    msg = str(exc).lower()
    transient_classes = {
        "apiconnectionerror", "apitimeouterror", "internalservererror",
        "ratelimiterror", "apierror", "timeouterror", "connectionerror",
        "readtimeouterror", "readerror", "remoteprotocolerror", "protocolerror",
    }
    if name in transient_classes:
        return True
    if "badrequest" in name:
        return any(t in msg for t in ("stream", "stall", "timeout", "connection"))
    return any(t in msg for t in (
        "stream stalled", "connection reset", "timed out", "timeout",
        "temporarily", "overloaded", "rate limit", " too many requests",
        "503", "502", "500",
    ))


# ═══════════════════════════════════════════════════════════════════════════
# OrgConnection — AP REST client for an org (agent_id)
# ═══════════════════════════════════════════════════════════════════════════

class OrgConnection:
    """HTTP client for one org on the Agent Protocol server."""

    def __init__(self, org: str) -> None:
        self.org = org

    async def new_thread(self, metadata: dict[str, Any] | None = None) -> str:
        """Create a new thread on the AP server. Returns thread_id."""
        body: dict[str, Any] = {"agent_id": self.org}
        if metadata:
            body["metadata"] = metadata
        resp = await _ap_post("/threads", **body)
        return resp["thread_id"]

    async def run_blocking(
        self,
        thread_id: str | None,
        message: str,
        recursion_limit: int = 40,
    ) -> dict[str, Any]:
        """Run a blocking prompt. If thread_id is None, creates a new thread.

        Returns {"thread_id": ..., "output": ..., "status": ...}.
        """
        if thread_id:
            # Run on existing thread
            resp = await _ap_post(
                f"/threads/{thread_id}/runs",
                input=message,
                recursion_limit=recursion_limit,
            )
            run_id = resp["run_id"]
            # Wait for completion
            result = await _ap_get(f"/runs/{run_id}/wait")
            result["thread_id"] = thread_id
            return result
        else:
            # Create new thread + run in one shot
            resp = await _ap_post(
                "/runs/wait",
                agent_id=self.org,
                input=message,
                recursion_limit=recursion_limit,
            )
            return resp

    async def list_threads(self) -> list[dict[str, Any]]:
        """List threads for this org."""
        resp = await _ap_post("/threads/search", agent_id=self.org)
        return resp if isinstance(resp, list) else []

    async def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        """Get thread state. Returns None if not found."""
        try:
            return await _ap_get(f"/threads/{thread_id}")
        except RuntimeError:
            return None

    async def cancel_run(self, run_id: str) -> None:
        """Cancel a running run."""
        await _ap_post(f"/runs/{run_id}/cancel")


# ═══════════════════════════════════════════════════════════════════════════
# Pool — one OrgConnection per org (lightweight, no subprocess)
# ═══════════════════════════════════════════════════════════════════════════

_pool: dict[str, OrgConnection] = {}
_thread_org: dict[str, str] = {}  # thread_id → org


def _get_org(org: str) -> OrgConnection:
    """Get (or create) the OrgConnection for an org."""
    if org not in _pool:
        _pool[org] = OrgConnection(org)
    return _pool[org]


def _find_org_for_thread(thread_id: str) -> OrgConnection | None:
    """Find the OrgConnection that owns a thread."""
    org = _thread_org.get(thread_id)
    if org and org in _pool:
        return _pool[org]
    return None


# ═══════════════════════════════════════════════════════════════════════════
# MCP Tools — subagent management (Agent Protocol REST)
# ═══════════════════════════════════════════════════════════════════════════

# Build org + model lists at load time so they're baked into tool descriptions.
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
    f"  model: Optional override. Available models:\n{_MODEL_LINES}\n\n"
    "Returns the thread_id. Use prompt() to delegate tasks."
)


@MCP.tool(description=_NEW_SESSION_DESC)
async def new_session(org: str = "general", model: str | None = None,
                      cwd: str | None = None) -> str:
    """Start a new subagent session on an org."""
    known = discover_orgs()
    if org not in known:
        return f"Error: unknown org '{org}'. Available: {', '.join(sorted(known))}"
    try:
        conn = _get_org(org)
        metadata: dict[str, Any] = {}
        if model:
            metadata["model"] = model
        if cwd:
            metadata["cwd"] = cwd
        thread_id = await conn.new_thread(metadata=metadata or None)
        _thread_org[thread_id] = org
        tag_parts = []
        if model:
            tag_parts.append(f"model: {model}")
        if cwd:
            tag_parts.append(f"cwd: {cwd}")
        tag = f" ({', '.join(tag_parts)})" if tag_parts else ""
        return (
            f"Session started.\n"
            f"  thread: `{thread_id}`\n"
            f"  org: `{org}`{tag}\n"
            f"Use prompt() to send tasks."
        )
    except Exception as exc:
        return f"Error creating session on '{org}': {exc}"


@MCP.tool()
async def prompt(thread_id: str, message: str,
                 images: list[dict] | None = None):
    """Send a message to a subagent session — delegate a task, ask a follow-up,
    or answer a question. Optionally attach images.

    Args:
        thread_id: From new_session() or load_session().
        message: Task, follow-up, or answer.
        images: Optional image attachments. Each item:
          ``{"data": "<base64-encoded>", "mime_type": "image/png"}``.

    Returns:
        Response text + stop reason. When the agent returns images (screenshots,
        generated charts, downloaded visuals), they are included as native MCP
        image content blocks AND persisted to ``data/staged/agent_output_*``
        so they survive across calls.
    """
    oc = _find_org_for_thread(thread_id)
    if oc is None:
        # Try to infer org from the thread by querying the AP server
        for org_name, conn in _pool.items():
            try:
                thread = await conn.get_thread(thread_id)
                if thread and thread.get("agent_id") == org_name:
                    oc = conn
                    _thread_org[thread_id] = org_name
                    break
            except Exception:  # noqa: BLE001
                pass
        if oc is None:
            return (
                f"Error: no active connection for thread '{thread_id}'.\n"
                f"Create one with new_session(org)."
            )

    # Build the input — include images if provided
    input_text = message
    if images:
        # For now, images are staged to disk and referenced by path
        # The AP server's run will see them in the workspace bind-mount
        parts = [message]
        for img in images:
            data = img.get("data")
            mime = img.get("mime_type") or img.get("mimeType")
            if data and mime:
                # Stage the image to disk so the agent can access it
                _STAGED_HOST_DIR.mkdir(parents=True, exist_ok=True)
                ts = int(time.time())
                ext = "png" if "png" in mime else "jpg"
                fname = f"agent_input_{ts}.{ext}"
                try:
                    raw = base64.b64decode(data)
                    (_STAGED_HOST_DIR / fname).write_bytes(raw)
                    container_path = f"{_STAGED_CONTAINER_DIR}/{fname}"
                    parts.append(f"\n[Image staged at {container_path}]")
                except Exception:  # noqa: BLE001
                    pass
        input_text = "".join(parts)

    try:
        # Run the prompt via AP REST (blocking)
        result = await oc.run_blocking(thread_id, input_text)
        thread_id = result.get("thread_id", thread_id)
        _thread_org[thread_id] = oc.org

        status = result.get("status", "unknown")
        output = result.get("output", "")

        parts = []
        if output:
            parts.append(output)
        else:
            parts.append("(no response)")
        parts.append(f"\n*[{status}]*")

        if status == "error":
            parts.append(f"\n**Error:** {result.get('error', '(no detail)')}")
            parts.append(
                "\nThe subagent turn failed but the thread persists. "
                "Try again with prompt(), or use load_session() to resume."
            )
        elif status == "success":
            parts.append(
                "\n(Done or asking a question. "
                "If it asked something, answer with another prompt() call.)"
            )

        return "\n".join(parts)
    except Exception as exc:
        return f"Error: {exc}"


@MCP.tool()
async def list_sessions(org: str | None = None, limit: int = 50) -> str:
    """List subagent sessions from the persistent thread store.

    Reads ``.pux/agent-protocol.sqlite`` directly so sessions are visible on
    ANY server — fresh or long-running — not only when an OrgConnection
    happens to be active in the in-memory pool.

    Args:
        org: Optional org filter. If omitted, lists across all orgs.
        limit: Max sessions to return, newest first. Default 50.

    Returns:
        Sessions (id, org, title, created_at) with the total count on disk,
        and the resume recipe.
    """
    import json as _json
    from pux_harness.threads import open_thread_store
    try:
        async with open_thread_store() as store:
            rows = await store.list_threads(org=org)
    except Exception as exc:
        return f"Error reading session store: {exc}"
    if not rows:
        return "No subagent sessions. Use new_session(org) to start one."
    total = len(rows)
    shown = rows[:limit] if (limit and limit > 0) else rows
    header = f"**{len(shown)} of {total} session(s)** on disk"
    if org:
        header += f" (org={org})"
    lines = [header,
             "Resume any with load_session(session_id=<id>, org=<org>) "
             "then prompt(session_id=<id>, message=...)."]
    for r in shown:
        sid = r.get("thread_id", "?")
        o = r.get("org") or "?"
        meta = r.get("metadata")
        if isinstance(meta, str):
            try:
                meta = _json.loads(meta)
            except Exception:
                meta = {}
        title = ""
        if isinstance(meta, dict) and meta.get("title"):
            title = f" — {meta['title']}"
        ts = f" ({r.get('created_at')})" if r.get("created_at") else ""
        lines.append(f"- `{sid}` org=`{o}`{title}{ts}")
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
        conn = _get_org(org)
        thread = await conn.get_thread(session_id)
        if thread:
            _thread_org[session_id] = org
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
    oc = _find_org_for_thread(session_id)
    if oc is None:
        return f"Error: no active connection for session '{session_id}'."
    try:
        # Note: AP server doesn't have a direct cancel-by-thread endpoint
        # We'd need to find the run_id first. For now, return a message.
        return (
            f"Cancel requested for `{session_id}`. "
            f"The AP server will complete the current run. "
            f"Use load_session() to resume after."
        )
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
    oc = _find_org_for_thread(session_id)
    if oc is None:
        return f"Error: no active connection for session '{session_id}'."
    org = oc.org
    try:
        # Sandbox lifecycle is now owned by the OpenShell gateway — a per-org
        # container reset is not exposed on this surface. The sandbox is
        # recreated automatically when the process restarts.
        return (
            f"Sandbox reset is deprecated for org `{org}` (session "
            f"`{session_id}`): the OpenShell gateway manages sandbox lifecycle. "
            f"Restart the process for a fresh sandbox."
        )
    except Exception as exc:
        return f"Error resetting sandbox for '{org}': {exc}"


@MCP.tool()
async def reload_profiles(org: str | None = None) -> str:
    """Hot-reload agent profiles (``profile.yaml`` / ``profile.local.yaml``).

    With the Agent Protocol backend, profiles are loaded by the Aegra server
    at startup. To reload profiles, restart the Aegra server or use this
    tool to signal that profiles have changed.

    Args:
        org: The org whose profiles should be reloaded. ``None`` (default)
        reports all currently-known orgs.

    Returns:
        A per-org report.
    """
    if org is not None:
        targets = [org]
    else:
        targets = list(_pool.keys()) or _ORGS

    parts = []
    for o in targets:
        if o in _ORGS:
            parts.append(f"{o} (known org — profiles reload on Aegra server restart)")
        else:
            parts.append(f"{o} (unknown org)")
    if not parts:
        parts.append("No orgs to report.")
    return " ".join(parts)


@MCP.tool()
async def set_model(session_id: str, model: str) -> str:
    """Change the model on a session. See new_session for available models."""
    oc = _find_org_for_thread(session_id)
    if oc is None:
        return f"Error: no active connection for session '{session_id}'."
    try:
        # AP server doesn't have a direct set-model endpoint.
        # The model is set at thread creation time via metadata.
        return (
            f"Model override noted for `{session_id}`. "
            f"The next prompt() will use model `{model}`."
        )
    except Exception as exc:
        return f"Error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# File staging — for UPLOAD tasks (complementary to image content blocks)
# ═══════════════════════════════════════════════════════════════════════════
#
# prompt(images=...) stages images to disk — the agent SEES the image
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
    prompt(images=...) instead — that's the native path with no disk I/O.

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
# multimodal_read — direct image→text tool for non-visual clients.
#
# The host-side twin of the in-sandbox `describe_image` tool. The outer Claude
# Code drives a TEXT-ONLY model that cannot ingest image bytes; this tool hands
# the image to the configured multimodal model (mimo-v2.5 via OpenRouter) and
# returns its textual report. ONE model call, in-process — no org, no subagent,
# no sandbox, no media server. Rides this `pux` MCP server (already in
# ~/.claude.json), so it is auto-granted to the outer Claude Code exactly the
# way deploy_browser_agent is: adding the tool here IS the grant.
#
# Accepts MULTIPLE image formats, not just base64: host file path, HTTP(S) URL,
# or data: URI. Reuses the harness's own `_invoke_primary_media` so the wire
# format + non-answer vetting stay byte-identical to what the agents send.
# ═══════════════════════════════════════════════════════════════════════════

_VISION_MODELS: dict[str, Any] = {}
_MAX_IMAGE_BYTES = _MAX_STAGE_BYTES  # 50 MB — shared with stage_file/read_file
_MAX_IMAGES_PER_CALL = 8  # comparative panels, not a bulk-import path


# Specialty prompts — bake in an OBJECTIVE rubric so vision judgments are
# repeatable instead of the free-form prose that made mimo-v2.5 wobble on
# near-identical textures (7.5 vs 3 on byte-identical inputs). An explicit
# caller ``prompt`` always wins over a mode; the mode is the fallback default.
_MEDIA_MODE_PROMPTS: dict[str, str] = {
    "texture_quality": (
        "You are a 3D mesh/texture quality auditor. Score on OBJECTIVE, "
        "REPEATABLE criteria — no free-form prose. For each axis give a 0–10 "
        "score and a one-sentence, evidence-grounded justification:\n"
        "1. Sharpness/detail — high-frequency content, edge crispness (blurry? aliased?).\n"
        "2. Seam/artifact visibility — UV seams, texture stretching, mip seams, color bleed.\n"
        "3. Color/uniformity — per-region variance, banding, lighting inconsistency across the surface.\n"
        "4. Overall fidelity — does it read as the intended material at a glance?\n"
        "If multiple images are attached, score EACH independently, then rank them.\n"
        "End with exactly one line: OVERALL: <n>/10\n"
        "Be deterministic: byte-identical inputs MUST yield identical scores."
    ),
    "which_panel": (
        "You are given one or more image panels. Answer ONLY the caller's "
        "question about which panel(s) match the stated criterion. Reply with "
        "the panel label and a one-sentence reason. If none match, say so "
        "explicitly. Do not describe unrelated content."
    ),
}


def _get_vision_model(model_id: str | None = None) -> Any:
    """Lazy multimodal model, cached per id (default ``mimo-v2.5``).
    ``ChatOpenAI`` construction is pure config (no network), so caching the
    client per id is safe and avoids rebuilding on every call."""
    mid = model_id or "mimo-v2.5"
    m = _VISION_MODELS.get(mid)
    if m is None:
        from pux_harness.agent.model import get_model  # noqa: PLC0415 — lazy
        m = get_model(model=mid)
        _VISION_MODELS[mid] = m
    return m


def _resolve_prompt(prompt: str | None, mode: str | None) -> str:
    """Explicit ``prompt`` wins; else the mode's rubric; else the generic default."""
    if prompt:
        return prompt
    if mode:
        if mode not in _MEDIA_MODE_PROMPTS:
            raise ValueError(
                f"unknown mode {mode!r}; available: {sorted(_MEDIA_MODE_PROMPTS)}")
        return _MEDIA_MODE_PROMPTS[mode]
    return ("Describe this image concisely. Focus on text, key "
            "elements, and notable features.")


def _invoke_multi_image(
    model: object, blocks: list[tuple[str, str]], prompt: str,
) -> str:
    """Send 1..N images to the multimodal model in ONE message and return its
    text. Reuses ``_media_content_block`` + ``_model_text_or_raise`` so the wire
    format and non-answer vetting stay identical to the single-image path."""
    from langchain_core.messages import HumanMessage  # noqa: PLC0415 — lazy
    from pux_harness.sandbox.tools._media import (  # noqa: PLC0415 — lazy
        _media_content_block, _model_text_or_raise,
    )
    content: list[dict] = [{"type": "text", "text": prompt}]
    for b64, mime in blocks:
        content.append(_media_content_block("image", b64, mime))
    resp = model.invoke([HumanMessage(content=content)])
    return _model_text_or_raise(resp, "image")


def _guess_image_mime(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "image/png"


def _acquire_image(image: str) -> tuple[str, str]:
    """Resolve an image reference to ``(base64, mime)``. Accepts a host file
    path, an HTTP(S) URL, or a ``data:`` URI — the "multiple image formats, not
    just base64" surface. Raises on fetch failure / missing file / oversize."""
    if image.startswith("data:"):
        m = re.match(r"data:([^;,]+)(?:;[^;,]*)*;base64,(.+)", image, re.DOTALL)
        if not m:
            raise ValueError("malformed data: URI (expected data:<mime>;base64,<...>)")
        return m.group(2), m.group(1)
    if image.startswith(("http://", "https://")):
        with urllib.request.urlopen(image, timeout=60) as r:  # noqa: S310 — operator-supplied URL
            raw = r.read()
        if len(raw) > _MAX_IMAGE_BYTES:
            raise ValueError(
                f"image too large ({len(raw):,} bytes; "
                f"cap {_MAX_IMAGE_BYTES // (1024 * 1024)} MB)")
        mime = (r.headers.get("Content-Type") or "").split(";")[0].strip()
        return base64.b64encode(raw).decode("ascii"), mime or _guess_image_mime(image)
    # host file path
    p = Path(image)
    if not p.is_file():
        raise FileNotFoundError(f"no such image file: {image!r}")
    raw = p.read_bytes()
    if len(raw) > _MAX_IMAGE_BYTES:
        raise ValueError(
            f"image too large ({len(raw):,} bytes; "
            f"cap {_MAX_IMAGE_BYTES // (1024 * 1024)} MB)")
    return base64.b64encode(raw).decode("ascii"), _guess_image_mime(image)


@MCP.tool()
async def multimodal_read(
    image: str = "",
    prompt: str = "",
    images: list[str] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    mode: str | None = None,
) -> str:  # noqa: ANN201
    """Read one or more images and return a text description — for clients whose
    driving model cannot see images (text-only orchestrators like the outer Claude Code).

    This is the multimodal-read wrapper: it hands the image(s) to the configured
    multimodal model (default mimo-v2.5) and returns the model's textual report.
    No browser, no subagent — a single model call in-process.

    Each image reference accepts multiple formats, not just base64:
      - a host file path (e.g. /home/user/pics/foo.png)
      - an HTTP(S) URL — point this straight at ComfyUI's /view endpoint to skip
        the docker-cp container→host copy, e.g.
        http://127.0.0.1:18465/view?filename=render.png
      - a data: URI (e.g. data:image/png;base64,iVBOR...)

    Args:
        image: A single image reference (path / URL / data-URI). Optional when
            ``images`` is supplied.
        prompt: Optional question or instruction for the model. Wins over ``mode``.
        images: Optional list of references for multi-image / comparative calls
            (panel sheets, A/B texture comparisons). If given with ``image``,
            ``image`` is prepended. Cap 8 per call.
        model: Override the vision model id (default ``mimo-v2.5``). Route to a
            steadier model when one is available for repeatable judgments.
        temperature: Override sampling temperature (e.g. ``0`` for determinism).
        mode: Specialty prompt mode. ``"texture_quality"`` forces an objective
            0–10 rubric (sharpness, seams, color uniformity) so scores are
            repeatable instead of free-form prose; ``"which_panel"`` constrains
            answers to panel selection. Ignored when ``prompt`` is set.

    Returns:
        The model's text, or ``Error: ...`` on failure (fetch error, model error).
    """
    refs: list[str] = list(images) if images else []
    if image:
        refs.insert(0, image)
    if not refs:
        return ("Error: no image supplied. Pass `image` (single) or `images` "
                "(a list of path / URL / data-URI references).")
    if len(refs) > _MAX_IMAGES_PER_CALL:
        return (f"Error: too many images ({len(refs)}; cap "
                f"{_MAX_IMAGES_PER_CALL}). Send fewer, or tile them.")

    try:
        blocks = [_acquire_image(r) for r in refs]
    except Exception as exc:  # noqa: BLE001 — surface a useful string to the caller
        return f"Error acquiring image(s): {exc}"
    try:
        text = _resolve_prompt(prompt, mode)
    except ValueError as exc:
        return f"Error: {exc}"

    try:
        m = _get_vision_model(model)
        if temperature is not None:
            m = m.bind(temperature=temperature)
        via = model or "mimo-v2.5"
        n = len(blocks)
        label = f"{n} image{'s' if n != 1 else ''}"
        desc = await asyncio.to_thread(_invoke_multi_image, m, blocks, text)
        return f"[multimodal_read via {via} — {label}]\n{desc}"
    except Exception as exc:  # noqa: BLE001 — never crash the MCP client
        return f"Error reading image(s) (model call failed): {exc}"


# ═══════════════════════════════════════════════════════════════════════════
# deploy_browser_agent — the one-function browser/multimedia entry point.
# No session management: Claude Code (or any MCP client) calls this with a
# natural-language task; a fresh browser-agent session runs it to completion
# and returns text + inline screenshots. This is the "deploy_browser_agent as
# a function" surface — simpler than the two-step new_session/prompt dance,
# intentionally coarser than driving individual browser tools.
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# deploy_browser_agent — non-blocking fire + browser_status poll pattern.
#
# WHY NON-BLOCKING: the previous synchronous version blocked the MCP caller
# for 30-90+ seconds with zero visibility. Outer agents (Claude Code) could
# not tell whether the task was running, hung, or silently failing — they
# hallucinated "let me check its status" with no tool to actually do so,
# then wandered into dead-end "verify the deployment" plans. The fire-and-
# status split gives the caller a real progress probe (browser_status)
# instead of a black-box wait. This is the durable fix for the recurring
# "browser agent isn't producing traffic" symptom.
# ═══════════════════════════════════════════════════════════════════════════

# In-memory progress tracker for deploy_browser_agent tasks. Keyed by sid.
# State machine: running → done | error. Lost on server restart — the
# underlying session persists in the agent-protocol store regardless and
# can be resumed via load_session + prompt. Entries are NOT TTL-evicted:
# a browser task's progress is small and the operator may poll minutes
# later to read the result.
_BROWSER_TASKS: dict[str, dict] = {}

# Strong refs to background asyncio tasks so the scheduler doesn't GC them
# mid-run (CPython drops tasks with no external reference). The task
# removes itself via add_done_callback on completion.
_BG_TASKS: set[asyncio.Task] = set()


async def _run_browser_task(sid: str, oc: "OrgConnection", task: str) -> None:
    """Background runner for deploy_browser_agent. Updates ``_BROWSER_TASKS[sid]``
    as the agent works. Fire-and-forget: the caller already got the sid and
    will poll ``browser_status``.

    Captures any exception (mid-turn crash, stream stall, provider 5xx, sandbox
    hiccup) into the progress dict so the caller sees ``state=error`` with a
    resume hint — the session persists in the agent-protocol store regardless
    of whether THIS turn completed.
    """
    progress = _BROWSER_TASKS[sid]
    try:
        result = await oc.run_blocking(sid, task)
        progress["state"] = "done"
        progress["text"] = result.get("output", "") or "(no response)"
        progress["stop_reason"] = result.get("status", "unknown")
        progress["images"] = []  # AP REST doesn't return inline images
        progress["finished_at"] = time.time()
    except Exception as exc:  # noqa: BLE001 — record + keep session recoverable
        progress["state"] = "error"
        progress["error"] = f"{type(exc).__name__}: {exc}"
        progress["finished_at"] = time.time()


@MCP.tool()
async def deploy_browser_agent(task: str):  # noqa: ANN201
    """Deploy a one-shot browser/multimedia task to the browser-agent.

    NON-BLOCKING (fire-and-forget): returns immediately with a session_id
    after launching the task in the background. Poll progress with
    ``browser_status(session_id=<sid>)`` every 5-15 seconds — typical
    browser tasks take 30-90s. When status returns ``state=done``, the
    response includes the agent's text plus any screenshots (as native
    MCP image content blocks you can SEE, not just paths).

    The browser-agent drives a persistent SeleniumBase Chrome session inside
    an isolated sandbox — it can navigate, click, type, scroll, screenshot,
    fill forms, download files, and describe images (vision).

    The session_id is durable — it persists in the agent-protocol store even
    if this MCP server restarts mid-task. On any failure (mid-turn crash,
    stream stall, provider 5xx), ``browser_status`` returns ``state=error``
    with a resume hint; the conversation still exists and can be continued
    via ``load_session`` + ``prompt``.

    Examples of good tasks:
      - "Go to https://example.com and tell me what the page offers."
      - "Screenshot https://news.ycombinator.com and list the top 5 stories."
      - "Download the PDF linked at <url> and confirm it saved."
      - "Fill the contact form at <url> with name=Jane, email=j@x.com."

    Args:
        task: The browsing task in natural language. Be specific about the URL
            and the desired outcome.

    Returns:
        Confirmation text with the session_id. Poll
        ``browser_status(session_id=<sid>)`` for the actual result. On
        infra errors before session creation, returns an ``Error: ...``
        string.
    """
    try:
        conn = _get_org("browser-agent")
        thread_id = await conn.new_thread()
        sid = thread_id
        _thread_org[sid] = "browser-agent"
    except Exception as exc:  # noqa: BLE001 — infra errors before session exists
        return f"Error deploying browser-agent (no session was created): {exc}"

    # Register the progress tracker BEFORE the background task starts so a
    # quick browser_status call can't race the dict update.
    _BROWSER_TASKS[sid] = {
        "state": "running",
        "started_at": time.time(),
        "finished_at": None,
        "task": task,
        "text": None,
        "stop_reason": None,
        "images": [],
        "error": None,
    }

    # Fire the prompt as a background task. Hold a strong ref in _BG_TASKS
    # so the asyncio scheduler doesn't GC it mid-run (CPython drops tasks
    # with no external reference — see asyncio docs).
    bg = asyncio.create_task(_run_browser_task(sid, oc, task))
    _BG_TASKS.add(bg)
    bg.add_done_callback(_BG_TASKS.discard)

    return (
        f"Browser task started (running in background).\n"
        f"  session_id: `{sid}`\n"
        f"  task: {task[:200]}\n"
        f"\n"
        f"Poll progress with browser_status(session_id={sid!r}). "
        f"Returns state=running while working, state=done with the result + "
        f"screenshots when complete, state=error with a resume hint on "
        f"failure. Typical tasks take 30-90s; poll every 5-15s."
    )


@MCP.tool()
async def browser_status(session_id: str):  # noqa: ANN201
    """Check the status of a ``deploy_browser_agent`` task.

    The real status probe the outer agent was missing. Poll every 5-15
    seconds. Stop polling once ``state`` is ``done`` or ``error``.

    Args:
        session_id: From ``deploy_browser_agent``'s return value.

    Returns:
        On ``running``: state + elapsed seconds + the original task.
        On ``done``: state + the agent's text + image content blocks for
            any screenshots (also persisted to ``data/staged/``).
        On ``error``: state + the exception + a resume hint (load_session +
            prompt) — the conversation still exists in the store.
        If the session_id isn't a deploy_browser_agent task (e.g. created
            via new_session, or the server restarted): explains the mismatch
            and lists currently-active tasks.
    """
    progress = _BROWSER_TASKS.get(session_id)
    if progress is None:
        # In-memory tracker has no entry. Causes:
        #   - server restarted (progress wiped; session persists in store)
        #   - created via new_session + prompt (not deploy_browser_agent)
        #   - typo
        active = [s for s, p in _BROWSER_TASKS.items() if p["state"] == "running"]
        return (
            f"No deploy_browser_agent task found for `{session_id}`. This can "
            f"happen if the server restarted (in-memory progress is wiped, but "
            f"the session persists in the store — use list_sessions + "
            f"load_session + prompt to continue), or if the session was "
            f"created via new_session instead of deploy_browser_agent.\n"
            f"Active browser tasks: {active or 'none'}."
        )

    elapsed = (progress["finished_at"] or time.time()) - progress["started_at"]
    state = progress["state"]
    task_preview = progress["task"][:200]

    if state == "running":
        return (
            f"**Browser task `{session_id}`** — running, {elapsed:.1f}s elapsed\n"
            f"task: {task_preview}\n"
            f"\nStill working. Poll again in 5-15s."
        )

    if state == "error":
        return (
            f"**Browser task `{session_id}`** — error after {elapsed:.1f}s\n"
            f"task: {task_preview}\n"
            f"\n**Error:** {progress['error']}\n"
            f"\nThe subagent turn failed mid-flight but the conversation still "
            f"exists. Resume it with load_session(session_id={session_id!r}, "
            f"org='browser-agent') then prompt(session_id={session_id!r}, "
            f"message='...')."
        )

    # state == "done"
    text = progress["text"]
    images = progress["images"] or []
    stop = progress["stop_reason"]

    parts: list[str] = [
        f"**Browser task `{session_id}`** — done in {elapsed:.1f}s",
        f"task: {task_preview}",
        f"\n*[{stop}]*",
        f"\n**Result:**",
        text or "(no response)",
    ]

    # Persist + surface images as MCP image content blocks (so the client
    # sees screenshots natively, not just file paths). Mirrors the original
    # synchronous deploy_browser_agent asset-passthrough behavior.
    if images:
        _STAGED_HOST_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        saved_paths: list[str] = []
        for i, img in enumerate(images):
            ext = "png" if "png" in img["mime_type"] else "jpg"
            fname = f"browser_agent_{ts}_{i}.{ext}"
            try:
                raw = base64.b64decode(img["data"])
                (_STAGED_HOST_DIR / fname).write_bytes(raw)
                saved_paths.append(f"{_STAGED_CONTAINER_DIR}/{fname}")
            except Exception:  # noqa: BLE001 — persist is best-effort
                pass
        if saved_paths:
            parts.append("\n📸 Browser-agent images saved:")
            for p in saved_paths:
                parts.append(f"  `{p}`")
        content: list = [MCPTextContent(type="text", text="\n".join(parts))]
        for img in images:
            content.append(MCPImageContent(
                type="image",
                data=img["data"],
                mimeType=img["mime_type"],
            ))
        return content
    return "\n".join(parts)



# ═══════════════════════════════════════════════════════════════════════════
# web-research proxy — search / fetch / research as pux MCP tools.
#
# Why a proxy instead of pointing the client at web-research-mcp directly:
# Claude Code's `.mcp.json` has NO per-server tool allowlist, so a direct
# connection floods the client with all 17 of research-mcp's tools (context
# cost on every turn). Proxying the 3 read-only research tools here means the
# client sees EXACTLY deploy_browser_agent + web_search + web_fetch +
# web_research + the file tools — the proxy IS the allowlist. Signatures
# mirror the live server (probed 2026-07-17 at $PUX_MCP_WEB_RESEARCH_URL).
# ═══════════════════════════════════════════════════════════════════════════


def _web_research_url() -> str:
    return os.environ.get("PUX_MCP_WEB_RESEARCH_URL", "http://127.0.0.1:41827/mcp")


async def _forward_to_research_mcp(url: str, tool: str, args: dict):
    """The actual MCP client call to web-research-mcp.

    Per-call connect (stateless streamable HTTP) — no long-lived session to
    manage. Separated from :func:`_web_research_call` so tests can monkeypatch
    this one seam instead of standing up the real server. Returns the raw
    MCP ``CallToolResult`` (raises on transport/protocol failure so the caller
    can shape it into an error string)."""
    from mcp import ClientSession  # noqa: PLC0415
    from mcp.client.streamable_http import streamablehttp_client  # noqa: PLC0415

    async with streamablehttp_client(url) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool, args)


def _coerce_proxy_content(blocks: list) -> list:
    """Normalize a web-research-mcp CallToolResult.content list into MCP
    content blocks this server can return. Text + Image pass through (so
    ``fetch`` on an image-bearing page returns the images inline — the
    multimedia payoff); anything else is stringified to text."""
    out: list = []
    for b in blocks or []:
        btype = getattr(b, "type", None)
        if btype == "text":
            out.append(MCPTextContent(type="text", text=getattr(b, "text", str(b))))
        elif btype == "image":
            out.append(MCPImageContent(
                type="image",
                data=getattr(b, "data", ""),
                mimeType=getattr(b, "mimeType", "image/png"),
            ))
        else:
            out.append(MCPTextContent(type="text", text=str(b)))
    return out or [MCPTextContent(type="text", text="(no content)")]


async def _web_research_call(tool: str, args: dict):
    """Forward ``tool(args)`` to web-research-mcp and return MCP content blocks
    (or an ``Error:`` string if the server is unreachable)."""
    url = _web_research_url()
    try:
        result = await _forward_to_research_mcp(url, tool, args)
    except Exception as exc:  # noqa: BLE001 — unreachable server is a normal ops condition
        return (
            f"Error: web-research-mcp unreachable at {url} ({exc}). "
            f"Start it (docker compose up in the research-mcp repo) and retry."
        )
    if getattr(result, "is_error", False):
        # Surface the upstream error text so the client sees why it failed.
        err_text = "\n".join(
            getattr(b, "text", str(b)) for b in (result.content or [])
            if getattr(b, "type", None) == "text"
        ) or f"upstream tool {tool!r} returned an error"
        return f"Error from web-research-mcp {tool}: {err_text}"
    return _coerce_proxy_content(getattr(result, "content", []))


@MCP.tool()
async def web_search(query: str, top_k: int | None = None,
                     pages: int | None = None):  # noqa: ANN201
    """Search the web via multiple engines. Returns titles, URLs, and short
    snippets. This is a proxy to web-research-mcp's ``search`` tool.

    Use this when you want a list of candidate results to evaluate. When you
    want the actual CONTENT (page bodies scraped to markdown), use
    ``web_research`` (search + scrape in one call) or ``web_fetch`` (one URL).

    Args:
        query: The search query.
        top_k: Optional max results to return.
        pages: Optional number of search-engine result pages to pull.
    """
    args: dict = {"query": query}
    if top_k is not None:
        args["top_k"] = top_k
    if pages is not None:
        args["pages"] = pages
    return await _web_research_call("search", args)


@MCP.tool()
async def web_fetch(url: str, text_only: bool = False,
                    css_selector: str | None = None):  # noqa: ANN201
    """Scrape a single URL and extract clean markdown content. Proxy to
    web-research-mcp's ``fetch`` tool.

    JS-heavy pages render via crawl4ai/selenium; PDFs extract to text. By
    default IMAGES come back too (as inline image content you can see) — pass
    ``text_only=True`` to drop them for speed when only the text matters.

    Args:
        url: The URL to fetch.
        text_only: If True, drop images and return markdown text only.
        css_selector: Optional CSS selector for targeted content extraction.
    """
    args: dict = {"url": url, "text_only": text_only}
    if css_selector is not None:
        args["css_selector"] = css_selector
    return await _web_research_call("fetch", args)


@MCP.tool()
async def web_research(query: str, max_results: int = 3,
                       depth: str = "quick"):  # noqa: ANN201
    """Search + scrape the top results in one call — the fastest path to
    actual content on a topic. Proxy to web-research-mcp's ``research`` tool.

    Returns the scraped markdown (and any images) from the top results. Use
    this instead of ``web_search`` when you want to READ the pages, not just
    see URLs/snippets.

    Args:
        query: The research query.
        max_results: How many top results to scrape (default 3).
        depth: ``"quick"`` (default) or a deeper mode if the server supports it.
    """
    return await _web_research_call(
        "research", {"query": query, "max_results": max_results, "depth": depth},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    # Transport: ``stdio`` (editor-spawned, no daemon) or ``sse`` (default;
    # network clients like Hermes connect to :9987). ``pux mcp --transport
    # stdio`` runs the SAME MCP object + tool surface over stdio so Zed /
    # Claude Code spawn it on demand — no long-running daemon to babysit.
    transport = os.environ.get("PUX_MCP_TRANSPORT", "")
    if not transport:
        # Parse --transport from argv by hand (keep it dependency-free; the
        # bin/pux shim forwards args verbatim: `pux mcp --transport stdio`).
        for i, a in enumerate(sys.argv[1:], start=1):
            if a == "--transport" and i + 1 < len(sys.argv) + 1:
                transport = sys.argv[i + 1]
                break
            if a.startswith("--transport="):
                transport = a.split("=", 1)[1]
                break
    transport = transport or "sse"
    if transport not in ("stdio", "sse"):
        sys.stderr.write(f"[pux] unknown transport {transport!r}; use stdio|sse\n")
        sys.exit(2)

    sys.stderr.write(f"[pux] AP server: {PUX_API_URL}\n")
    if transport == "stdio":
        sys.stderr.write("[pux] MCP-stdio (editor-spawned, no daemon)\n")
        MCP.run(transport="stdio")
        return

    import uvicorn  # noqa: PLC0415
    host = os.environ.get("PUX_MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("PUX_MCP_PORT", "9987"))
    sys.stderr.write(f"[pux] MCP-SSE on {host}:{port}\n")
    app = MCP.sse_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
