"""pux_sandbox_browser_* — in-sandbox SeleniumBase Chrome via sb_server.py.

Each pux process gets its OWN ephemeral sb_server + Chrome inside the
container — no sharing between concurrent orgs or subagents. The default
sb_server (port 9876) remains for warmup/status; all browser TOOL calls
route to the process's ephemeral instance on a unique port pair.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shlex
import time

from pydantic import BaseModel, Field, field_validator

from langchain_core.tools import StructuredTool

from pux_harness.sandbox.docker_exec import DockerExecClient, ExecTimeout
from pux_harness.sandbox.tools._shared import PUX_PREFIX, _tail, _result, _NoArgs


# --- ephemeral per-process browser isolation --------------------------------
# Each pux process (one per org via `pux acp --org X`) gets its own
# sb_server + Chrome on a unique port pair. Deterministic from PID so
# restarts of the same process reuse the same ports (stale detection
# cleans up dead instances). Range: 50 concurrent browser processes.
_EPHEMERAL_HTTP_BASE = 9900
_EPHEMERAL_CDP_BASE = 9300
_EPHEMERAL_RANGE = 50
_process_http_port: int | None = None  # cached for this process lifetime

# --- RELIABILITY FALLBACK ---------------------------------------------------
# The supervisord-managed default sb_server on port 9876 is ALWAYS running
# (started at container boot by supervisord — see sandbox/supervisord.conf).
# It is the NEVER-FAIL fallback when the per-process ephemeral spawn breaks.
# The ephemeral path can fail because: Chromium download glitch, Chrome cold-
# start timeout, port-kill race, /tmp pressure, sb_server Python error, etc.
# When that happens we fall back to 9876 — losing per-process isolation and
# stealth (shared default Chrome profile) but GUARANTEEING a working browser.
# The contract: the browser tool returns a working browser, always. Period.
_SUPERVISORD_SB_PORT = 9876


def _supervisord_browser_ready(exec_client: DockerExecClient) -> bool:
    """Is the supervisord-managed default sb_server up AND Chrome alive?

    Checks BOTH ``ok`` (sb_server process) AND ``alive`` (Chrome attached) in
    the /status response — a server with a dead Chrome is NOT a fallback
    candidate. If the server is up but Chrome cold, hit /warmup once and
    re-check (Chrome cold-start is ~10s, bounded by the server's own
    /warmup handler)."""
    out, _ = exec_client.exec(
        f"curl -sS --max-time 3 http://127.0.0.1:{_SUPERVISORD_SB_PORT}/status "
        f"2>/dev/null || true"
    )
    if '"alive": true' not in out and '"alive":true' not in out:
        # Chrome not yet alive — try to warm it up (supervisord sb_server
        # lazily inits Chrome on first request). One shot; if warmup fails
        # the fallback isn't usable.
        if '"ok": true' not in out and '"ok":true' not in out:
            return False  # sb_server itself is down — no fallback possible
        exec_client.exec(
            f"curl -sS --max-time 20 http://127.0.0.1:{_SUPERVISORD_SB_PORT}/warmup "
            f">/dev/null 2>&1 || true"
        )
        out, _ = exec_client.exec(
            f"curl -sS --max-time 3 http://127.0.0.1:{_SUPERVISORD_SB_PORT}/status "
            f"2>/dev/null || true"
        )
    return '"alive": true' in out or '"alive":true' in out

_BROWSER_TIMEOUT = 60

# Browser spawn deadline: each call waits up to this many seconds for the
# ephemeral sb_server + Chrome cold start before returning a TRANSIENT error
# for THAT call. Every browser tool call attempts spawn fresh — no sticky
# circuit breaker, no permanent "browser dead" state. If spawn fails, the
# agent gets a transient error and can retry; the underlying cause (broken
# container, dead Chrome, sb_server crash) is also being self-healed at the
# SandboxContainer.ensure() layer (network probe + auto-recreate). The
# previous sticky circuit breaker was a reliability defect: once tripped it
# instructed the agent to TELL THE USER "browser is down" and give up — the
# exact opposite of reliable. Removed.
_BROWSER_SPAWN_TIMEOUT = int(os.environ.get("PUX_BROWSER_SPAWN_TIMEOUT", "20"))
_warmup_started = False  # guards the background warmup so it fires once

# Human-like pacing: random delay before each browser tool call so the
# action cadence looks natural to antibot services. 250-700ms mimics
# human reaction lag. Set PUX_BROWSER_MIN_PACING=0 to disable.
_PACING_MIN_MS = int(os.environ.get("PUX_BROWSER_MIN_PACING", "250"))
_PACING_MAX_MS = int(os.environ.get("PUX_BROWSER_MAX_PACING", "700"))


def _pace():
    """Sleep a random human-like amount before each browser tool call."""
    if _PACING_MIN_MS > 0:
        delay = random.uniform(_PACING_MIN_MS / 1000.0, _PACING_MAX_MS / 1000.0)
        time.sleep(delay)


def _alloc_port_pair() -> tuple[int, int]:
    """Deterministic (http_port, cdp_port) pair based on PID."""
    h = int(hashlib.md5(str(os.getpid()).encode()).hexdigest(), 16)
    offset = h % _EPHEMERAL_RANGE
    return _EPHEMERAL_HTTP_BASE + offset, _EPHEMERAL_CDP_BASE + offset


def _is_server_alive(exec_client: DockerExecClient, http_port: int) -> bool:
    out, _ = exec_client.exec(
        f"curl -s -o /dev/null -w '%{{http_code}}' --max-time 2 "
        f"http://127.0.0.1:{http_port}/status 2>/dev/null || true"
    )
    return out.strip() == "200"


# Shell fragment (no fuser — busybox skips it; ss is always present). Walks
# the process tree by ppid so Chrome's setsid helpers are reaped too. The
# recursion via ``_kt`` is bounded by Chrome's actual tree depth (~3-4 levels).
_KILL_STALE_TEMPLATE = (
    "_kt() { "
    "  for c in $(ps -eo pid,ppid --noheaders | "
    "             awk -v P=$1 '$2==P{print $1}'); do "
    "    _kt $c; "
    "  done; "
    "  kill -9 $1 2>/dev/null; "
    "}; "
    "for port in %d %d; do "
    "  pid=$(ss -tlnp 2>/dev/null | grep \":$port \" | "
    "        grep -oP 'pid=\\K\\d+' | head -1); "
    "  [ -n \"$pid\" ] && _kt $pid; "
    "done; sleep 0.5 || true"
)


def _spawn_one_attempt(exec_client: DockerExecClient, http_port: int, cdp_port: int) -> bool:
    """One spawn attempt: kill stale, launch sb_server, poll for ready.

    Returns True on ready, False on timeout. Single-attempt; the caller
    decides whether to retry. Split out from ``_ensure_ephemeral_server`` so
    the retry loop is explicit and testable."""
    # Kill stale processes on our ports AND all their descendants.
    #
    # Two non-obvious facts drove this shape:
    #
    #   1. The container's busybox does NOT ship ``fuser`` — earlier code
    #      that called ``fuser -k`` silently no-op'd (stderr was discarded).
    #      Every kill was a lie. Orphaned Chrome helpers piled up across
    #      spawns and tripped the container's PID cgroup limit ("Resource
    #      temporarily unavailable"), killing the sandbox mid-task. We use
    #      ``ss`` (always present in the image) to find the listener PID.
    #
    #   2. Chrome calls ``setsid()`` on some helper subprocesses, detaching
    #      them from the spawner's process group. ``kill -9 -PGID`` therefore
    #      misses them. Only a recursive descendant walk by PPID reaches
    #      every Chrome helper (renderer, GPU process, zygote, crashpad).
    exec_client.exec(_KILL_STALE_TEMPLATE % (http_port, cdp_port))

    # Spawn ephemeral sb_server: own Chrome, own CDP port, own profile dir.
    # The --cdp-port flag makes sb_server NOT kill other instances' Chrome.
    display = os.environ.get("DISPLAY", ":99")
    cmd = (
        f"DISPLAY={display} SB_SERVER_PORT={http_port} "
        f"SB_CDP_PORT={cdp_port} "
        f"python3 /usr/local/bin/sb_server.py --stealth --use-chromium "
        f"> /tmp/sb_ephemeral_{http_port}.log 2>&1 &"
    )
    exec_client.exec(cmd)

    # Wait for Chrome cold start + sb_server ready (capped at _BROWSER_SPAWN_TIMEOUT)
    deadline = time.monotonic() + _BROWSER_SPAWN_TIMEOUT
    while time.monotonic() < deadline:
        if _is_server_alive(exec_client, http_port):
            return True
        time.sleep(0.5)
    return False


# Number of full spawn attempts per _ensure_ephemeral_server call. Each attempt
# is bounded by _BROWSER_SPAWN_TIMEOUT; on failure the next attempt kills stale
# processes (including a half-spawned Chrome from the prior attempt) and tries
# fresh. Reliable-by-default: the browser MUST come up. Only after this many
# genuine attempts do we surface a transient error to the agent.
_BROWSER_SPAWN_ATTEMPTS = int(os.environ.get("PUX_BROWSER_SPAWN_ATTEMPTS", "3"))


def _ensure_ephemeral_server(exec_client: DockerExecClient) -> int:
    """Ensure THIS PROCESS has its own ephemeral sb_server running. Returns HTTP port.

    On first browser tool call in the process:
      1. Allocates a unique port pair (http + cdp) from PID
      2. Spawns a fresh sb_server with its own Chrome (--cdp-port, unique profile)
      3. Waits for /status to answer (Chrome cold start ~10-15s)
      4. Restores saved session cookies if a session file exists

    RELIABILITY CONTRACT: this function tries ``_BROWSER_SPAWN_ATTEMPTS`` times
    before surfacing a transient error. Each attempt is a full kill-stale +
    spawn + wait-for-ready cycle — a half-dead Chrome from one attempt gets
    cleaned up by the next. The agent should essentially never see a spawn
    failure; when it does, the error says "transient — retry" and the next
    tool call will spawn fresh again (no sticky dead state).

    The underlying sandbox is also self-healed at SandboxContainer.ensure()
    (network probe + auto-recreate on container-level breakage)."""
    global _process_http_port

    if _process_http_port is not None:
        if _is_server_alive(exec_client, _process_http_port):
            return _process_http_port
        # Dead — fall through to respawn

    http_port, cdp_port = _alloc_port_pair()

    last_log = ""
    for attempt in range(_BROWSER_SPAWN_ATTEMPTS):
        if _spawn_one_attempt(exec_client, http_port, cdp_port):
            _process_http_port = http_port
            # Restore saved cookies so the ephemeral browser has the org's session
            _restore_session_cookies(exec_client, http_port)
            return http_port
        # Capture log for the final error; keep latest.
        log_out, _ = exec_client.exec(
            f"tail -5 /tmp/sb_ephemeral_{http_port}.log 2>/dev/null || true"
        )
        last_log = _tail(log_out, 200)

    # All ephemeral attempts failed — FALL BACK to the supervisord-managed
    # default sb_server (port 9876). It's always running at container boot,
    # so if it's up + Chrome alive, we use it. Loses per-process isolation
    # + stealth but delivers the RELIABILITY CONTRACT: browser tool always
    # returns a working browser. The fallback path is logged so operators
    # can see ephemeral is broken (and fix the root cause) without the agent
    # ever seeing a failure.
    if _supervisord_browser_ready(exec_client):
        # Use stderr-style logging via the exec client (module-level `log`
        # isn't imported here; the tail of the ephemeral log already shows
        # the failure for diagnosis).
        exec_client.exec(
            f"echo '[pux browser] ephemeral spawn failed ({_BROWSER_SPAWN_ATTEMPTS} "
            f"attempts); falling back to supervisord sb_server on port "
            f"{_SUPERVISORD_SB_PORT}' >> /tmp/pux_browser_fallback.log 2>&1 || true"
        )
        _process_http_port = _SUPERVISORD_SB_PORT
        return _SUPERVISORD_SB_PORT

    # Truly catastrophic: ephemeral spawn failed AND the supervisord
    # fallback isn't usable. This should essentially never happen — the
    # supervisord sb_server is started at container boot and re-spawns on
    # crash. If we get here, the container itself is broken; the next
    # SandboxContainer.ensure() call will detect it (network probe) and
    # recreate. Return a transient error so the agent retries.
    raise RuntimeError(
        f"ALL browser spawn paths failed: ephemeral ({_BROWSER_SPAWN_ATTEMPTS} "
        f"attempts × {_BROWSER_SPAWN_TIMEOUT}s) AND supervisord fallback "
        f"(port {_SUPERVISORD_SB_PORT} not ready). Transient — the container "
        f"will be recreated on the next ensure() call. Retry the browser tool. "
        f"Last ephemeral log: {last_log}"
    )


def warmup_ephemeral_browser(exec_client: DockerExecClient) -> None:
    """Kick off the ephemeral browser spawn in a BACKGROUND thread.

    Called at graph-build time (before the agent loop starts) so Chrome is
    already warming while the CTO boots + thinks. By the time the CTO
    dispatches to web-agent, Chrome should be ready — the first
    ``browser_navigate`` call hits a warm server (instant) instead of
    cold-starting (15-20s inside the LLM turn budget).

    Fire-and-forget: the result populates ``_process_http_port``. If the
    sandbox is down, the background thread fails silently — the first real
    tool call will discover the failure via the normal path (and retry, since
    there's no longer a sticky circuit breaker). No exception escapes to the
    caller.
    """
    global _warmup_started
    if _warmup_started:
        return
    _warmup_started = True

    import threading

    def _warm():
        try:
            _ensure_ephemeral_server(exec_client)
        except Exception:  # noqa: BLE001 — background warmup must never crash the agent
            pass

    t = threading.Thread(target=_warm, daemon=True, name="browser-warmup")
    t.start()


def _restore_session_cookies(exec_client: DockerExecClient, http_port: int) -> None:
    """Restore saved session cookies to the ephemeral browser if a file exists.

    Ephemeral browsers start with a fresh profile — no cookies. If the org
    has a saved session (e.g. from brave_cookie_bridge for twitter-agent),
    restore it so the browser is immediately authenticated."""
    for session_file in (
        "/sandbox/workspace/data/.twitter-session.json",
        "/sandbox/workspace/data/.browser-session.json",
    ):
        out, _ = exec_client.exec(f"test -f {session_file} && echo exists || echo missing")
        if "exists" in out:
            exec_client.exec(
                f"curl -s -X POST http://127.0.0.1:{http_port}/restore_session "
                f"-H 'Content-Type: application/json' "
                f"-d '{{\"path\":\"{session_file}\"}}' --max-time 10 2>/dev/null || true"
            )
            break  # one session file per browser


def _sb_post(exec_client: DockerExecClient, endpoint: str, body_obj: dict | None,
             *, timeout: int = _BROWSER_TIMEOUT) -> str:
    """POST ``body_obj`` to THIS PROCESS's ephemeral sb_server endpoint, return the
    parsed JSON re-serialized via ``_result``.

    Every call attempts spawn fresh — no circuit breaker, no sticky "dead"
    state. If spawn fails for this call, the agent gets a transient error it
    can retry. The underlying sandbox is self-healed at the container layer."""
    _pace()
    # Ensure we have our own isolated browser instance
    try:
        http_port = _ensure_ephemeral_server(exec_client)
    except Exception as exc:
        # Transient spawn failure for THIS call only — agent can retry.
        return _result({"success": False, "reason": "browser_spawn_failed",
                        "error": f"ephemeral browser: {exc}"})
    addr = f"http://127.0.0.1:{http_port}"
    max_time = max(1, timeout)
    parts = [
        "curl -s -S",
        f"--max-time {max_time}",
        "-X POST",
        f"{addr}{endpoint}",
        "-H 'Content-Type: application/json'",
    ]
    body = ""
    if body_obj is not None:
        body = json.dumps(body_obj)
        parts += ["-d", shlex.quote(body)]
    cmd = " ".join(parts)
    try:
        out, exit_code = exec_client.exec(cmd, timeout=timeout)
    except ExecTimeout:
        return _result({"success": False, "reason": "timeout",
                        "error": f"browser {endpoint}: timed out after {timeout}s"})
    except Exception as exc:
        return _result({"success": False, "reason": "exec_failed",
                        "error": f"browser {endpoint}: {exc}"})
    if exit_code != 0:
        return _result({"success": False, "reason": "exec_failed",
                        "error": f"browser {endpoint}: curl exit {exit_code}",
                        "detail": _tail(out, 400)})
    try:
        parsed = json.loads(out)
    except json.JSONDecodeError:
        return _result({"success": False, "reason": "malformed_response",
                        "error": f"browser {endpoint}: non-JSON response",
                        "detail": _tail(out, 400)})
    return _result(parsed)


# --- navigate ---------------------------------------------------------------

_BROWSER_NAVIGATE_DESC = (
    "Open a URL in the persistent Chrome. Returns page state + labeled screenshot."
)


class _BrowserNavigateArgs(BaseModel):
    url: str = Field(..., description="Absolute URL including scheme (https://example.com)")


def _browser_navigate_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(url: str) -> str:
        if not url:
            return _result({"success": False, "error": "url is required"})
        return _sb_post(exec_client, "/navigate", {"url": url})

    return StructuredTool(
        name=PUX_PREFIX + "browser_navigate", description=_BROWSER_NAVIGATE_DESC,
        args_schema=_BrowserNavigateArgs, func=_run,
    )


# --- click ------------------------------------------------------------------

_BROWSER_CLICK_DESC = (
    "Click an element by SoM label or CSS selector. trusted=true for anti-bot sites that ignore synthetic clicks."
)


class _BrowserClickArgs(BaseModel):
    index: int | None = Field(None, description="SoM label (numbered box on interactive elements from the last screenshot)")
    selector: str | None = Field(None, description="CSS selector (e.g. 'button#submit'). Used when index is omitted.")
    trusted: bool = Field(False, description="Drive the real cursor via CDP (isTrusted=true). Use when a normal click silently no-ops on anti-bot sites.")


def _browser_click_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(index: int | None = None, selector: str | None = None,
             trusted: bool = False) -> str:
        body: dict = {}
        if index is not None:
            body["index"] = index
        if selector is not None:
            body["selector"] = selector
        if trusted:
            body["trusted"] = True
        return _sb_post(exec_client, "/click", body)

    return StructuredTool(
        name=PUX_PREFIX + "browser_click", description=_BROWSER_CLICK_DESC,
        args_schema=_BrowserClickArgs, func=_run,
    )


# --- type -------------------------------------------------------------------

_BROWSER_TYPE_DESC = (
    "Type text into a form field by SoM label or selector. trusted=true for keystroke-fingerprinting defenses."
)


class _BrowserTypeArgs(BaseModel):
    text: str = Field(..., description="Text to type into the field")
    index: int | None = Field(None, description="SoM label of the target input")
    selector: str | None = Field(None, description="CSS selector of the target input")
    trusted: bool = Field(False, description="Type via CDP Input.insertText (isTrusted events). Use for keystroke-fingerprinting defenses.")


def _browser_type_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(text: str, index: int | None = None, selector: str | None = None,
             trusted: bool = False) -> str:
        if not text:
            return _result({"success": False, "error": "text is required"})
        if index is None and not selector:
            return _result({"success": False, "error": "either index or selector is required"})
        body = {"text": text}
        if index is not None:
            body["index"] = index
        if selector is not None:
            body["selector"] = selector
        if trusted:
            body["trusted"] = True
        return _sb_post(exec_client, "/type", body)

    return StructuredTool(
        name=PUX_PREFIX + "browser_type", description=_BROWSER_TYPE_DESC,
        args_schema=_BrowserTypeArgs, func=_run,
    )


# --- screenshot -------------------------------------------------------------

_BROWSER_SCREENSHOT_DESC = (
    "Capture a fresh labeled screenshot with SoM labels on interactive elements."
)


def _browser_screenshot_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run() -> str:
        return _sb_post(exec_client, "/read", {})

    return StructuredTool(
        name=PUX_PREFIX + "browser_screenshot", description=_BROWSER_SCREENSHOT_DESC,
        args_schema=_NoArgs, func=_run,
    )


# --- evaluate ---------------------------------------------------------------

_BROWSER_EVALUATE_DESC = (
    "Evaluate JavaScript on the page, return the result. Escape hatch when dedicated tools don't fit."
)


class _BrowserEvaluateArgs(BaseModel):
    code: str = Field(
        ...,
        description="JavaScript to evaluate. Use `return <value>` for the result. CRITICAL: wrap object-literal returns in parens: `return ({a: 1})`, not `return {a: 1}` (syntax error). The REPL persists across calls.",
    )


def _browser_evaluate_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(code: str) -> str:
        if not code:
            return _result({"success": False, "error": "code is required"})
        return _sb_post(exec_client, "/evaluate", {"code": code})

    return StructuredTool(
        name=PUX_PREFIX + "browser_evaluate", description=_BROWSER_EVALUATE_DESC,
        args_schema=_BrowserEvaluateArgs, func=_run,
    )


# --- search -----------------------------------------------------------------

_BROWSER_SEARCH_DESC = (
    "Search via DuckDuckGo and land on the results page. Entry point when you have a query, not a URL."
)


class _BrowserSearchArgs(BaseModel):
    query: str = Field(..., description="Natural-language search query (the engine URL-encodes it)")


def _browser_search_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(query: str) -> str:
        if not query:
            return _result({"success": False, "error": "query is required"})
        return _sb_post(exec_client, "/search", {"query": query})

    return StructuredTool(
        name=PUX_PREFIX + "browser_search", description=_BROWSER_SEARCH_DESC,
        args_schema=_BrowserSearchArgs, func=_run,
    )


# --- scroll -----------------------------------------------------------------

_BROWSER_SCROLL_DESC = (
    "Scroll the page (direction or pixel amount). Elements below the fold have no SoM label until scrolled into view."
)


class _BrowserScrollArgs(BaseModel):
    direction: str = Field("down", description="'down' or 'up' (viewport-sized); ignored when amount>0")
    amount: int = Field(0, description="Pixel count to scroll (sign follows direction). 0 = use direction for a viewport jump.")


def _browser_scroll_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(direction: str = "down", amount: int = 0) -> str:
        return _sb_post(exec_client, "/scroll", {"direction": direction, "amount": amount})

    return StructuredTool(
        name=PUX_PREFIX + "browser_scroll", description=_BROWSER_SCROLL_DESC,
        args_schema=_BrowserScrollArgs, func=_run,
    )


# --- go_back ----------------------------------------------------------------

_BROWSER_GO_BACK_DESC = (
    "Navigate back to the previous page in history."
)


def _browser_go_back_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run() -> str:
        return _sb_post(exec_client, "/go_back", {})

    return StructuredTool(
        name=PUX_PREFIX + "browser_go_back", description=_BROWSER_GO_BACK_DESC,
        args_schema=_NoArgs, func=_run,
    )


# --- wait -------------------------------------------------------------------

_BROWSER_WAIT_DESC = (
    "Pause for async content to load (default 2s, max 30), then return a fresh screenshot."
)


class _BrowserWaitArgs(BaseModel):
    seconds: int = Field(2, description="How long to wait; server clamps to 30")


def _browser_wait_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(seconds: int = 2) -> str:
        return _sb_post(exec_client, "/wait", {"seconds": seconds})

    return StructuredTool(
        name=PUX_PREFIX + "browser_wait", description=_BROWSER_WAIT_DESC,
        args_schema=_BrowserWaitArgs, func=_run,
    )


# --- find_text --------------------------------------------------------------

_BROWSER_FIND_TEXT_DESC = (
    "Scroll to and highlight the first occurrence of the given text on the page."
)


class _BrowserFindTextArgs(BaseModel):
    text: str = Field(..., description="Substring to locate on the page")


def _browser_find_text_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(text: str) -> str:
        if not text:
            return _result({"success": False, "error": "text is required"})
        return _sb_post(exec_client, "/find_text", {"text": text})

    return StructuredTool(
        name=PUX_PREFIX + "browser_find_text", description=_BROWSER_FIND_TEXT_DESC,
        args_schema=_BrowserFindTextArgs, func=_run,
    )


# --- extract ----------------------------------------------------------------

_BROWSER_EXTRACT_DESC = (
    "Extract structured text from the page (title, headings, paragraphs, lists, tables, forms). Returns {extracted:{...}}."
)


class _BrowserExtractArgs(BaseModel):
    query: str = Field("extract all text content", description="Free-text note of what you want (the engine extracts the same DOM structures regardless)")


def _browser_extract_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(query: str = "extract all text content") -> str:
        return _sb_post(exec_client, "/extract", {"query": query})

    return StructuredTool(
        name=PUX_PREFIX + "browser_extract", description=_BROWSER_EXTRACT_DESC,
        args_schema=_BrowserExtractArgs, func=_run,
    )


# --- extract_images ---------------------------------------------------------

_BROWSER_EXTRACT_IMAGES_DESC = (
    "List every <img> on the page with src + alt text. Returns {images:[{src,alt}], url}."
)


def _browser_extract_images_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run() -> str:
        return _sb_post(exec_client, "/extract_images", {})

    return StructuredTool(
        name=PUX_PREFIX + "browser_extract_images", description=_BROWSER_EXTRACT_IMAGES_DESC,
        args_schema=_NoArgs, func=_run,
    )


# --- save_screenshot --------------------------------------------------------

_BROWSER_SAVE_SCREENSHOT_DESC = (
    "Save the page as a clean PNG file at a path (for evidence/attachments, not for acting on). Returns {screenshot_path, url}."
)


class _BrowserSaveScreenshotArgs(BaseModel):
    path: str | None = Field(None, description="Absolute sandbox path incl. .png extension. If omitted the engine generates one and returns it.")


def _browser_save_screenshot_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(path: str | None = None) -> str:
        body: dict = {}
        if path:
            body["path"] = path
        return _sb_post(exec_client, "/screenshot", body)

    return StructuredTool(
        name=PUX_PREFIX + "browser_save_screenshot", description=_BROWSER_SAVE_SCREENSHOT_DESC,
        args_schema=_BrowserSaveScreenshotArgs, func=_run,
    )


# --- download ---------------------------------------------------------------

_BROWSER_DOWNLOAD_DESC = (
    "Download a file from a direct URL to a sandbox path. Returns {url, path, size}. Not for pages that need interaction to produce the file."
)


class _BrowserDownloadArgs(BaseModel):
    url: str = Field(..., description="Direct file URL to fetch")
    path: str = Field(..., description="Absolute sandbox output path (incl. extension)")


def _browser_download_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(url: str, path: str) -> str:
        if not url or not path:
            return _result({"success": False, "error": "url and path are both required"})
        return _sb_post(exec_client, "/download", {"url": url, "path": path})

    return StructuredTool(
        name=PUX_PREFIX + "browser_download", description=_BROWSER_DOWNLOAD_DESC,
        args_schema=_BrowserDownloadArgs, func=_run,
    )


# --- upload -----------------------------------------------------------------

_BROWSER_UPLOAD_DESC = (
    "Upload a local file into an <input type='file'> by CSS selector. Returns {uploaded, selector, file}."
)


class _BrowserUploadArgs(BaseModel):
    selector: str = Field(..., description="CSS selector of the <input type='file'>")
    file_path: str = Field(..., description="Absolute sandbox path of the file to upload (must exist)")


def _browser_upload_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(selector: str, file_path: str) -> str:
        if not selector or not file_path:
            return _result({"success": False, "error": "selector and file_path are both required"})
        return _sb_post(exec_client, "/upload", {"selector": selector, "file_path": file_path})

    return StructuredTool(
        name=PUX_PREFIX + "browser_upload", description=_BROWSER_UPLOAD_DESC,
        args_schema=_BrowserUploadArgs, func=_run,
    )


# --- tabs -------------------------------------------------------------------

_BROWSER_TABS_DESC = (
    "List all open tabs with index, url, title, active flag. Returns {tabs:[{index,url,title,active}]}."
)


def _browser_tabs_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run() -> str:
        return _sb_post(exec_client, "/tabs", {})

    return StructuredTool(
        name=PUX_PREFIX + "browser_tabs", description=_BROWSER_TABS_DESC,
        args_schema=_NoArgs, func=_run,
    )


# --- new_tab ----------------------------------------------------------------

_BROWSER_NEW_TAB_DESC = (
    "Open a new tab to a URL and switch to it."
)


class _BrowserNewTabArgs(BaseModel):
    url: str = Field("about:blank", description="URL to open in the new tab")


def _browser_new_tab_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(url: str = "about:blank") -> str:
        return _sb_post(exec_client, "/new_tab", {"url": url})

    return StructuredTool(
        name=PUX_PREFIX + "browser_new_tab", description=_BROWSER_NEW_TAB_DESC,
        args_schema=_BrowserNewTabArgs, func=_run,
    )


# --- switch_tab -------------------------------------------------------------

_BROWSER_SWITCH_TAB_DESC = (
    "Switch to the tab at the given 0-based index."
)


class _BrowserSwitchTabArgs(BaseModel):
    index: int = Field(0, description="0-based tab index (see browser_tabs)")


def _browser_switch_tab_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(index: int = 0) -> str:
        return _sb_post(exec_client, "/switch_tab", {"index": index})

    return StructuredTool(
        name=PUX_PREFIX + "browser_switch_tab", description=_BROWSER_SWITCH_TAB_DESC,
        args_schema=_BrowserSwitchTabArgs, func=_run,
    )


# --- close_tab --------------------------------------------------------------

_BROWSER_CLOSE_TAB_DESC = (
    "Close the current tab and switch to the last remaining one."
)


def _browser_close_tab_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run() -> str:
        return _sb_post(exec_client, "/close_tab", {})

    return StructuredTool(
        name=PUX_PREFIX + "browser_close_tab", description=_BROWSER_CLOSE_TAB_DESC,
        args_schema=_NoArgs, func=_run,
    )


# --- dropdown_options -------------------------------------------------------

_BROWSER_DROPDOWN_OPTIONS_DESC = (
    "Read the options of a <select> dropdown by SoM label or selector. Call before browser_select_dropdown."
)


class _BrowserDropdownOptionsArgs(BaseModel):
    index: int | None = Field(None, description="SoM label of the <select> element")
    selector: str | None = Field(None, description="CSS selector of the <select> element")


def _browser_dropdown_options_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(index: int | None = None, selector: str | None = None) -> str:
        if index is None and not selector:
            return _result({"success": False, "error": "either index or selector is required"})
        body: dict = {}
        if index is not None:
            body["index"] = index
        if selector is not None:
            body["selector"] = selector
        return _sb_post(exec_client, "/dropdown_options", body)

    return StructuredTool(
        name=PUX_PREFIX + "browser_dropdown_options", description=_BROWSER_DROPDOWN_OPTIONS_DESC,
        args_schema=_BrowserDropdownOptionsArgs, func=_run,
    )


# --- select_dropdown --------------------------------------------------------

_BROWSER_SELECT_DROPDOWN_DESC = (
    "Choose an option in a <select> by value attribute or visible text (XOR). Identify the select by SoM label or selector."
)


class _BrowserSelectDropdownArgs(BaseModel):
    index: int | None = Field(None, description="SoM label of the <select> element")
    selector: str | None = Field(None, description="CSS selector of the <select> element")
    value: str | None = Field(None, description="value attribute of the option to select (use XOR with text)")
    text: str | None = Field(None, description="Visible text of the option to select (use XOR with value)")


def _browser_select_dropdown_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(index: int | None = None, selector: str | None = None,
             value: str | None = None, text: str | None = None) -> str:
        if index is None and not selector:
            return _result({"success": False, "error": "either index or selector is required"})
        if value is None and text is None:
            return _result({"success": False, "error": "either value or text is required"})
        body: dict = {}
        if index is not None:
            body["index"] = index
        if selector is not None:
            body["selector"] = selector
        if value is not None:
            body["value"] = value
        if text is not None:
            body["text"] = text
        return _sb_post(exec_client, "/select_dropdown", body)

    return StructuredTool(
        name=PUX_PREFIX + "browser_select_dropdown", description=_BROWSER_SELECT_DROPDOWN_DESC,
        args_schema=_BrowserSelectDropdownArgs, func=_run,
    )


# --- save_session -----------------------------------------------------------

_BROWSER_SAVE_SESSION_DESC = (
    "Save cookies + localStorage to a JSON file. Call after login."
)


class _BrowserSaveSessionArgs(BaseModel):
    path: str = Field("/tmp/browser-session.json", description="Absolute sandbox path to write the session JSON")


def _browser_save_session_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(path: str = "/tmp/browser-session.json") -> str:
        return _sb_post(exec_client, "/save_session", {"path": path})

    return StructuredTool(
        name=PUX_PREFIX + "browser_save_session", description=_BROWSER_SAVE_SESSION_DESC,
        args_schema=_BrowserSaveSessionArgs, func=_run,
    )


# --- restore_session --------------------------------------------------------

_BROWSER_RESTORE_SESSION_DESC = (
    "Restore a saved session (cookies + localStorage). Call after navigating to the domain, before other actions."
)


class _BrowserRestoreSessionArgs(BaseModel):
    path: str = Field("/tmp/browser-session.json", description="Absolute sandbox path of a session JSON written by browser_save_session")


def _browser_restore_session_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(path: str = "/tmp/browser-session.json") -> str:
        return _sb_post(exec_client, "/restore_session", {"path": path})

    return StructuredTool(
        name=PUX_PREFIX + "browser_restore_session", description=_BROWSER_RESTORE_SESSION_DESC,
        args_schema=_BrowserRestoreSessionArgs, func=_run,
    )


# --- drag (SOTA drag-and-drop) ------------------------------------------------

_BROWSER_DRAG_DESC = (
    "Drag-and-drop: sortable lists, sliders, file drop-zones. Source: index/selector/coords; target: index/selector/coords or dx/dy offset. strategy auto/html5/physics."
)


class _BrowserDragArgs(BaseModel):
    from_index: int | None = Field(None, description="SoM label of the drag source")
    from_selector: str | None = Field(None, description="CSS selector of the drag source")
    from_x: float | None = Field(None, description="x coord of the drag source (use instead of index/selector)")
    from_y: float | None = Field(None, description="y coord of the drag source")
    to_index: int | None = Field(None, description="SoM label of the drop target")
    to_selector: str | None = Field(None, description="CSS selector of the drop target")
    to_x: float | None = Field(None, description="x coord of the drop target")
    to_y: float | None = Field(None, description="y coord of the drop target")
    dx: float | None = Field(None, description="drop this many px right(+) of the source (offset mode — sliders). Use XOR with to_*.")
    dy: float | None = Field(None, description="drop this many px down(+) of the source (offset mode — sliders)")
    strategy: str = Field("auto", description="'auto' | 'html5' | 'physics'")
    steps: int = Field(25, description="mouse-move interpolation steps for the physics path (ignored by html5)")


def _browser_drag_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(from_index: int | None = None, from_selector: str | None = None,
             from_x: float | None = None, from_y: float | None = None,
             to_index: int | None = None, to_selector: str | None = None,
             to_x: float | None = None, to_y: float | None = None,
             dx: float | None = None, dy: float | None = None,
             strategy: str = "auto", steps: int = 25) -> str:
        has_src = from_index is not None or from_selector or from_x is not None
        has_tgt = (to_index is not None or to_selector or to_x is not None
                   or dx is not None or dy is not None)
        if not has_src:
            return _result({"success": False, "error": "drag needs a source: from_index/from_selector OR from_x/from_y"})
        if not has_tgt:
            return _result({"success": False, "error": "drag needs a target: to_index/to_selector, to_x/to_y, or dx/dy"})
        body: dict = {"strategy": strategy, "steps": steps}
        if from_index is not None:
            body["from_index"] = from_index
        if from_selector:
            body["from_selector"] = from_selector
        if from_x is not None:
            body["from_x"] = from_x
        if from_y is not None:
            body["from_y"] = from_y
        if to_index is not None:
            body["to_index"] = to_index
        if to_selector:
            body["to_selector"] = to_selector
        if to_x is not None:
            body["to_x"] = to_x
        if to_y is not None:
            body["to_y"] = to_y
        if dx is not None:
            body["dx"] = dx
        if dy is not None:
            body["dy"] = dy
        return _sb_post(exec_client, "/drag", body)

    return StructuredTool(
        name=PUX_PREFIX + "browser_drag", description=_BROWSER_DRAG_DESC,
        args_schema=_BrowserDragArgs, func=_run,
    )


# --- hover ------------------------------------------------------------------

_BROWSER_HOVER_DESC = (
    "Hover an element (by index/selector/coords) to reveal dropdowns, tooltips, fly-out panels."
)


class _BrowserHoverArgs(BaseModel):
    index: int | None = Field(None, description="SoM label of the element to hover")
    selector: str | None = Field(None, description="CSS selector of the element to hover")
    x: float | None = Field(None, description="x coord to hover (use instead of index/selector)")
    y: float | None = Field(None, description="y coord to hover")


def _browser_hover_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(index: int | None = None, selector: str | None = None,
             x: float | None = None, y: float | None = None) -> str:
        has_el = index is not None or selector
        if not has_el and (x is None or y is None):
            return _result({"success": False, "error": "hover needs index/selector OR x,y"})
        body: dict = {}
        if index is not None:
            body["index"] = index
        if selector:
            body["selector"] = selector
        if x is not None:
            body["x"] = x
        if y is not None:
            body["y"] = y
        return _sb_post(exec_client, "/hover", body)

    return StructuredTool(
        name=PUX_PREFIX + "browser_hover", description=_BROWSER_HOVER_DESC,
        args_schema=_BrowserHoverArgs, func=_run,
    )


# --- press (keys / hotkeys) -------------------------------------------------

_BROWSER_PRESS_DESC = (
    "Press a key or hotkey (Enter, Escape, Tab, Control+a, etc.). For non-character keys browser_type can't send."
)


class _BrowserPressArgs(BaseModel):
    keys: str = Field(..., description="Key or '+'-joined combo, e.g. 'Enter', 'Control+a', 'ArrowDown'. Modifiers: Control/Shift/Alt/Cmd.")
    index: int | None = Field(None, description="SoM label of the element to focus before pressing")
    selector: str | None = Field(None, description="CSS selector of the element to focus before pressing")


def _browser_press_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(keys: str, index: int | None = None, selector: str | None = None) -> str:
        if not keys:
            return _result({"success": False, "error": "keys is required"})
        body: dict = {"keys": keys}
        if index is not None:
            body["index"] = index
        if selector:
            body["selector"] = selector
        return _sb_post(exec_client, "/press", body)

    return StructuredTool(
        name=PUX_PREFIX + "browser_press", description=_BROWSER_PRESS_DESC,
        args_schema=_BrowserPressArgs, func=_run,
    )


# --- click_at (coords / right / double) -------------------------------------

_BROWSER_CLICK_AT_DESC = (
    "Click at exact pixel coordinates — for targets with no SoM label/selector (canvas, chart points). Also right=true / double=true."
)


class _BrowserClickAtArgs(BaseModel):
    x: float | None = Field(None, description="x coord to click (omit to use index/selector center)")
    y: float | None = Field(None, description="y coord to click")
    index: int | None = Field(None, description="SoM label whose center to click (alternative to x,y)")
    selector: str | None = Field(None, description="CSS selector whose center to click")
    button: int = Field(0, description="mouse button: 0=left (default), 1=middle, 2=right")
    double: bool = Field(False, description="true → double-click")
    right: bool = Field(False, description="true → right-click (dispatches contextmenu)")
    trusted: bool = Field(False, description="Drive the real cursor via CDP (isTrusted=true). Use for anti-bot sites.")


def _browser_click_at_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(x: float | None = None, y: float | None = None,
             index: int | None = None, selector: str | None = None,
             button: int = 0, double: bool = False, right: bool = False,
             trusted: bool = False) -> str:
        has_target = x is not None or y is not None or index is not None or selector
        if not has_target:
            return _result({"success": False, "error": "click_at needs x,y OR index/selector"})
        body: dict = {"button": button, "double": double, "right": right}
        if x is not None:
            body["x"] = x
        if y is not None:
            body["y"] = y
        if index is not None:
            body["index"] = index
        if selector:
            body["selector"] = selector
        if trusted:
            body["trusted"] = True
        return _sb_post(exec_client, "/click_at", body)

    return StructuredTool(
        name=PUX_PREFIX + "browser_click_at", description=_BROWSER_CLICK_AT_DESC,
        args_schema=_BrowserClickAtArgs, func=_run,
    )


# --- scroll_into_view -------------------------------------------------------

_BROWSER_SCROLL_INTO_VIEW_DESC = (
    "Scroll a specific element (by index/selector) into the viewport, centered."
)


class _BrowserScrollIntoViewArgs(BaseModel):
    index: int | None = Field(None, description="SoM label of the element to bring into view")
    selector: str | None = Field(None, description="CSS selector of the element to bring into view")


def _browser_scroll_into_view_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(index: int | None = None, selector: str | None = None) -> str:
        if index is None and not selector:
            return _result({"success": False, "error": "either index or selector is required"})
        body: dict = {}
        if index is not None:
            body["index"] = index
        if selector:
            body["selector"] = selector
        return _sb_post(exec_client, "/scroll_into_view", body)

    return StructuredTool(
        name=PUX_PREFIX + "browser_scroll_into_view", description=_BROWSER_SCROLL_INTO_VIEW_DESC,
        args_schema=_BrowserScrollIntoViewArgs, func=_run,
    )


# --- a11y (accessibility tree) ----------------------------------------------

_BROWSER_A11Y_DESC = (
    "Read the page as an accessibility tree: {role, name, selector} per element. Cheaper than screenshots on dense pages. Selectors are usable by click/type."
)


def _browser_a11y_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run() -> str:
        return _sb_post(exec_client, "/a11y", {})

    return StructuredTool(
        name=PUX_PREFIX + "browser_a11y", description=_BROWSER_A11Y_DESC,
        args_schema=_NoArgs, func=_run,
    )


# --- iframe -----------------------------------------------------------------

_BROWSER_IFRAME_DESC = (
    "Act on elements inside iframes: action=list/click/evaluate. Cross-origin iframes blocked by SOP. Legacy enter/exit retired."
)


class _BrowserIframeArgs(BaseModel):
    action: str = Field("list", description="'list' | 'click' | 'evaluate' (legacy 'enter'/'exit' retired)")
    index: int | None = Field(None, description="SoM label of the iframe element")
    selector: str | None = Field(None, description="CSS selector of the iframe element")
    inner_selector: str | None = Field(None, description="action='click': CSS selector inside the iframe of the element to click")
    code: str | None = Field(None, description="action='evaluate': JS to run inside the iframe (use 'return' for a value)")


def _browser_iframe_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(action: str = "list", index: int | None = None, selector: str | None = None,
             inner_selector: str | None = None, code: str | None = None) -> str:
        body: dict = {"action": action}
        if index is not None:
            body["index"] = index
        if selector:
            body["selector"] = selector
        if inner_selector:
            body["inner_selector"] = inner_selector
        if code:
            body["code"] = code
        return _sb_post(exec_client, "/iframe", body)

    return StructuredTool(
        name=PUX_PREFIX + "browser_iframe", description=_BROWSER_IFRAME_DESC,
        args_schema=_BrowserIframeArgs, func=_run,
    )


# --- UC mode (Cloudflare / Turnstile / hCaptcha bypass) ---------------------

_BROWSER_UC_DESC = (
    "Cloudflare/Turnstile/hCaptcha bypass via SB(uc=True) + real pyautogui click. Actions: open|click|type|read|evaluate|cookies|close. See AGENTS.md captcha ladder."
)


class _BrowserUcArgs(BaseModel):
    action: str = Field("open", description="open|click|type|read|evaluate|cookies|close")
    url: str | None = Field(None, description="URL for action=open")
    click_captcha: bool | None = Field(True, description="action=open: run uc_gui_click_captcha")
    handoff: bool | None = Field(True, description="action=open: hand cookies to persistent browser")
    selector: str | None = Field(None, description="action=click/type: CSS selector")
    text: str | None = Field(None, description="action=click: link text | action=type: text to type")
    by: str | None = Field("css", description="action=click: selector strategy (css|text|xpath)")
    submit: bool | None = Field(False, description="action=type: press Enter after typing")
    clear: bool | None = Field(True, description="action=type: clear field first")
    code: str | None = Field(None, description="action=evaluate: JS to run")
    cookie_action: str | None = Field("get", description="action=cookies: get|inject_persistent")


def _browser_uc_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(action: str = "open", url: str | None = None, click_captcha: bool | None = True,
             handoff: bool | None = True, selector: str | None = None, text: str | None = None,
             by: str | None = "css", submit: bool | None = False, clear: bool | None = True,
             code: str | None = None, cookie_action: str | None = "get") -> str:
        body: dict = {"action": action}
        if url is not None:
            body["url"] = url
        if click_captcha is not None:
            body["click_captcha"] = click_captcha
        if handoff is not None:
            body["handoff"] = handoff
        if selector is not None:
            body["selector"] = selector
        if text is not None:
            body["text"] = text
        if by is not None:
            body["by"] = by
        if submit is not None:
            body["submit"] = submit
        if clear is not None:
            body["clear"] = clear
        if code is not None:
            body["code"] = code
        if cookie_action is not None:
            body["cookie_action"] = cookie_action
        # UC open can take ~30-60s (spawns Chrome + captcha click). read/click
        # are fast. Scale the curl max-time to the action.
        timeout = 150 if action == "open" else _BROWSER_TIMEOUT
        return _sb_post(exec_client, "/uc", body, timeout=timeout)
    return StructuredTool(
        name=PUX_PREFIX + "browser_uc", description=_BROWSER_UC_DESC,
        args_schema=_BrowserUcArgs, func=_run,
    )


# --- accept cookies (GDPR / CCPA banner dismissal) --------------------------

_BROWSER_ACCEPT_COOKIES_DESC = (
    "Dismiss GDPR/CCPA cookie-consent banners. Call right after navigate. Returns cookies_accepted + method. See AGENTS.md."
)


class _BrowserAcceptCookiesArgs(BaseModel):
    pass


def _browser_accept_cookies_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run() -> str:
        return _sb_post(exec_client, "/accept_cookies", {})
    return StructuredTool(
        name=PUX_PREFIX + "browser_accept_cookies", description=_BROWSER_ACCEPT_COOKIES_DESC,
        args_schema=_BrowserAcceptCookiesArgs, func=_run,
    )


# --- warmup history (fingerprint legitimacy) --------------------------------

_BROWSER_WARMUP_HISTORY_DESC = (
    "Build browsing-history legitimacy before sensitive targets (login, job apps). Call ONCE at session start. Optional urls=[...] and dwell=N."
)


class _BrowserWarmupHistoryArgs(BaseModel):
    urls: list[str] | None = Field(None, description="Custom site list (default: 6 benign sites)")
    dwell: float | None = Field(3.0, description="Seconds per site (randomized ±50%)")

    @field_validator("urls", mode="before")
    @classmethod
    def _coerce_urls(cls, v):
        """Models frequently pass a JSON array as a STRING ('[\"a\",\"b\"]')
        rather than a real array, and Pydantic's strict list[str] rejects it
        with "Input should be a valid list" — costing the agent 3-4 retry
        turns before it gives up. Accept str | list, JSON-decode strings,
        comma-split fallback, drop blanks. Returns None for empty input so
        the server uses its default site list."""
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            try:
                v = json.loads(s)
            except json.JSONDecodeError:
                v = [u.strip() for u in s.split(",")]
        if not isinstance(v, list):
            return None
        out = [str(u).strip() for u in v if str(u).strip()]
        return out or None


def _browser_warmup_history_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run(urls: list[str] | None = None, dwell: float | None = 3.0) -> str:
        body: dict = {}
        if urls is not None:
            body["urls"] = urls
        if dwell is not None:
            body["dwell"] = dwell
        # 6 sites × ~4s dwell = ~30s budget
        timeout = max(_BROWSER_TIMEOUT, int((len(urls) if urls else 6) * (dwell or 3) + 20))
        return _sb_post(exec_client, "/warmup_history", body, timeout=timeout)
    return StructuredTool(
        name=PUX_PREFIX + "browser_warmup_history", description=_BROWSER_WARMUP_HISTORY_DESC,
        args_schema=_BrowserWarmupHistoryArgs, func=_run,
    )


# --- solve captcha (honest best-effort on persistent browser) ----------------

_BROWSER_SOLVE_CAPTCHA_DESC = (
    "Best-effort captcha click on persistent Chrome. Returns honest captcha_solved=false if it can't pass — then use browser_uc."
)


class _BrowserSolveCaptchaArgs(BaseModel):
    pass


def _browser_solve_captcha_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run() -> str:
        return _sb_post(exec_client, "/solve_captcha", {})
    return StructuredTool(
        name=PUX_PREFIX + "browser_solve_captcha", description=_BROWSER_SOLVE_CAPTCHA_DESC,
        args_schema=_BrowserSolveCaptchaArgs, func=_run,
    )


# --- reset (fresh Chrome — clears stale tabs/captcha state) ------------------

_BROWSER_RESET_DESC = (
    "Reset the browser: close UC session, re-init Chrome, clear tabs + cookies. "
    "Use when the browser is stuck on a captcha/error page or has stale tabs from a previous task."
)


class _BrowserResetArgs(BaseModel):
    pass


def _browser_reset_tool(exec_client: DockerExecClient) -> StructuredTool:
    def _run() -> str:
        return _sb_post(exec_client, "/reset", {})
    return StructuredTool(
        name=PUX_PREFIX + "browser_reset", description=_BROWSER_RESET_DESC,
        args_schema=_BrowserResetArgs, func=_run,
    )
