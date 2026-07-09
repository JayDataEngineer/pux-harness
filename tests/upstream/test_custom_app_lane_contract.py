"""Contract test for the langgraph-api ``user_router`` custom-app seam (P3 C1).

Pins the contract that ONE ``langgraph dev`` process serves BOTH:

* the upstream Agent Protocol surface (CRUD; proven via ``assistants.search``), AND
* pux's CUSTOM surfaces mounted under it via ``http.app``
  (``pux_harness.runtime.custom_app:app``):
    - ``/events/health`` + ``/events`` + ``/events/stream``  (run-completion EventBus)
    - ``/jobs/{org}/status``                                 (prep/warmup jobs)

This is the C1 proof of the rely-on-upstream cutover vehicle
([[rely-on-upstream]], [[plan-p3-server-rest-retirement]]): pux contributes ONLY
its unique routes as a custom app and lets ``langgraph serve`` own the rest — the
two surfaces coexist on one process, which is what makes retiring ``server.py``'s
29 duplicate CRUD routes safe ([[no-legacy-left-behind]]).

LIVE: skipped unless ``langgraph-cli`` is importable. Run with::

    uv run --with 'langgraph-cli[inmem]' --project pux-harness \\
        pytest tests/upstream/test_custom_app_lane_contract.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

# --- live-guard: only run when the upstream runtime is actually available -----
try:  # pragma: no cover - import guard
    import langgraph_api  # noqa: F401

    _HAS_LANGGRAPH_API = True
except Exception:  # noqa: BLE001
    _HAS_LANGGRAPH_API = False

_LANGGRAPH_BIN = shutil.which("langgraph")

pytestmark = pytest.mark.skipif(
    not (_HAS_LANGGRAPH_API and _LANGGRAPH_BIN),
    reason="langgraph-cli not installed (run with --with 'langgraph-cli[inmem]')",
)

# pux-harness/ (this file: tests/upstream/test_custom_app_lane_contract.py)
HARNESS_ROOT = Path(__file__).resolve().parents[2]
CONFIG = HARNESS_ROOT / "langgraph.json"
REPO_ROOT = HARNESS_ROOT.parent  # .../auto-developer-orchestrator (owns orgs/)


def _free_port() -> int:
    """An ephemeral port the kernel guarantees free at bind time."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ok(base: str, timeout: float = 120.0) -> None:
    """Poll the upstream ``/ok`` health endpoint; raise on timeout."""
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/ok", timeout=2) as r:
                if r.status == 200 and json.loads(r.read()).get("ok"):
                    return
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(1.0)
    raise RuntimeError(f"upstream server never became healthy at {base}/ok: {last}")


def _get_json(base: str, path: str) -> tuple[int, object]:
    with urllib.request.urlopen(f"{base}{path}", timeout=10) as r:
        return r.status, json.loads(r.read())


@pytest.fixture(scope="module")
def serve() -> str:
    """Launch ``langgraph dev`` (custom app wired via langgraph.json ``http.app``);
    yield its base URL. Keystone regime: keyless + Dockerless contract proof."""
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "PUX_PROJECT_ROOT": str(REPO_ROOT),
        "PUX_UPSTREAM_GRAPH": "keystone",
    }
    cmd = [
        _LANGGRAPH_BIN, "dev",
        "--config", str(CONFIG),
        "--port", str(port),
        "--host", "127.0.0.1",
        "--no-browser",
        "--no-reload",
    ]
    proc = subprocess.Popen(
        cmd, cwd=str(HARNESS_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    log_buf: list[str] = []
    try:
        _wait_ok(base)
        yield base
    finally:
        proc.terminate()
        try:
            log_buf.append(proc.communicate(timeout=10)[0])
        except Exception:  # noqa: BLE001
            proc.kill()
            with __import__("contextlib").suppress(Exception):
                log_buf.append(proc.communicate(timeout=5)[0])
        with __import__("contextlib").suppress(Exception):
            Path(f"/tmp/pux_custom_app_test_{port}.log").write_text("".join(log_buf))


# --- 1. the custom app is the langgraph-api base app --------------------------


def test_http_app_is_wired() -> None:
    """langgraph.json declares the custom app via ``http.app`` (the seam)."""
    manifest = json.loads(CONFIG.read_text())
    assert manifest.get("http", {}).get("app") == "pux_harness.runtime.custom_app:app"


# --- 2. upstream CRUD is STILL served (mounted on the custom app) -------------


def test_upstream_assistants_still_served(serve: str) -> None:
    """Upstream AP CRUD rides ON the custom app — assistants.search still works."""

    async def _go() -> set[str]:
        from langgraph_sdk import get_client

        c = get_client(url=serve)
        return {a["graph_id"] for a in await c.assistants.search()}

    served = asyncio.run(_go())
    declared = set(json.loads(CONFIG.read_text())["graphs"])
    assert declared <= served, f"upstream CRUD missing graph_ids: {declared - served}"


# --- 3. the pux-unique custom surfaces ARE served (the C1 claim) --------------


def test_events_health_served(serve: str) -> None:
    """``/events/health`` — the EventBus lifespan ran (langgraph-api combined it)."""
    status, body = _get_json(serve, "/events/health")
    assert status == 200, body
    assert isinstance(body, dict)
    assert body.get("ok") is True, body
    assert "subscribers" in body and "events" in body, body


def test_events_list_served(serve: str) -> None:
    """``/events`` — recent run-completion events poll surface."""
    status, body = _get_json(serve, "/events")
    assert status == 200, body
    assert isinstance(body, dict) and isinstance(body.get("events"), list), body


def test_events_stream_served(serve: str) -> None:
    """``/events/stream`` — SSE feed opens + emits the metadata frame.

    Read frame-by-frame (the stream is an infinite keep-alive loop, so a fixed
    ``read(N)`` would block waiting for N bytes that never arrive)."""
    req = urllib.request.Request(f"{serve}/events/stream")
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.status == 200
        assert "text/event-stream" in r.headers.get("content-type", ""), r.headers.get("content-type")
        # the generator's first yield is the metadata frame — read its lines
        first = r.readline().decode("utf-8", "replace")
    assert first.startswith("event: metadata"), f"no metadata frame in SSE stream: {first!r}"


def test_jobs_status_served(serve: str) -> None:
    """``/jobs/{org}/status`` — prep/warmup job spec surface (uses ``general``)."""
    status, body = _get_json(serve, "/jobs/general/status")
    assert status == 200, body
    assert isinstance(body, dict) and body.get("org") == "general", body
    assert "jobs" in body, body


def test_jobs_status_404_unknown_org(serve: str) -> None:
    """``/jobs/{org}/status`` 404s on an unknown org (the handler bites)."""
    import urllib.error

    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(f"{serve}/jobs/__nope__/status", timeout=10)
    assert ei.value.code == 404
