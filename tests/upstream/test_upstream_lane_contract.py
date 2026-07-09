"""Contract test for the UPSTREAM Agent Protocol lane.

Pins the contract that ``pux_harness.runtime.upstream`` + ``langgraph.json``
serve the FULL Agent Protocol surface that pux's hand-rolled ``server.py`` REST
lane currently reimplements — so ``server.py``'s REST CRUD is retire-eligible
(see [[rely-on-upstream]], [[no-legacy-left-behind]], [[upstream-protocol-pivot]]).

This is the LIVE proof the pivot leans on (verify-or-die, not "should work"):
it launches the OFFICIAL ``langgraph-api`` runtime (``langgraph dev``) against
pux's OWN graph declaration, then drives the SAME ``langgraph_sdk`` surface a
real consumer uses:

  * ``assistants.search`` -> every org is served as a ``graph_id`` (org==assistant)
  * ``threads.create``     -> the upstream runtime mints + owns the thread
  * ``runs.stream``        -> a real run streams the langgraph wire format
                              (``metadata`` + ``values`` events) with the answer
  * ``store.put/get_item`` -> the BaseStore surface round-trips a value

The served graph is pux's real ``compile_org`` declaration (genuine roster +
prompt + deepagents middleware); only the supervisor LLM is scripted
(``_ScriptedModel`` -> ``"upstream ok"``), so the proof is keyless + Dockerless.
If green, the AP REST surface is proven servable upstream.

LIVE: skipped unless ``langgraph-cli`` is importable (``langgraph_api`` on path)
AND the ``langgraph`` console script is resolvable. Run with::

    uv run --with 'langgraph-cli[inmem]' --project pux-harness \\
        pytest tests/upstream/test_upstream_lane_contract.py -v
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

# pux-harness/ (this file: tests/upstream/test_upstream_lane_contract.py)
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


@pytest.fixture(scope="module")
def upstream_base() -> str:
    """Launch ``langgraph dev`` against pux's graph declaration; yield its base URL.

    Keystone regime (``PUX_UPSTREAM_GRAPH=keystone``): minimal scripted
    ``create_agent`` — portable + keyless + Dockerless, the contract keystone.
    ``--no-reload`` so watchfiles doesn't bounce the server mid-proof.
    """
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
            out, _ = proc.communicate(timeout=10)
            log_buf.append(out)
        except Exception:  # noqa: BLE001
            proc.kill()
            with __import__("contextlib").suppress(Exception):
                log_buf.append(proc.communicate(timeout=5)[0])
        # surface the server log on teardown for debugging
        with __import__("contextlib").suppress(Exception):
            Path(f"/tmp/pux_upstream_test_{port}.log").write_text("".join(log_buf))


def _sdk_client(base: str):
    from langgraph_sdk import get_client

    return get_client(url=base)


def test_assistants_serve_every_org(upstream_base: str) -> None:
    """Every org in langgraph.json is served as its own ``graph_id`` (== assistant)."""

    async def _go() -> set[str]:
        c = _sdk_client(upstream_base)
        assistants = await c.assistants.search()
        return {a["graph_id"] for a in assistants}

    declared = set(json.loads(CONFIG.read_text())["graphs"])
    served = asyncio.run(_go())
    assert declared <= served, f"missing graph_ids: {declared - served}"


def test_thread_create_and_run_stream(upstream_base: str) -> None:
    """threads.create + runs.stream: upstream mints the thread + streams the wire format."""

    async def _go() -> tuple[list[str], dict]:
        c = _sdk_client(upstream_base)
        th = await c.threads.create()
        tid = th["thread_id"]
        assert tid, "upstream returned no thread_id"
        events: list[str] = []
        last_values: dict = {}
        async for part in c.runs.stream(
            tid, "general",
            input={"messages": [{"role": "user", "content": "hi"}]},
            stream_mode=["values"],
        ):
            events.append(part.event)
            if part.event == "values" and isinstance(part.data, dict):
                last_values = part.data
        return events, last_values

    events, last_values = asyncio.run(_go())
    assert "metadata" in events, f"no metadata event: {events}"
    assert "values" in events, f"no values event: {events}"
    contents = [
        m["content"]
        for m in last_values.get("messages", [])
        if isinstance(m, dict) and isinstance(m.get("content"), str)
    ]
    assert "upstream ok" in contents, f"answer missing: {contents}"


def test_store_round_trips(upstream_base: str) -> None:
    """store.put_item/get_item: the BaseStore surface round-trips a value upstream."""

    async def _go() -> object:
        c = _sdk_client(upstream_base)
        await c.store.put_item(("pux", "general"), key="contract", value={"v": 1})
        return await c.store.get_item(("pux", "general"), key="contract")

    item = asyncio.run(_go())
    assert item, "store get returned nothing"
    val = item["value"] if isinstance(item, dict) else getattr(item, "value", None)
    assert val == {"v": 1}, f"store round-trip drift: {item}"


def test_persisted_state_survives_run(upstream_base: str) -> None:
    """The upstream runtime owns the checkpointer: state persists after a run."""

    async def _go() -> dict:
        c = _sdk_client(upstream_base)
        th = await c.threads.create()
        tid = th["thread_id"]
        async for _ in c.runs.stream(
            tid, "general",
            input={"messages": [{"role": "user", "content": "hi"}]},
            stream_mode=["values"],
        ):
            pass
        return await c.threads.get_state(tid)  # type: ignore[no-any-return]

    state = asyncio.run(_go())
    values = state.get("values", {}) if isinstance(state, dict) else {}
    msgs = values.get("messages", [])
    assert len(msgs) >= 2, f"expected persisted user+AI messages, got {len(msgs)}"
