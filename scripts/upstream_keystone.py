#!/usr/bin/env python3
"""KEYSTONE PROOF for the upstream-contract pivot.

Proves the OFFICIAL upstream ``langgraph-api`` runtime (launched by
``langgraph dev``) serves a pux-declared graph with the FULL Agent Protocol
surface that pux's hand-rolled ``server.py`` currently reimplements — so we can
retire the downstream REST lane and bind upstream (see [[rely-on-upstream]],
[[no-legacy-left-behind]]).

What it does (verify-or-die, not "should work"):
  1. launches ``langgraph dev --config langgraph.json`` in a subprocess
     (PUX_UPSTREAM_DEV=1 -> the deterministic keystone graph; no Docker, no key);
  2. waits for the server's ``/ok`` health endpoint;
  3. drives the SAME ``langgraph_sdk`` surface pux's tests already pin:
       assistants.search  -> the ``general`` graph is listed (org==assistant)
       threads.create     -> a thread is minted by the upstream runtime
       runs.stream        -> a real run streams the langgraph wire format
                             (metadata + values events) with the graph's answer
       store.put/get_item -> the BaseStore surface round-trips a value
  4. prints ``PROVEN`` + the evidence, or ``FAILED`` + the cause.

The graph served is pux's OWN declaration (``pux_harness.runtime.upstream``);
only the LLM is scripted. If this is green, the Agent Protocol REST surface is
proven servable upstream and the ``server.py`` REST lane is retire-eligible.

Usage:  uv run python scripts/upstream_keystone.py [--port 2024]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
from langgraph_sdk import get_client

ROOT = Path(__file__).resolve().parents[1]  # pux-harness/
CONFIG = ROOT / "langgraph.json"


def _wait_for_server(base: str, timeout: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base}/ok", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)
    return False


async def _drive(base: str) -> dict:
    """Exercise the SDK surface against the live upstream server. Returns a
    dict of evidence; raises AssertionError with a clear cause on any drift."""
    c = get_client(url=base)
    out: dict = {}

    # 1. assistants.search -> general is served as an assistant (org==assistant)
    assistants = await c.assistants.search()
    graph_ids = {a["graph_id"] for a in assistants}
    assert "general" in graph_ids, f"general not served: {graph_ids}"
    out["assistants"] = sorted(graph_ids)

    # 2. threads.create -> upstream mints the thread
    th = await c.threads.create()
    tid = th["thread_id"]
    assert tid, "no thread_id"
    out["thread_id"] = tid

    # 3. runs.stream -> real langgraph wire format (metadata + values + answer)
    events: list[str] = []
    last_values: dict = {}
    async for part in c.runs.stream(
        tid, "general", input={"messages": [{"role": "user", "content": "hi"}]},
        stream_mode=["values"],
    ):
        events.append(part.event)
        if part.event == "values" and isinstance(part.data, dict):
            last_values = part.data
    assert "metadata" in events, f"no metadata event: {events}"
    assert "values" in events, f"no values event: {events}"
    msgs = last_values.get("messages", [])
    contents = [
        m.get("content") for m in msgs
        if isinstance(m, dict) and isinstance(m.get("content"), str)
    ]
    assert "upstream ok" in contents, f"answer missing: {contents}"
    out["stream_events"] = events
    out["answer"] = "upstream ok"

    # 4. store.put/get -> BaseStore surface round-trips
    await c.store.put_item(("pux", "keystone"), key="k", value={"v": 1})
    item = await c.store.get_item(("pux", "keystone"), key="k")
    assert item, "store get returned nothing"
    # Item is a TypedDict (dict-like): value lives under the "value" key.
    val = item["value"] if isinstance(item, dict) else getattr(item, "value", None)
    assert val == {"v": 1}, f"store round-trip failed: {item}"
    out["store"] = val

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=2024)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    env = {**os.environ, "PUX_UPSTREAM_DEV": "1"}
    cmd = [
        "uv", "run", "langgraph", "dev",
        "--config", str(CONFIG),
        "--port", str(args.port),
        "--host", args.host,
        "--no-browser",
        "--no-reload",
    ]
    print(f"[keystone] launching upstream runtime: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    log_buf: list[str] = []
    try:
        if not _wait_for_server(base):
            # capture whatever the server printed before giving up
            try:
                out, _ = proc.communicate(timeout=3)
                log_buf.append(out)
            except Exception:  # noqa: BLE001
                pass
            print("FAILED: upstream server did not become healthy in time", file=sys.stderr)
            print("--- server log ---\n" + "".join(log_buf)[-4000:], file=sys.stderr)
            return 2

        print(f"[keystone] server healthy at {base}/ok; driving SDK surface", flush=True)
        evidence = asyncio.run(_drive(base))
        print("PROVEN — upstream langgraph-api serves pux's graph via the SDK surface:")
        for k, v in evidence.items():
            print(f"  {k}: {v}")
        return 0
    except AssertionError as e:
        print(f"FAILED (contract drift): {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"FAILED (unexpected): {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    finally:
        with contextlib.suppress(Exception):
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            with contextlib.suppress(Exception):
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
