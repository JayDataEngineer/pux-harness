#!/usr/bin/env python3
"""Aegra live smoke — prove the FREE prod AP runtime serves pux's real graphs +
custom_app end-to-end via the upstream ``langgraph_sdk`` wire format.

Aegra ([[aegra-verified]], [[langgraph-api-license-gate]]) is the OSS
self-hosted langgraph-api drop-in (FastAPI + PostgreSQL, Apache-2.0). This
script drives the SAME surface a real AP consumer (Hermes, dev-bot, MCP) uses:

  1. ``assistants.search``  — every discovered org is registered as a graph_id.
  2. ``threads.create``     — AP thread CRUD.
  3. ``runs.stream``        — the org-mode scripted graph EXECUTES end-to-end
     (keyless proof the deepagents graph shape serves up under Aegra's runtime).
  4. ``store.put_item``/``get_item`` — AP /store round-trip (memory/persistence).
  5. ``/events/health`` + ``/events`` — pux's CUSTOM EventBus (run-completion
     receiver-of-last-resort for webhook-less clients) mounted via http.app.

PREREQ — boot Aegra once (from pux-harness/), then run this probe::

    # 1. postgres sidecar (aegra's compose; creds must match the env below)
    docker run -d --name pux_harness-postgres -p 5432:5432 \\
      -e POSTGRES_USER=pux_harness -e POSTGRES_PASSWORD=pux_harness_secret \\
      -e POSTGRES_DB=pux_harness pgvector/pgvector:pg18

    # 2. aegra dev (reads aegra.json with PRIORITY over langgraph.json)
    PUX_UPSTREAM_GRAPH=org PUX_PROJECT_ROOT=$(git rev-parse --show-toplevel) \\
    AEGRA_CONFIG=aegra.json RUN_MIGRATIONS_ON_STARTUP=true \\
    POSTGRES_USER=pux_harness POSTGRES_PASSWORD=pux_harness_secret \\
    POSTGRES_DB=pux_harness POSTGRES_HOST=localhost POSTGRES_PORT=5432 \\
    uv run aegra dev --port 2026 --no-reload

    # 3. this probe
    AEGRA_URL=http://127.0.0.1:2026 uv run python scripts/aegra_smoke.py

Exit 0 = all 5 surfaces green; 1 = failure (printed).
"""

from __future__ import annotations

import os
import sys
import traceback
import urllib.request

URL = os.environ.get("AEGRA_URL", "http://127.0.0.1:2026").rstrip("/")


def _probe(path: str) -> tuple[int, str]:
    with urllib.request.urlopen(f"{URL}{path}", timeout=10) as r:
        return r.status, r.read(200).decode("utf-8", "replace")[:160]


def main() -> int:
    from langgraph_sdk import get_sync_client

    failures: list[str] = []
    c = get_sync_client(url=URL)

    # 1. assistants — every org registered as a graph_id
    asst = c.assistants.search(limit=100)
    graph_ids = sorted(a.get("graph_id") for a in asst)
    print(f"[1] assistants.search -> {len(graph_ids)} graph_ids: {graph_ids}")
    if "general" not in graph_ids:
        failures.append("no 'general' graph_id registered")

    # 2. thread
    tid = c.threads.create()["thread_id"]
    print(f"[2] threads.create   -> {tid}")

    # 3. run.stream — org-mode scripted graph must execute + reply
    general = next(a for a in asst if a.get("graph_id") == "general")
    final_state = None
    snapshots = 0
    for ev in c.runs.stream(
        tid,
        general["assistant_id"],
        input={"messages": [{"role": "user", "content": "reply: PUX-ALIVE"}]},
        stream_mode="values",
    ):
        snapshots += 1
        if (
            isinstance(ev, tuple)
            and len(ev) > 1
            and ev[0] == "values"
            and isinstance(ev[1], dict)
        ):
            final_state = ev[1]
    msgs = (final_state or {}).get("messages", [])
    last_ai = next(
        (
            m
            for m in reversed(msgs)
            if (getattr(m, "type", None) or (m.get("type") if isinstance(m, dict) else None))
            == "ai"
        ),
        None,
    )
    last_content = (
        getattr(last_ai, "content", None)
        or (last_ai.get("content") if isinstance(last_ai, dict) else None)
        if last_ai
        else None
    )
    print(f"[3] runs.stream      -> {snapshots} snapshots; final ai msg: {last_content!r}")
    if not last_content:
        failures.append("run produced no final ai message (graph did not execute)")

    # 4. store round-trip (namespace, key, value)
    c.store.put_item(("aegra_smoke", "ns"), "v1", {"v": 42})
    got = c.store.get_item(("aegra_smoke", "ns"), "v1")
    val = got.get("value") if isinstance(got, dict) else getattr(got, "value", None)
    print(f"[4] store put+get    -> {val}")
    if val != {"v": 42}:
        failures.append(f"store round-trip failed: got {val!r}")

    # 5. custom_app EventBus surfaces (mounted via http.app under Aegra)
    try:
        code, body = _probe("/events/health")
        print(f"[5] /events/health   -> {code} {body.strip()}")
        if code != 200 or '"ok"' not in body:
            failures.append(f"/events/health not ok: {code} {body}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"/events/health error: {exc!r}")

    if failures:
        print("\nAEGRA SMOKE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAEGRA SMOKE OK: AP wire + custom_app live; org graph executed end-to-end.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("AEGRA SMOKE ERROR (is aegra dev running on %s?):", URL, file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(2)
