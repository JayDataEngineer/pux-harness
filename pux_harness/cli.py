"""Unified ``pux`` CLI — replaces ``bin/pux``, ``main.py``, ``cli.py``, ``acp.py``.

Usage:
  pux direct                   in-process deepagents runner (no server)
  pux acp [--org X]            ACP stdio server for editor integration
  pux tui [--org X]            launch dcode TUI with org branding + agents
  pux sandbox <cmd>            Docker sandbox lifecycle (start/stop/status/ensure/pause/unpause/dump-persist)
  pux agents                   list orgs (agents)
  pux dispatch [--org X] ...   ephemeral blocking run
  pux resume [--org X]         list recent threads (+ task snippets; offline-capable)
  pux show <thread_id>         thread state (last message) + the resume command
  pux history <thread_id>      revision history
  pux direct --thread <id> ... resume a thread in-process (checkpointer restores every prior turn)
  pux run <thread_id> ...      background run on an existing thread
  pux wait <run_id>            block for a background run's output
  pux bundle <thread_id>       (optional) bundle transcript + artifacts + memos into a tarball
  pux list                     list discovered orgs
  pux check [--org X]          docker-exec + specialist smoke test (no model)
  pux check-contract           validate the declarative org contract
  pux check-policy [--org X]   resolve + report an org's policy
  pux jobs run --org X         run prep jobs inside the sandbox
  pux jobs status --org X      show declared prep jobs
  pux capabilities list        unified catalog of tools/skills/mcp/middleware/jobs
  pux pack --org X             pack org as a manifest-driven portable archive
  pux lock --org X             regenerate org.lock.yaml (pip + MCP pins)
  pux promote-function --org X NAME   graduate a lib function to git-tracked sandbox/ (c->b)
  pux archive-function --org X NAME   retire a lib function to lib/.archive/ (reversible)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import httpx

# --- Agent Protocol client helpers (shared by client subcommands) -------------

PUX_API_URL = os.environ.get("PUX_API_URL", "http://127.0.0.1:9988").rstrip("/")
TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=30.0)


def _post(path: str, **json_body: Any) -> Any:
    try:
        r = httpx.post(f"{PUX_API_URL}{path}", json=json_body, timeout=TIMEOUT)
    except httpx.ConnectError as exc:
        _die(f"can't reach the Agent Protocol server at {PUX_API_URL} "
             f"({exc}). Start the Agent Protocol server (prod: `aegra serve` "
             f"via scripts/start_pux_aegra.sh; dev: `langgraph dev` or "
             f"`aegra dev`)")
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:  # noqa: BLE001
            detail = r.text
        _die(f"server returned {r.status_code}: {detail}")
    return r.json()


def _get(path: str) -> Any:
    try:
        r = httpx.get(f"{PUX_API_URL}{path}", timeout=TIMEOUT)
    except httpx.ConnectError as exc:
        _die(f"can't reach the Agent Protocol server at {PUX_API_URL} "
             f"({exc}). Start the Agent Protocol server (prod: `aegra serve` "
             f"via scripts/start_pux_aegra.sh; dev: `langgraph dev` or "
             f"`aegra dev`)")
    if r.status_code >= 400:
        _die(f"server returned {r.status_code}: {r.text}")
    return r.json()


def _die(msg: str) -> None:
    print(f"pux: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _print_block(label: str, body: str) -> None:
    print(f"=== {label} ===")
    print(body.rstrip() if isinstance(body, str) else body)


def _add_tier_flags(p: Any) -> None:
    """Add ``--tier <name>`` + ``--fast`` (mutually exclusive) to an in-process
    subparser. ``acp``/``direct``/``tui`` build orgs (and so resolve models) in
    THIS process, so a tier flag here is authoritative. Client commands
    (``dispatch``/``run``) talk to an already-running server that resolved its
    tier at startup — a client flag would be misleading, so they don't get one.
    ``--fast`` is sugar for ``--tier fast``."""
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--tier", default=None, metavar="NAME",
        help="model tier to resolve roles from (sets PUX_TIER for this process; "
             "e.g. 'default' = SOTA supervisor + cheap workers, 'fast' = all "
             "cheap). Known tiers are read from models.yaml; an unknown name "
             "fails loud. Default: models.yaml's default_tier.",
    )
    g.add_argument(
        "--fast", action="store_true",
        help="shorthand for --tier fast (cheap models throughout — the rate-limit "
             "fallback / trivial-task mode)",
    )


def _apply_tier_flag(args: Any) -> None:
    """Translate ``--tier``/``--fast`` into ``PUX_TIER`` in-process, then validate
    it eagerly (``active_tier`` raises ValueError on an unknown tier) so a typo
    dies at the CLI, not at the first model build. An explicit flag overrides a
    pre-existing ``PUX_TIER`` env; passing neither leaves the env (or the default
    tier) in control. No-op for subcommands that don't carry the flags."""
    fast = getattr(args, "fast", False)
    tier = getattr(args, "tier", None)
    if not fast and not tier:
        return
    if fast:
        os.environ["PUX_TIER"] = "fast"
    else:
        os.environ["PUX_TIER"] = tier
    from pux_harness.agent.model import active_tier  # noqa: PLC0415 — lazy import

    active_tier()  # raises ValueError on an unknown PUX_TIER


# --- Client subcommands (Agent Protocol REST) ---------------------------------


def cmd_agents() -> None:
    agents = _post("/agents/search")
    print(f"{len(agents)} agents (orgs):")
    for a in agents:
        print(f"  {a['agent_id']:<22} {a['description']}")


def cmd_dispatch(org: str, task: str, recursion_limit: int, rubric: str | None = None) -> None:
    if rubric:
        payload: Any = {
            "messages": [{"role": "user", "content": task}],
            "rubric": rubric,
        }
    else:
        payload = task
    res = _post("/runs/wait", agent_id=org, input=payload, recursion_limit=recursion_limit)
    status = res.get("status")
    if status == "error":
        _print_block("ERROR", res.get("error", "(no detail)"))
        raise SystemExit(1)
    _print_block("FINAL ANSWER", res.get("output", "(empty)"))
    print(f"\n[thread] {res.get('thread_id')}   [agent] {res.get('agent_id')}   "
          f"[status] {status}")
    print("(resume with: pux show <thread_id>)")
    _print_output_dirs(res.get("thread_id"))


def cmd_resume(org: str | None) -> None:
    """List recent threads with their task snippets so you can find the one
    you want to resume.

    Two paths: live AP server first (rich), then on-disk sqlite fallback so
    listing works even after the operator has shut everything down — the
    thread index lives in ``<project>/.pux/agent-protocol.sqlite`` regardless
    of whether the server is up. Each row shows the thread_id, the org, the
    creation time, and the task string (the literal thing you asked the agent
    to do) so you can tell sessions apart."""
    threads = _list_threads_with_fallback(org)
    if not threads:
        print("(no threads)")
        return
    # Column-width math: cap thread_id at 30, agent at 18, task at 60.
    print(f"  {'thread_id':<30} {'agent':<18} {'created':<21} task")
    for t in threads:
        meta = t.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:  # noqa: BLE001
                meta = {}
        task = (meta.get("task") or "").replace("\n", " ").strip()
        if len(task) > 60:
            task = task[:57] + "..."
        created = (t.get("created_at") or "")[:19]
        print(f"  {t['thread_id']:<30} {t.get('org') or t.get('agent_id') or '?':<18} "
              f"{created:<21} {task}")
    print(f"\n{len(threads)} thread(s). Resume with:")
    print("  pux direct --org <name> --thread <thread_id> --task \"follow up\"")


def _list_threads_with_fallback(org: str | None) -> list[dict[str, Any]]:
    """Try the AP server for the thread list; fall back to the on-disk sqlite
    store when the server is unreachable. Never raises — returns [] on total
    failure (with a one-line note on stderr)."""
    # Path A: live server.
    try:
        body: dict[str, Any] = {}
        if org:
            body["agent_id"] = org
        return _post("/threads/search", **body)
    except SystemExit:
        pass  # _die raised; fall through
    except Exception:  # noqa: BLE001
        pass
    # Path B: on-disk sqlite store.
    try:
        import asyncio  # noqa: PLC0415
        from pux_harness.threads import open_thread_store  # noqa: PLC0415

        async def _list() -> list[dict[str, Any]]:
            async with open_thread_store() as store:
                return await store.list_threads(org)

        rows = asyncio.run(_list())
        # The on-disk store uses 'org'; the server returns 'agent_id'. Normalize
        # so the renderer can handle either.
        for r in rows:
            r.setdefault("agent_id", r.get("org"))
        print("(server unreachable — read from disk)", file=sys.stderr)
        return rows
    except Exception as exc:  # noqa: BLE001
        print(f"(could not list threads: {exc})", file=sys.stderr)
        return []


def cmd_show(thread_id: str) -> None:
    """Show a thread's last message, agent, status, message count, and the
    exact command to resume it. Two paths: live server first; on-disk sqlite
    fallback so `pux show` works even when the server is down."""
    state = _fetch_thread_state_with_fallback(thread_id)
    if not state:
        print(f"thread {thread_id!r} not found "
              f"(neither server nor disk had it).", file=sys.stderr)
        raise SystemExit(1)
    agent_id = state.get("agent_id") or state.get("org")
    status = state.get("status", "unknown")
    vals = state.get("values") or {}
    msgs = vals.get("messages") or []
    n_msgs = len(msgs)
    last = msgs[-1] if msgs else {}
    content = last.get("content") if isinstance(last, dict) else last
    _print_block(
        f"thread {thread_id} (agent={agent_id}, status={status}, "
        f"messages={n_msgs})",
        content or "(no messages)",
    )
    # The resume hint — the thing that was missing.
    print(
        f"\nresume with:  pux direct --org {agent_id} "
        f"--thread {thread_id} --task \"<your follow-up>\""
    )
    print(f"or via server: pux run {thread_id} \"<your follow-up>\"  "
          f"(then: pux wait <run_id>)")


def _fetch_thread_state_with_fallback(thread_id: str) -> dict[str, Any] | None:
    """Best-effort thread state fetch. Server first (rich — has messages +
    status), then on-disk sqlite (returns the pux_threads row without
    checkpoint state). Returns None if neither path has the thread."""
    try:
        return _get(f"/threads/{thread_id}")
    except SystemExit:
        pass
    except Exception:  # noqa: BLE001
        pass
    try:
        import asyncio  # noqa: PLC0415
        from pux_harness.threads import open_thread_store  # noqa: PLC0415

        async def _get_row() -> dict[str, Any] | None:
            async with open_thread_store() as store:
                return await store.get_thread(thread_id)

        row = asyncio.run(_get_row())
        if row is None:
            return None
        print("(server unreachable — read from disk; no message bodies)",
              file=sys.stderr)
        return {
            "thread_id": row["thread_id"],
            "agent_id": row["org"],
            "values": {},  # disk path has no checkpoint state inline
            "status": "archived",
        }
    except Exception:  # noqa: BLE001
        return None


def cmd_history(thread_id: str) -> None:
    hist = _get(f"/threads/{thread_id}/history")
    print(f"{len(hist)} revisions for {thread_id}:")
    for h in hist:
        nexts = ",".join(h.get("next") or []) or "-"
        print(f"  {h.get('checkpoint_id')}  next={nexts}")


def cmd_run(thread_id: str, task: str, recursion_limit: int) -> None:
    res = _post(f"/threads/{thread_id}/runs", input=task, recursion_limit=recursion_limit)
    print(f"[run] {res.get('run_id')}  status={res.get('status')}  "
          f"thread={thread_id}  (wait with: pux wait {res.get('run_id')})")


def cmd_wait(run_id: str) -> None:
    res = _get(f"/runs/{run_id}/wait")
    if res.get("status") == "error":
        _print_block("ERROR", res.get("error", "(no detail)"))
        raise SystemExit(1)
    _print_block(f"run {run_id} (status={res.get('status')})", res.get("output", "(empty)"))


# --- Output discovery + bundle (persistence-audit gaps 1-3, 7-8) -------------

# The workspace convention dirs the agent writes into during a run. Every dir
# is bind-mounted to the host (so it's on disk the moment the agent writes),
# but every dir is gitignored (runtime state, not source). ``pux bundle`` is
# the canonical way to extract one thread's worth of work from these dirs.
WORKSPACE_DIRS = ("artifacts", "memos", ".pux/memos", ".pux/sessions", "wild-runs")


def _project_root() -> Any:
    """Resolve the project root from the kit's location-independent resolver.

    Used by ``cmd_dispatch`` (to print where files land) and ``cmd_bundle`` (to
    walk the workspace dirs). Lives in the kit so the client process and the
    harness server agree on the same path."""
    from pux_harness.kit._paths import project_root  # noqa: PLC0415

    return project_root()


def _print_output_dirs(thread_id: str | None = None) -> None:
    """Print the resume command + where files live.

    The hint users actually need: how to pick this session back up tomorrow
    (the resume flow), plus where artifacts land on disk (the workspace
    bind-mount). Bundle is mentioned only as the export option, not the
    primary path — resuming is."""
    project = _project_root()
    if thread_id:
        print(f"resume with:   pux direct --thread {thread_id} --task \"<follow-up>\"")
        print(f"               (or: pux run {thread_id} \"<follow-up>\")")
    print(f"files under:   {project}")
    print("  artifacts/  memos/  .pux/sessions/")
    if thread_id:
        print(f"bundle (opt):  pux bundle {thread_id}   (one-tarball export)")


def _parse_iso8601(ts: str) -> float:
    """Parse an ISO 8601 timestamp to epoch seconds. Returns 0.0 on failure."""
    from datetime import datetime  # noqa: PLC0415

    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return 0.0


def _iso8601(epoch: float) -> str:
    """Render an epoch seconds float as an ISO 8601 UTC timestamp."""
    from datetime import datetime, timezone  # noqa: PLC0415

    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _now_iso() -> str:
    return _iso8601(__import__("time").time())


def _fetch_thread_transcript(thread_id: str) -> dict[str, Any]:
    """Best-effort fetch of thread state + history.

    Tries the Agent Protocol server first (full fidelity). On any failure
    (server down, 404, connection refused), falls back to reading the on-disk
    thread store directly so the bundle still works after the operator has
    shut everything down — the typical 'I stopped the server, now archive'
    case. Records which path was used in the returned dict (``_source``)."""
    out: dict[str, Any] = {"thread_id": thread_id, "state": None, "history": [], "_source": None}
    # Path A: live server. Use httpx directly (not the _get helper) so a 404 /
    # connection-refused falls through silently to the disk path — printing
    # "pux: server returned 404" via _die would mislead the user into thinking
    # the bundle failed when it's about to succeed via the disk fallback.
    try:
        state_resp = httpx.get(
            f"{PUX_API_URL}/threads/{thread_id}", timeout=TIMEOUT,
        )
        history_resp = httpx.get(
            f"{PUX_API_URL}/threads/{thread_id}/history", timeout=TIMEOUT,
        )
        if state_resp.status_code < 400 and history_resp.status_code < 400:
            out["state"] = state_resp.json()
            hist = history_resp.json()
            out["history"] = hist if isinstance(hist, list) else []
            out["_source"] = "server"
            return out
    except (httpx.ConnectError, httpx.TimeoutException):
        pass  # server down — fall through to disk
    # Path B: on-disk sqlite thread store.
    try:
        import asyncio  # noqa: PLC0415
        from pux_harness.threads import open_thread_store  # noqa: PLC0415

        async def _read() -> tuple[Any, Any]:
            async with open_thread_store() as store:
                row = await store.get_thread(thread_id)
                return row, None

        row, _ = asyncio.run(_read())
        if row is not None:
            out["state"] = {
                "thread_id": row["thread_id"],
                "agent_id": row["org"],
                "metadata": row["metadata"],
                "created_at": row["created_at"],
            }
            out["_source"] = "disk"
    except Exception:  # noqa: BLE001
        pass
    return out


def _add_bytes_to_tar(tar: Any, name: str, data: bytes) -> None:
    """Add an in-memory bytes payload as a member of a tarfile."""
    import io  # noqa: PLC0415
    import tarfile  # noqa: PLC0415
    import time  # noqa: PLC0415

    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = time.time()
    tar.addfile(info, io.BytesIO(data))


def cmd_bundle(
    thread_id: str,
    output: str | None = None,
    all_files: bool = False,
    since: str | None = None,
    no_files: bool = False,
) -> None:
    """Bundle a thread's transcript + workspace files into a portable tarball.

    The bundle is the answer to "how does a user get their research back":
    one ``<thread_id>.tgz`` containing the transcript (state + revision
    history), every file the agent wrote into a convention dir during the
    run (filtered by mtime > thread.created_at, or ``--all`` / ``--since``
    overrides), and a ``MANIFEST.json`` describing the contents.

    Works OFFLINE: if the Agent Protocol server is down, the transcript is
    read from the on-disk thread store + the per-thread ``.meta.json`` that
    ``pux direct`` writes (gap 2 fix). The MANIFEST records which path was
    used so a consumer knows whether to trust the transcript as canonical
    (server) or as a last-known-good snapshot (disk)."""
    import json as _json  # noqa: PLC0415
    import tarfile  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    project = _project_root()
    transcript = _fetch_thread_transcript(thread_id)
    state = transcript.get("state") or {}
    history = transcript.get("history") or []
    created_at = state.get("created_at") if isinstance(state, dict) else None
    agent_id = state.get("agent_id") if isinstance(state, dict) else None

    # Determine mtime floor for the file scan.
    if all_files:
        since_ts = 0.0
    elif since:
        since_ts = _parse_iso8601(since)
    elif created_at:
        since_ts = _parse_iso8601(created_at)
    else:
        # No anchor — last 24 h (avoids accidentally bundling the entire
        # workspace on a thread we know nothing about).
        since_ts = __import__("time").time() - 86400.0

    file_records: list[dict[str, Any]] = []
    files: list[Path] = []
    if not no_files:
        for sub in WORKSPACE_DIRS:
            root = project / sub
            if not root.exists():
                continue
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                if since_ts and p.stat().st_mtime < since_ts:
                    continue
                # Skip the bundle's own output if it's already in the tree
                # (defensive — keeps `pux bundle X` idempotent).
                if p.suffix == ".tgz" and thread_id in p.name:
                    continue
                files.append(p)

    out_path = Path(output) if output else Path(f"{thread_id}.tgz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tar:
        # Workspace files first (preserve paths relative to project root).
        for p in files:
            try:
                arcname = p.relative_to(project)
            except ValueError:
                arcname = Path(p.name)
            tar.add(p, arcname=str(arcname))
            st = p.stat()
            file_records.append({
                "path": str(arcname),
                "size": st.st_size,
                "mtime": _iso8601(st.st_mtime),
            })

        # Transcript (state + revision history, JSON-serialized).
        transcript_bytes = _json.dumps(
            {
                "thread_id": thread_id,
                "agent_id": agent_id,
                "created_at": created_at,
                "state": state,
                "history_revisions": len(history),
                "history": history,
            },
            indent=2,
            default=str,
        ).encode()
        _add_bytes_to_tar(tar, "transcript.json", transcript_bytes)

        # MANIFEST (written last so it carries the full file list).
        manifest = {
            "bundle_schema": "pux-bundle/1",
            "thread_id": thread_id,
            "agent_id": agent_id,
            "created_at": created_at,
            "bundled_at": _now_iso(),
            "transcript_source": transcript.get("_source"),
            "source_server": PUX_API_URL,
            "project_root": str(project),
            "since": _iso8601(since_ts) if since_ts else None,
            "history_revisions": len(history),
            "file_count": len(file_records),
            "files": file_records,
        }
        _add_bytes_to_tar(tar, "MANIFEST.json", _json.dumps(manifest, indent=2).encode())

    size = out_path.stat().st_size
    print(f"bundle:  {out_path} ({size:,} bytes)")
    print(f"  thread:     {thread_id}  agent: {agent_id or '(unknown)'}")
    print(f"  files:      {len(file_records)}  "
          f"(filtered by mtime > {_iso8601(since_ts) if since_ts else 'n/a'})")
    print(f"  transcript: {len(history)} revisions via {transcript.get('_source')}")
    if transcript.get("_source") != "server":
        print("  (transcript is from disk — start the AP server for the canonical copy)")


def cmd_jobs_run(org: str, job: str | None) -> None:
    body: dict[str, Any] = {}
    if job:
        body["job"] = job
    res = _post(f"/jobs/{org}/run", **body)
    jobs = res.get("jobs", [])
    if not jobs:
        print(res.get("message", "no jobs"))
        return
    failed = [j for j in jobs if j["status"] != "ok"]
    for j in jobs:
        status_icon = "ok" if j["status"] == "ok" else "FAIL"
        err = f"  error={j['error'][:120]}" if j.get("error") else ""
        print(f"  {j['name']:<24} {status_icon:<6} {j['duration']}s{err}")
    print(f"\n{len(jobs)} jobs run, {len(failed)} failed")
    if failed:
        raise SystemExit(1)


def cmd_jobs_status(org: str) -> None:
    res = _get(f"/jobs/{org}/status")
    jobs = res.get("jobs", [])
    if not jobs:
        print(res.get("message", "no jobs declared"))
        return
    print(f"{'NAME':<24} {'SCRIPT':<40} {'TIMEOUT':<8} DESCRIPTION")
    for j in jobs:
        timeout = f"{j['timeout']}s" if j["timeout"] else "none"
        print(f"  {j['name']:<22} {j['script']:<40} {timeout:<8} {j.get('description', '')}")
def cmd_capabilities_list(org: str | None, kind: str | None) -> None:
    """Print the merged ``CapabilityIndex`` — the ONE discovery view over the
    capability fleet (tools / skills / mcp / middleware / jobs), grouped by
    ``kind`` then ``provenance`` (the unified vocabulary from CU-2).

    Fleet-wide by default; ``--org`` folds in the org's declared channels
    (declared tools, authored lib/ functions, skill roots, jobs); ``--kind``
    filters to one surface. Docker-free — reads catalogs + declarations only."""
    from pux_harness.agent.capabilities import CapabilityIndex

    rows = CapabilityIndex.load(org)
    if kind:
        rows = [r for r in rows if r.kind == kind]
    scope = f"org {org!r}" if org else "fleet-wide"
    if not rows:
        print(f"no capabilities ({scope})")
        return
    by_kind: dict[str, list[Any]] = {}
    for r in rows:
        by_kind.setdefault(r.kind, []).append(r)
    print(f"capabilities ({scope}): {len(rows)} rows")
    for k in sorted(by_kind):
        provs = sorted({r.provenance for r in by_kind[k]})
        print(f"  {k:<11} {len(by_kind[k]):>3}  [{', '.join(provs)}]")
        for r in sorted(by_kind[k], key=lambda x: x.ref):
            print(f"      {r.ref}")


def cmd_prompt_show(
    org: str, scope: str, raw: bool, project_root: str | None,
    with_ask_user: bool = False, with_interpreter: bool = False,
) -> None:
    """Render the assembled system prompt for ``org`` with provenance.

    ``--scope supervisor`` (default) shows the CTO prompt (4 parts + any extras).
    ``--scope subagent:<slug>`` shows one subagent's prompt (3 parts + any extras).
    ``--raw`` prints just the assembled text (no part labels).
    ``--with-ask-user`` / ``--with-interpreter`` simulate the runtime-on state
    for the two conditional parts (preview what they'd emit over a real transport).

    Docker-free: walks the same part registries (``SUPERVISOR_PROMPT_PARTS`` /
    ``SUBAGENT_PROMPT_PARTS``) that ``assemble_prompt`` uses at runtime, but
    statically. See ``docs/prompt-system.md``.
    """
    from pathlib import Path

    from pux_harness.agent.orgs import _orgs_dir
    from pux_harness.agent.prompt_show import show_subagent, show_supervisor

    root = Path(project_root) if project_root else _orgs_dir().parent
    try:
        if scope.startswith("subagent:"):
            slug = scope.split(":", 1)[1]
            print(show_subagent(org, slug, root, raw=raw))
        else:
            print(show_supervisor(
                org, root, raw=raw,
                ask_user=with_ask_user, interpreter=with_interpreter,
            ))
    except FileNotFoundError as exc:
        import sys

        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_org_chain(org: str, project_root: str | None) -> None:
    """Print the inheritance chain for ``org``: the extends-chain (root→child),
    which files each org in the chain contributes, and the per-file merge rule
    that applies. Read-only introspection — no behavior change."""
    from pathlib import Path

    from pux_harness.agent.orgs import _orgs_dir
    from pux_harness.agent.org_chain import render_org_chain

    root = Path(project_root) if project_root else _orgs_dir().parent
    try:
        print(render_org_chain(org, root))
    except FileNotFoundError as exc:
        import sys

        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


# --- Unified CLI dispatcher ---------------------------------------------------


def main() -> None:
    # Bootstrap FIRST, before argparse: some argparse defaults read os.environ
    # at parser-build time (e.g. ``--org`` default = $PUX_ORG), so ``./.env`` must
    # be loaded NOW for those defaults to see it. This is the shared kit seam
    # (``pux_harness.kit.bootstrap_env_and_logging``) — the SAME function
    # direct/acp and exported runners use, so pux is seamless when run
    # from a FOREIGN codebase: the consumer's ``.env`` is picked up without an
    # ``export``. ``pin_stderr=False`` here (direct logs to stdout fine);
    # ``pux acp`` re-invokes with ``pin_stderr=True`` to defend its stdio wire.
    from pux_harness.kit import bootstrap_env_and_logging  # noqa: PLC0415

    bootstrap_env_and_logging()
    ap = argparse.ArgumentParser(
        prog="pux",
        description="Pux — deepagents-based agent orchestrator.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Server-mode commands. (The Agent Protocol HTTP server is no longer a `pux`
    # subcommand — prod is Aegra via scripts/start_pux_aegra.sh, dev is
    # `langgraph dev` / `aegra dev`. `pux` builds orgs in-process for ACP/TUI/direct.)
    p_acp = sub.add_parser("acp", help="ACP stdio server for editor integration")
    p_acp.add_argument("--org", default=os.environ.get("PUX_ORG", "general"))
    _add_tier_flags(p_acp)

    # MCP server (FastMCP SSE wrapper over the Agent Protocol). Port via
    # PUX_MCP_PORT (default 9987); requires the Agent Protocol server running
    # (prod: `aegra serve`; dev: `langgraph dev` / `aegra dev`).
    sub.add_parser("mcp", help="FastMCP server (SSE) wrapping the Agent Protocol")

    # TUI launcher (dcode wrapper)
    p_tui = sub.add_parser("tui", help="launch dcode TUI with org branding + agents")
    p_tui.add_argument("--org", default=os.environ.get("PUX_ORG", "general"))
    p_tui.add_argument("--list-orgs", action="store_true", help="list available orgs and exit")
    p_tui.add_argument("--model", default=None,
                       help="dcode model (provider:model), e.g. pux-openai:kimi-k2.7-code")
    p_tui.add_argument("-y", "--auto-approve", action="store_true",
                       help="auto-approve tool calls (interactive); always on when task given")
    p_tui.add_argument("-S", "--shell-allow", default=None,
                       help="shell allow-list (dcode -S); headless defaults to 'recommended'")
    p_tui.add_argument("task", nargs="?", default=None, help="optional direct task (headless)")
    _add_tier_flags(p_tui)

    # In-process runner
    p_dir = sub.add_parser("direct", help="in-process deepagents runner (no server)")
    p_dir.add_argument("--org", default="general")
    p_dir.add_argument("--task", required=True,
                       help="the objective ONLY — never embed data paths here; "
                            "use --data for the data folder")
    p_dir.add_argument("--data", default=None, metavar="DIR",
                       help="data folder for the org to process. Sets DATA_DIR "
                            "(absolute) in the agent's environment so the org's "
                            "pipeline can preprocess it — the path never enters "
                            "the task/prompt. This is the structural input hand-off.")
    p_dir.add_argument("--rubric", default=None)
    p_dir.add_argument("--recursion-limit", type=int, default=60)
    p_dir.add_argument("--thread", default=None,
                       help="continue an existing thread id (resume in-process)")
    _add_tier_flags(p_dir)

    # Sandbox lifecycle
    p_sb = sub.add_parser("sandbox", help="Docker sandbox lifecycle")
    p_sb.add_argument(
        "action",
        choices=[
            "start", "stop", "status", "ensure",
            "pause", "unpause",           # session preservation (no teardown)
            "dump-persist",
        ],
        metavar="CMD",
    )
    p_sb.add_argument(
        "--output", "-o", default=None,
        help="dump-persist output path (default: ./sandbox-<id>-persist-<ts>.tgz)",
    )

    # Client-mode commands
    sub.add_parser("agents", help="list orgs (agents)")

    p_disp = sub.add_parser("dispatch", help="ephemeral blocking run on an org")
    p_disp.add_argument("--org", default="general")
    p_disp.add_argument("--recursion-limit", type=int, default=60)
    p_disp.add_argument("--rubric", default=None)
    p_disp.add_argument("task")

    p_res = sub.add_parser("resume", help="list recent threads")
    p_res.add_argument("--org", default=None)

    p_show = sub.add_parser("show", help="show a thread's last message")
    p_show.add_argument("thread_id")

    p_hist = sub.add_parser("history", help="show a thread's revision history")
    p_hist.add_argument("thread_id")

    p_run = sub.add_parser("run", help="background run on an existing thread")
    p_run.add_argument("--recursion-limit", type=int, default=60)
    p_run.add_argument("thread_id")
    p_run.add_argument("task")

    p_wait = sub.add_parser("wait", help="block for a background run's output")
    p_wait.add_argument("run_id")

    # Bundle — package a thread's transcript + artifacts + memos into one
    # tarball (gap 1+2+7+8 of the persistence audit: the canonical "give me my
    # research back" command). Server is tried first for the live transcript;
    # falls back to the on-disk thread store + per-thread meta.json if the
    # server is down (the typical "I stopped everything, now archive" case).
    p_bundle = sub.add_parser(
        "bundle",
        help="bundle a thread's transcript + artifacts + memos into a tarball",
    )
    p_bundle.add_argument("thread_id")
    p_bundle.add_argument(
        "--output", "-o", default=None,
        help="output path (default: ./<thread_id>.tgz)",
    )
    p_bundle.add_argument(
        "--all", action="store_true",
        help="include every file in the convention dirs (ignore mtime filter)",
    )
    p_bundle.add_argument(
        "--since",
        default=None,
        help="only include files newer than this ISO 8601 timestamp (overrides thread created_at)",
    )
    p_bundle.add_argument(
        "--no-files", action="store_true",
        help="transcript only — skip the workspace file scan",
    )

    # Diagnostic / offline commands
    sub.add_parser("list", help="list discovered orgs and their agents")

    p_check = sub.add_parser("check", help="docker-exec + specialist smoke test (no model)")
    p_check.add_argument("--org", default="general")

    sub.add_parser("check-contract", help="validate the declarative org contract")

    p_cp = sub.add_parser("check-policy", help="resolve + report an org's policy")
    p_cp.add_argument("--org", default="general")

    # Jobs subcommands
    p_jobs = sub.add_parser("jobs", help="run or show prep jobs")
    jobs_sub = p_jobs.add_subparsers(dest="jobs_cmd", required=True)

    p_jr = jobs_sub.add_parser("run", help="run prep jobs in the sandbox")
    p_jr.add_argument("--org", required=True)
    p_jr.add_argument("--job", default=None)

    p_js = jobs_sub.add_parser("status", help="show declared prep jobs")
    p_js.add_argument("--org", required=True)

    # Capabilities — the unified fleet/per-org capability catalog (CU-2).
    # `pux capabilities list` prints the merged CapabilityIndex: ONE discovery
    # view over tools / skills / mcp / middleware / jobs.
    p_caps = sub.add_parser(
        "capabilities", help="inspect the unified capability catalog")
    caps_sub = p_caps.add_subparsers(dest="caps_cmd", required=True)
    p_cl = caps_sub.add_parser(
        "list", help="list capabilities (fleet-wide, or one org's resolved channels)")
    p_cl.add_argument(
        "--org", default=None,
        help="fold in this org's declared channels (declared tools, lib/ "
             "functions, skill roots, jobs); default = fleet-wide catalog only")
    p_cl.add_argument(
        "--kind", default=None,
        choices=["tool", "skill", "mcp", "middleware", "job"],
        help="filter to one capability surface")

    # Prompt introspection (D8) — render the assembled system prompt with
    # provenance. Docker-free: walks the same part registries the runtime uses.
    p_prompt = sub.add_parser(
        "prompt", help="inspect the assembled system prompt")
    prompt_sub = p_prompt.add_subparsers(dest="prompt_cmd", required=True)
    p_ps = prompt_sub.add_parser(
        "show",
        help="render the assembled prompt with part-by-part provenance")
    p_ps.add_argument("--org", required=True, help="the org to render")
    p_ps.add_argument(
        "--scope", default="supervisor",
        help="'supervisor' (default) or 'subagent:<slug>' for one subagent")
    p_ps.add_argument(
        "--raw", action="store_true",
        help="print just the assembled text (no part labels)")
    p_ps.add_argument(
        "--project-root", default=None,
        help="repo root containing orgs/ (default: auto-detected)")
    p_ps.add_argument(
        "--with-ask-user", action="store_true",
        help="simulate ask_user ACTIVE (preview the HITL suffix that would "
             "emit over a turn-based transport: direct/tui/acp/agui)")
    p_ps.add_argument(
        "--with-interpreter", action="store_true",
        help="simulate dynamic_dispatch ACTIVE (preview the eval-tool suffix "
             "that would emit when CodeInterpreterMiddleware is mounted)")

    # Org inheritance introspection — the extends-chain, per-file merge rules,
    # and which files each org in the chain contributes. Read-only.
    p_org = sub.add_parser(
        "org", help="inspect org inheritance and structure")
    org_sub = p_org.add_subparsers(dest="org_cmd", required=True)
    p_oc = org_sub.add_parser(
        "chain",
        help="print the extends-chain + per-file merge rules for an org")
    p_oc.add_argument("--org", required=True, help="the org to inspect")
    p_oc.add_argument(
        "--project-root", default=None,
        help="repo root containing orgs/ (default: auto-detected)")

    # Pack — manifest-driven default-deny portable archive (successor to the
    # deprecated `export`). ``pux pack`` is the validated path; the legacy
    # ``pux export`` verb HARD-ERRORS (Decision 5: no silent alias).
    p_pack = sub.add_parser("pack", help="pack org as a manifest-driven portable archive")
    p_pack.add_argument("--org", required=True)
    p_pack.add_argument("--output", "-o", default=None,
                        help="output path (default: <org>.tar.gz)")
    p_pack.add_argument(
        "--project-root", default=None,
        help="tree containing orgs/ + root AGENTS.md to pack FROM (default: "
             "$PUX_PROJECT_ROOT or CWD). Lets a standalone consumer app pack "
             "an org that lives in ITS OWN tree, not the orchestrator's.")
    p_pack.add_argument(
        "--oci", action="store_true",
        help="also emit a layered OCI artifact (oras) — the content-addressed, "
             "registry-pushable form; provenance.json records SHA-256 layer "
             "digests (tamper-evident). Fail-clear if oras is absent. (P5)")
    p_pack.add_argument(
        "--oci-layout", default=None,
        help="oci-layout dir to emit into (default: sibling of the tarball, "
             "<org>.oci). Implies --oci.")

    # `export` is retained as a PARSER so `pux export ...` yields the clear
    # deprecation message in dispatch (not an opaque argparse "invalid choice").
    # Its dispatch HARD-ERRORS — there is NO silent alias to `pack`.
    p_export = sub.add_parser(
        "export", help="DEPRECATED: use `pack` (hard error — no alias)")
    p_export.add_argument("--org", required=True)
    p_export.add_argument("--output", "-o", default=None)
    p_export.add_argument("--project-root", default=None)

    # Dynamic-function lifecycle (OPERATOR commands — not agent tools). Graduate
    # an agent-authored lib function to git-tracked sandbox/ (c->b), or retire
    # one to lib/.archive/ (reversible). See sandbox/tools/dynamic.py.
    p_promote = sub.add_parser(
        "promote-function",
        help="graduate a lib function to git-tracked sandbox/functions/ (c->b)")
    p_promote.add_argument("--org", required=True)
    p_promote.add_argument("name", help="function name to promote")

    p_archive = sub.add_parser(
        "archive-function",
        help="retire a lib function to lib/.archive/ (reversible)")
    p_archive.add_argument("--org", required=True)
    p_archive.add_argument("name", help="function name to archive")

    # Lock — regenerate org.lock.yaml: pin github MCP refs to commit SHAs
    # (best-effort git ls-remote) + snapshot declared pip/apt deps. The lock
    # travels with the org (committed by default — Decision 4) so its dep set is
    # reproducible. Always writable (offline → unresolved refs recorded, not
    # fatal). See pux_harness/lockfile.py.
    p_lock = sub.add_parser(
        "lock", help="regenerate org.lock.yaml (pip + MCP pin snapshot)")
    p_lock.add_argument("--org", required=True)
    p_lock.add_argument(
        "--project-root", default=None,
        help="tree containing orgs/ to lock against (default: "
             "$PUX_PROJECT_ROOT or CWD)")

    # Verify a packed OCI layout (P5 close-the-loop: ``emit`` records the tamper
    # anchor, ``verify`` checks it). Stdlib-only — reads index.json + blobs directly,
    # no ``oras`` needed at verify time (a trust op minimizes its toolchain). Checks
    # manifest + blob integrity, optional trust anchors, and optional source
    # attestation (does the packed library match ``orgs/<org>/lib/`` right now?).
    p_verify = sub.add_parser(
        "verify",
        help="verify a packed OCI layout's integrity (manifest + layer digests)")
    p_verify.add_argument(
        "--oci", required=True,
        help="oci-layout dir to verify (the <org>.oci from `pux pack --oci`)")
    p_verify.add_argument(
        "--org", default=None,
        help="enable source attestation: re-derive the library layer from "
             "orgs/<org>/lib/ and confirm it matches the packed layer")
    p_verify.add_argument(
        "--source-root", default=None,
        help="tree containing orgs/ for --org attestation (default: "
             "$PUX_PROJECT_ROOT or CWD)")
    p_verify.add_argument(
        "--expected", default=None,
        help="trusted manifest digest (sha256:...) to assert against")
    p_verify.add_argument(
        "--expected-library", default=None,
        help="trusted agent-library layer digest (sha256:...) to assert against")

    args = ap.parse_args()
    _apply_tier_flag(args)  # PUX_TIER for in-process model resolution (acp/direct/tui)

    # --- ACP stdio ---
    if args.cmd == "acp":
        from pux_harness.acp import run_acp

        run_acp(args.org)

    # --- MCP server (FastMCP SSE over the Agent Protocol) ---
    elif args.cmd == "mcp":
        from pux_harness.mcp_server import main as _mcp

        _mcp()

    # --- TUI launcher ---
    elif args.cmd == "tui":
        from pux_harness.tui import list_orgs, run_tui

        if args.list_orgs:
            list_orgs()
        else:
            run_tui(args.org, args.task, args.model,
                    args.auto_approve, args.shell_allow)

    # --- In-process runner ---
    elif args.cmd == "direct":
        from pux_harness.main import run_direct

        # --data: the structural data-folder hand-off. Sets DATA_DIR (absolute)
        # in the process environment so the org's pipeline reads it via
        # ``$DATA_DIR`` — the path NEVER enters the task/prompt string.
        # This is how Pux keeps the objective clean: --task is the mission,
        # --data is the input. They are separate concerns.
        data_dir = getattr(args, "data", None)
        if data_dir:
            # Relative to project root so it resolves in both host (CWD=repo)
            # and container (CWD=/sandbox/workspace) contexts.
            abs_path = os.path.abspath(data_dir)
            try:
                os.environ["DATA_DIR"] = os.path.relpath(abs_path, os.getcwd())
            except ValueError:
                os.environ["DATA_DIR"] = abs_path

        run_direct(args.org, args.task, args.rubric, args.recursion_limit, args.thread)

    # --- Sandbox ---
    elif args.cmd == "sandbox":
        from pux_harness.main import run_sandbox

        run_sandbox(args.action, output=args.output)

    # --- Client mode (requires running server) ---
    elif args.cmd == "agents":
        cmd_agents()
    elif args.cmd == "dispatch":
        cmd_dispatch(args.org, args.task, args.recursion_limit, args.rubric)
    elif args.cmd == "resume":
        cmd_resume(args.org)
    elif args.cmd == "show":
        cmd_show(args.thread_id)
    elif args.cmd == "history":
        cmd_history(args.thread_id)
    elif args.cmd == "run":
        cmd_run(args.thread_id, args.task, args.recursion_limit)
    elif args.cmd == "wait":
        cmd_wait(args.run_id)
    elif args.cmd == "bundle":
        cmd_bundle(
            args.thread_id,
            output=args.output,
            all_files=args.all,
            since=args.since,
            no_files=args.no_files,
        )

    # --- Diagnostic / offline ---
    elif args.cmd == "list":
        from pux_harness.main import run_list_orgs

        run_list_orgs()
    elif args.cmd == "check":
        from pux_harness.main import run_check_smoke

        run_check_smoke(args.org)
    elif args.cmd == "check-contract":
        from pux_harness.main import run_check_contract

        run_check_contract()
    elif args.cmd == "check-policy":
        from pux_harness.main import run_check_policy

        run_check_policy(args.org)

    # --- Jobs ---
    elif args.cmd == "jobs":
        if args.jobs_cmd == "run":
            cmd_jobs_run(args.org, args.job)
        elif args.jobs_cmd == "status":
            cmd_jobs_status(args.org)

    # --- Capabilities (unified catalog; CU-2) ---
    elif args.cmd == "capabilities":
        if args.caps_cmd == "list":
            cmd_capabilities_list(args.org, args.kind)

    # --- Prompt introspection (D8) ---
    elif args.cmd == "prompt":
        if args.prompt_cmd == "show":
            cmd_prompt_show(
                args.org, args.scope, args.raw, args.project_root,
                getattr(args, "with_ask_user", False),
                getattr(args, "with_interpreter", False),
            )

    # --- Org inheritance introspection ---
    elif args.cmd == "org":
        if args.org_cmd == "chain":
            cmd_org_chain(args.org, args.project_root)

    # --- Pack (manifest-driven default-deny archive; successor to `export`) ---
    elif args.cmd == "pack":
        from pathlib import Path as _Path
        from pux_harness.oci import OciError
        from pux_harness.pack import pack_org
        from pux_harness.pack_hooks import PackHookError

        output = _Path(args.output) if args.output else None
        project_root = _Path(args.project_root) if args.project_root else None
        # --oci-layout implies --oci; an explicit layout Path is passed through,
        # else a bool (pack_org derives a sibling <org>.oci layout). (P5)
        oci_arg = _Path(args.oci_layout) if args.oci_layout else bool(args.oci)
        try:
            result = pack_org(args.org, output, project_root=project_root, oci=oci_arg)
        except PackHookError as exc:
            # A pack-time hook REFUSED the pack (P4) — a syntax-broken function
            # or a leaked secret. Print the failing hook + its findings + every
            # hook run so far, then exit non-zero (no archive written).
            import sys as _sys
            _sys.stderr.write(
                f"pack REFUSED — hook {exc.result.name!r} failed for org "
                f"{args.org!r} (no archive written):\n"
            )
            for finding in exc.result.findings:
                _sys.stderr.write(f"  - {finding}\n")
            _sys.stderr.write("  hooks run:\n")
            for r in exc.all_results:
                flag = "OK" if r.ok else "FAIL"
                _sys.stderr.write(f"    [{flag}] {r.name}\n")
            raise SystemExit(1)
        except OciError as exc:
            # pack_org writes the .tar.gz BEFORE the OCI emit, so the validated
            # archive still shipped — only the ADDITIONAL OCI artifact was refused
            # (typically ``oras`` absent). Honest: report both, exit non-zero
            # since the requested --oci output was not produced.
            import sys as _sys
            _sys.stderr.write(
                f"oci emit refused for org {args.org!r} (the .tar.gz pack is "
                f"unaffected — OCI is additional):\n  {exc}\n")
            raise SystemExit(1)
        print(f"packed {args.org!r} -> {result}")
        # Print summary
        import tarfile as _tar
        with _tar.open(result, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            print(f"  {len(members)} files")
            manifest_member = f"{args.org}/manifest.json"
            if any(m.name == manifest_member for m in tar.getmembers()):
                mf = tar.extractfile(manifest_member)
                if mf:
                    manifest = json.loads(mf.read())
                    cats = manifest.get("categories", {})
                    for cat, items in cats.items():
                        if items:
                            print(f"  {cat}: {len(items)} files")
                    # Surface the hook provenance (P4) — the validation gate's
                    # audit record that every shipped file passed AST + gitleaks.
                    prov = manifest.get("provenance")
                    if prov and prov.get("hooks"):
                        names = ", ".join(
                            h["name"] for h in prov["hooks"] if h.get("ok")
                        )
                        print(f"  hooks OK: {names}")
        # Surface the OCI artifact (P5) — the layered, content-addressed form's
        # manifest digest + the agent-library layer digest (the tamper anchor).
        if oci_arg:
            layout = (_Path(args.oci_layout) if args.oci_layout
                      else result.parent / f"{args.org}.oci")
            prov_path = layout / "provenance.json"
            if prov_path.is_file():
                oci_prov = json.loads(prov_path.read_text())
                art = oci_prov.get("artifact", {})
                print(f"  oci artifact: {art.get('digest')}  (layout: {layout})")
                lib = next((layer for layer in oci_prov.get("layers", [])
                            if layer.get("type") == "agent-library"), None)
                if lib:
                    print(f"  library layer (integrity): {lib.get('digest')}  "
                          f"[{lib.get('size')} B]")

    # --- Verify a packed OCI layout (P5: emit records the anchor, verify checks it) ---
    elif args.cmd == "verify":
        from pathlib import Path as _Path
        from pux_harness.kit._paths import project_root as _project_root
        from pux_harness.oci import OciError, verify_oci_layout

        layout = _Path(args.oci)
        # Source attestation needs a tree to re-derive lib/ from: an explicit
        # --source-root wins; else the live project root when --org is given.
        source_root = (_Path(args.source_root) if args.source_root
                       else (_project_root() if args.org else None))
        try:
            result = verify_oci_layout(
                layout, org=args.org, source_root=source_root,
                expected=args.expected, expected_library=args.expected_library)
        except OciError as exc:
            import sys as _sys
            _sys.stderr.write(f"verify failed for {layout}: {exc}\n")
            raise SystemExit(1)
        print(result.summary())
        raise SystemExit(0 if result.ok else 1)

    # --- `export` is DEPRECATED → HARD ERROR (Decision 5: no silent alias) ---
    # The verb is retained as a parser so `pux export ...` yields this clear
    # migration message (not an opaque argparse "invalid choice"). It never
    # reaches pack logic — using it is always a deliberate, failing act, which
    # is what forces updated scripts/muscle memory off the un-validated path.
    elif args.cmd == "export":
        import sys as _sys
        _sys.stderr.write(
            "`pux export` has been deprecated and replaced by `pux pack` to "
            "enforce manifest validation, secrets scanning, and OCI packaging "
            "standards. Please use `pux pack` instead.\n"
        )
        raise SystemExit(1)

    # --- Dynamic-function lifecycle (operator commands) ---
    elif args.cmd == "promote-function":
        from pux_harness.agent.orgs import _org_path
        from pux_harness.sandbox.tools import dynamic as _dyn

        res = _dyn.promote_function(_org_path(args.org) / "lib", args.name)
        _print_block("promote-function", json.dumps(res, indent=2))
        if not res.get("success"):
            raise SystemExit(1)

    elif args.cmd == "archive-function":
        from pux_harness.agent.orgs import _org_path
        from pux_harness.sandbox.tools import dynamic as _dyn

        res = _dyn.archive_function(_org_path(args.org) / "lib", args.name)
        _print_block("archive-function", json.dumps(res, indent=2))
        if not res.get("success"):
            raise SystemExit(1)

    # --- Lock — regenerate org.lock.yaml (Decision 4) ---
    elif args.cmd == "lock":
        from pathlib import Path as _Path
        from pux_harness.lockfile import lock_org

        project_root = _Path(args.project_root) if args.project_root else None
        result = lock_org(args.org, project_root=project_root)
        print(f"locked {args.org!r} -> {result}")
        # The on-disk file is the source of truth — re-read for a faithful
        # summary (best-effort SHA resolution may have left refs unresolved).
        import yaml as _yaml
        data = _yaml.safe_load(result.read_text()) or {}
        deps = data.get("dependencies", {})
        mcp = deps.get("mcp_servers", [])
        resolved = sum(1 for s in mcp if s.get("resolved"))
        print(f"  mcp_servers: {len(mcp)} ({resolved} resolved to a commit SHA)")
        for s in mcp:
            sha = s.get("sha") or "(unresolved)"
            print(f"    {s['name']}: {s['repo']}@{s['version']} -> {sha}")
        print(f"  pip: {len(deps.get('pip', []))} declared")
        print(f"  apt: {len(deps.get('apt', []))} declared")


if __name__ == "__main__":
    main()
