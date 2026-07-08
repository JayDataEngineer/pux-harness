"""Unified ``pux`` CLI — replaces ``bin/pux``, ``main.py``, ``cli.py``, ``acp.py``.

Usage:
  pux serve                    start the Agent Protocol server (uvicorn)
  pux direct                   in-process deepagents runner (no server)
  pux acp [--org X]            ACP stdio server for editor integration
  pux tui [--org X]            launch dcode TUI with org branding + agents
  pux sandbox <cmd>            Docker sandbox lifecycle (start/stop/status/ensure)
  pux agents                   list orgs (agents)
  pux dispatch [--org X] ...   ephemeral blocking run
  pux resume [--org X]         list recent threads
  pux show <thread_id>         thread state (last message)
  pux history <thread_id>      revision history
  pux run <thread_id> ...      background run on an existing thread
  pux wait <run_id>            block for a background run's output
  pux list                     list discovered orgs
  pux check [--org X]          docker-exec + specialist smoke test (no model)
  pux check-contract           validate the declarative org contract
  pux check-policy [--org X]   resolve + report an org's policy
  pux jobs run --org X         run prep jobs inside the sandbox
  pux jobs status --org X      show declared prep jobs
  pux export --org X           export org as standalone portable archive
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
             f"({exc}). Start it with: pux serve")
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
             f"({exc}). Start it with: pux serve")
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
    subparser. ``serve``/``acp``/``direct``/``tui`` build orgs (and so resolve
    models) in THIS process, so a tier flag here is authoritative. Client
    commands (``dispatch``/``run``) talk to an already-running server that
    resolved its tier at startup — a client flag would be misleading, so they
    don't get one. ``--fast`` is sugar for ``--tier fast``."""
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


def cmd_resume(org: str | None) -> None:
    body: dict[str, Any] = {}
    if org:
        body["agent_id"] = org
    threads = _post("/threads/search", **body)
    if not threads:
        print("(no threads)")
        return
    for t in threads:
        print(f"  {t['thread_id']}   [agent] {t['agent_id']:<16} {t['created_at']}")


def cmd_show(thread_id: str) -> None:
    state = _get(f"/threads/{thread_id}")
    vals = state.get("values") or {}
    msgs = vals.get("messages") or []
    last = msgs[-1] if msgs else {}
    content = last.get("content") if isinstance(last, dict) else last
    _print_block(f"thread {thread_id} (agent={state.get('agent_id')}, "
                 f"status={state.get('status')})",
                 content or "(no messages)")


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


# --- Unified CLI dispatcher ---------------------------------------------------


def main() -> None:
    # Bootstrap FIRST, before argparse: some argparse defaults read os.environ
    # at parser-build time (e.g. ``--org`` default = $PUX_ORG), so ``./.env`` must
    # be loaded NOW for those defaults to see it. This is the shared kit seam
    # (``pux_harness.kit.bootstrap_env_and_logging``) — the SAME function
    # serve/direct/acp and exported runners use, so pux is seamless when run
    # from a FOREIGN codebase: the consumer's ``.env`` is picked up without an
    # ``export``. ``pin_stderr=False`` here (serve/direct log to stdout fine);
    # ``pux acp`` re-invokes with ``pin_stderr=True`` to defend its stdio wire.
    from pux_harness.kit import bootstrap_env_and_logging  # noqa: PLC0415

    bootstrap_env_and_logging()
    ap = argparse.ArgumentParser(
        prog="pux",
        description="Pux — deepagents-based agent orchestrator.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Server-mode commands
    p_serve = sub.add_parser("serve", help="start the Agent Protocol server (uvicorn)")
    _add_tier_flags(p_serve)

    p_acp = sub.add_parser("acp", help="ACP stdio server for editor integration")
    p_acp.add_argument("--org", default=os.environ.get("PUX_ORG", "general"))
    _add_tier_flags(p_acp)

    # MCP server (FastMCP SSE wrapper over the Agent Protocol). Port via
    # PUX_MCP_PORT (default 9987); requires `pux serve` running.
    sub.add_parser("mcp", help="FastMCP server (SSE) wrapping the Agent Protocol")

    # TUI launcher (dcode wrapper)
    p_tui = sub.add_parser("tui", help="launch dcode TUI with org branding + agents")
    p_tui.add_argument("--org", default=os.environ.get("PUX_ORG", "general"))
    p_tui.add_argument("--list-orgs", action="store_true", help="list available orgs and exit")
    p_tui.add_argument("--model", default=None,
                       help="dcode model (provider:model), e.g. opencode-go-openai:kimi-k2.7-code")
    p_tui.add_argument("-y", "--auto-approve", action="store_true",
                       help="auto-approve tool calls (interactive); always on when task given")
    p_tui.add_argument("-S", "--shell-allow", default=None,
                       help="shell allow-list (dcode -S); headless defaults to 'recommended'")
    p_tui.add_argument("task", nargs="?", default=None, help="optional direct task (headless)")
    _add_tier_flags(p_tui)

    # In-process runner
    p_dir = sub.add_parser("direct", help="in-process deepagents runner (no server)")
    p_dir.add_argument("--org", default="general")
    p_dir.add_argument("--task", required=True)
    p_dir.add_argument("--rubric", default=None)
    p_dir.add_argument("--recursion-limit", type=int, default=60)
    p_dir.add_argument("--thread", default=None,
                       help="continue an existing thread id (resume in-process)")
    _add_tier_flags(p_dir)

    # Sandbox lifecycle
    p_sb = sub.add_parser("sandbox", help="Docker sandbox lifecycle")
    p_sb.add_argument("action", choices=["start", "stop", "status", "ensure"], metavar="CMD")

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

    # Export
    p_export = sub.add_parser("export", help="export org as standalone portable archive")
    p_export.add_argument("--org", required=True)
    p_export.add_argument("--output", "-o", default=None,
                          help="output path (default: <org>.tar.gz)")
    p_export.add_argument(
        "--project-root", default=None,
        help="tree containing orgs/ + root AGENTS.md to export FROM (default: "
             "$PUX_PROJECT_ROOT or CWD). Lets a standalone consumer app export "
             "an org that lives in ITS OWN tree, not the orchestrator's.")

    args = ap.parse_args()
    _apply_tier_flag(args)  # PUX_TIER for in-process model resolution (serve/acp/direct/tui)

    # --- Server mode ---
    if args.cmd == "serve":
        from pux_harness.server import main as _serve

        _serve()

    # --- ACP stdio ---
    elif args.cmd == "acp":
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

        run_direct(args.org, args.task, args.rubric, args.recursion_limit, args.thread)

    # --- Sandbox ---
    elif args.cmd == "sandbox":
        from pux_harness.main import run_sandbox

        run_sandbox(args.action)

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

    # --- Export ---
    elif args.cmd == "export":
        from pathlib import Path as _Path
        from pux_harness.export import export_org

        output = _Path(args.output) if args.output else None
        project_root = _Path(args.project_root) if args.project_root else None
        result = export_org(args.org, output, project_root=project_root)
        print(f"exported {args.org!r} -> {result}")
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


if __name__ == "__main__":
    main()
