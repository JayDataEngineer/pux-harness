"""In-process deepagents runner — drive any org against the harness directly.

  uv run pux list                  # discovered orgs + agents
  uv run pux check                 # docker-exec + native specialist smoke (no tokens)
  uv run pux direct --org general --task "..."   # one-shot run

Proves per-org: deepagents drives the pux sandbox through the shared
``BaseSandbox`` (OpenShell by default — native
``ls/read_file/write_file/edit_file/glob/grep/execute`` via the sandbox), the CTO delegates to its specialist via ``task(subagent_type=...)``, and
specialists fall back to those same native tools. The 13 specialist capabilities
(browser/desktop/vision/skills/python) are native ``pux_sandbox_*`` Python tools
too — there is no Go bridge. Prints a message trace + token usage so
cost/output is comparable.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid

from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError

from pux_harness.agent.org_validation import audit_org, has_errors
from pux_harness.agent.graph import build_graph, shared_backend
from pux_harness.agent.observability import build_invoke_config
from pux_harness.agent.profile import default_rubric
from pux_harness.agent.stack import RuntimeFacts, autonomous_from_env
from pux_harness.threads import open_thread_store
from pux_harness.sandbox.tools import (
    LEGACY_TOOL_NAMES,
    NATIVE_FS_TOOLS,
    make_specialist_tools,
)
from pux_harness.agent.orgs import (
    discover_orgs,
    org_agent_slugs,
)


def _build_agent(
    org: str,
    saver,
    mcp_tools: list[BaseTool] | None = None,
):
    """Build the per-org graph against *saver* (the shared persistent
    checkpointer from ``open_thread_store``). ``pux direct`` no longer
    uses an ephemeral ``MemorySaver`` — it shares the same
    ``.pux/agent-protocol.sqlite`` as ``serve`` / ``acp``, so threads survive the
    process and show up in ``pux resume`` / ``pux show``.

    ``direct`` is the interactive CLI runner, so an opted-in ``ask_user`` uses
    the turn-based branch (pose the question + end the turn; the user's next
    input is the answer). ``PUX_AUTONOMOUS`` drops ask_user entirely (headless
    batch runs)."""
    agent = build_graph(
        org,
        checkpointer=saver,
        facts=RuntimeFacts(transport="direct", autonomous=autonomous_from_env()),
        mcp_tools=mcp_tools or (),
    )
    return agent, shared_backend()


def _usage(messages: list) -> dict:
    tot = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for msg in messages:
        meta = getattr(msg, "usage_metadata", None) or {}
        for k in tot:
            tot[k] += meta.get(k, 0) or 0
    return tot


def _trace(messages: list) -> None:
    """One line per message so delegation + tool calls are honestly visible."""
    for i, m in enumerate(messages):
        t = getattr(m, "type", type(m).__name__)
        name = getattr(m, "name", "") or ""
        tool_calls = getattr(m, "tool_calls", None) or []
        content = getattr(m, "content", "")
        if isinstance(content, list):
            content = " | ".join(str(c)[:140] for c in content)
        cstr = str(content).replace("\n", " ")[:200]
        tcstr = ""
        if tool_calls:
            parts = []
            for tc in tool_calls:
                tcname = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
                tcargs = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                parts.append(f"{tcname}({tcargs})")
            tcstr = " TOOLS=[" + "; ".join(parts)[:240] + "]"
        tag = f":{name}" if name else ""
        print(f"  [{i}] {t}{tag}{tcstr}: {cstr}")


async def _recover_partial(agent, config) -> tuple[list, object]:
    """Fetch the partial checkpoint after a ``GraphRecursionError``.

    langgraph persists state at the end of every super-step, so when the step
    cap fires the last checkpoint holds whatever the agent had produced. We
    surface that instead of dying on a bare traceback. Returns ``(messages,
    todos)``; ``( [], None )`` when nothing was checkpointed yet (the caller
    must not assume ``messages`` is non-empty).
    """
    partial = await agent.aget_state(config)
    values = partial.values if partial is not None else {}
    return list(values.get("messages", [])), values.get("todos")


# --- Thread provenance + output discovery (persistence-audit gaps 2-3) --------
#
# Two small but load-bearing helpers that close the "how do I get my run back?"
# gap. ``_write_thread_meta`` drops a tiny JSON file at
# ``<project>/.pux/sessions/<thread_id>.meta.json`` every time ``pux direct``
# starts + finishes a run — this is the THREAD → FILES MAPPING anchor that was
# missing (nothing else records "thread X wrote artifacts/Y.md"). ``pux bundle``
# reads it as a fallback when the server is down. ``_print_output_dirs`` is the
# CLI-hint counterpart: the dispatch/direct output never told the user where
# their files landed — now it does.


def _write_thread_meta(
    thread_id: str,
    org: str,
    task: str,
    status: str,
) -> None:
    """Write/update the per-thread provenance file.

    Idempotent on ``thread_id``; later writes (status=finished) merge into the
    same file. Never raises — provenance is best-effort, not a run-blocking
    concern."""
    import json  # noqa: PLC0415
    from datetime import datetime, timezone  # noqa: PLC0415

    try:
        from pux_harness.kit._paths import project_root  # noqa: PLC0415

        meta_path = project_root() / ".pux" / "sessions" / f"{thread_id}.meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing.update(
            {
                "thread_id": thread_id,
                "org": org,
                "task": task,
                "status": status,
                f"{status}_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        meta_path.write_text(json.dumps(existing, indent=2))
    except OSError:
        # Best-effort — never block the run on provenance write failure.
        pass


def _print_output_dirs(thread_id: str | None = None, stream=None) -> None:
    """Print the resume command + where files live.

    Mirrors ``pux_harness.cli._print_output_dirs`` but callable from main.py
    paths (direct runner, stderr-print) without importing the CLI module
    (avoids a circular import — cli.py imports main.py for run_* dispatch)."""
    from pux_harness.kit._paths import project_root  # noqa: PLC0415

    stream = stream or sys.stdout
    project = project_root()
    if thread_id:
        print(f'resume with:   pux direct --thread {thread_id} --task "<follow-up>"', file=stream)
        print(f'               (or: pux run {thread_id} "<follow-up>")', file=stream)
    print(f"files under:   {project}", file=stream)
    print("  artifacts/  memos/  .pux/sessions/", file=stream)


async def _run(
    org: str,
    task: str,
    recursion_limit: int,
    rubric: str | None = None,
    thread: str | None = None,
) -> None:
    from pux_harness.agent.mcp_client import open_org_mcp  # noqa: PLC0415

    mcp_tools: list[BaseTool] = []
    try:
        mcp_tools = await open_org_mcp(org)
    except ValueError as exc:
        print(f"  [mcp] tool_servers resolution failed: {exc}")
    thread_id = thread or f"{org}-{uuid.uuid4().hex[:8]}"
    # Persistent thread store: direct shares the SAME
    # .pux/agent-protocol.sqlite as serve/acp, so a thread created here is
    # visible to `pux show <id>` / `pux resume`. The thread_id is printed to
    # stderr so stdout (the message trace) stays clean.
    _write_thread_meta(thread_id, org, task, status="running")
    async with open_thread_store() as store:
        agent, backend = _build_agent(org, saver=store.saver, mcp_tools=mcp_tools)
        await store.register_thread(thread_id, org, metadata={"source": "direct", "task": task})
        print(
            f"\n[thread] {thread_id}   (resume via: pux show {thread_id} | pux resume)",
            file=sys.stderr,
        )
        _print_output_dirs(thread_id, stream=sys.stderr)
        # Run prep jobs after the sandbox is up, before the agent loop.
        from pux_harness.sandbox.exec import prepare  # noqa: PLC0415

        job_results = prepare(org, sandbox=shared_backend())
        if job_results:
            failed = [r for r in job_results if r["status"] != "ok"]
            for r in job_results:
                tag = "ok" if r["status"] == "ok" else "FAIL"
                print(
                    f"  [job] {r['name']:<24} {tag} {r['duration']}s"
                    + (f"  {r['error'][:80]}" if r.get("error") else "")
                )
            if failed:
                print(f"\n  {len(failed)} prep job(s) failed (continuing to agent)")
        # Surface the watchable-desktop URL when sandbox.display.watch is on.
        # (OpenShell does not expose a VNC watch URL — this is a no-op stub
        # retained for prompt-output parity; restore when a watch surface lands.)
        _watch = None
        if _watch:
            print(f"  [watch] live desktop → {_watch}")
        print(f"[org] {org}   [task] {task}\n")
        # Arm the org's RubricMiddleware gate. An explicit ``rubric``
        # (the ``--rubric`` override) wins; otherwise fall back to the org's shipped
        # default (profile.yaml ``rubric.default``). No rubric → the gate stays a
        # no-op (upstream RubricMiddleware contract).
        state: dict = {"messages": [{"role": "user", "content": task}]}
        if rubric:
            state["rubric"] = rubric
        else:
            dr = default_rubric(org)
            if dr:
                state["rubric"] = dr
        invoke_config = build_invoke_config(thread_id, recursion_limit, org, transport="direct")
        try:
            result = await agent.ainvoke(state, config=invoke_config)
        except GraphRecursionError:
            # The run hit the step cap (--recursion-limit). The partial
            # checkpoint is still in the checkpointer, so surface what the
            # agent had produced so far instead of dying on a bare traceback.
            # Resume with `pux resume <thread_id>` (same thread).
            print(
                f"\n=== RECURSION LIMIT HIT (--recursion-limit {recursion_limit}) ===",
                file=sys.stderr,
            )
            print(
                "  The agent exceeded its step cap. Partial progress is "
                "surfaced below; resume with `pux resume "
                f"{thread_id}` (same thread) to continue.\n",
                file=sys.stderr,
            )
            partial_messages, todos = await _recover_partial(agent, invoke_config)
            if todos:
                print("=== OPEN TODOS (partial) ===", file=sys.stderr)
                print(f"  {todos}", file=sys.stderr)
            result = {"messages": partial_messages}
    messages = result["messages"]
    print("=== MESSAGE TRACE ===")
    _trace(messages)
    print("\n=== FINAL ANSWER ===")
    # ``messages`` is empty only in the recursion path when nothing checkpointed
    # before the cap — guard so the handler degrades gracefully, not via IndexError.
    if not messages:
        print("(no messages surfaced — the run produced no partial output)")
    else:
        final = messages[-1]
        content = getattr(final, "content", final)
        print(content if content else "(empty content)")
    print("\n=== USAGE ===")
    print(_usage(messages))
    print(f"messages in thread: {len(messages)}")

    # Top-level surface check: only the MAIN agent's tool_calls live in
    # `messages` — subagent calls run in a nested thread the main trace can't
    # see. So this proves the CTO didn't leak a legacy tool, but the real
    # native-flip proof for the subagent is `backend.execute_log` below.
    # ``native`` + ``legacy`` come from the single tool REGISTRY now (derived
    # ``NATIVE_FS_TOOLS`` + the ``LEGACY_TOOL_NAMES`` denylist) — no second
    # hand-maintained copy of either surface here.
    used: set[str] = set()
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            used.add(tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", ""))
    leaked = used & LEGACY_TOOL_NAMES
    print("\n=== SURFACE CHECK (main agent only) ===")
    print(f"  tools used: {sorted(used)}")
    print(f"  native fs used: {sorted(used & NATIVE_FS_TOOLS) or 'NONE'}")
    print(f"  legacy pux_sandbox fs/shell leaked: {sorted(leaked) or 'NONE'}")

    # Native-flip proof across the WHOLE tree: every native fs/shell call (the
    # main agent's, AND every subagent's) routes through backend.execute() —
    # incl. the inherited ls/read/glob/grep/write/edit. pux_sandbox_bash is
    # never bound, so ANY entry here is a native call by construction. Empty
    # would mean the tree did no shell/fs work at all (it delegated and the
    # specialist answered from memory only) — for these forcing tasks that's
    # a red flag, not a pass.
    print(f"\n=== NATIVE EXECUTE LOG (whole agent tree, {len(backend.execute_log)} calls) ===")
    for cmd in backend.execute_log:
        one = " ".join(cmd.split())
        print(f"  $ {one[:140]}")
    if not getattr(backend, "execute_log", ()):
        print("  (none — no native fs/shell call was made this run)")
    _write_thread_meta(thread_id, org, task, status="finished")


def _jobs_run(org: str, job: str | None) -> int:
    """Run prep jobs inside the org's sandbox."""
    from pux_harness.sandbox.jobs import run_jobs  # noqa: PLC0415
    from pux_harness.sandbox import policy as policy_mod  # noqa: PLC0415
    from pux_harness.kit._paths import project_root  # noqa: PLC0415

    try:
        pol = policy_mod.load(org, project_root())
    except policy_mod.NoPolicy:
        print(f"{org}: no policy.yaml — no jobs declared")
        return 0

    specs = policy_mod.job_specs(pol)
    if not specs:
        print(f"{org}: no jobs declared")
        return 0

    if job:
        specs = [s for s in specs if s.name == job]
        if not specs:
            print(f"{org}: no job named {job!r}")
            return 1

    print(f"[jobs] {org}: running {len(specs)} job(s)")
    results = run_jobs(pol, shared_backend())

    if job:
        results = [r for r in results if r.name == job]

    for r in results:
        icon = "ok" if r.status == "ok" else "FAIL"
        err = f"  error={r.error[:120]}" if r.error else ""
        print(f"  {r.name:<24} {icon:<6} {r.duration:.1f}s{err}")

    failed = [r for r in results if r.status != "ok"]
    print(f"\n{len(results)} jobs run, {len(failed)} failed")
    return 1 if failed else 0


def _jobs_status(org: str) -> int:
    """Show declared prep jobs for this org."""
    from pux_harness.sandbox import policy as policy_mod  # noqa: PLC0415
    from pux_harness.kit._paths import project_root  # noqa: PLC0415

    try:
        pol = policy_mod.load(org, project_root())
    except policy_mod.NoPolicy:
        print(f"{org}: no policy.yaml — no jobs declared")
        return 0

    specs = policy_mod.job_specs(pol)
    if not specs:
        print(f"{org}: no jobs declared")
        return 0

    print(f"{'NAME':<24} {'SCRIPT':<40} {'TIMEOUT':<8} DESCRIPTION")
    for s in specs:
        timeout = f"{s.timeout}s" if s.timeout else "none"
        print(f"  {s.name:<22} {s.script:<40} {timeout:<8} {s.description}")
    return 0


def _check_policy(org: str) -> int:
    """Resolve + report an org's policy WITHOUT running the model — a dry-run of
    what container-side enforcement will do. Prints expanded mounts,
    credential presence (names only — never values; ``.env`` holds live keys),
    the rendered egress allowlist (DNS resolved now), and tier/image overrides.

    Exits 1 if any required credential is missing — the same gate a real
    container create would enforce, so this is a usable pre-flight."""
    from pux_harness.sandbox import policy
    from pux_harness.kit._paths import project_root

    try:
        p = policy.load(org, project_root())
    except policy.NoPolicy:
        print(
            f"{org}: no policy.yaml — today's behavior "
            "(full egress, default image/tier, no required creds)."
        )
        return 0

    print(f"## {org} policy")
    mounts = policy.resolve_mounts(p)
    if mounts:
        print("workspace.mounts:")
        for m in mounts:
            print(f"  {m.host} -> {m.container} ({m.mode})")
    else:
        print("workspace.mounts: (none)")

    present = [n for n in p.credentials.required if os.environ.get(n, "")]
    missing = [n for n in p.credentials.required if not os.environ.get(n, "")]
    opt_present = [n for n in p.credentials.optional if os.environ.get(n, "")]
    print(f"credentials.required: {p.credentials.required or '(none)'}")
    print(f"  present:  {present or '(none)'}")
    print(f"  MISSING:  {missing or '(none)'}")
    print(f"credentials.optional present: {opt_present or '(none)'}")

    if p.egress.allow:
        try:
            rules = policy.egress_rules(p)
            print("egress.allow (DNS-resolved now):")
            for line in rules.rstrip("\n").split("\n"):
                print(f"  {line}")
        except policy.PolicyError as e:
            print(f"egress.allow: RESOLUTION ERROR — {e}")
            missing = missing or ["<egress-unresolvable>"]  # fail the gate
    else:
        print("egress.allow: (none — full egress)")

    print(f"sandbox.image: {p.sandbox.image or '(default pux-sandbox:latest)'}")
    print(f"sandbox.tier:  {policy.resolve_tier(p, 'isolated')!r} (effective)")

    if p.browser.cookies_env:
        state = "set" if os.environ.get(p.browser.cookies_env, "") else "UNSET"
        print(f"browser.cookies_env: {p.browser.cookies_env} ({state})")
    if p.browser.proxy:
        print(f"browser.proxy: {p.browser.proxy}")

    return 1 if missing else 0


def _sandbox(
    cmd: str,
    output: str | None = None,
    sandbox_id: str | None = None,
    prune_days: int = 14,
    dry_run: bool = False,
) -> int:
    """Sandbox-side CLI. Container lifecycle (start/stop/ensure/pause/unpause/
    status/dump-persist) is now owned by the OpenShell gateway — the sandbox
    auto-starts on first use, so those subcommands print a pointer instead of
    managing a Docker container. The two host-side commands survive:

    * ``projects``       — wayfinding: every project path ever bound.
    * ``prune-sessions`` — bound the inert ``.pux/`` session history.
    """
    from pux_harness.sandbox.exec import resolve_project_path

    project = resolve_project_path()

    if cmd == "projects":
        from pux_harness.sandbox import projects as _projects  # noqa: PLC0415

        rows = _projects.list_projects()
        if not rows:
            print("(no projects recorded yet)")
            return 0
        print(f"  {'last_used (UTC)':<21} {'org':<14} {'sandbox':<14} path")
        for r in rows:
            mark = " " if r["exists"] else "!"
            ts = (r["last_used"] or "")[:19]
            org = (r["org"] or "-")[:14]
            sid = (r["sandbox_id"] or "-")[:14]
            print(f"{mark} {ts:<21} {org:<14} {sid:<14} {r['path']}")
        print(f"\n{len(rows)} project(s).  (! = host path no longer exists)")
        prev = _projects.previous_project(project)
        if prev:
            print("\nmost recent OTHER project:")
            print(f"  {prev['path']}")
            print("  resume with:")
            print(f"    cd {prev['path']} && pux acp")
        return 0

    if cmd == "prune-sessions":
        from pux_harness.sandbox.prune import prune_pux_dir, summarize  # noqa: PLC0415
        from pux_harness.kit._paths import project_root  # noqa: PLC0415

        pux_dir = project_root() / ".pux"
        report = prune_pux_dir(pux_dir, days=prune_days, dry_run=dry_run)
        print(summarize(report))
        if dry_run:
            print("\n(re-run without --dry-run to apply)")
        return 0

    if cmd in {"start", "stop", "ensure", "pause", "unpause", "status", "dump-persist"}:
        print(
            f"`pux sandbox {cmd}` is deprecated — the OpenShell gateway now "
            f"manages sandbox lifecycle (the sandbox auto-starts on first use). "
            f"See ~/.local/share/pux/openshell/README.md."
        )
        return 0

    raise SystemExit(
        f"unknown sandbox subcommand {cmd!r}; use: projects | prune-sessions"
    )


def _check_contract() -> int:
    """Run the declarative org contract — fully offline (no server, no tokens).
    Rule 4 (tool-resolution) resolves against the static native surface (fs
    tools ∪ the specialist registry), so it runs identically in pytest and
    here with nothing live. Per-org audit only: the fleet/AST tripwires live in
    the pytest suite (``tests/harness/tripwire_checks.py``) — optional, never
    deployed."""
    per_org = {org: audit_org(org) for org in discover_orgs()}
    for org in sorted(per_org):
        vs = per_org[org]
        print(f"\n## {org}")
        if vs:
            for x in vs:
                print(f"  {x}")
        else:
            print("  OK")

    n_orgs = len(per_org)
    error_orgs = [o for o, vs in per_org.items() if has_errors(vs)]
    print(f"\n{n_orgs} orgs checked.")
    if error_orgs:
        print(f"BLOCKING errors in: {error_orgs}")
    return 1 if error_orgs else 0


# --- Public API (called from the unified CLI) ---------------------------------


def run_direct(
    org: str = "general",
    task: str = "",
    rubric: str | None = None,
    recursion_limit: int = 60,
    thread: str | None = None,
) -> None:
    """In-process deepagents run for an org + task."""
    if org not in discover_orgs():
        raise SystemExit(f"unknown org {org!r}; discovered: {discover_orgs()}")
    if not task:
        raise SystemExit(
            f"--task is required for --org {org}. "
            "See tests/integration/default_tasks.py for per-org forcing tasks."
        )
    asyncio.run(_run(org, task, recursion_limit, rubric=rubric, thread=thread))


def run_list_orgs() -> None:
    """List discovered orgs and their agents."""
    orgs = discover_orgs()
    print(f"{len(orgs)} orgs:")
    for org in orgs:
        print(f"  {org}: {', '.join(org_agent_slugs(org)) or '(no agents)'}")


def run_sandbox(
    cmd: str,
    output: str | None = None,
    sandbox_id: str | None = None,
    prune_days: int = 14,
    dry_run: bool = False,
) -> None:
    """Docker sandbox lifecycle."""
    raise SystemExit(
        _sandbox(cmd, output=output, sandbox_id=sandbox_id, prune_days=prune_days, dry_run=dry_run)
    )


def run_check_smoke(org: str = "general") -> None:
    """Shared-backend + native specialist smoke test (no model call)."""
    backend = shared_backend()
    specialists = make_specialist_tools(shared_backend(), org=org)
    print(
        f"backend OK: {len(specialists)} native pux_sandbox_* specialists + "
        f"native fs (ls/read_file/write_file/edit_file/glob/grep/execute)"
    )
    ex = backend.execute("echo pux-ok")
    print(f"  backend.execute: exit={ex.exit_code} output={ex.output!r}")
    ls = backend.ls("/sandbox/workspace")
    print(f"  backend.ls: {len(ls.entries or [])} entries, error={ls.error}")


def run_check_contract() -> None:
    """Validate the declarative org contract; exit 1 on error."""
    raise SystemExit(_check_contract())


def run_check_policy(org: str = "general") -> None:
    """Resolve + report this org's policy; exit 1 if required creds missing."""
    raise SystemExit(_check_policy(org))


def run_jobs(org: str, job: str | None = None) -> None:
    """Run prep jobs for this org inside the sandbox."""
    raise SystemExit(_jobs_run(org, job))


def run_jobs_status(org: str) -> None:
    """Show declared prep jobs for this org."""
    raise SystemExit(_jobs_status(org))


def main() -> None:
    """Legacy CLI entry point (argparse). Replaced by ``pux_harness.cli.main``."""
    ap = argparse.ArgumentParser(description="deepagents Pux harness")
    ap.add_argument("--org", default="general", help="org to run (default: general)")
    ap.add_argument(
        "--task",
        help="the objective ONLY — never embed data paths here; use --data for the data folder",
    )
    ap.add_argument(
        "--data",
        default=None,
        metavar="DIR",
        help="data folder for the org to process. Sets DATA_DIR "
        "(absolute) in the agent's environment so the org's "
        "pipeline can preprocess it — the path never enters "
        "the task/prompt. This is the structural input hand-off.",
    )
    ap.add_argument(
        "--rubric",
        default=None,
        help="override the org's shipped rubric (arms the RubricMiddleware "
        "verify-gate for an opted-in org). Default: the org's "
        "profile.yaml `rubric.default`.",
    )
    ap.add_argument(
        "--thread", default=None, help="continue an existing thread id (resume in-process)"
    )
    ap.add_argument("--recursion-limit", type=int, default=60)
    ap.add_argument(
        "--check",
        action="store_true",
        help="docker-exec backend + native specialist smoke, no model call",
    )
    ap.add_argument("--list", action="store_true", help="list discovered orgs + their agents")
    ap.add_argument(
        "--check-contract",
        action="store_true",
        help="validate the declarative org contract; exit 1 on error",
    )
    ap.add_argument(
        "--check-policy",
        action="store_true",
        help="resolve + report this org's policy (mounts/creds/egress/tier); "
        "exit 1 if required creds are missing. No model call.",
    )
    ap.add_argument(
        "--sandbox",
        metavar="CMD",
        help="Docker sandbox lifecycle: start | stop | status | ensure | "
        "projects (harness-owned; replaces `task start/stop/status`). "
        "Default sandbox_id is now derived from the project path — "
        "use --sandbox-id only to override (e.g. two sessions on one project).",
    )
    ap.add_argument(
        "--sandbox-id",
        default=None,
        help="override the per-project sandbox id (container name + persist "
        "volume key). Default: derived from PUX_PROJECT_PATH so different "
        "projects never collide. Set this ONLY when running two concurrent "
        "sessions against the SAME project.",
    )
    ap.add_argument(
        "--prune-days",
        type=int,
        default=14,
        metavar="N",
        help="with --sandbox prune-sessions: retention window in days "
        "(default 14). Entries older than N days are swept. 0 = all.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="with --sandbox prune-sessions: preview what would be swept "
        "without deleting anything.",
    )
    ap.add_argument(
        "--jobs-run", action="store_true", help="run prep jobs for this org inside the sandbox"
    )
    ap.add_argument(
        "--jobs-status", action="store_true", help="show declared prep jobs for this org"
    )
    ap.add_argument("--job", default=None, help="with --jobs-run: run only this named job")
    args = ap.parse_args()

    if args.sandbox is not None:
        run_sandbox(
            args.sandbox,
            sandbox_id=args.sandbox_id,
            prune_days=args.prune_days,
            dry_run=args.dry_run,
        )

    if args.jobs_run:
        run_jobs(args.org, args.job)

    if args.jobs_status:
        run_jobs_status(args.org)

    if args.check_contract:
        run_check_contract()

    if args.list:
        run_list_orgs()
        return

    if args.check_policy:
        run_check_policy(args.org)
        return

    if args.check:
        run_check_smoke(args.org)
        return

    # --data: the structural data-folder hand-off. Sets DATA_DIR in the
    # agent's environment. The org's AGENTS.md references ``$DATA_DIR``
    # — the path NEVER enters the task/prompt string. The objective is
    # --task; the input folder is --data. They are separate concerns.
    #
    # Path form: RELATIVE to the project root when possible. The agent runs
    # inside a Docker sandbox where the project is at /sandbox/workspace,
    # not the host path. A relative path (data/telegram-dump/...) resolves
    # correctly in BOTH contexts (host CWD = repo root, container CWD =
    # /sandbox/workspace). An absolute host path would be invisible inside
    # the container.
    data_dir = getattr(args, "data", None)
    if data_dir:
        abs_path = os.path.abspath(data_dir)
        project_root = os.getcwd()
        try:
            rel = os.path.relpath(abs_path, project_root)
            # If the data is under the project (normal case), use relative.
            # If it's outside (../../tmp/foo), still relative — the container
            # bind-mount may cover it.
            os.environ["DATA_DIR"] = rel
        except ValueError:
            # Cross-drive (Windows) — fall back to absolute.
            os.environ["DATA_DIR"] = abs_path

    run_direct(args.org, args.task, args.rubric, args.recursion_limit, args.thread)


if __name__ == "__main__":
    main()
